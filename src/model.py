import flax
import optax
from flax import linen as nn 
from flax.training import train_state
from typing import Any, Sequence

import jax
import jax.numpy as jnp
from jax import random

import wandb

from src.ctc_objectives import ctc_loss
from fast_ctc_decode import viterbi_search

import chex

sample_input_shape = (64,2000,1)

class CNN_feature_extractor(nn.Module):
    # ToDo: initial parameters were initialized via a uniform distribution with values ranging from -0.08 to 0.08
    # https://flax.readthedocs.io/en/latest/api_reference/flax.linen/initializers.html#flax.linen.initializers.uniform
    def setup(self):
        self.conv1 = nn.Conv(features=4, kernel_size=(5,), strides=1, padding="SAME", use_bias=True)
        self.conv2 = nn.Conv(features=16, kernel_size=(5,), strides=1, padding="SAME", use_bias=True)
        self.conv3 = nn.Conv(features=384, kernel_size=(19,), strides=5, padding="SAME", use_bias=True)
    
    def __call__(self, x):
        x = self.conv1(x)
        x = nn.silu(x)
        x = self.conv2(x)
        x = nn.silu(x)
        x = self.conv3(x)
        x = nn.silu(x)
        return x

class LSTM_encoder(nn.Module):
    # initial parameters were initialized via a uniform distribution with values ranging from -0.08 to 0.08

    def setup(self):
        self.lstm_layer_1 = nn.RNN(nn.LSTMCell(features=384), reverse=True, keep_order=True)
        self.lstm_layer_2 = nn.RNN(nn.LSTMCell(features=384), reverse=False)
        self.lstm_layer_3 = nn.RNN(nn.LSTMCell(features=384), reverse=True, keep_order=True)
        self.lstm_layer_4 = nn.RNN(nn.LSTMCell(features=384), reverse=False)
        self.lstm_layer_5 = nn.RNN(nn.LSTMCell(features=384), reverse=True, keep_order=True)

    def __call__(self, x):
        x = self.lstm_layer_1(x)
        x = self.lstm_layer_2(x)
        x = self.lstm_layer_3(x)
        x = self.lstm_layer_4(x)
        x = self.lstm_layer_5(x)
        return x

class CTC_decoder(nn.Module):
    def setup(self):
        self.linear = nn.Dense(features=5, use_bias=True)
    
    def __call__(self, x):
        x = self.linear(x)
        # x = nn.log_softmax(x, axis=-1)
        # ctc loss function wants logits as input
        return x
    
class CNN_LSTM_CTC(nn.Module):
    def setup(self):
        self.cnn = CNN_feature_extractor()
        self.encoder = LSTM_encoder()
        self.decoder = CTC_decoder()

    def __call__(self, x):
        # x: input of shape (Batch, Length, Features) with Length probably being 2000 and features being 1
        x = self.cnn(x)
        x = self.encoder(x)
        x = self.decoder(x)
        # output should be of shape (Batch, Length, alphabet_size=5)
        return x


def decode(logits, alphabet, qstring = False, qscale = 1.0, qbias = 1.0, collapse_repeats = True):
    """
    inputs:
        logits:
            - has shape (Batch, length, features=5)
        Alphabet:
            - string of length 5, probably "NACGT"
    returns: 
        list of strings:[Batch]
    """
    # ToDo: rewite the loopy implementation in a vmapped or pmapped way, this is not very Jax-thonic
    chex.assert_rank(logits, 3)
    # process the logits to be predictions
    predictions = nn.log_softmax(logits, axis=-1)
    decoded_predictions = []
    for sample_num in range(predictions.shape[0]):
        seq, _path = viterbi_search(predictions[sample_num,:,:], alphabet,  qstring = qstring, qscale = qscale, qbias = qbias, collapse_repeats = collapse_repeats)
        decoded_predictions.append(seq)

@jax.jit
def train_step(state, samples, ground_truth_labels, labels_padding_mask, alphabet, epoch):
    def loss_fn(params):
        # apply can take both a single input or a vector of inputs
        epsilon = 1e-10
        x = jnp.expand_dims(samples, axis=2)
        logits = jnp.clip(state.apply_fn({'params': params}, x), a_min=epsilon, a_max=1-epsilon)
        loss = ctc_loss(logits=logits,
                        logitpaddings=jnp.zeros((logits.shape[0], logits.shape[1])),
                        labels=ground_truth_labels,
                        labelpaddings=labels_padding_mask,
                        blank_id=0)
        mean_batch_loss = jnp.mean(loss)
        return mean_batch_loss, logits
  
    (_mean_batch_loss, logits), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)   # gradient descent is performed here

    # this abstracts away the optimizer function call and optimizer implementation behind the state class api
    state = state.apply_gradients(grads=grads) # update is performed here

    metrics = compute_metrics(logits=logits, 
                              ground_truth_labels=ground_truth_labels, 
                              labels_padding_mask=labels_padding_mask,
                              alphabet=alphabet)
    return state, metrics

@jax.jit
def eval_step(state, samples, ground_truth_labels, labels_padding_mask, alphabet):
    x = jnp.expand_dims(samples, axis=2)
    logits = state.apply_fn({'params': state.params}, x)
    return compute_metrics(logits=logits, 
                           ground_truth_labels=ground_truth_labels, 
                           labels_padding_mask=labels_padding_mask,
                           alphabet=alphabet)

def train_one_epoch(state, train_dataloader, alphabet, epoch):
    """Train for 1 epoch on the training set."""
    batch_metrics = []
    for count, batch in enumerate(train_dataloader):
        features = batch["x"]
        labels = batch["y"]
        labels_padding_mask = jnp.zeros_like(labels)
        labels_padding_mask = labels_padding_mask.at[labels == 0].set(1)
        # labels_padding_mask[labels == 0] = 1
        state, metrics = train_step(state=state, samples=features, ground_truth_labels=labels, 
                                    labels_padding_mask=labels_padding_mask, alphabet=alphabet, 
                                    epoch=epoch)
        batch_metrics.append(metrics)
        print("epoch: ", epoch, ", batch: ", count)
        print(metrics)

    # Aggregate the metrics
    batch_metrics_np = jax.device_get(batch_metrics)  # pull from the accelerator onto host (CPU)
    epoch_metrics_np = {
        k: jnp.mean(jnp.array([metrics[k] for metrics in batch_metrics_np]))
        for k in batch_metrics_np[0]
    }

    # wandb.log({
    #         "Train Loss": epoch_metrics_np['loss'],
    #         "Train Accuracy": epoch_metrics_np['accuracy'],
    #     }, step=epoch)


    return state, epoch_metrics_np


def evaluate_model(state, test_dataloader, alphabet):
    """Evaluate on the validation set."""
    batch_metrics = []
    for num, batch in enumerate(test_dataloader):
        print("batch num: ", num)
        test_features = batch["x"]
        test_labels = batch["y"]

        test_labels_padding_mask = jnp.zeros_like(test_labels)
        test_labels_padding_mask = test_labels_padding_mask.at[test_labels == 0].set(1)
        # test_labels_padding_mask[test_labels == 0] = 1
        metrics = eval_step(state=state, samples=test_features, ground_truth_labels=test_labels, 
                            labels_padding_mask=test_labels_padding_mask, alphabet=alphabet)
        metrics = jax.device_get(metrics)  # pull from the accelerator onto host (CPU)
        metrics = jax.tree_map(lambda x: x.item(), metrics)  # np.ndarray -> scalar
        batch_metrics.append(metrics)
    print("bnatch metrics element type: ", type(batch_metrics[0]))
    # print("shape of elem accuracy: ", batch_metrics[0]["loss"].shape)
    mean_metrics_over_test_set = {}
    mean_metrics_over_test_set["loss"] = jnp.mean(jnp.array([item["loss"] for item in batch_metrics]))
    mean_metrics_over_test_set["accuracy"] = jnp.mean(jnp.array([item["accuracy"] for item in batch_metrics]))
    return mean_metrics_over_test_set


def create_train_state(key, learning_rate):
    model = CNN_LSTM_CTC()
    params = model.init(key, jnp.ones(sample_input_shape))['params']
    
    # ToDo: add optimizer with:
    #   - gradient clipping -2 to 2
    #   - linear learning rate warmup
    #   _ cosine learning rate decay
    adam_opt = optax.adam(learning_rate=learning_rate, b1=0.9, b2=0.999, eps=1e-08, eps_root=0.0, mu_dtype=None)
    # adam_cliped_opt = optax.chain(
    #     optax.clip(1.0),
    #     optax.adam(learning_rate=learning_rate, b1=0.9, b2=0.999, eps=1e-08, eps_root=0.0, mu_dtype=None)
    # )

    # TrainState is a simple built-in wrapper class that makes things a bit cleaner
    return flax.training.train_state.TrainState.create(apply_fn=model.apply, params=params, tx=adam_opt)

def compute_metrics(*, logits, ground_truth_labels, labels_padding_mask, alphabet):
    per_sequence_loss = ctc_loss(logits=logits,
                        logitpaddings=jnp.zeros((logits.shape[0], logits.shape[1])),
                        labels=ground_truth_labels,
                        labelpaddings=labels_padding_mask,
                        blank_id=0)
    mean_batch_loss = jnp.mean(per_sequence_loss)
    # ToDo: compute accuracy as well as loss
    list_decoded_predictions = decode(logits=logits, alphabet=alphabet)
    metrics = {
        'loss': mean_batch_loss,
        'accuracy': 0.5,
    }
    return metrics




# class MLP(nn.Module):
#     num_features: Sequence[int]

#     def setup(self):
#         self.layers = [nn.Dense(features=num, kernel_init=flax.linen.initializers.glorot_normal()) for num in self.num_features]
#         # self.layers = [nn.Dense(features=num) for num in self.num_features]
    
#     def __call__(self, x):
#         activation = x
#         for i, layer in enumerate(self.layers):
#             activation = layer(activation)
#             if i != len(self.layers) - 1:
#                 # activation = nn.relu(activation)
#                 activation = nn.leaky_relu(activation)
#         # return nn.sigmoid(activation)
#         return nn.softmax(activation)


# @jax.jit
# def train_step(state, samples, ground_truth_labels, epoch):
#     def loss_fn(params):
#         # apply can take both a single input or a vector of inputs
#         # predictions = MLP().apply({'params': params}, samples)
#         epsilon = 1e-10
#         predictions = jnp.clip(state.apply_fn({'params': params}, samples), a_min=epsilon, a_max=1-epsilon)
#         loss = -jnp.mean(jnp.sum(nn.one_hot(ground_truth_labels, num_classes=2) * jnp.log(predictions), axis=-1))
#         return loss, predictions
  
#     (_, predictions), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)   # gradient descent is performed here

#     # this abstracts away the optimizer function call and iptimizes implementation behind the state class api
#     state = state.apply_gradients(grads=grads) # update is performed here

#     metrics = compute_metrics(predictions=predictions, ground_truth_labels=ground_truth_labels)
#     return state, metrics

# @jax.jit
# def eval_step(state, samples, ground_truth_labels):
#     # predictions = MLP().apply({'params': state.params}, samples)
#     predictions = state.apply_fn({'params': state.params}, samples)
#     return compute_metrics(predictions=predictions, ground_truth_labels=ground_truth_labels)


# def train_one_epoch(state, dataloader, epoch):
#     """Train for 1 epoch on the training set."""
#     batch_metrics = []
#     for count, (features, labels) in enumerate(dataloader):
#         state, metrics = train_step(state, features, labels, epoch)
#         batch_metrics.append(metrics)

#     # Aggregate the metrics
#     batch_metrics_np = jax.device_get(batch_metrics)  # pull from the accelerator onto host (CPU)
#     epoch_metrics_np = {
#         k: jnp.mean(jnp.array([metrics[k] for metrics in batch_metrics_np]))
#         for k in batch_metrics_np[0]
#     }

#     wandb.log({
#             "Train Loss": epoch_metrics_np['loss'],
#             "Train Accuracy": epoch_metrics_np['accuracy'],
#         }, step=epoch)


#     return state, epoch_metrics_np

# def evaluate_model(state, test_samples, test_labels):
#     """Evaluate on the validation set."""
#     metrics = eval_step(state, test_samples, test_labels)
#     metrics = jax.device_get(metrics)  # pull from the accelerator onto host (CPU)
#     metrics = jax.tree_map(lambda x: x.item(), metrics)  # np.ndarray -> scalar
#     return metrics


# def create_train_state(key, num_features, learning_rate, momentum):
#     mlp = MLP(num_features=num_features)
#     # params = mlp.init(key, jnp.ones([1, input_feature_size]))['params']
#     params = mlp.init(key, jnp.ones([input_feature_size]))['params']
#     # print("========params at initialization==========")
#     # print(params)
#     # sgd_opt = optax.sgd(learning_rate, momentum)
#     adam_opt = optax.adam(learning_rate=learning_rate, b1=0.9, b2=0.999, eps=1e-08, eps_root=0.0, mu_dtype=None)
#     # adam_opt = optax.adam(learning_rate=learning_rate, b1=0.9, b2=0.999, eps=1e-03, eps_root=0.0, mu_dtype=None)
#     # adam_cliped_opt = optax.chain(
#     #     optax.clip(1.0),
#     #     optax.adam(learning_rate=learning_rate, b1=0.9, b2=0.999, eps=1e-08, eps_root=0.0, mu_dtype=None)
#     # )
#     # adagrad_opt = optax.adagrad(learning_rate=learning_rate, initial_accumulator_value=0.1, eps=1e-07)
#     # lion_opt = optax.lion(learning_rate=learning_rate)
#     # rms_opt = optax.rmsprop(learning_rate=learning_rate, momentum=momentum)
#     # TrainState is a simple built-in wrapper class that makes things a bit cleaner
#     return flax.training.train_state.TrainState.create(apply_fn=mlp.apply, params=params, tx=adam_opt)

# def compute_metrics(*, predictions, ground_truth_labels):
#     # loss = jnp.mean(-ground_truth_labels*jnp.log(predictions) - (1-ground_truth_labels)*jnp.log(1-predictions))
#     loss = -jnp.mean(jnp.sum(nn.one_hot(ground_truth_labels, num_classes=2) * jnp.log(predictions), axis=-1))
#     # accuracy = jnp.mean(jnp.round(predictions) == ground_truth_labels)
#     accuracy = jnp.mean(jnp.argmax(predictions, axis=-1) == ground_truth_labels)

#     metrics = {
#         'loss': loss,
#         'accuracy': accuracy,
#     }
#     return metrics


# def batch_generator(feature_matrix, label_matrix, batch_size):
#     assert feature_matrix.shape[0] == label_matrix.shape[0], "feature matrix and label matrix should have the same number of items"
#     num_samples = feature_matrix.shape[0]
#     num_batches = num_samples // batch_size

#     for i in range(num_batches):
#         start = i * batch_size
#         end = (i + 1) * batch_size
#         features_batch = feature_matrix[start:end, :]
#         labels_batch = label_matrix[start:end]
#         yield features_batch, labels_batch

#     # Yield the remaining samples in a smaller batch if any
#     if num_samples % batch_size != 0:
#         start = num_batches * batch_size
#         features_batch = feature_matrix[start:, :]
#         labels_batch = label_matrix[start:]
#         yield features_batch, labels_batch


# seed = 0  # needless to say these should be in a config or defined like flags
# learning_rate = .01
# momentum = 0.9
# num_epochs = 30
# # batch_size = 3578
# batch_size = 32
# num_classes = 2
# # num_features = [100,100,num_classes]
# num_features = [100,100,100,num_classes]


# start a new wandb run to track this script
# wandb.init(
#     # set the wandb project where this run will be logged
#     project="CS271_Midterm_202301101",
    
#     # track hyperparameters and run metadata
#     config={
#     "learning_rate": learning_rate,
#     "architecture": "MLP",
#     "dataset": "Buisness usecase",
#     "batch_size": batch_size,
#     "epochs": num_epochs,
#     "num_features": num_features,
#     "optimizer": "adam",
#     "activation":"leaky_relu",
#     "initialization":"XG"
#     }
# )


# train_state = create_train_state(key=jax.random.PRNGKey(seed), num_features=num_features, learning_rate=learning_rate, momentum=momentum)

# for epoch in range(1, num_epochs + 1):
#     train_state, train_metrics, flag = train_one_epoch(train_state, batch_generator(train_inputs, train_targets, batch_size), epoch)
#     if flag:
#         break
#     print(f"Train epoch: {epoch}, loss: {train_metrics['loss']}, accuracy: {train_metrics['accuracy'] * 100}")

#     test_metrics = evaluate_model(train_state, test_inputs, test_targets)
#     print(f"\tTest epoch: {epoch}, loss: {test_metrics['loss']}, accuracy: {test_metrics['accuracy'] * 100}")


# wandb.finish()