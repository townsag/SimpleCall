import flax
import optax
from flax import linen as nn 
from flax.training import train_state
from typing import Any, Sequence

import jax
import jax.numpy as jnp
from jax import random
import numpy as np

import wandb

from src.ctc_objectives import ctc_loss
from fast_ctc_decode import viterbi_search
from scripts.evaluation import alignment_accuracy

import chex

sample_input_shape = (64,2000,1)
NON_RECURRENT_ENCODING_DICT = {'A':1, 'C':2, 'G':3, 'T':4, '':0}
NON_RECURRENT_DECODING_DICT = {1:'A', 2:'C', 3:'G', 4:'T', 0:''}
PADDING_NUM = 0

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

@jax.jit
def train_step(state, samples, ground_truth_labels, labels_padding_mask, batch):
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
  
    (mean_batch_loss, logits), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)   # gradient descent is performed here

    # this abstracts away the optimizer function call and optimizer implementation behind the state class api
    state = state.apply_gradients(grads=grads) # update is performed here
    return state, logits, mean_batch_loss

@jax.jit
def eval_step(state, samples, ground_truth_labels, labels_padding_mask):
    x = jnp.expand_dims(samples, axis=2)
    logits = state.apply_fn({'params': state.params}, x)
    per_seq_loss = ctc_loss(logits=logits,
                            logitpaddings=jnp.zeros((logits.shape[0], logits.shape[1])),
                            labels=ground_truth_labels,
                            labelpaddings=labels_padding_mask,
                            blank_id=0)
    mean_batch_loss = jnp.mean(per_seq_loss)
    return logits, mean_batch_loss

def decode_batch(logits, alphabet, qstring = False, qscale = 1.0, qbias = 1.0, collapse_repeats = True):
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
    # this is questionable, I am not sure if jax can vmap over non pure python code
    chex.assert_rank(logits, 3)
    # process the logits to be predictions
    predictions = nn.log_softmax(logits, axis=-1)
    decoded_predictions = []
    for sample_num in range(predictions.shape[0]):
        seq, _path = viterbi_search(np.array(predictions[sample_num,:,:]), alphabet,  qstring = qstring, qscale = qscale, qbias = qbias, collapse_repeats = collapse_repeats)
        decoded_predictions.append(seq)
    return decoded_predictions

def accuracy_batch(logits, ground_truth_labels, alphabet):
    decoded_predictions = decode_batch(logits=logits, alphabet=alphabet)
    # decode labels
    ground_truth_labels = ground_truth_labels.astype(str)
    ground_truth_labels[ground_truth_labels == str(PADDING_NUM)] = ""
    for key, value in NON_RECURRENT_DECODING_DICT.items():
        ground_truth_labels[ground_truth_labels == str(key)] = value
    ground_truth_labels_str_list = ["".join(sample) for sample in ground_truth_labels.tolist()]
    # compute the accuracy
    assert len(decoded_predictions) == len(ground_truth_labels_str_list), \
        f"there should be as many label sequences: {len(ground_truth_labels_str_list)} as there are prediction sequences: {len(decoded_predictions)}"
    per_sequence_accuracies = list()
    for label_seq, pred_seq in zip(ground_truth_labels_str_list, decoded_predictions):
        per_sequence_accuracies.append(alignment_accuracy(label_seq, pred_seq))
    
    return jnp.mean(jnp.array(per_sequence_accuracies))

def train_one_epoch(state, train_dataloader, alphabet, epoch):
    """Train for 1 epoch on the training set."""
    batch_metrics = []
    for count, batch in enumerate(train_dataloader):
        features = batch["x"]
        labels = batch["y"]
        labels_padding_mask = jnp.zeros_like(labels)
        labels_padding_mask = labels_padding_mask.at[labels == 0].set(1)
        # labels_padding_mask[labels == 0] = 1
        state, batch_logits, mean_batch_loss = train_step(state=state, samples=features, ground_truth_labels=labels, 
                                    labels_padding_mask=labels_padding_mask, batch=count)
        mean_batch_accuracy = accuracy_batch(logits=batch_logits, ground_truth_labels=labels, alphabet=alphabet)

        batch_metrics.append({"loss": mean_batch_loss, "accuracy": mean_batch_accuracy})
        print("epoch: ", epoch, ", batch: ", count, ", mean_batch_accuracy: ", mean_batch_accuracy, "mean_batch_loss: ", mean_batch_loss)
        # print(metrics)

    # Aggregate the metrics over the epoch
    batch_metrics = jax.device_get(batch_metrics)  # pull from the accelerator onto host (CPU)
    epoch_metrics = {
        "mean_epoch_loss": jnp.mean(jnp.array([item["loss"] for item in batch_metrics])),
        "mean_epoch_accuracy": jnp.mean(jnp.array([item["accuracy"] for item in batch_metrics]))
    }

    # wandb.log({
    #         "Train Loss": epoch_metrics_np['loss'],
    #         "Train Accuracy": epoch_metrics_np['accuracy'],
    #     }, step=epoch)


    return state, epoch_metrics


def evaluate_model(state, test_dataloader, alphabet):
    """Evaluate on the validation set."""
    batch_metrics = []
    for num, batch in enumerate(test_dataloader):
        print("batch num: ", num)
        test_features = batch["x"]
        test_labels = batch["y"]

        test_labels_padding_mask = jnp.zeros_like(test_labels)
        test_labels_padding_mask = test_labels_padding_mask.at[test_labels == 0].set(1)
        batch_logits, mean_batch_loss = eval_step(state=state, samples=test_features, ground_truth_labels=test_labels, 
                            labels_padding_mask=test_labels_padding_mask)
        mean_batch_accuracy = accuracy_batch(logits=batch_logits, ground_truth_labels=test_labels, alphabet=alphabet)
        mean_batch_loss = jax.device_get(mean_batch_loss)  # pull from the accelerator onto host (CPU)
        mean_batch_accuracy = jax.device_get(mean_batch_accuracy)
        batch_metrics.append({"loss": mean_batch_loss, "accuracy": mean_batch_accuracy})
    # print("bnatch metrics element type: ", type(batch_metrics[0]))
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

# def compute_metrics(*, logits, ground_truth_labels, labels_padding_mask, alphabet):
#     per_sequence_loss = ctc_loss(logits=logits,
#                         logitpaddings=jnp.zeros((logits.shape[0], logits.shape[1])),
#                         labels=ground_truth_labels,
#                         labelpaddings=labels_padding_mask,
#                         blank_id=0)
#     mean_batch_loss = jnp.mean(per_sequence_loss)
#     list_decoded_predictions = decode_batch(logits=logits, alphabet=alphabet)
#     metrics = {
#         'loss': mean_batch_loss,
#         'accuracy': 0.5,
#     }
#     return metrics

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