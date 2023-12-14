from src.nanopore_dataset import BaseNanoporeDataset, custom_collate
from torch.utils.data import DataLoader
import jax
from jax import numpy as jnp
from flax.training import checkpoints
import time
import wandb

NON_RECURRENT_ENCODING_DICT = {'A':1, 'C':2, 'G':3, 'T':4, '':0}
NON_RECURRENT_DECODING_DICT = {1:'A', 2:'C', 3:'G', 4:'T', 0:''}
BASES = ['A', 'C', 'G', 'T']
BASES_CTC = 'N' + ''.join(BASES)

data_dir = "/home/townsag/nanopore_project/test_nn_input"
#data_dir = "/home/townsag/nanopore_project/SimpleCall/practice_data/training_input"
checkpoint_dir = "/home/townsag/nanopore_project/SimpleCall/checkpoints"
batch_size = 64
s2s = False

dataset = BaseNanoporeDataset(
    data_dir = data_dir, 
    decoding_dict = NON_RECURRENT_DECODING_DICT, 
    encoding_dict = NON_RECURRENT_ENCODING_DICT, 
    split = 0.95, 
    shuffle = True, 
    seed = 1,
    s2s = s2s,
)
print(f"total num samples in dataset: {len(dataset)}")

dataloader_train = DataLoader(
    dataset, 
    batch_size = batch_size, 
    sampler = dataset.train_sampler,
    collate_fn=custom_collate,
    num_workers = 1
)
dataloader_validation = DataLoader(
    dataset, 
    batch_size = batch_size, 
    sampler = dataset.validation_sampler,
    collate_fn=custom_collate,
    num_workers = 1
)
print(f"num train batches {len(dataloader_train)}")
print(f"num test batches{len(dataloader_validation)}")

seed = 0
learning_rate = 1e-3
min_learning_rate = 1e-5
num_epochs = 5
linear_warmup_steps = 10

from ml_collections import config_dict
initial_dictionary = {
    'linear_warmup_steps': linear_warmup_steps,
    'num_batches_per_epoch': len(dataloader_train),
    'num_epochs': num_epochs,
    'base_learning_rate': learning_rate,
    'min_learning_rate': min_learning_rate
}
cfg = config_dict.ConfigDict(initial_dictionary)

wandb.init(
    # set the wandb project where this run will be logged
    project="SimpleCall-20231213",
    
    # track hyperparameters and run metadata
    config={
        "base_learning_rate": learning_rate,
        "min_learning_rate": min_learning_rate,
        "architecture": "CNN-LSTM-CTC",
        "dataset": "demo_data",
        "batch_size": batch_size,
        "epochs": num_epochs,
        "optimizer": "adam with linear warmup and cosine decay",
        "num_linear_warmup_steps": linear_warmup_steps,
        "batches_per_epoch": len(dataloader_train)
    }
)


from src.model import create_train_state, train_one_epoch, evaluate_model
from jax import device_put

print("initializing train_state")
start = time.time()
train_state = create_train_state(key=jax.random.PRNGKey(seed), config=cfg)
end = time.time()
print("time to create train state: ", end - start)

print("starting training")
for epoch in range(1, num_epochs + 1):
    start = time.time()
    train_state, train_metrics = train_one_epoch(state=train_state, train_dataloader=dataloader_train, 
            alphabet=BASES_CTC, epoch=epoch, batches_per_epoch=cfg.num_batches_per_epoch, checkpoint_dir=checkpoint_dir)
    end = time.time()
    print(f"========== Total Epoch Train Time: {end - start} ==========")
    print(f"Train epoch: {epoch}, loss: {train_metrics['mean_epoch_loss']}, accuracy: {train_metrics['mean_epoch_accuracy'] * 100}")


    start = time.time()
    print(f"Starting evaluation for epoch: {epoch}")
    test_metrics = evaluate_model(state=train_state, test_dataloader=dataloader_validation, alphabet=BASES_CTC, epoch=epoch)
    print(f"\Validation epoch: {epoch}, loss: {test_metrics['loss']}, accuracy: {test_metrics['accuracy'] * 100}")
    end = time.time()
    print(f"========== Total Epoch Test Time: {end - start} ==========")

    checkpoints.save_checkpoint(ckpt_dir=checkpoint_dir, target=train_state.params, step=train_state.step, keep=3)

wandb.finish()
"""
first_batch = next(iter(dataloader_train))
features = device_put(first_batch["x"])
labels = device_put(first_batch["y"])
labels_padding_mask = jnp.zeros_like(labels)
labels_padding_mask = labels_padding_mask.at[labels == 0].set(1)

print(f"features size: {features.shape}")
print(f"labels size: {labels.shape}")

from src.model import train_step

print("features type and device: ", type(features), features.devices())
print("labels type and device: ", type(labels), labels.devices())
print("padding type and device: ", type(labels_padding_mask), labels_padding_mask.devices())


start = time.time()
state, batch_logits, mean_batch_loss = train_step(state=train_state, samples=features, ground_truth_labels=labels,labels_padding_mask=labels_padding_mask, batch=1)
end = time.time()

print("time to train one batch: ", end - start)
"""
