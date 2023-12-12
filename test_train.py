from src.nanopore_dataset import BaseNanoporeDataset, custom_collate
from torch.utils.data import DataLoader
import jax
from jax import numpy as jnp

NON_RECURRENT_ENCODING_DICT = {'A':1, 'C':2, 'G':3, 'T':4, '':0}
NON_RECURRENT_DECODING_DICT = {1:'A', 2:'C', 3:'G', 4:'T', 0:''}
BASES = ['A', 'C', 'G', 'T']
BASES_CTC = 'N' + ''.join(BASES)

data_dir = "/home/atownsend/nanopore_project/SimpleCall/practice_data/training_input"
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
num_epochs = 5

from src.model import create_train_state, train_one_epoch, evaluate_model

train_state = create_train_state(key=jax.random.PRNGKey(seed), learning_rate=learning_rate)
