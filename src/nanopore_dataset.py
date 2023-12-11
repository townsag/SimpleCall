# copied from /basecalling_architectures
import numpy as np
import zipfile
import os
from torch.utils.data import Dataset, Sampler, DataLoader
import random

S2S_PAD = 0
S2S_SOS = 1
S2S_EOS = 2

def read_metadata(file_name):
    """Read the metadata of a npz file
    
    Args:
        filename (str): .npz file that we want to read the metadata from
        
    Returns:
        (list): with as many items as arrays in the file, each item in the list
        is filename (within the zip), shape, fortran order, dtype
    """
    zip_file=zipfile.ZipFile(file_name, mode='r')
    arr_names=zip_file.namelist()

    metadata=[]
    for arr_name in arr_names:
        fp=zip_file.open(arr_name,"r")
        version=np.lib.format.read_magic(fp)

        if version[0]==1:
            shape,fortran_order,dtype=np.lib.format.read_array_header_1_0(fp)
        elif version[0]==2:
            shape,fortran_order,dtype=np.lib.format.read_array_header_2_0(fp)
        else:
            print("File format not detected!")
        metadata.append((arr_name,shape,fortran_order,dtype))
        fp.close()
    zip_file.close()
    return metadata

class BaseNanoporeDataset(Dataset):
    """Base dataset class that contains Nanopore data
    
    The simplest class that handles a hdf5 file that has two datasets
    named 'x' and 'y'. The first one contains an array of floats with
    the raw data normalized. The second one contains an array of 
    byte-strings with the bases appended with ''.
    
    This dataset already takes case of shuffling, for the dataloader set
    shuffling to False.
    
    Args:
        data (str): dir with the npz files
        decoding_dict (dict): dictionary that maps integers to bases
        encoding_dict (dict): dictionary that maps bases to integers
        split (float): fraction of samples for training
        randomizer (bool): whether to randomize the samples order
        seed (int): seed for reproducible randomization
        s2s (bool): whether to encode for s2s models
        token_sos (int): value used for encoding start of sequence
        token_eos (int): value used for encoding end of sequence
        token_pad (int): value used for padding all the sequences (s2s and not s2s)
    """

    def __init__(self, data_dir, decoding_dict, encoding_dict, 
                 split = 0.95, shuffle = True, seed = None,
                 s2s = False, token_sos = S2S_SOS, token_eos = S2S_EOS, token_pad = S2S_PAD, transform=None):
        super(BaseNanoporeDataset, self).__init__()
        
        self.data_dir = data_dir
        self.decoding_dict = decoding_dict
        self.encoding_dict = encoding_dict
        self.split = split
        self.shuffle = shuffle
        self.seed = seed
        
        self.files_list = self._find_files()
        self.num_samples_per_file = self._get_samples_per_file()
        self.total_num_samples = np.sum(np.array(self.num_samples_per_file))
        self.train_files_idxs = set()
        self.validation_files_idxs = set()
        self.train_idxs = list()
        self.validation_idxs = list()
        self.train_sampler = None
        self.validation_sampler = None
        self._split_train_validation()
        self._get_samplers()
        
        self.loaded_train_data = None
        self.loaded_validation_data = None
        self.current_loaded_train_idx = None
        self.current_loaded_validation_idx = None

        self.s2s = s2s
        self.token_sos = token_sos
        self.token_eos = token_eos
        self.token_pad = token_pad

        self.transform = transform

        self._check()
    
    def __len__(self):
        """Number of samples
        """
        return self.total_num_samples
        
    def __getitem__(self, idx):
        """Get a set of samples by idx
        
        If the datafile is not loaded it loads it, otherwise
        it uses the already in memory data.
        
        Returns a dictionary
        """
        if idx[0] in self.train_files_idxs:
            if idx[0] != self.current_loaded_train_idx:
                self.loaded_train_data = self.load_file_into_memory(idx[0])
                self.current_loaded_train_idx = idx[0]
            # if self.transform:
            #     return self.transform(self.get_data(data_dict = self.loaded_train_data, idx = idx[1]))
            # else:
            return self.get_data(data_dict = self.loaded_train_data, idx = idx[1])
            
        elif idx[0] in self.validation_files_idxs:
            if idx[0] != self.current_loaded_validation_idx:
                self.loaded_validation_data = self.load_file_into_memory(idx[0])
                self.current_loaded_validation_idx = idx[0]
            # if self.transform:
            #     return self.transform(self.get_data(data_dict = self.loaded_validation_data, idx = idx[1]))
            # else:
            return self.get_data(data_dict = self.loaded_validation_data, idx = idx[1])
        else:
            raise IndexError('Given index not in train or validation files indices: ' + str(idx[0]))
    
    def _check(self):
        """Check for possible problems
        """

        # check that the encoding dict does not conflict with S2S tokens
        if self.s2s:
            s2s_tokens = (self.token_eos, self.token_sos, self.token_pad)
            for v in self.encoding_dict.values():
                assert v not in s2s_tokens

    def _find_files(self):
        """Finds list of files to read
        """
        l = list()
        for f in os.listdir(self.data_dir):
            if f.endswith('.npz'):
                l.append(f)
        l = sorted(l)
        return l
    
    def _get_samples_per_file(self):
        """Gets the number of samples per file from the file name
        """
        l = list()
        for f in self.files_list:
            metadata = read_metadata(os.path.join(self.data_dir, f))
            l.append(metadata[0][1][0]) # [array_num, shape, first elem shape]
        return l
    
    def _split_train_validation(self):
        """Splits datafiles and idx for train and validation according to split
        """
        
        # split train and validation data based on files
        num_train_files = int(len(self.files_list) * self.split)
        num_validation_files = len(self.files_list) - num_train_files
        print("num train: ", num_train_files)
        print("num valid: ", num_validation_files)
        
        files_idxs = list(range(len(self.files_list)))
        if self.shuffle:
            if self.seed:
                random.seed(self.seed)
            random.shuffle(files_idxs)
            
        self.train_files_idxs = set(files_idxs[:num_train_files])
        self.validation_files_idxs = set(files_idxs[num_train_files:])
        
        # shuffle indices within each file and make a list of indices (file_idx, sample_idx)
        # as tuples that can be iterated by the sampler
        for idx in self.train_files_idxs:
            sample_idxs = list(range(self.num_samples_per_file[idx]))
            if self.shuffle:
                if self.seed:
                    random.seed(self.seed)
                random.shuffle(sample_idxs)
            for i in sample_idxs:
                self.train_idxs.append((idx, i))
        
        for idx in self.validation_files_idxs:
            sample_idxs = list(range(self.num_samples_per_file[idx]))
            if self.shuffle:
                if self.seed:
                    random.seed(self.seed)
                random.shuffle(sample_idxs)
            for i in sample_idxs:
                self.validation_idxs.append((idx, i))
                
        return None
    
    def _get_samplers(self):
        """Add samplers
        """
        self.train_sampler = IdxSampler(self.train_idxs, data_source = self)
        self.validation_sampler = IdxSampler(self.validation_idxs, data_source = self)
        return None
            
    def load_file_into_memory(self, idx):
        """Loads a file into memory and processes it
        """
        arr = np.load(os.path.join(self.data_dir, self.files_list[idx]))
        x = arr['x']
        y = arr['y']
        return self.process({'x':x, 'y':y})
    
    def get_data(self, data_dict, idx):
        """Slices the data for given indices
        """
        return {'x': data_dict['x'][idx], 'y': data_dict['y'][idx]}
    
    def process(self, data_dict):
        """Processes the data into a ready for training format
        """
        
        y = data_dict['y']
        if y.dtype != 'U1':
            y = y.astype('U1')
        if self.s2s:
            y = self.encode_s2s(y)
        else:
            y = self.encode(y)
        data_dict['y'] = y
        return data_dict
    
    def encode(self, y_arr):
        """Encode the labels
        """
        
        new_y = np.full(y_arr.shape, self.token_pad, dtype=int)
        for k, v in self.encoding_dict.items():
            new_y[y_arr == k] = v
        return new_y

    def encode_s2s(self, y_arr):
    
        new_y = np.full(y_arr.shape, self.token_pad, dtype=int)
        # get the length of each sample to add eos token at the end
        sample_len = np.sum(y_arr != '', axis = 1)
        # array with sos_token to append at the begining
        sos_token = np.full((y_arr.shape[0], 1), self.token_sos, dtype=int)
        # replace strings for integers according to encoding dict
        for k, v in self.encoding_dict.items():
            if v is None:
                continue
            new_y[y_arr == k] = v
        # replace first padding for eos token
        for i, s in enumerate(sample_len):
            new_y[i, s] = self.token_eos
        # add sos token and slice of last padding to keep same shape
        new_y = np.concatenate([sos_token, new_y[:, :-1]], axis = 1)
        return new_y
    
    def encoded_array_to_list_strings(self, y):
        """Convert an encoded array back to a list of strings

        Args:
            y (array): with shape [batch, len]
        """

        y = y.astype(str)
        if self.s2s:
            # replace tokens with nothing
            for k in [self.token_sos, self.token_eos, self.token_pad]:
                y[y == str(k)] = ''
        else:
            y[y == str(self.token_pad)] = ''
        # replace predictions with bases
        for k, v in self.decoding_dict.items():
            y[y == str(k)] = v

        # join everything
        decoded_sequences = ["".join(i) for i in y.tolist()]
        return decoded_sequences


class IdxSampler(Sampler):
    """Sampler class to not sample from all the samples
    from a dataset.
    """
    def __init__(self, idxs, *args, **kwargs):
        super(IdxSampler, self).__init__(*args, **kwargs)
        self.idxs = idxs

    def __iter__(self):
        return iter(self.idxs)

    def __len__(self):
        return len(self.idxs)
    
def custom_collate(batch):
    """
    A custom collate function that stacks NumPy arrays into a batch.
    This gets around the issue of the default collate function converting the
    numpy arrays into torch tensors
    """
    return {"x":np.stack([item["x"] for item in batch]),
            "y":np.stack([item["y"] for item in batch])}