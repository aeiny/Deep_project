import numpy as np
import torch
from torch.utils.data import IterableDataset, Dataset


class SignalsDataset(Dataset):

    def __init__(self, fileName):
        loader = np.load(fileName)
        train_clean_signal, train_noisy_signal, train_Pxx_dens = loader['arr_1'],  loader['arr_3'], loader['arr_4']
        train_clean_signal = torch.FloatTensor(train_clean_signal)
        train_noisy_signal = torch.FloatTensor(train_noisy_signal)
        train_Pxx_dens = torch.FloatTensor(train_Pxx_dens)
        self.data = list(zip(train_clean_signal, train_noisy_signal, train_Pxx_dens))

    def __getitem__(self, item):
        return self.data[item]

    def __iter__(self):
        return iter(self.data)

    def __len__(self):
        return len(self.data)