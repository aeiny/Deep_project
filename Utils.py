import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt
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

# def plotSignal():
#     loader = torch.load('linear_with_relu_with_STA_uniform_-1_1_E^sinx_amp-2_freq-2_uniform_-2_2_E^sinx_amp-2_freq-2_test.pth')
#     signal = loader['data'][2][0]
#     clean_signal = loader['data'][0][0]
#     plt.plot(signal.squeeze().cpu())
#     plt.plot(clean_signal.squeeze().cpu())
#     plt.show()

# def plotSig(fileName):
#     loader = np.load(fileName)
#     train_clean_signal, train_noisy_signal, train_Pxx_dens = loader['arr_1'],  loader['arr_3'], loader['arr_4']
#     plt.plot(train_clean_signal[0].squeeze())
#     # plt.plot(train_noisy_signal[0].squeeze())
#     plt.show()

# if __name__ == '__main__':
#     plotSig('val_data_signal__sinX__amp_2__freq_2__noise_normal__mu_0__sigma_1.npz')
