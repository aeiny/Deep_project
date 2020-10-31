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

def plotSignal():
    loader = torch.load('Train_sig-5cos0.2x_noise-Uni(-1,1)__Test_sig-5cos0.2x_noise-Uni(-1,1)___nonlinearities-True_gets_spectral_input-True_sta_enabled-True_test.pth')
    signal = loader['data'][2][0]
    noise = loader['data'][1][0]
    clean_signal = loader['data'][0][0]
    plt.plot(clean_signal.squeeze().cpu(),  label='The clean signal')
    plt.plot(noise.squeeze().cpu(),  label='The noisy signal')
    plt.plot(signal.squeeze().cpu(),  label='Result with STA')
    plt.legend()
    plt.show()
    
    loader = torch.load('Train_sig-5cos0.2x_noise-Uni(-1,1)__Test_sig-5cos0.2x_noise-Uni(-1,1)___nonlinearities-False_gets_spectral_input-False_sta_enabled-False_test.pth')
    signal = loader['data'][2][0]
    clean_signal = loader['data'][0][0]
    plt.plot(clean_signal.squeeze().cpu(),  label='The clean signal')
    plt.plot(noise.squeeze().cpu(),  label='The noisy signal')
    plt.plot(signal.squeeze().cpu(),  label='Result with STA')
    plt.legend()
    plt.show()

def plotSig(fileName):
    loader = np.load(fileName)
    train_clean_signal, train_noisy_signal, train_Pxx_dens = loader['arr_1'],  loader['arr_3'], loader['arr_4']
    plt.plot(train_clean_signal[0].squeeze())
    # plt.plot(train_noisy_signal[0].squeeze())
    plt.show()

if __name__ == '__main__':
    plotSignal()