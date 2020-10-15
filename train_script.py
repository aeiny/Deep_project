import torch
import sys
nonlinearities=True
gets_spectral_input=True
sta_enabled=True
batch_size=128
n_epochs=5000
lr=0.001
run_name = 'linear_with_relu_with_STA_uniform_-1_1_E^sinx_amp-2_freq-2'

from Models import Encoder
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
enc = Encoder(nonlinearities=nonlinearities, gets_spectral_input=gets_spectral_input, sta_enabled=sta_enabled)
enc = enc.to(device)

print(f"model length: {len(list(enc.parameters()))}")

# enc.load_state_dict(torch.load('checkpoints/linear_without_relu_ckpt_4800.pth')['model'])

from Train_sta import train
from Utils import SignalsDataset
from torch.utils.data import DataLoader
from torch.optim import SGD
from torch.optim import Adam

ds_train = SignalsDataset('train_data_signal__EsinX^2__amp_2__freq_2__noise_uniform__low_-2__high_2.npz')
dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True, drop_last=True)
ds_val = SignalsDataset('val_data_signal__EsinX^2__amp_2__freq_2__noise_uniform__low_-2__high_2.npz')
dl_val = DataLoader(ds_val, batch_size=batch_size, shuffle=True, drop_last=True)
ds_test = SignalsDataset('test_data_signal__EsinX^2__amp_2__freq_2__noise_uniform__low_-2__high_2.npz')
dl_test = DataLoader(ds_test, batch_size=batch_size, shuffle=True, drop_last=True)
optimizer = SGD(enc.parameters(), lr=lr)
loss_func = torch.nn.MSELoss()

print(f"starting run: {run_name}")
train(enc, run_name, dl_train, dl_val, dl_test, n_epochs, optimizer, loss_func, print_every=1, checkpoint_every=50, with_freq=gets_spectral_input)