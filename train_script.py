import torch
import sys


nonlinearities=False
gets_spectral_input=False
sta_enabled=False
n_epochs=5000

sig_name_train='5cos0.2x'#2exp(sin(2x^2))
noise_name_train='Uni(-1,1)' #'N(0,1)'#Uni(-1,1)
train_name='data_signal__cosX__amp-5_freq-0.2__noise_uniform_low--1_high-1.npz'

sig_name_test='5cos0.2x'
noise_name_test='Uni(0,4)'
test_name='data_signal__cosX__amp-5_freq-0.2__noise_uniform_low-0_high-4.npz'

batch_size=128
lr=0.001
run_name = f'Train_sig-{sig_name_train}_noise-{noise_name_train}__Test_sig-{sig_name_test}_noise-{noise_name_test}___nonlinearities-{nonlinearities}_gets_spectral_input-{gets_spectral_input}_sta_enabled-{sta_enabled}'

from Models import Encoder
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
enc = Encoder(nonlinearities=nonlinearities, gets_spectral_input=gets_spectral_input, sta_enabled=sta_enabled)
enc = enc.to(device)

print(f"model length: {len(list(enc.parameters()))}")

# enc.load_state_dict(torch.load(f'checkpoints/{run_name}_ckpt_1000.pth')['model'])

from Train_sta import train
from Utils import SignalsDataset
from torch.utils.data import DataLoader
from torch.optim import SGD
from torch.optim import Adam

ds_train = SignalsDataset(f'data/train_{train_name}')
dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True, drop_last=True)
ds_val = SignalsDataset(f'data/val_{test_name}')
dl_val = DataLoader(ds_val, batch_size=batch_size, shuffle=True, drop_last=True)
ds_test = SignalsDataset(f'data/test_{test_name}')
dl_test = DataLoader(ds_test, batch_size=batch_size, shuffle=True, drop_last=True)
optimizer = SGD(enc.parameters(), lr=lr)
loss_func = torch.nn.MSELoss()

print(f"starting run: {run_name}")
train(enc, run_name, dl_train, dl_val, dl_test, n_epochs, optimizer, loss_func, print_every=1, checkpoint_every=50, with_freq=gets_spectral_input)
print(f"end run: {run_name}")