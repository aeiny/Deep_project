import datetime
import os
import pickle
import time

import torch
from tqdm import tqdm

from Models import STA, EncoderSTA
from Utils import makeSignalAndNoise
import numpy as np


# clean_signal, noise, signal, Pxx_den = makeSignalAndNoise(np.cos, num_of_points=10000, noise_type="white noise")
# signal, Pxx_den = torch.unsqueeze(torch.tensor(signal, dtype=torch.float32), dim=0), torch.unsqueeze(torch.tensor(Pxx_den, dtype=torch.float32), dim=0)

def calculate_loss():
    loader = np.load('test_data.npz')
    test_clean_signals, test_noisy_signals, test_Pxx_dens = loader['arr_0'], loader['arr_1'], loader['arr_2']

    losses = []
    with torch.no_grad():
        sta.eval()
        encoder.eval()
        for _clean_signal, _noisy_signal, _pxx_den in tqdm(
                zip(test_clean_signals, test_noisy_signals, test_Pxx_dens)):
            time0 = time.time()

            # send them to device
            _clean_signal = torch.Tensor(_clean_signal).to(device)
            _noisy_signal = torch.Tensor(_noisy_signal).to(device)
            _pxx_den = torch.Tensor(_pxx_den).to(device)

            noisy_signal, pxx_den, clean_signal = torch.unsqueeze(_noisy_signal, dim=0), torch.unsqueeze(_pxx_den,
                                                                                                         dim=0), torch.unsqueeze(
                _clean_signal, dim=0)

            sta_output = sta(noisy_signal, pxx_den)

            # forward + backward + optimize
            output = encoder(sta_output)  # forward pass
            mse_loss = torch.nn.MSELoss()
            loss = mse_loss(clean_signal, output)
            losses.append(loss.data.item())

    return np.mean(losses)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"device: {device}")
checkpoint = f'./checkpoints/sta_ckpt_{1600030206.2596262}.pth'

sta = STA(1, 1000, 1).to(device)
sta.load_state_dict(torch.load(checkpoint)['sta'])
encoder = EncoderSTA().to(device)
encoder.load_state_dict(torch.load(checkpoint)['encoder'])

# parameters
epochs = 500
learning_rate = 0.00001
optimizer = torch.optim.Adam(list(sta.parameters()) + list(encoder.parameters()), lr=learning_rate)

train_clean_signals, train_noisy_signals, train_Pxx_dens = None, None, None
loader = np.load('train_data.npz')
train_clean_signals, train_noisy_signals, train_Pxx_dens = loader['arr_0'], loader['arr_1'], loader['arr_2']

# training loop
for epoch in range(1, epochs + 1):
    print(f"start epoch {epoch}")
    sta.train()
    encoder.train()  # put in training mode
    running_loss = []
    epoch_time = time.time()
    for _clean_signal, _noisy_signal, _pxx_den in tqdm(zip(train_clean_signals, train_noisy_signals, train_Pxx_dens)):
        time0 = time.time()

        # send them to device
        _clean_signal = torch.Tensor(_clean_signal).to(device)
        _noisy_signal = torch.Tensor(_noisy_signal).to(device)
        _pxx_den = torch.Tensor(_pxx_den).to(device)

        noisy_signal, pxx_den, clean_signal = torch.unsqueeze(_noisy_signal, dim=0), torch.unsqueeze(_pxx_den,
                                                                                                     dim=0), torch.unsqueeze(
            _clean_signal, dim=0)

        sta_output = sta(noisy_signal, pxx_den)

        # forward + backward + optimize
        output = encoder(sta_output)  # forward pass
        mse_loss = torch.nn.MSELoss()
        loss = mse_loss(clean_signal, output)

        # always the same 3 steps
        optimizer.zero_grad()  # zero the parameter gradients
        loss.backward()  # backpropagation
        optimizer.step()  # update parameters

        # print statistics
        running_loss.append(loss.data.item())

    # Calculate training/test set accuracy of the existing model
    test_loss = calculate_loss()

    log = "Epoch: {} | Loss: {:.4f} | Test loss: {:.3f} | ".format(epoch, np.mean(running_loss), test_loss)
    epoch_time = time.time() - epoch_time
    log += "Epoch Time: {:.2f} secs".format(epoch_time)
    print(log)

    # save model
    if epoch % 5 == 0:
        print('==> Saving model ...')
        state = {
            'encoder': encoder.state_dict(),
            'sta': sta.state_dict(),
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoints'):
            os.mkdir('checkpoints')
        torch.save(state, f'./checkpoints/sta_ckpt_{datetime.datetime.now()}.pth')

print('==> Finished Training ...')
