import datetime
import os
import time

import torch
from tqdm import tqdm

from Models import Encoder
import numpy as np



def calculate_loss():
    loader = np.load('test_data.npz')
    test_clean_signals, test_noisy_signals = loader['arr_0'], loader['arr_1']

    losses = []
    with torch.no_grad():
        encoder.eval()
        for _clean_signal, _noisy_signal in tqdm(
                zip(test_clean_signals, test_noisy_signals)):

            # send them to device
            _clean_signal = torch.Tensor(_clean_signal).to(device)
            _noisy_signal = torch.Tensor(_noisy_signal).to(device)

            noisy_signal, clean_signal = torch.unsqueeze(_noisy_signal, dim=0), torch.unsqueeze(_clean_signal, dim=0)


            # forward + backward + optimize
            output = encoder(noisy_signal)  # forward pass
            mse_loss = torch.nn.MSELoss()
            loss = mse_loss(clean_signal, output)
            losses.append(loss.data.item())

    return np.mean(losses)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"device: {device}")
checkpoint = f'./checkpoints/nonSta_ckpt_{1600030206.2596262}.pth'

encoder = Encoder().to(device)
# encoder.load_state_dict(torch.load(checkpoint)['encoder'])

# parameters
epochs = 500
learning_rate = 0.0001
optimizer = torch.optim.Adam(encoder.parameters(), lr=learning_rate)

train_clean_signals, train_noisy_signals, train_Pxx_dens = None, None, None
loader = np.load('train_data.npz')
train_clean_signals, train_noisy_signals, train_Pxx_dens = loader['arr_0'], loader['arr_1'], loader['arr_2']

# training loop
for epoch in range(1, epochs + 1):
    print(f"start epoch {epoch}")
    encoder.train()  # put in training mode
    running_loss = []
    epoch_time = time.time()
    for _clean_signal, _noisy_signal, _pxx_den in tqdm(zip(train_clean_signals, train_noisy_signals, train_Pxx_dens)):
        time0 = time.time()

        # send them to device
        _clean_signal = torch.Tensor(_clean_signal).to(device)
        _noisy_signal = torch.Tensor(_noisy_signal).to(device)
        _pxx_den = torch.Tensor(_pxx_den).to(device)

        noisy_signal, clean_signal = torch.unsqueeze(_noisy_signal, dim=0), torch.unsqueeze(_clean_signal, dim=0)

        # forward + backward + optimize
        output = encoder(noisy_signal)  # forward pass
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
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoints'):
            os.mkdir('checkpoints')
        torch.save(state, f'./checkpoints/nonSta_ckpt_{datetime.datetime.now()}.pth')

print('==> Finished Training ...')
