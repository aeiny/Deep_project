import os
import pickle
import time

import torch
from tqdm import tqdm

from Models import Encoder
from Utils import makeSignalAndNoise
import numpy as np

def calculate_loss():
    test_clean_signals, test_noisy_signals = None, None
    train_clean_signals, train_noisy_signals = np.load('test_data.npz')

    losses = []
    with torch.no_grad():
        encoder.eval()
        for clean_signal, noisy_signal in zip(test_clean_signals, test_noisy_signals):
            # send them to device
            test_clean_signal = clean_signal.to(device)
            test_noisy_signal = noisy_signal.to(device)

            signal = torch.unsqueeze(torch.tensor(signal, dtype=torch.float32)

            # forward + backward + optimize
            output = encoder(signal)  # forward pass
            mse_loss = torch.nn.MSELoss()
            loss = mse_loss(clean_signal, output)
            losses.append(loss.item())

    return np.mean(losses)


encoder = Encoder()

#parameters
epochs = 500
learning_rate = 0.001
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
optimizer = torch.optim.Adam(encoder.parameters(), lr=learning_rate)

train_clean_signals, train_noisy_signals = None, None
loader = np.load('train_data.npz')
train_clean_signals, train_noisy_signals = loader['arr_0'], loader['arr_1']


# training loop
for epoch in range(1, epochs + 1):
    print(f"start epoch {epoch}")
    encoder.train()  # put in training mode
    running_loss = 0.0
    epoch_time = time.time()
    for _clean_signal, _noisy_signal in tqdm(zip(train_clean_signals, train_noisy_signals)):
        # send them to device
        _clean_signal = torch.Tensor(_clean_signal, device=device)
        _noisy_signal = torch.Tensor(_noisy_signal, device=device)

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
        running_loss += loss.data.item()

    # Calculate training/test set accuracy of the existing model
    test_loss = calculate_loss()

    log = "Epoch: {} | Loss: {:.4f} | Test loss: {:.3f}% | ".format(epoch, running_loss, test_loss)
    epoch_time = time.time() - epoch_time
    log += "Epoch Time: {:.2f} secs".format(epoch_time)
    print(log)

    # save model
    if epoch % 20 == 0:
        print('==> Saving model ...')
        state = {
            'encoder': encoder.state_dict(),
            'sta': sta.state_dict(),
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoints'):
            os.mkdir('checkpoints')
        torch.save(state, './checkpoints/cifar_cnn_ckpt.pth')

print('==> Finished Training ...')
