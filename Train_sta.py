import datetime
import os
import torch
from tqdm import tqdm
# from tqdm.notebook import tqdm
import numpy as np

def load_batch(batch, device):
    clean, noisy, freq = batch
    return clean.to(device), noisy.to(device), freq.to(device)

def calculate_loss(model, device, dl, loss_fn, with_freq=False):
    losses = []
    with torch.no_grad():
        model.eval()
        for batch in tqdm(dl):
            clean, noisy, freq = load_batch(batch, device)

            if with_freq:
                result = model((noisy, freq))
            else:
                result = model(noisy)

            loss = loss_fn(clean, result)

            losses.append(loss.data.item())

    return np.mean(losses)

def check_model(model, device, dl, loss_fn, with_freq=False):
    losses = []
    noises = []
    results = []
    cleans = []
    with torch.no_grad():
        model.eval()
        for batch in tqdm(dl):
            clean, noisy, freq = load_batch(batch, device)

            if with_freq:
                result = model((noisy, freq))
            else:
                result = model(noisy)

            loss = loss_fn(clean, result)
            losses.append(loss.data.item())
            noises.extend(noisy)
            results.extend(result)
            cleans.extend(clean)

    return {"data": (cleans, noises, results), "avg loss": np.mean(losses)}


#
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print(f"device: {device}")
# checkpoint = f'./checkpoints/sta_ckpt_{1600030206.2596262}.pth'
#
# sta = STA(1, 1000, 1).to(device)
# sta.load_state_dict(torch.load(checkpoint)['sta'])
# encoder = EncoderSTA().to(device)
# encoder.load_state_dict(torch.load(checkpoint)['encoder'])
#
# # parameters
# epochs = 500
# learning_rate = 0.00001
# optimizer = torch.optim.Adam(list(sta.parameters()) + list(encoder.parameters()), lr=learning_rate)
#
# train_clean_signals, train_noisy_signals, train_Pxx_dens = None, None, None
# loader = np.load('train_data.npz')
# train_clean_signals, train_noisy_signals, train_Pxx_dens = loader['arr_0'], loader['arr_1'], loader['arr_2']
#
# # training loop
# for epoch in range(1, epochs + 1):
#     print(f"start epoch {epoch}")
#     sta.train()
#     encoder.train()  # put in training mode
#     running_loss = []
#     epoch_time = time.time()
#     for _clean_signal, _noisy_signal, _pxx_den in tqdm(zip(train_clean_signals, train_noisy_signals, train_Pxx_dens)):
#         time0 = time.time()
#
#         # send them to device
#         _clean_signal = torch.Tensor(_clean_signal).to(device)
#         _noisy_signal = torch.Tensor(_noisy_signal).to(device)
#         _pxx_den = torch.Tensor(_pxx_den).to(device)
#
#         noisy_signal, pxx_den, clean_signal = torch.unsqueeze(_noisy_signal, dim=0), torch.unsqueeze(_pxx_den,
#                                                                                                      dim=0), torch.unsqueeze(
#             _clean_signal, dim=0)
#
#         sta_output = sta(noisy_signal, pxx_den)
#
#         # forward + backward + optimize
#         output = encoder(sta_output)  # forward pass
#         mse_loss = torch.nn.MSELoss()
#         loss = mse_loss(clean_signal, output)
#
#         # always the same 3 steps
#         optimizer.zero_grad()  # zero the parameter gradients
#         loss.backward()  # backpropagation
#         optimizer.step()  # update parameters
#
#         # print statistics
#         running_loss.append(loss.data.item())
#
#     # Calculate training/test set accuracy of the existing model
#     test_loss = calculate_loss()
#
#     log = "Epoch: {} | Loss: {:.4f} | Test loss: {:.3f} | ".format(epoch, np.mean(running_loss), test_loss)
#     epoch_time = time.time() - epoch_time
#     log += "Epoch Time: {:.2f} secs".format(epoch_time)
#     print(log)
#
#     # save model
#     if epoch % 5 == 0:
#         print('==> Saving model ...')
#         state = {
#             'encoder': encoder.state_dict(),
#             'sta': sta.state_dict(),
#             'epoch': epoch,
#         }
#         if not os.path.isdir('checkpoints'):
#             os.mkdir('checkpoints')
#         torch.save(state, f'./checkpoints/sta_ckpt_{datetime.datetime.now()}.pth')
#
# print('==> Finished Training ...')

def train(model, model_name, dl_train, dl_val, dl_test, n_epochs, optimizer, loss_fn, print_every, checkpoint_every, with_freq=False):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    losses_per_epoch_train = []
    losses_per_epoch_validation = []
    # model = model.to(device)
    for epoch in range(1, n_epochs + 1):
        print(f"Start epoch {epoch}")
        avg_loss, test_loss = train_epoch(model, dl_train, dl_val, optimizer, loss_fn, device, with_freq)
        losses_per_epoch_train.append(avg_loss.item())
        losses_per_epoch_validation.append(test_loss)
        if epoch % print_every == 0:
            log = "Epoch: {} | Train Loss: {:.4f} | Val loss: {:.3f} | ".format(epoch, avg_loss, test_loss)
            print(log)

        if epoch % checkpoint_every == 0:
            print('==> Saving model ...')
            state = {
                'model': model.state_dict(),
                'epoch': epoch,
            }
            if not os.path.isdir('checkpoints'):
                os.mkdir('checkpoints')
            torch.save(state, os.path.join(os.path.abspath(os.curdir),'checkpoints',f'{model_name}_ckpt_{epoch}.pth'))

    print('==> Finished Training ...')
    print('Saving results on Train')
    if not os.path.isdir('results'):
        os.mkdir('results')

    torch.save((check_model(model, device, dl_train, loss_fn, with_freq)), os.path.join(os.path.abspath(os.curdir),'results',f'{model_name}_train.pth'))

    print('Saving results on Test')
    res = check_model(model, device, dl_test, loss_fn, with_freq)
    torch.save(res, os.path.join(os.path.abspath(os.curdir),'results',f'{model_name}_test.pth'))

    print(f"Average loss on test: {res['avg loss']}")
    print('Saving all losses per epoch')
    torch.save({'Train Losses': losses_per_epoch_train,'Val losses': losses_per_epoch_validation}, os.path.join(os.path.abspath(os.curdir),'results',f'{model_name}_losses.pth'))



def train_epoch(model, dl_train, dl_test, optimizer, loss_fn, device, with_freq=False):
    model.train()

    losses = []
    num_batches = len(dl_train.batch_sampler)

    for batch in tqdm(dl_train):
        loss = train_batch(model, batch, optimizer, loss_fn, device, with_freq)
        losses.append(loss)

    avg_loss = sum(losses) / num_batches
    test_loss = calculate_loss(model, device, dl_test, loss_fn, with_freq)

    return avg_loss, test_loss


def train_batch(model, batch, optimizer, loss_fn, device, with_freq=False):
    optimizer.zero_grad()  # zero the parameter gradients
    clean, noisy, freq = load_batch(batch, device)

    if with_freq:
        result = model((noisy, freq))
    else:
        result = model(noisy)

    clean = clean.view(clean.shape[0], 1, 450)
    loss = loss_fn(result, clean)
    loss.backward()  # backpropagation
    optimizer.step()  # update parameters

    return loss
