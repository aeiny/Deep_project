import numpy as np

from Utils import makeSignalAndNoise
import pickle
from tqdm import tqdm

TRAIN_DATA = 1000
TEST_DATA = 200

clean_signals = []
Pxx_dens = []
noisy_signals = []
for i in tqdm(range(TRAIN_DATA)):
    amp = np.random.uniform(0, 10)
    freq = np.random.uniform(0, 1000)
    clean_signal, noise, noisy_signal, Pxx_den = makeSignalAndNoise(np.cos, amp, freq)
    clean_signals.append(clean_signal)
    noisy_signals.append(noisy_signal)
    Pxx_dens.append(Pxx_den)

np.savez('train_data.npz', clean_signals, noisy_signals, Pxx_dens)

clean_signals = []
Pxx_dens = []
noisy_signals = []
for i in tqdm(range(TEST_DATA)):
    amp = np.random.uniform(0, 10)
    freq = np.random.uniform(0, 1000)
    clean_signal, noise, noisy_signal, Pxx_den = makeSignalAndNoise(np.cos, amp, freq)
    clean_signals.append(clean_signal)
    noisy_signals.append(noisy_signal)
    Pxx_dens.append(Pxx_den)

np.savez('test_data.npz', clean_signals, noisy_signals, Pxx_dens)
