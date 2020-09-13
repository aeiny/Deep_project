import numpy as np
from scipy import signal

def makeSignalAndNoise(function, amp, freq, num_of_points=1e4):
    # def createNoise(signal, noise_type=noise_type):
    #     noise_power = 0.001
    #     signal_noise = signal + noise_power * np.random.normal(0, 1, size=len(signal))
    #
    #     if noise_type == "white noise":
    #         noise_power = 0.001
    #         signal_noise = signal + noise_power * np.random.normal(0, 1, size=len(signal))
    #     elif noise_type == "peak noise":
    #         signal_noise = np.copy(signal)
    #         signal_noise[np.random.choice(len(signal), 0.05/2*len(signal), replace=False)] =-1
    #         signal_noise[np.random.choice(len(signal), 0.05/2*len(signal), replace=False)] = 1
    #     else:
    #         signal_noise = np.copy(signal)
    #
    #     return signal_noise

    # amp = 2 * np.sqrt(2)
    # freq = 1000.0
    noise_power = 0.001
    time = np.arange(int(num_of_points)) / 1000

    clean_signal = amp * function((2*np.pi) * freq * time)
    noise = noise_power * np.random.normal(0, 1, size=len(clean_signal))

    noisy_signal = clean_signal + noise
    _, Pxx_den = signal.welch(noisy_signal, axis=-1, return_onesided=False, nperseg=len(noisy_signal))
    
    return clean_signal, noise, noisy_signal, Pxx_den
