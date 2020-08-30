import numpy as np
from scipy import signal

def makeFunction(function, num_of_points=1e5, noise_type="white noise"):
    def createNoise(signal, noise_type=noise_type):
        if noise_type == "white noise":
            noise_power = 0.001
            signal_noise = signal + noise_power * np.random.normal(0, 1, size=len(signal))
        elif noise_type == "peak noise":
            signal_noise = np.copy(signal)
            signal_noise[np.random.choice(len(signal), 0.05/2*len(signal), replace=False)] =-1
            signal_noise[np.random.choice(len(signal), 0.05/2*len(signal), replace=False)] = 1
        else:
            signal_noise = np.copy(signal)

        return signal_noise

    amp = 2 * np.sqrt(2)
    freq = 1000.0
    time = np.arange(int(num_of_points)) / 1000

    x = createNoise(amp * function((2*np.pi) * freq * time), )
    _, Pxx_den = signal.welch(x, axis=-1, return_onesided=False, nperseg=len(x))
    
    return x, Pxx_den
