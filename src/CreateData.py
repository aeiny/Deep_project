import numpy as np
import tqdm
from scipy import signal

# Const:
rand_gen = np.random.RandomState(0)
SAMPLE_SIZE = 450
OFFSET_SIZE = 50

MIN_SIGNAL_X = int(-33e6/2)
MAX_SIGNAL_X = int(33e6/2)

TRAIN_SIZE = int(50e3)
TEST_SIZE = int(10e3)
VAL_SIZE = int(5e3)

class NOISE_TYPE:
    UNIFORM, NORMAL = range(2)

def crateCosSignal(amp, freq):
    return {
            "name" : f"cosX__amp-{amp}_freq-{freq}",
            "function" : lambda x: amp * np.cos( freq * x),
            }
def crateCosSquaredSignal(amp, freq):
    return {
            "name" : f"cosX^2__amp-{amp}_freq-{freq}",
            "function" : lambda x: amp * np.cos( freq * x * x),
            }

def crateExpCosSignal(amp, freq):
    return {
            "name" : f"exp(cosX)__amp-{amp}_freq-{freq}",
            "function" : lambda x: amp * np.exp(np.cos( freq * x)),
            }

def crateExpCosSquaredSignal(amp, freq):
    return {
            "name" : f"exp(cosX^2)__amp-{amp}_freq-{freq}",
            "function" : lambda x: amp * np.exp(np.cos( freq * x * x)),
            }

# def crateSinSignal(amp, freq):
#     return {
#             "name" : f"sinX__amp_{amp}__freq_{freq}",
#             "function" : lambda x: amp * np.cos((2 * np.pi) * freq * x),
#             }

# def crateSinSquaredSignal(amp, freq):
#     return {
#             "name" : f"sinX^2__amp_{amp}__freq_{freq}",
#             "function" : lambda x: amp * np.sin((2*np.pi) * freq * (x^2)),
#             }

# def crateESinSquaredSignal(amp, freq):
#     return {
#             "name" : f"exp(sinX^2)__amp_{amp}__freq_{freq}",
#             "function" : lambda x: amp * np.e ** (np.sin((2*np.pi) * freq * (x^2))),
#             }

def crateUniformNoise(low, high):
    return {
            "name" : f"uniform_low-{low}_high-{high}",
            "type" : NOISE_TYPE.UNIFORM,
            "low": low,
            "high" : high
            }

def crateNormalNoise(mu, sigma):
    return {
            "name" : f"normal_mu-{mu}_sigma-{sigma}",
            "type" : NOISE_TYPE.NORMAL,
            "mu": mu,
            "sigma" : sigma
            }

def makeSignalAndNoise(signal_info, noise_info):
    def trainTestSplitIndices():
        num_of_indices = int((MAX_SIGNAL_X - MIN_SIGNAL_X) / (SAMPLE_SIZE + OFFSET_SIZE))
        indices = np.arange(num_of_indices)
        rand_gen.shuffle(indices)
        train_indices = np.sort(indices[:TRAIN_SIZE])
        validation_indices = np.sort(indices[TRAIN_SIZE : TRAIN_SIZE + VAL_SIZE])
        test_indices = np.sort(indices[TRAIN_SIZE + VAL_SIZE : TRAIN_SIZE + VAL_SIZE + TEST_SIZE])
        assert(len(train_indices) == TRAIN_SIZE)
        assert(len(validation_indices) == VAL_SIZE)
        assert(len(test_indices) == TEST_SIZE)
        return train_indices, validation_indices, test_indices

    def creteCleanSignal(indices, signal_info=signal_info):
        def indicesToIndexes(indices):
            indexes = [[MIN_SIGNAL_X + (SAMPLE_SIZE + OFFSET_SIZE) * offset + x for x in range(SAMPLE_SIZE)] for offset in tqdm.tqdm(indices)]
            return np.array(indexes)
        indexes = indicesToIndexes(indices)
        indexes = np.expand_dims(indexes, axis=1)
        return indexes, signal_info["function"](indexes)
        

    def creteNoise(noise_shape, noise_info=noise_info):
        noise = None
        noise_type = noise_info["type"]
        if noise_type == NOISE_TYPE.UNIFORM:
            return np.random.uniform(noise_info["low"], noise_info["high"], noise_shape)
        elif noise_type == NOISE_TYPE.NORMAL:
            return np.random.normal(noise_info["mu"], noise_info["sigma"], noise_shape)
        else:
            return None

    # Find Indices:
    train_indices, val_indices, test_indices = trainTestSplitIndices()
    
    # Create Clean Signals:
    train_indexes, train_clean_signal = creteCleanSignal(train_indices)
    val_indexes, val_clean_signal = creteCleanSignal(val_indices)
    test_indexes, test_clean_signal = creteCleanSignal(test_indices)

    # Create Noise:
    train_noise = creteNoise(train_clean_signal.shape)
    val_noise = creteNoise(val_clean_signal.shape)
    test_noise = creteNoise(test_clean_signal.shape)
    
    # Create real Signals:
    train_real_signal = train_clean_signal + train_noise
    val_real_signal = val_clean_signal + val_noise
    test_real_signal = test_clean_signal + test_noise

    # Create Pxx dens:
    _, train_Pxx_dens = signal.welch(train_real_signal, axis=-1, return_onesided=False, nperseg=SAMPLE_SIZE)
    _, val_Pxx_dens = signal.welch(val_real_signal, axis=-1, return_onesided=False, nperseg=SAMPLE_SIZE)
    _, test_Pxx_dens = signal.welch(test_real_signal, axis=-1, return_onesided=False, nperseg=SAMPLE_SIZE)

    np.savez(f'data/train_data_signal__{signal_info["name"]}__noise_{noise_info["name"]}.npz', train_indexes, train_clean_signal, train_noise, train_real_signal, train_Pxx_dens)
    np.savez(f'data/val_data_signal__{signal_info["name"]}__noise_{noise_info["name"]}.npz', val_indexes, val_clean_signal, val_noise, val_real_signal, val_Pxx_dens)
    np.savez(f'data/test_data_signal__{signal_info["name"]}__noise_{noise_info["name"]}.npz', test_indexes, test_clean_signal, test_noise, test_real_signal, test_Pxx_dens)
    print(f'data/train_data_signal__{signal_info["name"]}__noise_{noise_info["name"]}.npz')
    print(f'data/val_data_signal__{signal_info["name"]}__noise_{noise_info["name"]}.npz')
    print(f'data/test_data_signal__{signal_info["name"]}__noise_{noise_info["name"]}.npz')


makeSignalAndNoise(crateExpCosSignal(5, 1/5), crateNormalNoise(0,1))
#crateUniformNoise(0,4)
#crateNormalNoise