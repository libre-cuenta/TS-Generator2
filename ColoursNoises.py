import numpy as np
import matplotlib.pyplot as plt

def plot_spectrum(s):
    f = np.fft.rfftfreq(len(s))
    return plt.loglog(f, np.abs(np.fft.rfft(s)))[0]

def noise_psd(N, noise_std, dtype=[float, int], psd = lambda f: 1):
    X_white = np.fft.rfft(np.random.normal(0, noise_std, N))
    S = psd(np.fft.rfftfreq(N))
    # Нормализация S
    S = S / np.sqrt(np.mean(S))
    X_shaped = X_white * S
    Y = np.fft.irfft(X_shaped)
    
    if (dtype == int):
        Y = list(map(int, Y))
    return Y

def NOISEGenerator(f):
    return lambda N, noise_std, dtype=[float, int]: noise_psd(N, noise_std, dtype, f)

@NOISEGenerator
def white_noise(f):
    return 1

@NOISEGenerator
def blue_noise(f):
    return f

@NOISEGenerator
def violet_noise(f):
    return f**2

@NOISEGenerator
def brownian_noise(f):
    return 1/np.where(f == 0, float('inf'), f**2)

@NOISEGenerator
def pink_noise(f):
    return 1/np.where(f == 0, float('inf'), f)
