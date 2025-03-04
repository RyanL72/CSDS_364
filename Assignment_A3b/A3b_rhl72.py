import numpy as np
import matplotlib.pyplot as plt

def movingavg(x, lam=0.5):
    y = np.zeros_like(x)
    for n in range(len(x)):
        if n == 0:
            y[n] = (1 - lam) * x[n]
        else:
            y[n] = lam * y[n-1] + (1 - lam) * x[n]
    return y

def randprocess(N, sigma=1):
    x = np.zeros(N)
    x[0] = np.random.randn()
    for n in range(1, N):
        x[n] = np.random.normal(loc=x[n-1], scale=sigma)
    return x

def noisy_sine(t, freq=5, noise_amp=0.5):
    sine_wave = np.sin(2 * np.pi * freq * t)
    noise = noise_amp * np.random.randn(len(t))
    return sine_wave + noise

def filterIIR(x, a, b):
    N = len(x)
    y = np.zeros(N)
    na = len(a)
    nb = len(b)
    for n in range(N):
        ff = sum(b[k] * x[n - k] for k in range(nb) if n - k >= 0)
        fb = sum(a[k] * y[n - k] for k in range(1, na) if n - k >= 0)
        y[n] = ff - fb
    return y

def convolve_signal(x, h, i0):
    N = len(x)
    L = len(h)
    y = np.zeros(N)
    for n in range(N):
        y[n] = sum(h[m] * x[n + i0 - m] for m in range(L) if 0 <= n + i0 - m < N)
    return y

def gabor_kernel(duration, fs, f, sigma):
    N = int(duration * fs)
    if N % 2 == 0:
        N += 1
    t = np.linspace(-duration/2, duration/2, N)
    h = np.exp(-t**2 / (2 * sigma**2)) * np.cos(2 * np.pi * f * t)
    return h, t

def gammatone(t, n=4, b=150, f=300):
    y = np.zeros_like(t)
    t_pos = np.maximum(t, 0)
    y = (t_pos**(n-1)) * np.exp(-2*np.pi*b*t_pos) * np.cos(2*np.pi*f*t_pos)
    y[t < 0] = 0
    return y