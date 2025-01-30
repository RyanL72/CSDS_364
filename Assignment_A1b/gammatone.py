import numpy as np
from trig import *


def erb(f):
    return 24.7*(((4.37 * f)/1000)+1)

def bandwidth(f):
    return 1.019 * erb(f)

def gammatone(t, f=1.0, n=4, d=0.0, a=1.0):
    t = np.maximum(t, 0) # t >= 0
    b = bandwidth(f)
    return a * (t**(n-1)) * np.exp(-2 * np.pi * b * t) * coswave(t, f, d)

def gammatone_norm(f=1.0, n=4, fs=1000):
    t = np.linspace(0, 0.1, int(0.1 * fs))  # 0 to 0.1 seconds
    g = gammatone(t, f, n, 0.0, 1.0)
    return 1 / np.sqrt(np.sum(g**2))