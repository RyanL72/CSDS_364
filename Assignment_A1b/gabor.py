import numpy as np
from Assignment_A1b.trigwaves import *

def gaussian_envelope(t, sigma):
    return np.e ** (- ((t**2)  / (2 * sigma**2)))

def gabor(t, sigma, f, phi=0, a=1):
    return a * gaussian_envelope(t, sigma) * coswave_phi(t, f, phi)

def gaboro(t, sigma, f, phi=0, a=1):
    return gabor(t, sigma, f, phi, a)

def gabore(t, sigma, f, phi, a):
    return a * gaussian_envelope(t, sigma) * sinwave_phi(t, f, phi)

def gabor_norm(t, sigma, f, phi=0, a=1):
    sum = 0
    for value in  t:
        g = gabor(value, sigma, f, phi, a)
        g_squared = g**2
        sum+=g_squared
    return 1/sum

def gabore_norm(t, sigma, f, phi=0, a=1):
    sum = 0
    for value in  t:
        g = gabore(value, sigma, f, phi, a)
        g_squared = g**2
        sum+=g_squared
    return 1/sum

def gaboro_norm(t, sigma, f, phi=0, a=1):
    sum = 0
    for value in  t:
        g = gaboro(value, sigma, f, phi, a)
        g_squared = g**2
        sum+=g_squared
    return 1/sum
