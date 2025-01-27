import numpy as np
from trig import *

def gaussian_envelope(t, sigma):
    return np.e ** (- ((t**2)  / (2 * sigma**2)))

def gabor(t, sigma, f, phi=0, a=1):
    return a * gaussian_envelope(t, sigma) * coswave_phi(t, f, phi)

def gaboro(t, sigma, f, phi=0, a=1):
    return a * gaussian_envelope(t, sigma) * sinwave_phi(t, f, phi)

def gabore(t, sigma, f, phi=0, a=1):
    return gabor(t , sigma, f, phi, a)

def gabor_norm(f, sigma, fs, d=0):
    """
    Calculate the normalizing constant 
    """

    # Calculate t vector
    t_range = 3 * sigma 
    num_samples = int(2 * t_range * fs)
    t = np.linspace(-t_range, t_range, num_samples)

    #Get g(t) values
    g_values = gaussian_envelope(t, sigma) * coswave(t, f, d)
    energy = np.trapz(g_values**2, t)

    return 1 / np.sqrt(energy)

def gabore_norm(f, sigma, fs, d=0):
    """
    Calculate the normalizing constant 
    """

    # Calculate t vector
    t_range = 3 * sigma 
    num_samples = int(2 * t_range * fs)
    t = np.linspace(-t_range, t_range, num_samples)

    #Get g(t) values
    # Custom Sine function will be equivelent to cos if phi = pi/4
    g_values = gaussian_envelope(t, sigma) * coswave(t, f, d)
    energy = np.trapz(g_values**2, t)

    return 1 / (np.sqrt(energy))

def gaboro_norm(f, sigma, fs, d=0):
    """
    Calculate the normalizing constant 
    """

    # Calculate t vector
    t_range = 3 * sigma 
    num_samples = int(2 * t_range * fs)
    t = np.linspace(-t_range, t_range, num_samples)

    #Get g(t) values
    # Custom Sine function will be equivelent to cos if phi = pi/4
    g_values = gaussian_envelope(t, sigma) * sinewave(t, f, d)
    energy = np.trapz(g_values**2, t)

    return 1 / np.sqrt(energy)



# Initial Implementations, not using samping frequency

# def gabor_norm(t, sigma, f, phi=0, a=1):
#     sum = 0
#     for value in  t:
#         g = gabor(value, sigma, f, phi, a)
#         g_squared = g**2
#         sum+=g_squared
#     return 1/sum

# def gabore_norm(t, sigma, f, phi=0, a=1):
#     sum = 0
#     for value in  t:
#         g = gabore(value, sigma, f, phi, a)
#         g_squared = g**2
#         sum+=g_squared
#     return 1/sum

# def gaboro_norm(t, sigma, f, phi=0, a=1):
#     sum = 0
#     for value in  t:
#         g = gaboro(value, sigma, f, phi, a)
#         g_squared = g**2
#         sum+=g_squared
#     return 1/sum
