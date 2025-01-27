import numpy as np


def sinewave(t, f=1.0, d=0.0):
    """
    Generate a sinewave value at a given time t with a frequency f and delay d.
    Phi must be calculated from d.
    
    Parameters:
        t (float or array-like): Time values at which to compute the sinewave.
        f (float): Frequency of the sinewave in Hertz. Default is 1.0.
        d (float): Delay in seconds. Default is 0.0.
    
    Returns:
        float or array-like: The sinewave value(s) at the given time(s).
    """
    phi = 2 * np.pi * f * d
    return np.sin((2 * np.pi * f * t) - phi)

def coswave_phi(t, f=1.0, phi=0.0):
    return np.cos((2 * np.pi * f * t) - phi)


def sinwave_phi(t, f=1.0, phi=0.0):
    return np.sin((2 * np.pi * f * t) - phi)