import numpy as np

def sinewave(t, f=1.0, d=0.0):
    phi = 2 * np.pi * f * d
    return np.sin((2 * np.pi * f * t) - phi)

def coswave(t, f=1.0, d=0.0):
    phi = 2 * np.pi * f * d
    return np.cos((2 * np.pi * f * t) - phi)

def coswave_phi(t, f=1.0, phi=0.0):
    """
    cosine but with t, f, and phi for parameters
    """
    return np.cos((2 * np.pi * f * t) - phi)


def sinwave_phi(t, f=1.0, phi=0.0):
    """
    sine but with t, f, and phi for parameters
    """
    return np.sin((2 * np.pi * f * t) - phi)