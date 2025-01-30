import numpy as np

def sinewave(f, t, d):
    """
    Takes in Frequency (Hz), Time (Seconds), and Delay(Seconds) and returns the Sine Value.
    Notice the delay must be translated to radians before entered into the sine function.
    """
    phi = 2 * np.pi * f * d
    return np.sin((2 * np.pi * f * t) + phi) 