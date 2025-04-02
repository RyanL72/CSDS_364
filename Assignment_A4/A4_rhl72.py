import numpy as np


def harmonic(t, f1=1, alist=1, philist=0):
    """
    generate a harmonic wave

    t ~ Time of a Harmonic Function 
    f1 ~ Fundamental Frequncy (hz)
    alist ~ Values specify the amplitudes of each harmonic starting with the fundament
    philist ~ Phase shifts (radians)
    use cosine
    """

    alist = np.atleast_1d(alist)
    phaselist = np.atleast_1d(philist)

    # Match lengths
    if phaselist.size == 1:
        phaselist = np.full_like(alist, phaselist)

    # Sum the cosine components
    waveform = sum(
        a * np.cos(2 * np.pi * (i + 1) * f1 * t + phi)
        for i, (a, phi) in enumerate(zip(alist, phaselist))
    )
    
    return waveform

def cosines(t, flist=1, alist=1, philist=0):
    """
    Generate a sum of cosine waves with arbitrary frequencies.
    
    Parameters:
    - t: time array
    - flist: list of frequencies (Hz)
    - alist: list of amplitudes
    - philist: list of phase shifts (radians)
    """
    flist = np.atleast_1d(flist)
    alist = np.atleast_1d(alist)
    philist = np.atleast_1d(philist)

    # Match lengths
    if alist.size == 1:
        alist = np.full_like(flist, alist)
    if philist.size == 1:
        philist = np.full_like(flist, philist)

    waveform = sum(
        a * np.cos(2 * np.pi * f * t + phi)
        for f, a, phi in zip(flist, alist, philist)
    )
    
    return waveform

    
