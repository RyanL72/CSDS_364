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

def gaussian_envelope(t, sigma):
    return np.e ** (- ((t**2)  / (2 * sigma**2)))

def gabor(t, sigma, f, phi=0, a=1):
    return a * gaussian_envelope(t, sigma) * coswave_phi(t, f, phi)

def gaboro(t, sigma, f, phi=0, a=1):
    return -1* a * gaussian_envelope(t, sigma) * sinwave_phi(t, f, phi)

def gabore(t, sigma, f, phi=0, a=1):
    return gabor(t , sigma, f, phi, a)

def gabor_norm(f, sigma, fs, d=0):
    """
    Calculate the normalizing constant 
    """

    # Calculate t vector
    t_range = 100 * sigma 
    num_samples = int(2 * t_range * fs)
    t = np.linspace(-t_range, t_range, num_samples)

    #Get g(t) values
    g_values = gaussian_envelope(t, sigma) * coswave(t, f, d)
    energy = np.trapezoid(g_values**2, t)

    return 1 / np.sqrt(energy)

def gabore_norm(f, sigma, fs, d=0):
    """
    Calculate the normalizing constant 
    """

    # Calculate t vector
    t_range = 100 * sigma 
    num_samples = int(2 * t_range * fs)
    t = np.linspace(-t_range, t_range, num_samples)

    #Get g(t) values
    # Custom Sine function will be equivelent to cos if phi = pi/4
    g_values = gaussian_envelope(t, sigma) * coswave(t, f, d)
    norm_L2 = np.linalg.norm(g_values, ord=2)
    
    return norm_L2

def gaboro_norm(f, sigma, fs, d=0):
    """
    Calculate the normalizing constant 
    """

    # Calculate t vector
    t_range = 100 * sigma 
    num_samples = int(2 * t_range * fs)
    t = np.linspace(-t_range, t_range, num_samples)

    #Get g(t) values
    # Custom Sine function will be equivelent to cos if phi = pi/4
    g_values = gaussian_envelope(t, sigma) * sinewave(t, f, d)

    norm_L2 = np.linalg.norm(g_values, ord=2)

    return norm_L2

def erb(f):
    return 24.7*(((4.37 * f)/1000)+1)

def bandwidth(f):
    return 1.019 * erb(f)

def gammatone(t, f=1.0, n=4, phi=0.0, a=1.0):
    t = np.maximum(t, 0)  # Ensure non-negative time
    b = bandwidth(f)
    return a * (t**(n-1)) * np.exp(-2 * np.pi * b * t) * np.cos(2*np.pi*t*f + phi)

def gammatone_norm(f, n=4, fs=10000):
    t = np.linspace(0, 1, fs)  # Ensure time starts from 0
    g = gammatone(t, f, n, 0.0, 100)
    norm_L2 = np.linalg.norm(g, ord=2) / np.sqrt(fs)  # Normalize by sqrt(fs)
    
    return norm_L2


def localmaxima(signal):
    
    # Convert input to NumPy array for easy slicing
    signal = np.asarray(signal)

    # Initialize an empty list to store indices of local maxima
    maxima_indices = []

    # Iterate through the array, ignoring the first and last elements (edges)
    for i in range(1, len(signal) - 1):
        # Check if the current element is a local maximum
        if signal[i - 1] < signal[i] > signal[i + 1]:
            maxima_indices.append(i)

    return maxima_indices

def crossings(signal, threshold, dir="both"):

    # Convert signal to a NumPy array
    signal = np.asarray(signal)
    
    # Initialize crossings
    crossings = []

    # Compute the conditions for each direction
    if dir == "negpos":
        # Crossing from below to above or equal
        crossings = np.where((signal[:-1] < threshold) & (signal[1:] >= threshold))[0] + 1
    elif dir == "posneg":
        # Crossing from above or equal to below
        crossings = np.where((signal[:-1] >= threshold) & (signal[1:] < threshold))[0] + 1
    elif dir == "both":
        # Crossing in either direction
        negpos_crossings = np.where((signal[:-1] < threshold) & (signal[1:] >= threshold))[0] + 1
        posneg_crossings = np.where((signal[:-1] >= threshold) & (signal[1:] < threshold))[0] + 1
        crossings = np.sort(np.concatenate((negpos_crossings, posneg_crossings)))
    else:
        raise ValueError("Invalid direction. Must be 'negpos', 'posneg', or 'both'.")

    return crossings

def envelope(y, nblocks=None):

    # Convert input to a NumPy array
    y = np.asarray(y)
    
    # Set default for `nblocks` if not provided
    if nblocks is None:
        nblocks = max(1, len(y) // 10)  # Default to 1/10th the length of `y`

    # Compute the size of each block
    block_size = len(y) // nblocks

    # Compute the starting indices for each block
    blockindices = np.arange(0, len(y), block_size)

    # Initialize lower and upper bounds arrays
    ylower = []
    yupper = []

    # Compute lower and upper bounds for each block
    for i in range(len(blockindices)):
        start = blockindices[i]
        end = blockindices[i + 1] if i + 1 < len(blockindices) else len(y)  # Handle last block
        block = y[start:end]
        ylower.append(np.min(block))  # Lower bound
        yupper.append(np.max(block))  # Upper bound

    return np.array(ylower), np.array(yupper), blockindices

def timeToIndex(t, fs):
    return t * fs