import numpy as np

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