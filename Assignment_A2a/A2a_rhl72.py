import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple
import scipy.stats as stats


def genwaveform(N=100, alpha=0.1, A=1, mu=0, sigma=1, noisetype='Gaussian'):
    """
    Generates a wave with 
    N ~ Length of time.
    alpha ~ Event probability of an event.
    A ~ Amplitude of event.
    mu ~ Center of wave.
    sigma ~ Noise Standard Deviation.
    noisetype ~ Gaussian or Uniform.
    """

    ## Generate Signal locations
    events = (np.random.rand(N) < alpha).astype(int)
    
    # Scale them up to A
    events*=A*np.random.choice([-1,1], size=N)

    ## Generate Noise

    if noisetype == 'Gaussian':
        noise = np.random.normal(mu, sigma, N)  
    elif noisetype == 'Uniform':
        noise = np.random.uniform(mu - sigma/2, mu + sigma/2, N)  
    else:
        print("Unsuppported Noise Type: {noisetype}")
    
    ## Create wave by adding them

    wave = noise + events

    return wave, np.where(events != 0)[0]


def plot_waveform(N=100, alpha=0.1, A=1, mu=0, sigma=1, noisetype='Gaussian', threshold=None):

    # Generate waveform and event indices
    wave, event_indices = genwaveform(N, alpha, A, mu, sigma, noisetype)

    # Create the plot
    plt.figure(figsize=(10, 4))
    plt.plot(wave, label=f'Waveform (Noise: {noisetype})', color='b')

    # Plot event markers (red dots)
    plt.scatter(event_indices, wave[event_indices], color='red', label="Events", marker='o')

    # Add threshold lines if specified
    if threshold is not None:
        plt.axhline(threshold, color='green', linestyle="dotted")
        plt.axhline(-threshold, color='green', linestyle="dotted")

    # Labels and title
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.title(f"Generated Waveform with Amplidtude {A}")
    plt.legend()
    plt.grid()
    plt.show()



def detectioncounts(si, y, theta):
    """
    - A named tuple with (TP, FN, FP, TN).
    """
    # Get threshold crossings in both directions
    detected_events_pos = crossings(y, theta, dir="negpos")  # Positive threshold crossings
    detected_events_neg = crossings(y, -theta, dir="posneg")  # Negative threshold crossings

    # Combine positive and negative crossings
    detected_events = np.sort(np.concatenate((detected_events_pos, detected_events_neg)))

    # Convert detected and true event indices to sets for comparison
    detected_set = set(detected_events)
    true_set = set(si)

    # Compute the four error counts
    TP = len(true_set & detected_set)  # Events correctly detected
    FN = len(true_set - detected_set)  # Events missed
    FP = len(detected_set - true_set)  # False detections (noise crossing threshold)
    TN = len(y) - (TP + FN + FP)       # Remaining samples are true negatives

    # Return results in a named tuple
    DetectionCounts = namedtuple("DetectionCounts", ["TP", "FN", "FP", "TN"])
    return DetectionCounts(TP, FN, FP, TN)


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

def plot_detection_results(N, wave, true_events, threshold):
    """
    Plots waveform with detected event locations and threshold.
    """
    detected_events_pos = crossings(wave, threshold, dir="negpos")
    detected_events_neg = crossings(wave, -threshold, dir="posneg")

    plt.figure(figsize=(12, 5))
    plt.plot(wave, label="Waveform", color='b')

    # Mark true event locations
    plt.scatter(true_events, wave[true_events], color='red', marker='o', label="True Events")

    # Mark detected crossings
    plt.scatter(detected_events_pos, wave[detected_events_pos], color='green', marker='x', label="Detected (Positive)")
    plt.scatter(detected_events_neg, wave[detected_events_neg], color='purple', marker='x', label="Detected (Negative)")

    # Threshold lines
    plt.axhline(threshold, color='black', linestyle="dotted")
    plt.axhline(-threshold, color='black', linestyle="dotted")

    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.title("Waveform with True and Detected Events")
    plt.legend()
    plt.grid()
    plt.show()

import scipy.stats as stats

def falsepos(theta, mu=0, sigma=1, noisetype='Gaussian'):
    """
    Computes the probability of a false positive: P(FP).
    
    - theta: float, detection threshold.
    - mu: float, mean of noise distribution.
    - sigma: float, standard deviation (Gaussian) or range width (Uniform).
    - noisetype: str, 'Gaussian' or 'Uniform'.
    
    Returns:
    - Probability of a false positive.
    """
    if noisetype == 'Gaussian':
        return 1 - stats.norm.cdf(theta, loc=mu, scale=sigma)  # P(ε >= θ)
    
    elif noisetype == 'Uniform':
        if theta < mu + sigma/2:
            return (mu + sigma/2 - theta) / sigma
        else:
            return 0  # Outside range of uniform noise
    
    else:
        raise ValueError("Unsupported noise type. Choose 'Gaussian' or 'Uniform'.")

def falseneg(theta, A=1, mu=0, sigma=1, noisetype='Gaussian'):
    """
    Computes the probability of a false negative: P(FN).
    
    - θ: float, detection threshold.
    - A: float, event amplitude.
    - mu: float, mean of noise distribution.
    - sigma: float, standard deviation (Gaussian) or range width (Uniform).
    - noisetype: str, 'Gaussian' or 'Uniform'.
    
    Returns:
    - Probability of a false negative.
    """
    if noisetype == 'Gaussian':
        return stats.norm.cdf(theta, loc=A + mu, scale=sigma)  # P(A + ε < θ)
    
    elif noisetype == 'Uniform':
        if theta > mu + A - sigma/2:
            return (theta - (mu + A) + sigma/2) / sigma
        else:
            return 0  # Outside range
    
    else:
        raise ValueError("Unsupported noise type. Choose 'Gaussian' or 'Uniform'.")