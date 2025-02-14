import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple
import scipy.stats as stats

def genwaveform(N=100, alpha=0.1, A=1, mu=0, sigma=1, noisetype='Gaussian'):
    """
    Generates a waveform with:
    - N: length of time.
    - alpha: probability of an event at each sample.
    - A: magnitude of event (always positive).
    - mu: center of the noise distribution.
    - sigma: noise standard deviation.
    - noisetype: 'Gaussian' or 'Uniform'.
    """
    # Generate signal locations (1 indicates an event, 0 no event)
    events = (np.random.rand(N) < alpha).astype(int)
    
    # Multiply by A (always positive, since we use np.random.choice([1], ...))
    events *= A * np.random.choice([1], size=N)
    
    # Generate noise
    if noisetype == 'Gaussian':
        noise = np.random.normal(mu, sigma, N)
    elif noisetype == 'Uniform':
        noise = np.random.uniform(mu - sigma/2, mu + sigma/2, N)
    else:
        raise ValueError(f"Unsupported Noise Type: {noisetype}")
    
    # Create the waveform by adding the events and noise
    wave = noise + events

    return wave, np.where(events != 0)[0]


def plot_waveform(N=100, alpha=0.1, A=1, mu=0, sigma=1, noisetype='Gaussian', threshold=None):
    """
    Generates and plots the waveform with event markers.
    If a threshold is provided, a dotted horizontal line is drawn at that level.
    """
    # Generate waveform and event indices
    wave, event_indices = genwaveform(N, alpha, A, mu, sigma, noisetype)
    
    # Create the plot
    plt.figure(figsize=(10, 4))
    plt.plot(wave, label=f'Waveform (Noise: {noisetype})', color='b')
    plt.scatter(event_indices, wave[event_indices], color='red', label="Events", marker='o')
    
    # Add a single threshold line if specified (only positive threshold needed)
    if threshold is not None:
        plt.axhline(threshold, color='green', linestyle="dotted")
    
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.title(f"Generated Waveform with Magnitude {A}")
    plt.legend()
    plt.grid()
    plt.show()


def crossings(signal, threshold, dir="both"):
    """
    Finds the indices where the signal crosses the threshold.
    For positive event detection, only "negpos" (rising) crossings are used.
    """
    signal = np.asarray(signal)
    
    if dir == "negpos":
        # Crossing from below to above or equal to threshold
        cross_idx = np.where((signal[:-1] < threshold) & (signal[1:] >= threshold))[0] + 1
    elif dir == "posneg":
        # Crossing from above or equal to below threshold
        cross_idx = np.where((signal[:-1] >= threshold) & (signal[1:] < threshold))[0] + 1
    elif dir == "both":
        negpos = np.where((signal[:-1] < threshold) & (signal[1:] >= threshold))[0] + 1
        posneg = np.where((signal[:-1] >= threshold) & (signal[1:] < threshold))[0] + 1
        cross_idx = np.sort(np.concatenate((negpos, posneg)))
    else:
        raise ValueError("Invalid direction. Must be 'negpos', 'posneg', or 'both'.")
    
    return cross_idx


def detectioncounts(si, y, theta):
    """
    Computes the detection counts for a given threshold theta.
    
    Parameters:
    - si: indices where true events occur.
    - y: observed waveform.
    - theta: detection threshold.
    
    Returns:
    - A named tuple (TP, FN, FP, TN) where:
        TP: true positives (events correctly detected),
        FN: false negatives (missed events),
        FP: false positives (detections where no event exists),
        TN: true negatives.
    
    Since events are always positive, only positive (rising) threshold crossings are used.
    """
    # Use only positive (rising) threshold crossings
    detected_events = crossings(y, theta, dir="negpos")
    
    # Convert detected and true event indices to sets
    detected_set = set(detected_events)
    true_set = set(si)
    
    TP = len(true_set & detected_set)  # correctly detected events
    FN = len(true_set - detected_set)  # missed events
    FP = len(detected_set - true_set)  # detections not corresponding to true events
    TN = len(y) - (TP + FN + FP)        # remaining samples
    
    DetectionCounts = namedtuple("DetectionCounts", ["TP", "FN", "FP", "TN"])
    return DetectionCounts(TP, FN, FP, TN)


def plot_detection_results(N, wave, true_events, threshold):
    """
    Plots the waveform with true event markers and detected event crossings.
    Only positive threshold crossings are considered.
    """
    detected_events = crossings(wave, threshold, dir="negpos")
    
    plt.figure(figsize=(12, 5))
    plt.plot(wave, label="Waveform", color='b')
    
    # Mark true event locations
    plt.scatter(true_events, wave[true_events], color='red', marker='o', label="True Events")
    
    # Mark detected crossings (positive threshold crossings only)
    plt.scatter(detected_events, wave[detected_events], color='green', marker='x', label="Detected Events")
    
    # Draw threshold line
    plt.axhline(threshold, color='black', linestyle="dotted")
    
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.title("Waveform with True and Detected Events")
    plt.legend()
    plt.grid()
    plt.show()


def falsepos(theta, mu=0, sigma=1, noisetype='Gaussian'):
    """
    Computes the probability of a false positive (P(FP)).
    
    A false positive occurs when noise (in the absence of an event) exceeds theta.
    """
    if noisetype == 'Gaussian':
        return 1 - stats.norm.cdf(theta, loc=mu, scale=sigma)
    elif noisetype == 'Uniform':
        if theta < mu + sigma/2:
            return (mu + sigma/2 - theta) / sigma
        else:
            return 0
    else:
        raise ValueError("Unsupported noise type. Choose 'Gaussian' or 'Uniform'.")


def falseneg(theta, A=1, mu=0, sigma=1, noisetype='Gaussian'):
    """
    Computes the probability of a false negative (P(FN)).
    
    A false negative occurs when an event (of magnitude A) plus noise fails to reach theta.
    """
    if noisetype == 'Gaussian':
        return stats.norm.cdf(theta, loc=A + mu, scale=sigma)
    elif noisetype == 'Uniform':
        if theta > mu + A - sigma/2:
            return (theta - (mu + A) + sigma/2) / sigma
        else:
            return 0
    else:
        raise ValueError("Unsupported noise type. Choose 'Gaussian' or 'Uniform'.")
