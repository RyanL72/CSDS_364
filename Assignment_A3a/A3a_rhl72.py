import numpy as np
import matplotlib.pyplot as plt

def sinewave(t, f=1.0, d=0.0):
    phi = 2 * np.pi * f * d
    return np.sin((2 * np.pi * f * t) - phi)


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


def plot_sampled_function(g, fs=1, tlim=(0, 2*np.pi), tscale=1, tunits="secs", **kwargs):
    """
    Plots the continuous function g (evaluated on a dense grid) and overlays
    the discrete samples computed at sampling frequency fs.
    
    Parameters:
      g      : function accepting an array t (in seconds) and optional kwargs.
      fs     : sampling frequency (samples per second).
      tlim   : tuple (t_min, t_max) specifying the time interval.
      tscale : scaling factor for the time axis (e.g., 1e3 for msecs).
      tunits : label for the time units.
      kwargs : additional keyword arguments passed to g.
    """
    # Dense time grid for continuous plot
    t_dense = np.linspace(tlim[0], tlim[1], 1000)
    y_dense = g(t_dense, **kwargs)
    
    # Discrete sample times (centered on each period)
    t_samples = np.arange(tlim[0], tlim[1], step=1/fs)
    y_samples = g(t_samples, **kwargs)
    
    plt.figure(figsize=(8,4))
    # Plot continuous function (scaled in time)
    plt.plot(t_dense * tscale, y_dense, label="Continuous", color="blue")
    # Overlay the samples using a stem plot
    markerline, stemlines, baseline = plt.stem(t_samples * tscale, y_samples, linefmt='r-', markerfmt='ro', basefmt="k-")
    plt.setp(stemlines, 'linewidth', 2)
    
    plt.xlabel(f"Time ({tunits})")
    plt.ylabel("Amplitude")
    plt.title("Sampled Function")
    plt.legend()
    plt.grid(True)
    plt.show()


