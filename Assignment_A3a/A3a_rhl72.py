import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import write

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


def delta(t, fs=1):
    """
    Discrete delta function.
    For a continuous time t, if |t| < half the sample period (0.5/fs),
    we consider it as the impulse sample.
    """
    return np.where(np.abs(t) < (0.5 / fs), 1.0, 0.0)

def u(t):
    """
    Unit step function (Heaviside step function).
    Returns 1 for t >= 0, 0 for t < 0.
    """
    return np.where(t >= 0, 1.0, 0.0)


def gensignal(t, g, tau=0, T=1, **kwargs):
    """
    Generate a signal based on function g delayed by tau and lasting for duration T.
    
    Parameters:
      t    : time vector (in seconds)
      g    : function of time (accepts t and kwargs)
      tau  : delay (in seconds)
      T    : duration of the signal (in seconds)
      kwargs: additional parameters for g
      
    Returns:
      Signal array: g(t-tau) for t in [tau, tau+T), else 0.
    """
    signal = np.where((t >= tau) & (t < tau + T), g(t - tau, **kwargs), 0)
    return signal


def energy(x):
    """Compute the energy of signal x (sum of squares)."""
    return np.sum(x**2)

def power(x):
    """Compute the average power of signal x."""
    return energy(x) / len(x)

def snr(Ps, Pn):
    """
    Compute SNR in decibels given signal power Ps and noise power Pn.
    SNR (dB) = 10*log10(Ps/Pn)
    """
    return 10 * np.log10(Ps / Pn)

def noisysignal(t, g, tau=0, T=1, sigma=0.1, **kwargs):
    """
    Generate a signal (using function g with delay tau and duration T) and add Gaussian noise.
    
    Parameters:
      t     : time vector (seconds)
      g     : signal function
      tau   : delay (seconds)
      T     : duration of the signal (seconds)
      sigma : standard deviation of the Gaussian noise
      kwargs: additional parameters for g
      
    Returns:
      y(t) = signal + noise
    """
    sig = gensignal(t, g, tau, T, **kwargs)
    noise = np.random.normal(0, sigma, size=t.shape)
    return sig + noise

def snr2sigma(x, xrange=None, dBsnr=10):
    """
    Given a signal x and an optional index range (xrange), compute the noise sigma
    required so that adding noise with that sigma gives the desired SNR (in dB).
    
    Parameters:
      x      : signal array
      xrange : indices (e.g., a slice or list) over which to compute the signal's std.
               Defaults to the whole array.
      dBsnr  : desired SNR in decibels
      
    Returns:
      sigma : noise standard deviation.
    """
    if xrange is None:
        x_signal = x
    else:
        x_signal = x[xrange]
    signal_std = np.std(x_signal)
    sigma = signal_std / (10**(dBsnr/20))
    return sigma


def extent(y, theta=0.01):
    """
    Returns a tuple (start_index, end_index) corresponding to the indices
    where |y| first and last exceed theta times the maximum absolute value of y.
    
    Parameters:
      y     : signal array
      theta : threshold fraction (default 0.01)
      
    Returns:
      (start, end) indices or None if no such indices are found.
    """
    thresh = theta * np.max(np.abs(y))
    indices = np.where(np.abs(y) >= thresh)[0]
    if len(indices) == 0:
        return None
    return indices[0], indices[-1]

def grand_synthesis(duration=5, fs=44100, num_tones=20, T=0.1, fmin=100, fmax=8000):
    """
    Synthesize a waveform composed of multiple random gammatone tones.
    
    Parameters:
      duration : total duration of the waveform in seconds.
      fs       : sampling frequency.
      num_tones: number of gammatone tones to include.
      T        : duration of each tone.
      fmin, fmax: frequency range for the gammatone tones.
      
    Returns:
      t        : time vector.
      waveform : synthesized waveform (normalized).
    """
    t = np.linspace(0, duration, int(fs*duration), endpoint=False)
    waveform = np.zeros_like(t)
    
    for _ in range(num_tones):
        # Random delay tau: ensure the tone fits inside the overall duration.
        tau = np.random.uniform(0, duration - T)
        # Random frequency between fmin and fmax.
        f = np.random.uniform(fmin, fmax)
        # (Optional) random amplitude; here we use a fixed amplitude of 1.
        A = 1.0
        
        # Generate the tone using gensignal.
        tone = gensignal(t, lambda tt: gammatone(tt, f=f, a=A, n=4), tau=tau, T=T)
        waveform += tone
    
    # Normalize the waveform to avoid clipping.
    waveform = waveform / np.max(np.abs(waveform))
    return t, waveform