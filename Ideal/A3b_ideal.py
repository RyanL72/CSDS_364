import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# Part 1: Filtering – Moving Average Filter and Random Process
# =============================================================================

def movingavg(x, lam=0.5):
    """
    Implements the recursive moving average filter:
      y[n] = lam * y[n-1] + (1 - lam)*x[n]
    """
    y = np.zeros_like(x)
    for n in range(len(x)):
        if n == 0:
            y[n] = (1 - lam) * x[n]
        else:
            y[n] = lam * y[n-1] + (1 - lam) * x[n]
    return y

def randprocess(N, sigma=1):
    """
    Generates a random process of length N such that
      x[0] is drawn from a normal distribution and
      for n>0, x[n] ~ N(x[n-1], sigma)
    """
    x = np.zeros(N)
    x[0] = np.random.randn()
    for n in range(1, N):
        x[n] = np.random.normal(loc=x[n-1], scale=sigma)
    return x

def noisy_sine(t, freq=5, noise_amp=0.5):
    """
    Generates a noisy sine wave.
      sine_wave = sin(2*pi*freq*t)
      noise is added with amplitude noise_amp.
    """
    sine_wave = np.sin(2 * np.pi * freq * t)
    noise = noise_amp * np.random.randn(len(t))
    return sine_wave + noise

# =============================================================================
# Part 2: IIR Filters
# =============================================================================

def filterIIR(x, a, b):
    """
    Implements a general IIR filter given the difference equation:
      y[n] = (b0*x[n] + b1*x[n-1] + ... ) - (a1*y[n-1] + a2*y[n-2] + ...)
    Assumes a[0] == 1.
    """
    N = len(x)
    y = np.zeros(N)
    na = len(a)
    nb = len(b)
    for n in range(N):
        # Feedforward part using b coefficients
        ff = 0
        for k in range(nb):
            if n - k >= 0:
                ff += b[k] * x[n - k]
        # Feedback part using a coefficients (skip a[0])
        fb = 0
        for k in range(1, na):
            if n - k >= 0:
                fb += a[k] * y[n - k]
        y[n] = ff - fb
    return y

# =============================================================================
# Part 3: Impulse Response Functions
# =============================================================================
# (This will be shown by filtering an impulse signal.)

# =============================================================================
# Part 4: Convolution and FIR Filtering
# =============================================================================

def convolve_signal(x, h, i0):
    """
    Convolve signal x with kernel h.
    Parameter i0 specifies the index of h corresponding to time zero.
    
    For each output sample n, the sum is:
      y[n] = sum_{m} h[m] * x[n + i0 - m]
    Only valid indices in x are summed.
    """
    N = len(x)
    L = len(h)
    y = np.zeros(N)
    for n in range(N):
        s = 0
        for m in range(L):
            j = n + i0 - m
            if 0 <= j < N:
                s += h[m] * x[j]
        y[n] = s
    return y

def gabor_kernel(duration, fs, f, sigma):
    """
    Generates a Gabor kernel (a Gaussian-modulated cosine).
      h(t) = exp(-t^2/(2*sigma^2)) * cos(2*pi*f*t)
    duration: total time span (seconds) of the kernel.
    fs: sampling rate.
    Returns the kernel h and the corresponding time vector t.
    """
    N = int(duration * fs)
    if N % 2 == 0:
        N += 1  # make odd so the center (t=0) is an integer index
    t = np.linspace(-duration/2, duration/2, N)
    h = np.exp(-t**2 / (2 * sigma**2)) * np.cos(2 * np.pi * f * t)
    return h, t

def gammatone(t, n=4, b=150, f=300):
    """
    Generates a gammatone function for t >= 0.
      g(t) = t^(n-1) * exp(-2*pi*b*t) * cos(2*pi*f*t)
    For t < 0, the output is zero.
    """
    y = np.zeros_like(t)
    # Only compute for nonnegative t
    t_pos = t.copy()
    t_pos[t_pos < 0] = 0
    y = (t_pos**(n-1)) * np.exp(-2*np.pi*b*t_pos) * np.cos(2*np.pi*f*t_pos)
    # Zero out negative times explicitly (if any nonzero due to t==0, it's fine)
    y[t < 0] = 0
    return y

# =============================================================================
# Demonstration and Testing of All Parts
# =============================================================================

if __name__ == "__main__":
    # ----- Part 1 Demonstration: Moving Average Filter on a Random Process -----
    N = 200
    sigma_val = 1
    lam = 0.8  # Example lambda value for the moving average filter
    
    # Generate random process and filter it
    x_rand = randprocess(N, sigma=sigma_val)
    y_movavg = movingavg(x_rand, lam=lam)
    
    plt.figure(figsize=(10,4))
    plt.plot(x_rand, label='Random Process')
    plt.plot(y_movavg, label=f'Moving Average (λ={lam})')
    plt.title("Moving Average Filter on Random Process")
    plt.xlabel("Sample")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.show()
    
    # Generate a noisy sine wave and smooth it with moving average filter
    fs = 2000  # Sampling frequency in Hz
    duration = 0.1  # seconds
    t = np.linspace(0, duration, int(fs*duration), endpoint=False)
    sine_noisy = noisy_sine(t, freq=50, noise_amp=0.5)
    y_movavg_sine = movingavg(sine_noisy, lam=lam)
    
    plt.figure(figsize=(10,4))
    plt.plot(t, sine_noisy, label="Noisy Sine")
    plt.plot(t, y_movavg_sine, label="Smoothed by Moving Average")
    plt.title("Noisy Sine Wave and Smoothed Output")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.show()
    
    # ----- Part 2 Demonstration: IIR Filter Implementations -----
    # (a) Show that filterIIR reproduces the moving average filter.
    # The recursive moving average filter: y[n] = λ*y[n-1] + (1-λ)*x[n]
    # In standard form: y[n] = (1-λ)*x[n] - (-λ)*y[n-1]
    # So use: a = [1, -λ] and b = [1-λ]
    a_mov = np.array([1, -lam])
    b_mov = np.array([1 - lam])
    y_iir_mov = filterIIR(x_rand, a_mov, b_mov)
    
    plt.figure(figsize=(10,4))
    plt.plot(x_rand, label="Random Process")
    plt.plot(y_iir_mov, label="IIR Moving Average")
    plt.title("IIR Implementation of Moving Average Filter")
    plt.xlabel("Sample")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.show()
    
    # (b) First order low-pass and high-pass IIR filters.
    # For demonstration we use:
    # Low-pass: y[n] = 0.1*x[n] + 0.9*y[n-1] -> in standard form: a = [1, -0.9], b = [0.1]
    # High-pass: y[n] = 0.1*x[n] - 0.9*y[n-1] -> in standard form: a = [1, 0.9], b = [0.1]
    a_low = np.array([1, -0.9])
    b_low = np.array([0.1])
    y_low = filterIIR(x_rand, a_low, b_low)
    
    a_high = np.array([1, 0.9])
    b_high = np.array([0.1])
    y_high = filterIIR(x_rand, a_high, b_high)
    
    plt.figure(figsize=(10,4))
    plt.plot(x_rand, label="Random Process", alpha=0.6)
    plt.plot(y_low, label="Low-pass Filter Output")
    plt.plot(y_high, label="High-pass Filter Output")
    plt.title("First Order IIR Filters")
    plt.xlabel("Sample")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.show()
    
    # (c) Second order bandpass filters.
    # Two example coefficient sets:
    # Set 1: a (excluding a0) = [-1.265, 0.81], so full a = [1, -1.265, 0.81], b = [0.135, 0, 0]
    # Set 2: a = [1, -1.702, 0.81], b = [0.063, 0, 0]
    a_band1 = np.array([1, -1.265, 0.81])
    b_band1 = np.array([0.135, 0, 0])
    y_band1 = filterIIR(x_rand, a_band1, b_band1)
    
    a_band2 = np.array([1, -1.702, 0.81])
    b_band2 = np.array([0.063, 0, 0])
    y_band2 = filterIIR(x_rand, a_band2, b_band2)
    
    plt.figure(figsize=(10,4))
    plt.plot(x_rand, label="Random Process", alpha=0.6)
    plt.plot(y_band1, label="Bandpass Filter Set 1")
    plt.plot(y_band2, label="Bandpass Filter Set 2")
    plt.title("Second Order Bandpass Filters on Random Process")
    plt.xlabel("Sample")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.show()
    
    # (d) Characterizing the filter response.
    # (i) Generate a noisy sine wave with different frequencies and noise levels,
    # then filter using Bandpass Filter Set 1.
    freqs = [50, 150, 300, 500]      # Frequencies (Hz) for the rows
    noise_levels = [1.0, 0.7, 0.4, 0.1]  # Noise amplitudes for the columns
    
    fig, axes = plt.subplots(4, 4, figsize=(12, 10), sharex=True, sharey=True)
    for i, f in enumerate(freqs):
        for j, noise_amp in enumerate(noise_levels):
            t = np.linspace(0, duration, int(fs*duration), endpoint=False)
            sine_noisy = noisy_sine(t, freq=f, noise_amp=noise_amp)
            y_filtered = filterIIR(sine_noisy, a_band1, b_band1)
            axes[i, j].plot(t, sine_noisy, label="Input", color='gray')
            axes[i, j].plot(t, y_filtered, label="Filtered", color='blue')
            if i == 0:
                axes[i, j].set_title(f"Noise={noise_amp}")
            if j == 0:
                axes[i, j].set_ylabel(f"Freq={f}Hz")
    plt.suptitle("Filter Response (Bandpass Set 1) for Various Frequencies and Noise Levels", y=0.92)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
    
    # (ii) Frequency response: vary input sine frequency from 0 to Nyquist (fs/2) and compute output power.
    freqs_resp = np.linspace(0, fs/2, 100)
    power_output = []
    for f in freqs_resp:
        t = np.linspace(0, duration, int(fs*duration), endpoint=False)
        sine_wave = np.sin(2 * np.pi * f * t)
        y_out = filterIIR(sine_wave, a_band1, b_band1)
        power = np.mean(y_out**2)
        power_output.append(power)
    plt.figure(figsize=(8,4))
    plt.plot(freqs_resp, power_output)
    plt.title("Output Signal Power vs Frequency (Bandpass Set 1)")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power")
    plt.show()
    
    # ----- Part 3: Impulse Response Functions -----
    # Compute impulse responses by filtering an impulse (delta function).
    impulse = np.zeros(N)
    impulse[0] = 1  # delta at n=0
    ir_movavg = filterIIR(impulse, a_mov, b_mov)
    ir_low = filterIIR(impulse, a_low, b_low)
    ir_high = filterIIR(impulse, a_high, b_high)
    ir_band1 = filterIIR(impulse, a_band1, b_band1)
    
    plt.figure(figsize=(10,6))
    plt.subplot(2,2,1)
    plt.stem(ir_movavg, use_line_collection=True)
    plt.title("Impulse Response: Moving Average")
    plt.subplot(2,2,2)
    plt.stem(ir_low, use_line_collection=True)
    plt.title("Impulse Response: Low-pass")
    plt.subplot(2,2,3)
    plt.stem(ir_high, use_line_collection=True)
    plt.title("Impulse Response: High-pass")
    plt.subplot(2,2,4)
    plt.stem(ir_band1, use_line_collection=True)
    plt.title("Impulse Response: Bandpass Set 1")
    plt.tight_layout()
    plt.show()
    
    # ----- Part 4: Convolution and FIR Filtering -----
    # (a) Demonstrate the convolution function by convolving x_rand with the impulse response of the moving average filter.
    # For the moving average filter the impulse response is causal so we use i0 = 0.
    y_conv = convolve_signal(x_rand, ir_movavg, i0=0)
    
    plt.figure(figsize=(10,4))
    plt.plot(x_rand, label="Original Signal")
    plt.plot(y_conv, label="Convolved Signal")
    plt.title("Convolution with Impulse Response (Moving Average)")
    plt.xlabel("Sample")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.show()
    
    # (b) FIR Filtering using a Gabor kernel.
    gabor_dur = 0.05  # seconds: duration of the Gabor kernel
    gabor_freq = 250  # Hz
    sigma_gabor = 3/250  # as suggested in the assignment
    h_gabor, t_kernel = gabor_kernel(gabor_dur, fs, gabor_freq, sigma_gabor)
    i0 = len(h_gabor) // 2  # index corresponding to time zero (center)
    # Generate a noise signal and filter it using the Gabor kernel via convolution.
    noise_signal = np.random.randn(N)
    y_fir = convolve_signal(noise_signal, h_gabor, i0=i0)
    
    plt.figure(figsize=(10,4))
    plt.subplot(2,1,1)
    plt.plot(t_kernel, h_gabor)
    plt.title("Gabor Kernel")
    plt.xlabel("Time (s)")
    plt.subplot(2,1,2)
    plt.plot(noise_signal, label="Noisy Signal", alpha=0.7)
    plt.plot(y_fir, label="FIR Filtered Output", linewidth=2)
    plt.title("FIR Filtering using Gabor Kernel")
    plt.xlabel("Sample")
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    # (c) Matched Filter Detection using a Gammatone Function.
    # Generate a gammatone signal (considered here as the signal of interest)
    t_gt = np.linspace(0, 0.05, int(fs*0.05), endpoint=False)
    gt_signal = gammatone(t_gt, n=4, b=150, f=300)
    # Test detection under different noise levels.
    noise_levels = [0.5, 1.0, 1.5]
    plt.figure(figsize=(10,6))
    for noise_amp in noise_levels:
        noisy_gt = gt_signal + noise_amp * np.random.randn(len(gt_signal))
        # The matched filter is the time-reversed gammatone signal.
        matched_filter = gt_signal[::-1]
        # Using np.convolve with 'same' mode for simplicity.
        detection = np.convolve(noisy_gt, matched_filter, mode='same')
        plt.plot(detection, label=f"Noise amp = {noise_amp}")
    plt.title("Matched Filter Detection on Gammatone Signal")
    plt.xlabel("Sample")
    plt.ylabel("Detection Output")
    plt.legend()
    plt.show()
    
    print("All code implementations have been demonstrated using only numpy and matplotlib.pyplot.")
