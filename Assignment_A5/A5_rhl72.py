import numpy as np
import matplotlib.pyplot as plt
import time

def plot_complex_basis(N, k):
    n = np.arange(N)
    w = np.exp(2j * np.pi * k * n / N)
    
    fig, ax = plt.subplots()
    # Plot the unit circle
    theta = np.linspace(0, 2 * np.pi, 200)
    ax.plot(np.cos(theta), np.sin(theta), label='Unit Circle')
    # Plot discrete points
    ax.scatter(w.real, w.imag, marker='o', label='w_k[n] points')
    
    ax.set_aspect('equal')
    ax.set_title(f'Complex Representation of Basis Function (N={N}, k={k})')
    ax.set_xlabel('Real Part')
    ax.set_ylabel('Imaginary Part')
    ax.legend()
    plt.show()

def w(n, k, N):
    """
    Compute the DFT basis function w_k[n] = exp(2πi * k * n / N).
    """
    return np.exp(2j * np.pi * k * n / N)

def plotw(k, N):
    """
    Plot the real and imaginary parts of the DFT basis function for given k and N.
    Uses discrete stem plots, and for the Nyquist frequency (k == N/2), overlays continuous sine/cosine.
    """
    n = np.arange(N)
    basis = w(n, k, N)
    
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    
    # Real part
    axes[0].stem(n, basis.real, basefmt=" ")
    axes[0].set_ylabel("Re{w[k][n]}")
    axes[0].set_title(f"Real Part (N={N}, k={k})")
    
    # Imaginary part
    axes[1].stem(n, basis.imag, basefmt=" ")
    axes[1].set_ylabel("Im{w[k][n]}")
    axes[1].set_xlabel("n")
    axes[1].set_title(f"Imag Part (N={N}, k={k})")
    
    # Overlay continuous functions if k is Nyquist frequency
    if k == N // 2:
        x_cont = np.linspace(0, N-1, 1000)
        axes[0].plot(x_cont, np.cos(2 * np.pi * k * x_cont / N))
        axes[1].plot(x_cont, np.sin(2 * np.pi * k * x_cont / N))
    
    plt.tight_layout()
    plt.show()

# 1c. Orthogonality check
def basis_vector(k, N):
    return np.exp(2j * np.pi * k * np.arange(N) / N)

# 2a. Fourier matrix construction
def fourier_matrix(N):
    n = np.arange(N)
    k = n.reshape((N, 1))
    return np.exp(2j * np.pi * k * n / N)


# 2d. Benchmarking
for N in [64, 128, 256]:
    x = np.random.randn(N)
    start = time.time()
    fourier_matrix(N).dot(x)
    t_mat = time.time() - start
    start = time.time()
    np.fft.fft(x)
    t_fft = time.time() - start
    print(f"\nN={N}: Matrix mult = {t_mat:.4f}s, FFT = {t_fft:.4f}s")


