import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple
import math


# 1a, 1b: Generating signals with events and additive noise


def genwaveform(N=100, alpha=0.05, A=1, mu=0, sigma=2, noisetype='uniform'):
    t = np.linspace(0, 1, N)
    signal = A * np.sin(2 * np.pi * t / alpha)
    
    if noisetype == 'Gaussian':
        noise = np.random.normal(mu, sigma, N)
    elif noisetype == 'uniform':
        noise = np.random.uniform(mu - sigma, mu + sigma, N)
    else:
        raise ValueError("Unsupported noise type. Use 'Gaussian' or 'uniform'.")
    
    waveform = signal + noise
    return t, waveform

def plot_waveform(y, event_indices, title="Waveform with Events"):
    plt.figure(figsize=(8,4))
    plt.plot(y, '-o', label='Waveform')
    plt.plot(event_indices, y[event_indices], 'ro', label='Events')
    plt.title(title)
    plt.xlabel('Sample index')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.tight_layout()
    plt.show()


# 2b: Detection counts (TP, FN, FP, TN)


def detectioncounts(event_indices, y, theta):
    DetectionCounts = namedtuple("DetectionCounts", ["tp", "fn", "fp", "tn"])
    N = len(y)

    actual = np.zeros(N, dtype=bool)
    actual[event_indices] = True
    detected = (y >= theta)

    tp = np.sum(actual & detected)
    fn = np.sum(actual & ~detected)
    fp = np.sum(~actual & detected)
    tn = np.sum(~actual & ~detected)

    return DetectionCounts(tp, fn, fp, tn)

def plot_detections(y, event_indices, theta, title="Detections"):
    plt.figure(figsize=(8,4))
    plt.plot(y, 'k.-', label='Waveform')
    plt.axhline(theta, color='r', linestyle='--', label=f'Threshold={theta:.2f}')

    N = len(y)
    actual = np.zeros(N, dtype=bool)
    actual[event_indices] = True
    detected = (y >= theta)

    tp_idx = np.where(actual & detected)[0]
    fn_idx = np.where(actual & ~detected)[0]
    fp_idx = np.where(~actual & detected)[0]

    plt.plot(tp_idx, y[tp_idx], 'go', label='True Positive')
    plt.plot(fn_idx, y[fn_idx], 'rx', label='False Negative')
    plt.plot(fp_idx, y[fp_idx], 'ms', mfc='none', label='False Positive')

    plt.title(title)
    plt.xlabel('Sample index')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.tight_layout()
    plt.show()


# 2c: False positive and false negative probabilities


def phi(z):
    return 0.5 * (1 + math.erf(z / np.sqrt(2)))

def falsepos(theta, A=1.0, sigma=1.0, noisetype='gaussian'):
    noisetype = noisetype.lower()
    if noisetype == 'gaussian':
        return 1.0 - phi(theta / sigma)
    elif noisetype == 'uniform':
        low, high = -sigma/2, sigma/2
        if theta <= low:
            return 1.0
        elif theta >= high:
            return 0.0
        else:
            return (high - theta) / (high - low)
    else:
        raise ValueError("noisetype must be 'gaussian' or 'uniform'.")

def falseneg(theta, A=1.0, sigma=1.0, noisetype='gaussian'):
    noisetype = noisetype.lower()
    if noisetype == 'gaussian':
        return phi((theta - A) / sigma)
    elif noisetype == 'uniform':
        low, high = -sigma/2, sigma/2
        delta = theta - A
        if delta >= high:
            return 1.0
        elif delta <= low:
            return 0.0
        else:
            return (delta - low) / (high - low)
    else:
        raise ValueError("noisetype must be 'gaussian' or 'uniform'.")


# 3b: ROC curve


def plotROC(A=1.0, sigma=1.0, noisetype='gaussian', theta_range=None, npoints=100):
    if theta_range is None:
        lower = -1.0 * sigma
        upper = A + 3.0 * sigma
        thetas = np.linspace(lower, upper, npoints)
    else:
        thetas = np.linspace(theta_range[0], theta_range[1], npoints)

    fpr_vals = []
    tpr_vals = []
    for th in thetas:
        fp_val = falsepos(th, A=A, sigma=sigma, noisetype=noisetype)
        fn_val = falseneg(th, A=A, sigma=sigma, noisetype=noisetype)
        tp_val = 1.0 - fn_val

        fpr_vals.append(fp_val)
        tpr_vals.append(tp_val)

    plt.figure(figsize=(5,5))
    plt.plot(fpr_vals, tpr_vals, 'b-o', label='ROC')
    plt.plot([0,1], [0,1], 'k--', label='Chance line')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve: {noisetype}, A={A}, sigma={sigma}')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# Run a demo if executed as a script


def main_demo():
    y_gauss, si_gauss = genwaveform(N=50, alpha=0.2, A=2, sigma=1.0, noisetype='gaussian', seed=0)
    plot_waveform(y_gauss, si_gauss, title="Gaussian Noise Example")

    y_uniform, si_uniform = genwaveform(N=50, alpha=0.2, A=2, sigma=1.0, noisetype='uniform', seed=0)
    plot_waveform(y_uniform, si_uniform, title="Uniform Noise Example")

    theta = 1.5
    dc_gauss = detectioncounts(si_gauss, y_gauss, theta)
    print("Gaussian noise detection counts (TP, FN, FP, TN):", dc_gauss)
    plot_detections(y_gauss, si_gauss, theta, title="Detections (Gaussian)")

    tp_rate_emp = dc_gauss.tp / (dc_gauss.tp + dc_gauss.fn)
    fp_rate_emp = dc_gauss.fp / (dc_gauss.fp + dc_gauss.tn)
    print(f"Empirical TPR (Gaussian) = {tp_rate_emp:.3f}, FPR (Gaussian) = {fp_rate_emp:.3f}")

    fp_theo = falsepos(theta, A=2, sigma=1.0, noisetype='gaussian')
    fn_theo = falseneg(theta, A=2, sigma=1.0, noisetype='gaussian')
    tp_theo = 1.0 - fn_theo
    print(f"Theoretical TPR (Gaussian) = {tp_theo:.3f}, FPR (Gaussian) = {fp_theo:.3f}")

    plotROC(A=2, sigma=1.0, noisetype='gaussian', theta_range=(0,4), npoints=50)

