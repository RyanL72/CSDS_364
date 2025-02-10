import numpy as np
import matplotlib.pyplot as plt
import random
import math
from scipy.stats import poisson


def randtimes(N, t1, t2):
    """
    Simulate a Poisson process 

    N - number of random times

    [ t1 , t2 ] - time interval
    """
    random_times = [random.uniform(t1,t2) for _ in range(N)]
    return random_times

def plotflash(times):
    """
    Plot the given times as a stem plot with unit heights.
    """
    plt.stem(times, [1] * len(times), markerfmt='')
    plt.xlabel('Time')
    plt.ylabel('Event Height')
    plt.title('Stem Plot of Random Times')
    plt.show()

def randintervals(N, lambd, t1):
    """
    Generate N random time intervals from t1 with lambda rate.
    """
    intervals = np.random.exponential(scale=1.0/lambd, size = N)

    event_times = t1 + np.cumsum(intervals)
    
    return event_times

def poisson_pdf(n, lambd, T):
    """
    Compute Poisson probability for n events in time T
    """
    mu = lambd * T

    poisson = ((mu**n)/(np.math.factorial(n))) * math.exp(-mu)

    return poisson


def detectionprob(K, lambd=40, T=0.1):
    """
    Probability of detecting the flash
    """

    mu = lambd * T

    if(K <=0):
        return 1

    return 1.0 - poisson.cdf(K-1, mu) 

def lightflash(lambd, t1=0.8, t2=2.2):
    """
    Simulate a Poisson process with rate `lambd` within the interval [t1, t2].
    
    lambd - Poisson rate (events per unit time)
    t1, t2 - Start and end time of the  
    """
    T = t2 - t1  # Duration of process

    # Sample the number of events from a Poisson distribution
    N = np.random.poisson(lambd * T)  # Number of events
    
    # Generate random event times
    event_times = randtimes(N, t1, t2)

    # Return sorted event times
    return np.sort(event_times)

def poisson_pdf_mu(n, mu):
    """
    Compute Poisson probability for n events given a mean mu.
    """
    return (mu**n / np.math.factorial(n)) * np.exp(-mu)

def detection_probability(K, lambd=40, T=0.1):
    """
    Compute the probability of detecting at least K events in time T
    for a Poisson process with rate lambd.
    """
    mu = lambd * T
    cumulative = sum(poisson_pdf_mu(n, mu) for n in range(K))
    return 1 - cumulative

def probseeing(I, alpha=0.06, K=6):
    """
    Given I photons at the cornea, absorption rate alpha, and a photon detection threshold K,
    return the probability of seeing the flash.
    (Assumes the number of absorbed photons is Poisson distributed with mean alpha * I.)
    """
    mu = alpha * I  # Here mu is the mean number of absorbed photons.
    cumulative = sum(poisson_pdf_mu(n, mu) for n in range(K))
    return 1 - cumulative


def plotdetectioncurve(alpha=0.5, K=6, ax=None, label=None):
    """
    Plot the detection probability as a function of intensity I.
    I is the total number of photons arriving at the cornea.
    The x-axis is on a logarithmic scale and ranges from 0.01 to 100.
    """
    I_values = np.logspace(-2, 3, 200)

    prob_values = [probseeing(I, alpha=alpha, K=K) for I in I_values]
    if ax is None:
        fig, ax = plt.subplots(figsize=(8,6))
    ax.plot(I_values, np.array(prob_values)*100, label=label)
    ax.set_xscale('log')
    ax.set_xlabel("Light intensity I (photons at cornea)")
    ax.set_ylabel("Detection probability (%)")
    ax.set_title("Detection Curve (α=%.2f, K=%d)" % (alpha, K))
    ax.grid(True, which="both", ls="--")
    return ax


def plotfit(alpha, K):
    """
    Plot the detection curve for given alpha and K together with the experimental data 
    from Hecht, Shlaer and Pirenne (HSP) subject SS.
    
    """
    # Experimental data points:
    I_data = np.array([24.1, 37.6, 58.6, 91.0, 141.9, 221.3])
    perc_seen = np.array([0.0, 4.0, 18.0, 54.0, 94.0, 100.0])
    
    fig, ax = plt.subplots(figsize=(8,6))
    plotdetectioncurve(alpha=alpha, K=K, ax=ax, 
                       label=r'Detection curve: $\alpha$=%.4f, K=%d' % (alpha, K))
    
    # Overlay experimental data.
    ax.plot(I_data, perc_seen, 'ko', label="HSP subject SS")
    ax.set_title(r"Detection curve fit: $\alpha$=%.4f, K=%d" % (alpha, K))
    ax.legend()

    plt.show()