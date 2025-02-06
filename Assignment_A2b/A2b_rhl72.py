import numpy as np
import matplotlib.pyplot as plt
import random

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