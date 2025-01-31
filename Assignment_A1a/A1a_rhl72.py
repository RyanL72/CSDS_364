import numpy as np


# Function to compute the normal probability density function
def g(x, mu=0, sigma=1.0):
    return (1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))


