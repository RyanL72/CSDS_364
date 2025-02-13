import numpy as np

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


    