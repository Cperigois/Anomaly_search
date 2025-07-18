import numpy as np

def gauss_noise(lenght, mu, sigma) :
    "Generate a gaussian noise centered on mu"
    gaussian_noise = np.random.normal(mu, sigma, lenght)
    return gaussian_noise

def rw_noise(lenght, amplitude, prob_up, prob_down) :
    # Probability for moving down and up
    prob = [prob_down, prob_up]

    # Start from position 2
    start = 2
    positions = [start]

    rr = np.random.random(lenght-1)
    downp = rr < prob[0]
    upp = rr > prob[1]

    for idownp, iupp in zip(downp, upp):
        down = idownp and positions[-1] > -amplitude
        up = iupp and positions[-1] < amplitude
        positions.append(positions[-1] - down + up)
    return np.array(positions)