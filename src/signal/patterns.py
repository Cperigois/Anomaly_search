import numpy as np


def triangle(length, plateau_size = 1, amplitude = 1):
    """ Generate a piece of perfect triangle signal"""


    variation_length = (length-plateau_size)/2
    slope = amplitude/variation_length

    values = np.concatenate((slope * np.arange(variation_length),
                             amplitude * np.ones(plateau_size),
                             -slope * np.arange(variation_length) + amplitude))

    return values









