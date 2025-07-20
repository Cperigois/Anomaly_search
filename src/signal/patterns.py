import numpy as np


def triangle(lenght, plateau_size = 1, amplitude = 1):
    """ Generate a piece of perfect triangle signal"""


    variation_lenght = (lenght-plateau_size)/2
    slope = amplitude/variation_lenght

    values = np.concatenate((slope * np.arange(variation_lenght),
                             amplitude * np.ones(plateau_size),
                             -slope * np.arange(variation_lenght) + amplitude))

    return values









