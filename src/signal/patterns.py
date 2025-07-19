import numpy as np
from scipy.interpolate import interp1d

def triangle(lenght, slope = 1, plateau = 1, before_tri = 1, after_tri = 1):
    """ Generate a piece of perfect triangle signal"""

    pattern_size = 2+before_tri+after_tri+plateau

    time_init = np.arange(pattern_size-1)
    values = slope * np.concatenate((np.zeros(before_tri+1), np.ones(plateau), np.zeros(after_tri+1)))

    time = np.linspace(0,pattern_size-1, lenght)
    interp_triangle = interp1d(time_init, values)

    return time, interp_triangle(time)

def insert_pattern(lenght, peak) :
    time  = np.arange(lenght)
    values = np.zeros(lenght)








