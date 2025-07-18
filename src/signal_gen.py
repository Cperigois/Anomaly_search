import numpy as np
from scipy.interpolate import interp1d

def triangle(lenght):
    """ Generate a piece of perfect triangle signal"""
    time_init = [0,1,2,3,4]
    values = [0,0,1,0,0]

    time = np.linspace(0,4, lenght)
    interp_triangle = interp1d(time_init, values)

    return time, interp_triangle(time)

def noising(data):
    return 0



