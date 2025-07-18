import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from signal_gen import triangle
from noises import gauss_noise, rw_noise

if __name__ == '__main__':

    outfunc_time , outfunc_u = triangle(200)
    gaussian_noise = gauss_noise(200, 0, 0.2)
    random_walk_noise = 0.1*rw_noise(200, 10, 0.5, 0.5,)

    #plt.plot(time_init, values, label = "triangle signal")
    plt.plot(outfunc_time, outfunc_u, label = "interpolation")
    plt.plot(outfunc_time, outfunc_u+gaussian_noise, label = "interp+gauss")
    plt.plot(outfunc_time, outfunc_u+random_walk_noise, label = "interp + symetric rw")
    plt.plot(outfunc_time, outfunc_u+random_walk_noise+gaussian_noise, label = "interp+ gauss + symmetric rw")
    plt.legend()
    plt.plot()

    plt.savefig("test_triangle.png")



