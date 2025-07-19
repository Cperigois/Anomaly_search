import matplotlib.pyplot as plt

from src.signal.patterns import triangle
from noises import gauss_noise, rw_noise

if __name__ == '__main__':

    outfunc_time , outfunc_u = triangle(500)
    gaussian_noise = gauss_noise(500, 0, 0.05)
    random_walk_noise = 0.1*rw_noise(500, 3, 0.5, 0.5,)

    #plt.plot(time_init, values, label = "triangle signal")
    plt.plot(outfunc_time, outfunc_u, label = "interpolation")
    plt.plot(outfunc_time, outfunc_u+gaussian_noise, label = "interp+gauss")
    plt.plot(outfunc_time, outfunc_u+random_walk_noise, label = "interp + symetric rw")
    plt.plot(outfunc_time, outfunc_u+random_walk_noise+gaussian_noise, label = "interp+ gauss + symmetric rw")
    plt.legend()
    plt.plot()

    plt.savefig("test_triangle.png")



