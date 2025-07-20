import matplotlib.pyplot as plt

from src.signal.signal import Signal

if __name__ == '__main__':

    signal = Signal(lenght = 500, unit ="s" )
    signal.insert_triangle(120,80, 5, 4)
    signal.insert_triangle(230,20,1,0)
    signal.insert_random_walk_noise(3,0.5,0.5, 0.1)
    signal.insert_gaussian_noise(0,0.05)

    #plt.plot(time_init, values, label = "triangle signal")
    plt.plot(signal.time, signal.values, label = "signal")
    plt.legend()

    plt.savefig("test_signal_class.png")



