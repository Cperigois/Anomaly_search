import numpy as np
from src.signal.patterns import triangle
from src.signal.noises import gauss_noise, rw_noise
from src.signal.utils import time_conversion

class Signal :

    def __init__(self, length: int, unit:str = "s", time = None, values = None):
        """Initializes a signal with a length and a time unit

        :param length: (int) length of the signal.
        :param unit: (str) time unit {'s','min','h','d','w','month','y'}.
        """

        self.length = length
        self.unit = unit
        if time :
            self.time = time
        else :
            self.time = np.arange(length)
        if values :
            self.values = values
        else :
            self.values = np.zeros(length)
        self.random_noises = np.zeros(length)

    def save(self, file_name):
        """Save the cosmological model in presets.json."""
        time_column_name = 'time'

        df = pd.DataFrame({
            "time"+self.unit: self.time,
            "values": self.values
        })
        df.to_csv(filename, index = None, sep = '\t')


    @classmethod
    def load(cls, time_array, values_array, unit):
        """Load an existing model from presets.json or create a new one with default values."""
        return cls(length=len(time_array), unit=unit, time = time_array , values = values_array)

    def insert_triangle(self, starting_time = 1, triangle_length=5, amplitude = 1, plateau = 1):
        """Generate a signal triangle and add it to the values of the instance.
        """

        new_values = np.concatenate((np.zeros(starting_time),triangle(length=triangle_length,plateau_size = plateau,amplitude = amplitude),
                                     np.zeros(self.length-starting_time-triangle_length)))
        self.values += new_values

    def insert_gaussian_noise(self,  mu, sigma):
        """Generate a signal triangle and add it to the values of the instance.
                """

        new_values = gauss_noise(length = self.length, mu = mu, sigma = sigma)
        self.random_noises += new_values
        self.values += new_values

    def insert_random_walk_noise(self, amplitude, prob_up, prob_down, rescale):

        new_values = rescale * rw_noise(length=self.length, amplitude = amplitude,prob_up=prob_up, prob_down = prob_down)
        self.random_noises += new_values
        self.values += new_values

    def insert_sinus(self, freq, amplitude, phase = 0):
        self.values += amplitude * np.sin(freq * self.time + phase)

    def insert_offset(self, offset):
        self.values += offset * np.ones(len(self.values))

    def insert_drift(self,drift_per_time_unit, time_unit):

        drift = 1./time_conversion(1/drift_per_time_unit, time_unit, self.unit) * self.length
        self.values += np.linspace(0, drift, self.length)

    def residual_mse_error(self):
        """
        Computes the residual values for the loss, from the noise.
        :param loss: function
        :return:
        """
        return np.mean(self.random_noises ** 2)


