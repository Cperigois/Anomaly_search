import numpy as np
from src.signal.patterns import triangle
from src.signal.noises import gauss_noise, rw_noise

class Signal :

    def __init__(self, lenght: int, unit:str = "s", time = None, values = None):
        """Initializes a signal with a lenght and a time unit

        :param lenght: (int) lenght of the signal.
        :param unit: (str) time unit {'s','min','h','d','month','y'}.
        """

        self.lenght = lenght
        self.unit = unit
        if time :
            self.time = time
        else :
            self.time = np.arange(lenght)
        if values :
            self.values = values
        else :
            self.values = np.zeros(lenght)

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
        return cls(lenght=len(time_array), unit=unit, time = time_array , values = values_array)

    def insert_triangle(self, starting_time = 1, triangle_lenght=5, amplitude = 1, plateau = 1):
        """Generate a signal triangle and add it to the values of the instance.
        """

        new_values = np.concatenate((np.zeros(starting_time),triangle(lenght=triangle_lenght,plateau_size = plateau,amplitude = amplitude),
                                     np.zeros(self.lenght-starting_time-triangle_lenght)))
        self.values += new_values

    def insert_gaussian_noise(self,  mu, sigma):
        """Generate a signal triangle and add it to the values of the instance.
                """

        new_values = gauss_noise(lenght = self.lenght, mu = mu, sigma = sigma)
        self.values += new_values

    def insert_random_walk_noise(self, amplitude, prob_up, prob_down, rescale):

        new_values = rescale * rw_noise(lenght=self.lenght, amplitude = amplitude,prob_up=prob_up, prob_down = prob_down)
        self.values += new_values


