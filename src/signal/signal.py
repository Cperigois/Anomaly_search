import numpy as np
from patterns import triangle


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


    def insert_triangle(self, starting_time = 1, triangle_lenght=5, slope=1, amplitude = 1, plateau = 1):
        """Generate a table of luminosity distance as a function of redshift.

        If a table for this cosmology already exists, it is not recomputed.
        """

        new_values = np.concatenate((np.zeros(starting_time),triangle(triangle_lenght,slope,plateau,amplitude),
                                     np.zeros(self.lenght-starting_time-triangle_lenght)))
        self.values += new_values



    def luminosity_distance(self, z):
        """Compute the luminosity distance for a given redshift using numerical integration."""
        c = 299792.458  # Speed of light in km/s
        integral = lambda zp: 1.0 / np.sqrt(self.Omega_m * (1 + zp) ** 3 + self.Omega_L)
        dz = np.linspace(0, z, 1000)
        integral_value = np.trapz([integral(zp) for zp in dz], dz)
        d_C = (c / self.H0) * integral_value  # Comoving distance
        return (1 + z) * d_C  # Luminosity distance

    def object_age(self, z):
        """Compute the age of an object at a given redshift."""
        H0_s = self.H0 / (3.0857e19)  # Convert H0 to s^-1
        integral = lambda zp: 1.0 / ((1 + zp) * np.sqrt(self.Omega_m * (1 + zp) ** 3 + self.Omega_L))
        dz = np.linspace(0, z, 1000)
        integral_value = np.trapz([integral(zp) for zp in dz], dz)
        age = integral_value / H0_s / (3.154e7 * 1e9)  # Convert to Gyr
        return age

    def comoving_volume(self, z):
        """Compute the comoving volume enclosed within a given redshift."""
        D_C = self.comoving_distance(z)  # Comoving distance in Mpc
        V_c = (4 / 3) * math.pi * (D_C ** 3)  # Comoving volume in Mpc^3
        return V_c

    def info(self):
        """Display and return the cosmology model information."""
        info_dict = {
            "Name": self.name,
            "H0": self.H0,
            "Omega_m": self.Omega_m,
            "Omega_Lambda": self.Omega_L,
            "Reference": preset_cosmologies.get(self.name, {}).get("reference", "N/A"),
            "Table": preset_cosmologies.get(self.name, {}).get("table", "N/A")
        }

        print("\n=== Cosmology Model Information ===")
        for key, value in info_dict.items():
            print(f"{key}: {value}")

        return info_dict
    def compute_z(self, dl_array):
        "Computes the redshift from a luminosity distance array."
        self.luminosity_distance_table(self, zmax = 30, steps = 50000)
        table_filename = f"AuxiliaryFiles/z_dl_table_{self.name}.csv"
        df = pd.read_csv(table_filename)
        interpolation = InterpolatedUnivariateSpline(df['Luminosity Distance (Mpc)'], df['Redshift'])
        return interpolation(dl_array)

    def compute_dl(self, z_array):
        "Computes the luminosity distance from a redshift array."
        d_L_values = [self.luminosity_distance(z) for z in z_array]
        return d_L_values