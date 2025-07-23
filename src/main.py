import matplotlib.pyplot as plt
import numpy as np
import torch

from src.utils import calculate_and_save_stats
from src.signal.signal import Signal
from src.train import train
from src.dataset import TimeSeriesDataset

if __name__ == '__main__':

    name = "test_step_by_step_learning"
    N = 12000  # dataset size
    signal_length = 500 # signal size
    all_signals = []
    residual_error = []

    model_name = "sinusoïd recognition last step only"

    #######
    # First training with only fixed pure sinusoïds
    step_name = 'pure_fixed_sinusoïd'
    print("########################")
    print(fr"step : {step_name}")
    print("########################")
    #######

    for _ in range(N):
        signal = Signal(length=signal_length, unit="s")
        signal.insert_sinus(0.05, 4, 0)
        #signal.insert_offset(0.5)
        #signal.insert_drift(0.1, 'min')
        #signal.insert_triangle(120, 20, 1, 0)
        #signal.insert_random_walk_noise(3, 0.5, 0.5, 0.1)
        #signal.insert_gaussian_noise(0, 0.05)

        all_signals.append(signal.values)  # assume .values is a NumPy array of size 500
        residual_error.append(signal.residual_mse_error())

    #
    dataset = TimeSeriesDataset(data = all_signals, signal_length = signal_length, dataset_size = N, split_ratio= 0.8, dataset_name='pure_fixed_sinusoïd')

    lr_params = {'init_learning_rate' : 1.e-4,'learning_rate_factor' : 0.5, 'learning_rate_patience':3, 'Evolution' : True}

    results_dir = train(epochs = 10, dataset= dataset, learning_rate_params= lr_params, encoded_size_ratio = 0.3, model_name = model_name,batch_size = 32)
    print("Min:", np.min(residual_error))
    print("Max:", np.max(residual_error))
    print("Has NaN?", np.isnan(residual_error).any())
    print("Has Inf?", np.isinf(residual_error).any())
    print("All zeros?", np.all(np.array(residual_error) == 0))
    calculate_and_save_stats(residual_error, f"{results_dir}/residual_error_stats.dat")
    print(len(residual_error))

    ######
    # Second training step, small variation on the amplitude
    step_name = 'small_amplitude_variation'
    print("########################")
    print(fr"step : {step_name}")
    print("########################")

    ######
    all_signals = []
    residual_error = []
    for _ in range(N):
        signal = Signal(length=signal_length, unit="s")
        signal.insert_sinus(0.05, 3.5+np.random.random(), 0)
        # signal.insert_offset(0.5)
        # signal.insert_drift(0.1, 'min')
        # signal.insert_triangle(120, 20, 1, 0)
        # signal.insert_random_walk_noise(3, 0.5, 0.5, 0.1)
        # signal.insert_gaussian_noise(0, 0.05)

        all_signals.append(signal.values)  # assume .values is a NumPy array of size 500
        residual_error.append(signal.residual_mse_error())

    #
    dataset = TimeSeriesDataset(data = all_signals, signal_length = signal_length, dataset_size = N, split_ratio= 0.8, dataset_name=step_name)

    lr_params = {'init_learning_rate' : 1.e-4,'learning_rate_factor' : 0.5, 'learning_rate_patience':5, 'Evolution' : True}

    results_dir = train(epochs = 30, dataset= dataset, learning_rate_params= lr_params, encoded_size_ratio = 0.3, model_name = model_name,batch_size = 32)


    calculate_and_save_stats(residual_error, f"{results_dir}/residual_error_stats.dat")
    print(len(residual_error))

    ######
    # Third training step, small changes in the phase
    step_name = 'small_amplitude_and_phase_variation'
    ######
    print("########################")
    print(fr"step : {step_name}")
    print("########################")
    all_signals = []
    residual_error = []
    for _ in range(N):
        signal = Signal(length=signal_length, unit="s")
        signal.insert_sinus(0.05, 3.5 + np.random.random(), 0.5 * np.random.random())
        # signal.insert_offset(0.5)
        # signal.insert_drift(0.1, 'min')
        # signal.insert_triangle(120, 20, 1, 0)
        # signal.insert_random_walk_noise(3, 0.5, 0.5, 0.1)
        # signal.insert_gaussian_noise(0, 0.05)

        all_signals.append(signal.values)  # assume .values is a NumPy array of size 500
        residual_error.append(signal.residual_mse_error())

    #
    dataset = TimeSeriesDataset(data = all_signals, signal_length = signal_length, dataset_size = N, split_ratio= 0.8, dataset_name=step_name)

    lr_params = {'init_learning_rate' : 1.e-5,'learning_rate_factor' : 0.5, 'learning_rate_patience':10, 'Evolution' : True}

    results_dir = train(epochs=30, dataset=dataset, learning_rate_params=lr_params, encoded_size_ratio=0.3,
                        model_name=model_name,batch_size = 32)

    calculate_and_save_stats(residual_error, f"{results_dir}/residual_error_stats.dat")
    print(len(residual_error))

    ######
    # Third training step, small changes in the frerquency
    step_name = 'small_amplitude_phase_frequency_variation'
    print("########################")
    print(fr"step : {step_name}")
    print("########################")
    ######
    all_signals = []
    residual_error = []
    for _ in range(N):
        signal = Signal(length=signal_length, unit="s")
        signal.insert_sinus(0.025+0.05 * np.random.random(), 3.5 + np.random.random(), 0.5 * np.random.random())
        # signal.insert_offset(0.5)
        # signal.insert_drift(0.1, 'min')
        # signal.insert_triangle(120, 20, 1, 0)
        # signal.insert_random_walk_noise(3, 0.5, 0.5, 0.1)
        # signal.insert_gaussian_noise(0, 0.05)

        all_signals.append(signal.values)  # assume .values is a NumPy array of size 500
        residual_error.append(signal.residual_mse_error())

    #
    dataset = TimeSeriesDataset(data=all_signals, signal_length=signal_length, dataset_size=N, split_ratio=0.8,
                                dataset_name=step_name)

    lr_params = {'init_learning_rate': 1.e-4, 'learning_rate_factor': 0.5, 'learning_rate_patience': 5,
                 'Evolution': True}

    results_dir = train(epochs=100, dataset=dataset, learning_rate_params=lr_params, encoded_size_ratio=0.3,
                        model_name=model_name, batch_size=32)

    calculate_and_save_stats(residual_error, f"{results_dir}/residual_error_stats.dat")
    print(len(residual_error))

    ######
    # Fourth training step, addition of small gaussian noise
    step_name = 'small_variation_and_small_gaussian_noise'
    ######
    print("########################")
    print(fr"step : {step_name}")
    print("########################")
    all_signals = []
    residual_error = []
    for _ in range(N):
        signal = Signal(length=signal_length, unit="s")
        signal.insert_sinus(0.025+0.05 * np.random.random(), 3.5 + np.random.random(), 0.5 * np.random.random())
        # signal.insert_offset(0.5)
        # signal.insert_drift(0.1, 'min')
        # signal.insert_triangle(120, 20, 1, 0)
        # signal.insert_random_walk_noise(3, 0.5, 0.5, 0.1)
        signal.insert_gaussian_noise(0, 0.05)

        all_signals.append(signal.values)  # assume .values is a NumPy array of size 500
        residual_error.append(signal.residual_mse_error())

    #
    dataset = TimeSeriesDataset(data = all_signals, signal_length = signal_length, dataset_size = N, split_ratio= 0.8, dataset_name=step_name)

    lr_params = {'init_learning_rate' : 1.e-4,'learning_rate_factor' : 0.5, 'learning_rate_patience':5, 'Evolution' : True}

    results_dir = train(epochs=100, dataset=dataset, learning_rate_params=lr_params, encoded_size_ratio=0.3,
                        model_name=model_name,batch_size = 32)

    calculate_and_save_stats(residual_error, f"{results_dir}/residual_error_stats.dat")
    print(len(residual_error))

    ######
    # Fifth training step, full random phase and high gaussian noise
    step_name = 'variation_and_small_gaussian_noise'
    ######
    print("########################")
    print(fr"step : {step_name}")
    print("########################")
    all_signals = []
    residual_error = []
    for _ in range(N):
        signal = Signal(length=signal_length, unit="s")
        signal.insert_sinus(0.025+0.2 * np.random.random(), 2 + 4*np.random.random(), 2*np.pi * np.random.random())
        # signal.insert_offset(0.5)
        # signal.insert_drift(0.1, 'min')
        # signal.insert_triangle(120, 20, 1, 0)
        # signal.insert_random_walk_noise(3, 0.5, 0.5, 0.1)
        signal.insert_gaussian_noise(0, 0.05)

        all_signals.append(signal.values)  # assume .values is a NumPy array of size 500
        residual_error.append(signal.residual_mse_error())

    dataset = TimeSeriesDataset(data = all_signals, signal_length = signal_length, dataset_size = N, split_ratio= 0.8, dataset_name=step_name)

    lr_params = {'init_learning_rate' : 1.e-4,'learning_rate_factor' : 0.5, 'learning_rate_patience':10, 'Evolution' : True}

    results_dir = train(epochs=250, dataset=dataset, learning_rate_params=lr_params, encoded_size_ratio=0.3,
                        model_name=model_name,batch_size = 32)

    calculate_and_save_stats(residual_error, f"{results_dir}/residual_error_stats.dat")
    print(len(residual_error))

    ######
    # Fifth training step, increase parameters (phase and amplidude) variation
    step_name = 'variation_and_gaussian_noise'
    ######
    print("########################")
    print(fr"step : {step_name}")
    print("########################")
    all_signals = []
    residual_error = []
    for _ in range(N):
        signal = Signal(length=signal_length, unit="s")
        signal.insert_sinus(0.025+0.2 * np.random.random(), 2 + 4*np.random.random(), np.pi * np.random.random())
        # signal.insert_offset(0.5)
        # signal.insert_drift(0.1, 'min')
        # signal.insert_triangle(120, 20, 1, 0)
        # signal.insert_random_walk_noise(3, 0.5, 0.5, 0.1)
        signal.insert_gaussian_noise(0, 0.25)

        all_signals.append(signal.values)  # assume .values is a NumPy array of size 500
        residual_error.append(signal.residual_mse_error())

    #
    dataset = TimeSeriesDataset(data = all_signals, signal_length = signal_length, dataset_size = N, split_ratio= 0.8, dataset_name=step_name)

    lr_params = {'init_learning_rate' : 0.1,'learning_rate_factor' : 0.5, 'learning_rate_patience':10, 'Evolution' : True}

    results_dir = train(epochs=920, dataset=dataset, learning_rate_params=lr_params, encoded_size_ratio=0.3, batch_size = 32,
                        model_name=model_name)

    calculate_and_save_stats(residual_error, f"{results_dir}/residual_error_stats.dat")
    print(len(residual_error))









