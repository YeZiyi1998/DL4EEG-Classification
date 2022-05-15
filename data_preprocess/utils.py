# official preprocessing code of search-brainwave
from math import log2
from functools import reduce
import numpy as np
import torch as t
from scipy.fftpack import fft


def get_window_function(type, window_points):
    pi = t.acos(t.zeros(1)).item() * 2
    if type == 'hanning':
        window_tensor = t.tensor(
            [0.5 - t.cos(t.tensor(2 * pi * x / (window_points + 1))) for x in range(1, window_points + 1)])
    else:
        window_tensor = None
    return window_tensor


def DE(trial, window_length, window_type, time_sample_rate, frequency_sample_rate, bands):
    # shape
    channels = trial.size(0)
    timesteps = trial.size(1)
    windows = int(timesteps / time_sample_rate / window_length)
    bands_num = len(bands['eeg']) - 1
    rate = int(frequency_sample_rate / time_sample_rate)  # 频时采样比率
    frequency_window_points = int(window_length * frequency_sample_rate)
    # Declare DE tensor
    DE = t.zeros(channels, windows, bands_num)
    # get window function
    time_window_points = int(window_length * time_sample_rate)
    window_function = get_window_function(window_type, time_window_points)
    # compute DE of a trial
    for i in range(channels):
        # Apply different band division strategies to the two data sets
        cur_bands = bands['eeg']
        for j in range(windows):
            # Apply window function
            data_mul_window = trial[i, j * time_window_points:(
                j + 1) * time_window_points] * window_function
            # Apply DFT
            fft_data = abs(fft(data_mul_window.numpy(), frequency_window_points)[
                           :int(frequency_window_points / 2)])
            # compute DE
            for k in range(bands_num):
                bands_list = fft_data[int(
                    cur_bands[k] * rate):int(cur_bands[k + 1] * rate - 1)]
                DE[i][j][k] = log2(100 * reduce(lambda x, y: x + y,
                                   map(lambda x: x * x, bands_list)) / len(bands_list))
    DE = DE.transpose(1, 0)
    return DE


def bandpower(data, sf, band, method='welch', window_sec=None, relative=False):
    from scipy.signal import welch
    from scipy.integrate import simps
    from mne.time_frequency import psd_array_multitaper
    band = np.asarray(band)
    low, high = band
    # Compute the modified periodogram (Welch)
    if method == 'welch':
        if window_sec is not None:
            nperseg = window_sec * sf
        else:
            nperseg = (2 / low) * sf
        freqs, psd = welch(data, sf, nperseg=nperseg)
    elif method == 'multitaper':
        psd, freqs = psd_array_multitaper(data, sf, adaptive=True,
                                          normalization='full', verbose=0)
    # Frequency resolution
    freq_res = freqs[1] - freqs[0]
    # Find index of band in frequency vector
    idx_band = np.logical_and(freqs >= low, freqs <= high)
    # Integral approximation of the spectrum using parabola (Simpson's rule)
    bp = simps(psd[idx_band], dx=freq_res)
    if relative:
        bp /= simps(psd, dx=freq_res)
    return bp
