"""
This module is useful to create synthetic data for testing and benchmarking purposes.
"""

import numpy as np
import pandas as pd
import scipy.signal


def synthetic101(fs=30, rl=1000, event_rate=.2):
    """
    Generates synthetic photometry data with
    :param fs: sampling frequency
    :param rl: recording length in seconds
    :param event_rate: rate of events generating transients in Hz
    """
    ns = fs * rl
    tscale = np.arange(ns) / fs
    photobleach = np.exp(-(tscale + 1) / 200)
    ric = scipy.signal.ricker(int(fs * 4), 8)

    event_times = np.cumsum(- np.log(np.random.rand(int(rl * event_rate))) / event_rate)
    event_times = event_times[:np.searchsorted(event_times, rl - 10)]

    transients = np.zeros(ns)
    transients[np.int32(event_times * fs - len(ric) / 2)] = 1
    transients = np.convolve(transients, ric / np.max(ric), mode='full')
    isosbestic = photobleach * 0.78 + 1.2
    calcium = photobleach * 1.00 + 3.21 + transients[:ns] * 0.05
    # plt.figure()
    # # plt.plot(ric)
    # plt.plot(isosbestic)
    # plt.plot(calcium)
    return pd.DataFrame({'times': tscale, 'raw_isosbestic': isosbestic, 'raw_calcium': calcium}), event_times