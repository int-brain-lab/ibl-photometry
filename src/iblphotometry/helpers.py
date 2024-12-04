import numpy as np
from scipy import signal
import pandas as pd
from ibldsp.utils import WindowGenerator
from iblutil.numerical import rcoeff


# TODO move to processing
def z(A: np.array, mode='classic'):
    """classic z-score. Deviation from sample mean in units of sd

    Args:
        A (np.array): _description_

    Returns:
        _type_: _description_
    """
    if mode == 'classic':
        return (A - np.average(A)) / np.std(A)
    if mode == 'median':
        return (A - np.median(A)) / np.std(A)


# TODO move to processing
def mad(A: np.array):
    """
    https://en.wikipedia.org/wiki/Median_absolute_deviation
    :the MAD is defined as the median of the absolute deviations from the data's median

    Args:
        A (np.array): _description_

    Returns:
        _type_: _description_
    """
    return np.median(np.absolute(A - np.median(A)), axis=-1)


# TODO move to processing
def madscore(F: pd.Series):
    y, t = F.values, F.index.values
    return pd.Series(mad(y), index=t)


# TODO move to processing
def zscore(F: pd.Series, mode='classic'):
    """pynapple friendly zscore

    Args:
        F (nap.Tsd): signal to be z-scored

    Returns:
        _type_: z-scored nap.Tsd
    """
    y, t = F.values, F.index.values
    # mu, sig = np.average(y), np.std(y)
    return pd.Series(z(y, mode=mode), index=t)


# TODO move to processing
def filt(F: pd.Series, N: int, Wn: float, fs: float = None, btype='low'):
    """a pynapple friendly wrapper for scipy.signal.butter and sosfiltfilt

    Args:
        F (nap.Tsd): _description_
        N (int): _description_
        Wn (float): _description_
        fs (float, optional): _description_. Defaults to None.
        btype (str, optional): _description_. Defaults to "low".

    Returns:
        _type_: _description_
    """
    y, t = F.values, F.index.values
    if fs is None:
        fs = 1 / np.median(np.diff(t))
    sos = signal.butter(N, Wn, btype, fs=fs, output='sos')
    y_filt = signal.sosfiltfilt(sos, y)
    return pd.Series(y_filt, index=t)


def sliding_rcoeff(signal_a, signal_b, nswin, overlap=0):
    """
    Computes the local correlation coefficient between two signals in sliding windows
    :param signal_a:
    :param signal_b:
    :param nswin: window size in samples
    :param overlap: overlap of successiv windows in samples
    :return: ix: indices of the center of the windows, r: correlation coefficients
    """
    wg = WindowGenerator(ns=signal_a.size, nswin=nswin, overlap=overlap)
    first_samples = np.array([fl[0] for fl in wg.firstlast])
    iwin = np.zeros([wg.nwin, wg.nswin], dtype=np.int32) + np.arange(wg.nswin)
    iwin += first_samples[:, np.newaxis]
    iwin[iwin >= signal_a.size] = signal_a.size - 1
    r = rcoeff(signal_a[iwin], signal_b[iwin])
    ix = first_samples + nswin // 2
    return ix, r
