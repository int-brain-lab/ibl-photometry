import numpy as np
from scipy import signal
import pynapple as nap
from ibldsp.utils import WindowGenerator
from iblutil.numerical import rcoeff


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


def madscore(F: nap.Tsd):
    y, t = F.values, F.times()
    return nap.Tsd(t=t, d=mad(y))


def zscore(F: nap.Tsd, mode='classic'):
    """pynapple friendly zscore

    Args:
        F (nap.Tsd): signal to be z-scored

    Returns:
        _type_: z-scored nap.Tsd
    """
    y, t = F.values, F.times()
    # mu, sig = np.average(y), np.std(y)
    return nap.Tsd(t=t, d=z(y, mode=mode))


def filt(F: nap.Tsd, N: int, Wn: float, fs: float = None, btype='low'):
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
    y, t = F.values, F.times()
    if fs is None:
        fs = 1 / np.median(np.diff(t))
    sos = signal.butter(N, Wn, btype, fs=fs, output='sos')
    y_filt = signal.sosfiltfilt(sos, y)
    return nap.Tsd(t=t, d=y_filt)


def psth(calcium, times, t_events, fs=None, peri_event_window=None):
    """
    Compute the peri-event time histogram of a calcium signal
    :param calcium:
    :param times:
    :param t_events:
    :param fs:
    :param peri_event_window:
    :return:
    """
    fs = 1 / np.median(np.diff(times)) if fs is None else fs
    peri_event_window = [-1, 2] if peri_event_window is None else peri_event_window
    # compute a vector of indices corresponding to the perievent window at the given sampling rate
    sample_window = np.round(
        np.arange(peri_event_window[0] * fs, peri_event_window[1] * fs + 1)
    ).astype(int)
    # we inflate this vector to a 2d array where each column corresponds to an event
    idx_psth = np.tile(sample_window[:, np.newaxis], (1, t_events.size))
    # we add the index of each event too their respective column
    idx_event = np.searchsorted(times, t_events)
    idx_psth += idx_event
    i_out_of_bounds = np.logical_or(idx_psth > (calcium.size - 1), idx_psth < 0)
    idx_psth[i_out_of_bounds] = -1
    psth = calcium[idx_psth]  # psth is a 2d array (ntimes, nevents)
    psth[i_out_of_bounds] = np.nan  # remove events that are out of bounds
    return psth, idx_psth


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
