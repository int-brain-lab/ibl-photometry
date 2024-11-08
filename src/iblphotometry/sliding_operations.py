"""this module holds a collection of processing pipelines for fiber photometry data"""

import numpy as np
import pandas as pd
import pynapple as nap
from iblphotometry.helpers import z, psth
from ibldsp.utils import WindowGenerator


def make_sliding_window(
    A: np.ndarray,
    w_size: int,
    pad_mode='edge',
    method='stride_tricks',
    warning=None,
):
    """use np.stride_tricks to make a sliding window view of a 1-d np.array A
    full overlap, step size 1
    assumes 8 byte numbers (to be exposed? but needs to be tested first)
    pads beginning and end of array with edge values

    Args:
        A (np.ndarray): Array to be windowed
        w_size (int): window size in samples

    Raises:
        ValueError: is raised if w_size is not an even number

    Returns:
        _type_: The view on the array, with shape (A.shape[0], w_size)
    """
    n_samples = A.shape[0]

    if method == 'stride_tricks':
        if w_size % 2 != 0:
            if warning == 'raise_exception':
                raise ValueError('w_size needs to be an even number')
            else:
                w_size += 1  # dangerous
        B = np.lib.stride_tricks.as_strided(A, ((n_samples - w_size), w_size), (8, 8))

    if method == 'window_generator':
        wg = WindowGenerator(n_samples - 1, w_size, w_size - 1)
        dtype = np.dtype((np.float64, w_size))
        B = np.fromiter(wg.slice_array(A), dtype=dtype)

    if pad_mode is not None:
        B = np.pad(B, ((int(w_size / 2), int(w_size / 2)), (0, 0)), mode=pad_mode)
    return B


def sliding_dFF(F: nap.Tsd, w_len: float, fs=None, weights=None):
    y, t = F.values, F.times()
    fs = 1 / np.median(np.diff(t)) if fs is None else fs
    w_size = int(w_len * fs)

    def _dFF(A: np.array):
        return (A - np.average(A)) / np.average(A)

    if weights is not None:
        # note: passing weights makes the stride trick not possible, or only with allocating a matrix of shape (n_samples * w_size)
        # true sliding operation implemented, much slower
        weights /= weights.sum()
        n_samples = y.shape[0]
        wg = WindowGenerator(n_samples - 1, w_size, w_size - 1)
        d = [
            np.sum(_dFF(F[first:last].values) * weights)
            for (first, last) in wg.firstlast
        ]
    else:
        B = make_sliding_window(y, w_size)
        mus = np.average(B, axis=1)
        d = (y - mus) / mus
    return nap.Tsd(t=t, d=d)


def sliding_z(F: nap.Tsd, w_len: float, fs=None, weights=None):
    """sliding window z-score of a pynapple time series with data with optional weighting

    Args:
        F (nap.Tsd): Signal to be zscored
        w_size (int): window size
        w_weights (_type_, optional): weights of the window . Defaults to None.

    Returns:
        _type_: _description_
    """
    y, t = F.values, F.times()
    fs = 1 / np.median(np.diff(t)) if fs is None else fs
    w_size = int(w_len * fs)

    if weights is not None:
        # note: passing weights makes the stride trick not possible, or only with allocating a matrix of shape (n_samples * w_size)
        # true sliding operation implemented, much slower
        weights /= weights.sum()
        n_samples = y.shape[0]
        wg = WindowGenerator(n_samples - 1, w_size, w_size - 1)
        d = [
            np.sum(z(F[first:last].values) * weights) for (first, last) in wg.firstlast
        ]
    else:
        B = make_sliding_window(y, w_size)
        mus, sds = np.average(B, axis=1), np.std(B, axis=1)
        d = (y - mus) / sds
    return nap.Tsd(t=t, d=d)


def sliding_mad(F: nap.Tsd, w_len: float = None, fs=None, overlap=90):
    y, t = F.values, F.times()
    fs = 1 / np.median(np.diff(t)) if fs is None else fs
    w_size = int(w_len * fs)

    n_samples = y.shape[0]
    wg = WindowGenerator(ns=n_samples, nswin=w_size, overlap=overlap)
    trms = np.array([first for first, last in wg.firstlast]) / fs + t[0]

    rmswin, _ = psth(y, t, t_events=trms, fs=fs, peri_event_window=[0, w_len])
    gain = np.nanmedian(np.abs(y)) / np.nanmedian(np.abs(rmswin), axis=0)
    gain = np.interp(t, trms, gain)
    return nap.Tsd(t=t, d=y * gain)


# def sliding_mad_new(F: nap.Tsd, w_len: float = None, fs=None, overlap=90):
#     y, t = F.values, F.times()
#     fs = 1 / np.median(np.diff(t)) if fs is None else fs
#     w_size = int(w_len * fs)

#     B = make_sliding_window(y, w_size)
#     gain = np.nanmedian(y) / np.nanmedian(B, axis=1)
#     return nap.Tsd(t=t, d=y * gain)

# wg = WindowGenerator(ns=n_samples, nswin=w_size, overlap=overlap)
# trms = np.array([first for first, last in wg.firstlast]) / fs + t[0]

# rmswin, _ = psth(y, t, t_events=trms, fs=fs, peri_event_window=[0, w_len])
# gain = np.nanmedian(np.abs(y)) / np.nanmedian(np.abs(rmswin), axis=0)
# gain = np.interp(t, trms, gain)
# return nap.Tsd(t=t, d=y * gain)
