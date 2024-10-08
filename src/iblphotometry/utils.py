import numpy as np
from scipy import signal
import pynapple as nap
from ibldsp.utils import WindowGenerator


def z(A: np.array):
    return (A - np.average(A)) / np.std(A)


def mad(A: np.array):
    return np.median(np.absolute(A - np.median(A)))


def zscore(F: nap.Tsd):
    """pynapple friendly zscore

    Args:
        F (nap.Tsd): signal to be z-scored

    Returns:
        _type_: z-scored nap.Tsd
    """
    y, t = F.values, F.times()
    # mu, sig = np.average(y), np.std(y)
    return nap.Tsd(t=t, d=z(y))


def filter(F: nap.Tsd, N: int, Wn: float, fs: float = None, btype="low"):
    """a pynapple friendly wrapper for scipy.signal.butter and subsequent sosfiltfilt"""
    y, t = F.values, F.times()
    if fs is None:
        fs = 1 / np.median(np.diff(t))
    sos = signal.butter(N, Wn, btype, fs=fs, output="sos")
    y_filt = signal.sosfiltfilt(sos, y)
    return nap.Tsd(t=t, d=y_filt)


def make_sliding_window(
    A: np.ndarray, w_size: int, pad_mode="edge", method="stride_tricks"
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

    if method == "stride_tricks":
        if w_size % 2 != 0:
            raise ValueError("w_size needs to be an even number")
        B = np.lib.stride_tricks.as_strided(A, ((n_samples - w_size), w_size), (8, 8))

    if method == "window_generator":
        wg = WindowGenerator(n_samples - 1, w_size, w_size - 1)
        dtype = np.dtype((np.float64, w_size))
        B = np.fromiter(wg.slice_array(A), dtype=dtype)

    if pad_mode is not None:
        B = np.pad(B, ((int(w_size / 2), int(w_size / 2)), (0, 0)), mode=pad_mode)
    return B


def sliding_z(F: nap.Tsd, w_size: int, weights=None):
    """sliding window z-score of a pynapple time series with data with optional weighting

    Args:
        F (nap.Tsd): Signal to be zscored
        w_size (int): window size
        w_weights (_type_, optional): weights of the window . Defaults to None.

    Returns:
        _type_: _description_
    """
    y, t = F.values, F.times()
    B = make_sliding_window(y, w_size)
    # this will not work with stride tricks!
    # if weights is not None:
    #     # normalize
    #     weights /= weights.sum()
    #     B = B * weights[np.newaxis, :]
    mus, sds = np.average(B, axis=1), np.std(B, axis=1)
    F_sz = (y - mus) / sds
    return nap.Tsd(t=t, d=F_sz)
