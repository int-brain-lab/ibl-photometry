import numpy as np
import pandas as pd
import pynapple as nap
from scipy import stats
from scipy.stats import ttest_ind

from iblphotometry.helpers import z
from iblphotometry.behavior import psth
from iblphotometry.bleach_corrections import Regression, ExponDecay
from iblphotometry.outlier_detection import detect_spikes, grubbs_sliding


## this approach works as well
# from functools import singledispatch
# @singledispatch
# def percentile_dist(A: nap.Tsd, pc: tuple = (50, 95), axis: None = None) -> float:
#     """the distance between two percentiles in units of z
#     should be proportional to SNR, assuming the signal is
#     in the positive 5th percentile
#     """
#     P = np.percentile(z(A.values), pc)
#     return P[1] - P[0]

# @percentile_dist.register
# def _(A: np.ndarray, pc: tuple = (50, 95), axis: int = -1) -> float:
#     P = np.percentile(z(A), pc, axis=axis)
#     return P[1] - P[0]


def percentile_dist(A: nap.Tsd | np.ndarray, pc: tuple = (50, 95), axis=-1) -> float:
    """the distance between two percentiles in units of z. Captures the magnitude of transients.

    Args:
        A (nap.Tsd | np.ndarray): the input data, np.ndarray for stride tricks sliding windows
        pc (tuple, optional): percentiles to be computed. Defaults to (50, 95).
        axis (int, optional): only for arrays, the axis to be computed. Defaults to -1.

    Returns:
        float: the value of the metric
    """
    if isinstance(A, nap.Tsd):  # "overloading"
        P = np.percentile(z(A.values), pc)
    elif isinstance(A, np.ndarray):
        P = np.percentile(z(A), pc, axis=axis)
    else:
        raise TypeError('A must be nap.Tsd or np.ndarray.')
    return P[1] - P[0]


def signal_asymmetry(A: nap.Tsd | np.ndarray, pc_comp: int = 95, axis=-1) -> float:
    """the ratio between the distance of two percentiles to the median. Proportional to the the signal to noise.

    Args:
        A (nap.Tsd | np.ndarray): _description_
        pc_comp (int, optional): _description_. Defaults to 95.
        axis (int, optional): _description_. Defaults to -1.

    Returns:
        float: _description_
    """
    if not (isinstance(A, nap.Tsd) or isinstance(A, np.ndarray)):
        raise TypeError('A must be nap.Tsd or np.ndarray.')

    a = np.absolute(percentile_dist(A, (50, pc_comp), axis=axis))
    b = np.absolute(percentile_dist(A, (100 - pc_comp, 50), axis=axis))
    return a / b


def signal_skew(A: nap.Tsd | np.ndarray, axis=-1) -> float:
    if isinstance(A, nap.Tsd):
        P = stats.skew(A.values)
    elif isinstance(A, np.ndarray):
        P = stats.skew(A, axis=axis)
    else:
        raise TypeError('A must be nap.Tsd or np.ndarray.')
    return P


def n_unique_samples(A: nap.Tsd | np.ndarray) -> int:
    """number of unique samples in the signal. Low values indicate that the signal during acquisition was not within the range of the digitizer."""
    if isinstance(A, nap.Tsd):
        return np.unique(A.values).shape[0]
    elif isinstance(A, np.ndarray):
        return A.shape[0]
    else:
        raise TypeError('A must be nap.Tsd or np.ndarray.')


def n_spikes(A: nap.Tsd | np.ndarray, sd: int = 5):
    """count the number of spike artifacts in the recording."""
    if isinstance(A, nap.Tsd):
        return detect_spikes(A.values, sd=sd).shape[0]
    elif isinstance(A, np.ndarray):
        return detect_spikes(A, sd=sd).shape[0]
    else:
        raise TypeError('A must be nap.Tsd or np.ndarray.')


def n_outliers(
    A: nap.Tsd | np.ndarray, w_size: int = 1000, alpha: float = 0.0005
) -> int:
    """counts the number of outliers as detected by grubbs test for outliers.
    int: _description_
    """
    if isinstance(A, nap.Tsd):
        return grubbs_sliding(A.values, w_size=w_size, alpha=alpha).shape[0]
    elif isinstance(A, np.ndarray):
        return grubbs_sliding(A, w_size=w_size, alpha=alpha).shape[0]
    else:
        raise TypeError('A must be nap.Tsd or np.ndarray.')


def bleaching_tau(A: nap.Tsd) -> float:
    """overall tau of bleaching."""
    y, t = A.values, A.t
    reg = Regression(model=ExponDecay())
    reg.fit(y, t)
    return reg.popt[1]


def ttest_pre_post(
    A: nap.Tsd,
    trials: pd.DataFrame,
    event_name: str,
    fs=None,
    pre_w=[-1, -0.2],
    post_w=[0.2, 1],
    alpha=0.001,
) -> bool:
    """
    :param calcium: np array, trace of the signal to be used
    :param times: np array, times of the signal to be used
    :param t_events: np array, times of the events to align to
    :param fs: float, sampling frequency of the signal
    :param pre_w: list of floats, pre window sizes in seconds
    :param post_w: list of floats, post window sizes in seconds
    :param confid: float, confidence level (alpha)
    :return: boolean, True if metric passes
    """
    y, t = A.values, A.t
    fs = 1 / np.median(np.diff(t)) if fs is None else fs

    t_events = trials[event_name].values

    psth_pre = psth(y, t, t_events, fs=fs, event_window=pre_w)[0]
    psth_post = psth(y, t, t_events, fs=fs, event_window=post_w)[0]

    # Take median value of signal over time
    pre = np.median(psth_pre, axis=0)
    post = np.median(psth_post, axis=0)
    # Paired t-test
    ttest = stats.ttest_rel(pre, post)
    passed_confg = ttest.pvalue < alpha
    return passed_confg


def has_response_to_event(
    A: nap.Tsd,
    event_times: nap.Ts,
    fs: float = None,
    window: tuple = (-1, 1),
    alpha: float = 0.005,
    mode='peak',
) -> bool:
    # checks if there is a significant response to an event

    # ibldsb way
    y, t = A.values, A.t
    fs = 1 / np.median(np.diff(t)) if fs is None else fs
    P = psth(y, t, event_times.t, fs=fs, event_window=window)[0]

    # or: pynapple style
    # in the long run, this will be the preferred way as this will
    # respect the .time_support of the pynapple object. # TODO verify this
    # P = nap.compute_perievent_continuous(A, event_times, window).values

    # temporally averages the samples in the window. Sensitive to window size!
    if mode == 'mean':
        sig_samples = np.average(P, axis=0)
    # takes the peak sample, minus one sd
    if mode == 'peak':
        sig_samples = np.max(P, axis=0) - np.std(y)

    # baseline is all samples that are not part of the response
    ts = event_times.t
    gaps = nap.IntervalSet(start=ts + window[0], end=ts + window[1])
    base_samples = A.restrict(A.time_support.set_diff(gaps)).values

    res = ttest_ind(sig_samples, base_samples)
    return res.pvalue < alpha


def has_responses(
    A: nap.Tsd,
    trials: pd.DataFrame,
    event_names: list = None,
    fs: float = None,
    window: tuple = (-1, 1),
    alpha: float = 0.005,
) -> bool:
    t = A.t
    fs = 1 / np.median(np.diff(t)) if fs is None else fs

    res = []
    for event_name in event_names:
        event_times = nap.Ts(t=trials[event_name].values)
        res.append(
            has_response_to_event(A, event_times, fs=fs, window=window, alpha=alpha)
        )

    return np.any(res)
