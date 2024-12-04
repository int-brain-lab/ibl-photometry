from typing import Optional
import numpy as np
import pandas as pd
from scipy import stats

from iblphotometry.processing import (
    z,
    Regression,
    ExponDecay,
    detect_spikes,
    detect_outliers,
)
from iblphotometry.behavior import psth


def percentile_dist(A: pd.Series | np.ndarray, pc: tuple = (50, 95), axis=-1) -> float:
    """the distance between two percentiles in units of z. Captures the magnitude of transients.

    Args:
        A (pd.Series | np.ndarray): the input data, np.ndarray for stride tricks sliding windows
        pc (tuple, optional): percentiles to be computed. Defaults to (50, 95).
        axis (int, optional): only for arrays, the axis to be computed. Defaults to -1.

    Returns:
        float: the value of the metric
    """
    if isinstance(A, pd.Series):  # "overloading"
        P = np.percentile(z(A.values), pc)
    elif isinstance(A, np.ndarray):
        P = np.percentile(z(A), pc, axis=axis)
    else:
        raise TypeError('A must be pd.Series or np.ndarray.')
    return P[1] - P[0]


def signal_asymmetry(A: pd.Series | np.ndarray, pc_comp: int = 95, axis=-1) -> float:
    """the ratio between the distance of two percentiles to the median. Proportional to the the signal to noise.

    Args:
        A (pd.Series | np.ndarray): _description_
        pc_comp (int, optional): _description_. Defaults to 95.
        axis (int, optional): _description_. Defaults to -1.

    Returns:
        float: _description_
    """
    if not (isinstance(A, pd.Series) or isinstance(A, np.ndarray)):
        raise TypeError('A must be pd.Series or np.ndarray.')

    a = np.absolute(percentile_dist(A, (50, pc_comp), axis=axis))
    b = np.absolute(percentile_dist(A, (100 - pc_comp, 50), axis=axis))
    return a / b


def signal_skew(A: pd.Series | np.ndarray, axis=-1) -> float:
    if isinstance(A, pd.Series):
        P = stats.skew(A.values)
    elif isinstance(A, np.ndarray):
        P = stats.skew(A, axis=axis)
    else:
        raise TypeError('A must be pd.Series or np.ndarray.')
    return P


def n_unique_samples(A: pd.Series | np.ndarray) -> int:
    """number of unique samples in the signal. Low values indicate that the signal during acquisition was not within the range of the digitizer."""
    a = A.values if isinstance(A, pd.Series) else A
    return np.unique(a).shape[0]


def n_spikes(A: pd.Series | np.ndarray, sd: int = 5):
    """count the number of spike artifacts in the recording."""
    a = A.values if isinstance(A, pd.Series) else A
    return detect_spikes(a, sd=sd).shape[0]


def n_outliers(
    A: pd.Series | np.ndarray, w_size: int = 1000, alpha: float = 0.0005
) -> int:
    """counts the number of outliers as detected by grubbs test for outliers.
    int: _description_
    """
    a = A.values if isinstance(A, pd.Series) else A
    return detect_outliers(a, w_size=w_size, alpha=alpha).shape[0]


def bleaching_tau(A: pd.Series) -> float:
    """overall tau of bleaching."""
    y, t = A.values, A.index.values
    reg = Regression(model=ExponDecay())
    reg.fit(y, t)
    return reg.popt[1]


def ttest_pre_post(
    A: pd.Series,
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
    y, t = A.values, A.index.values
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
    A: pd.Series,
    event_times: np.ndarray,
    fs: Optional[float] = None,
    window: tuple = (-1, 1),
    alpha: float = 0.005,
    mode='peak',
) -> bool:
    # checks if there is a significant response to an event

    # ibldsb way
    y, t = A.values, A.index.values
    fs = 1 / np.median(np.diff(t)) if fs is None else fs
    P, psth_ix = psth(y, t, event_times, fs=fs, event_window=window)

    # temporally averages the samples in the window. Sensitive to window size!
    if mode == 'mean':
        sig_samples = np.average(P, axis=0)
    # takes the peak sample, minus one sd
    if mode == 'peak':
        sig_samples = np.max(P, axis=0) - np.std(y)

    # baseline is all samples that are not part of the response
    base_ix = np.setdiff1d(np.arange(y.shape[0]), psth_ix.flatten())
    base_samples = y[base_ix]

    res = stats.ttest_ind(sig_samples, base_samples)
    return res.pvalue < alpha


def has_responses(
    A: pd.Series,
    trials: pd.DataFrame,
    event_names: list,
    fs: Optional[float] = None,
    window: tuple = (-1, 1),
    alpha: float = 0.005,
) -> bool:
    t = A.index.values
    fs = 1 / np.median(np.diff(t)) if fs is None else fs

    res = []
    for event_name in event_names:
        event_times = trials[event_name]
        res.append(
            has_response_to_event(A, event_times, fs=fs, window=window, alpha=alpha)
        )

    return np.any(res)
