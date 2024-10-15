import numpy as np
import pandas as pd
import pynapple as nap
from scipy import stats
from iblphotometry.utils import z, psth
from iblphotometry.bleach_corrections import ExponDecayBleachingModel
from iblphotometry.outlier_detection import detect_spikes, grubbs_sliding
from scipy.stats import ttest_ind
from functools import singledispatch


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
    """the distance between two percentiles in units of z
    should be proportional to SNR, assuming the signal is
    in the positive 5th percentile
    """
    if isinstance(A, nap.Tsd):  # "overloading"
        P = np.percentile(z(A.values), pc)
    if isinstance(A, np.ndarray):
        P = np.percentile(z(A), pc, axis=axis)
    return P[1] - P[0]


def signal_asymmetry(A: nap.Tsd | np.ndarray, pc_comp: int = 95, axis=-1) -> float:
    a = percentile_dist(A, (50, pc_comp), axis=axis)
    b = percentile_dist(A, (100 - pc_comp, 50), axis=axis)
    return a / b


def signal_skew(A: nap.Tsd | np.ndarray, axis=-1) -> float:
    if isinstance(A, nap.Tsd):
        P = stats.skew(A.values)
    if isinstance(A, np.ndarray):
        P = stats.skew(A, axis=axis)
    return P


def n_unique_samples(A: nap.Tsd) -> int:
    return np.unique(A.values).shape[0]


def n_spikes(A: nap.Tsd, sd: int):
    return detect_spikes(A.values, sd=sd).shape[0]


def ttest_pre_post(
    A: nap.Tsd,
    trials: pd.DataFrame,
    # t_events: np.array,
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
    y, t = A.values, A.times()
    fs = 1 / np.median(np.diff(t)) if fs is None else fs

    t_events = trials[event_name].values

    psth_pre = psth(y, t, t_events, fs=fs, peri_event_window=pre_w)[0]
    psth_post = psth(y, t, t_events, fs=fs, peri_event_window=post_w)[0]

    # Take median value of signal over time
    pre = np.median(psth_pre, axis=0)
    post = np.median(psth_post, axis=0)
    # Paired t-test
    ttest = stats.ttest_rel(pre, post)
    passed_confg = ttest.pvalue < alpha
    return passed_confg


def n_outliers(A: nap.Tsd, w_size: int = 1000, alpha: float = 0.0005) -> int:
    """implements a sliding version of using grubbs test to detect outliers."""
    return grubbs_sliding(A.values, w_size=w_size, alpha=alpha).shape[0]


def bleaching_tau(A: nap.Tsd) -> float:
    y, t = A.values, A.times()
    bleaching_model = ExponDecayBleachingModel()
    bleaching_model._fit(y, t)
    return bleaching_model.popt[1]


def has_response_to_event(
    A: nap.Tsd,
    event_times: nap.Ts,
    fs: float = None,
    window: tuple = (-1, 1),
    alpha: float = 0.005,
    mode='mean',
) -> bool:
    # checks if there is a significant response to an event

    # ibldsb way
    y, t = A.values, A.times()
    fs = 1 / np.median(np.diff(t)) if fs is None else fs
    P = psth(y, t, event_times.times(), fs=fs, peri_event_window=window)[0]

    # or: pynapple style
    P = nap.compute_perievent_continuous(A, event_times, window).values

    # assuming time is on dim 1
    if mode == 'mean':
        sig_samples = np.average(P, axis=1)
    if mode == 'peak':
        sig_samples = np.max(P, axis=1) - np.std(y)

    # baseline is all samples that are not part of the response
    ts = event_times.times()
    gaps = nap.Intervalset(start=ts + window[0], end=ts + window[1])
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
    y, t = A.values, A.times()
    fs = 1 / np.median(np.diff(t)) if fs is None else fs

    res = []
    for event_name in event_names:
        event_times = nap.Ts(t=trials[event_name])
        res.append(
            has_response_to_event(A, event_times, fs=fs, window=window, alpha=alpha)
        )

    return np.any(res)
