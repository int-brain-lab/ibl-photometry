import numpy as np
import pandas as pd
import pynapple as nap
from scipy import stats
from utils import z, psth
from bleach_corrections import (
    ExponDecayBleachingModel,
)
from outlier_detection import detect_spikes, grubbs_sliding
from sciy.stats import ttest_ind


def percentile_dist(A: nap.Tsd, pc: tuple = (50, 95), axis=-1):
    """the distance between two percentiles in units of z
    should be proportional to SNR, assuming the signal is
    in the positive 5th percentile

    Args:
        A (np.array): _description_
        pc (tuple, optional): _description_. Defaults to (50, 95).
        axis (int, optional): _description_. Defaults to -1.

    Returns:
        _type_: _description_
    """

    P = np.percentile(z(A.values), pc, axis=axis)
    return P[1] - P[0]


def signal_asymmetry(A: nap.Tsd, pc_comp: int = 95, axis=-1):
    """_summary_

    Args:
        A (np.array): _description_
        pc_comp (int, optional): _description_. Defaults to 95.
        axis (int, optional): _description_. Defaults to -1.

    Returns:
        _type_: _description_
    """
    a = percentile_dist(A, (50, pc_comp), axis=axis)
    b = percentile_dist(A, (100 - pc_comp, 50), axis=axis)
    return a / b


def n_unique_samples(A: nap.Tsd):
    """_summary_

    Args:
        A (np.array): _description_

    Returns:
        _type_: _description_
    """
    return np.unique(A.values).shape[0]


def n_spikes(A: nap.Tsd, sd: int):
    return detect_spikes(A.values, sd=sd).shape[0]


def ttest_pre_post(
    A: nap.Tsd,
    trials: pd.Dataframe,
    # t_events: np.array,
    event_name: str,
    fs=None,
    pre_w=[-1, -0.2],
    post_w=[0.2, 1],
    alpha=0.001,
):
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


def n_outliers(A: np.Tsd, w_size: int = 1000, alpha: float = 0.0005):
    """implements a sliding version of using grubbs test to detect outliers.

    Args:
        A (np.array): _description_
        w_size (int, optional): _description_. Defaults to 1000.
        alpha (float, optional): _description_. Defaults to 0.0005.

    Returns:
        _type_: _description_
    """
    return grubbs_sliding(A.values, w_size=w_size, alpha=alpha).shape[0]


def signal_skew(A: nap.Tsd):
    return stats.skew(A.values)


def bleaching_tau(A: nap.Tsd):
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
    mode="mean",
):
    # checks if there is a significant response to an event

    # ibldsb way
    y, t = A.values, A.times()
    fs = 1 / np.median(np.diff(t)) if fs is None else fs
    P = psth(y, t, event_times.times(), fs=fs, peri_event_window=window)[0]

    # or: pynapple style
    P = nap.compute_perievent_continuous(A, event_times, window).values

    # assuming time is on dim 1
    if mode == "mean":
        sig_samples = np.average(P, axis=1)
    if mode == "peak":
        sig_samples = np.max(P, axis=1) - np.std(y)

    # baseline is all samples that are not part of the response
    ts = event_times.times()
    gaps = nap.Intervalset(start=ts + window[0], end=ts + window[1])
    base_samples = A.restrict(A.time_support.set_diff(gaps)).values

    res = ttest_ind(sig_samples, base_samples)
    return res.pvalue < alpha


def has_responses(
    A: nap.Tsd,
    trials: pd.Dataframe,
    event_names: list = None,
    fs: float = None,
    window: tuple = (-1, 1),
    alpha: float = 0.005,
):
    y, t = A.values, A.times()
    fs = 1 / np.median(np.diff(t)) if fs is None else fs

    res = []
    for event_name in event_names:
        event_times = nap.Ts(t=trials[event_name])
        res.append(
            has_response_to_event(A, event_times, fs=fs, window=window, alpha=alpha)
        )

    return np.any(res)
