import numpy as np
import pynapple as nap
from scipy import stats
from sliding_operations import make_sliding_window
from utils import z, psth
from bleach_corrections import (
    ExponDecayBleachingModel,
    DoubleExponDecayBleachingModel,
)
from outlier_detection import detect_spikes, grubbs_sliding


def percentile_dist(A: np.array, pc: tuple = (50, 95), axis=-1):
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

    P = np.percentile(z(A), pc, axis=axis)
    return P[1] - P[0]


def signal_asymmetry(A: np.array, pc_comp: int = 95, axis=-1):
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


def n_unique_samples(A: np.array):
    """_summary_

    Args:
        A (np.array): _description_

    Returns:
        _type_: _description_
    """
    return np.unique(A).shape[0]


def n_spikes(A: np.array, sd: int):
    return detect_spikes(A, sd=sd).shape[0]


def ttest_pre_post(
    calcium, times, t_events, fs, pre_w=[-1, -0.2], post_w=[0.2, 1], confid=0.001
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
    psth_pre = psth(calcium, times, t_events, fs=fs, peri_event_window=pre_w)[0]
    psth_post = psth(calcium, times, t_events, fs=fs, peri_event_window=post_w)[0]

    # Take median value of signal over time
    pre = np.median(psth_pre, axis=0)
    post = np.median(psth_post, axis=0)
    # Paired t-test
    ttest = stats.ttest_rel(pre, post)
    passed_confg = ttest.pvalue < confid
    return passed_confg


def n_outliers(A: np.array, w_size: int = 1000, alpha: float = 0.0005):
    """implements a sliding version of using grubbs test to detect outliers.

    Args:
        A (np.array): _description_
        w_size (int, optional): _description_. Defaults to 1000.
        alpha (float, optional): _description_. Defaults to 0.0005.

    Returns:
        _type_: _description_
    """
    return grubbs_sliding(A, w_size=w_size, alpha=alpha).shape[0]


def signal_skew(A: np.array):
    return stats.skew(A)


def bleaching_tau(A: np.array, t: np.array):
    bleaching_model = ExponDecayBleachingModel()
    bleaching_model._fit(A, t)
    return bleaching_model.popt[1]
