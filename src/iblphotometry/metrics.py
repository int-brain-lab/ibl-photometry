import numpy as np
import scipy.stats as stats
from iblphotometry.preprocessing import psth

def ttest_pre_post(calcium, times, t_events, fs, pre_w=[-1, -0.2], post_w=[0.2, 1], confid=0.001):
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