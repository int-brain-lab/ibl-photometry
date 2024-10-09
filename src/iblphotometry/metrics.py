import numpy as np
import scipy.stats as stats
from iblphotometry.preprocessing import psth
import ibldsp.waveforms as waveforms

def ttest_pre_post(calcium, times, t_events, fs,
                   pre_w=np.array([-1, -0.2]), post_w=np.array([0.2, 1]), confid=0.001):
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


def modulation_index_peak(calcium, times, t_events, fs,
                          pre_w = np.array([-1, -0.2]), post_w = np.array([0.2, 20]), w_size=1.):
    """
    Steps:
    - Find the peak value post within a large window. For this, re-use the waveform peak-finder code,
    considering each trial as a waveform, i.e. N trace per waveform = 1.
    - Take a number of sample around that peak index per trial
    - Average those samples
    - Take the same amount of samples in the pre-condition, and average similarly
    - compute a modulation index pre/post: MI = (pre-post)/(pre+post) : if 0, similar
    - threshold the MI (TBD what is a good value), and count how many trials pass that threshold

    Then do the same, but use as fixed time window the max found on average PSTH
    """
    psth_pre = psth(calcium, times, t_events, fs=fs, peri_event_window=pre_w)[0]
    psth_post = psth(calcium, times, t_events, fs=fs, peri_event_window=post_w)[0]

    # Find peak index in post for each trial
    # 3D dimension have to be (wav, time, trace)
    arr_in = np.expand_dims(np.swapaxes(psth_post, axis1=1, axis2=0), axis=2)
    df = waveforms.find_peak(arr_in)

    # Find peak index in post for average PSTH
    # Average over trials
    avg_psth_post = np.median(psth_post, axis=1)
    arr_in = np.expand_dims(avg_psth_post,  axis=[0, 2])
    df_avg = waveforms.find_peak(arr_in)

    