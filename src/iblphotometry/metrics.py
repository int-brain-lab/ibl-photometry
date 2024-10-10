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


def peak_indx_post(psth_post):
    # Find peak index in post for each trial
    # 3D dimension have to be (wav, time, trace)
    arr_in = np.expand_dims(np.swapaxes(psth_post, axis1=1, axis2=0), axis=2)
    df_trial = waveforms.find_peak(arr_in)

    # Find peak index in post for average PSTH
    # Average over trials
    avg_psth_post = np.median(psth_post, axis=1)
    arr_in = np.expand_dims(avg_psth_post,  axis=[0, 2])
    df_avg = waveforms.find_peak(arr_in)

    return df_trial, df_avg



def modulation_index_peak(calcium, times, t_events, fs,
                          pre_w = np.array([-1, -0.2]), post_w = np.array([0.2, 20]),
                          wind_around=np.array([-0.2, 2])):
    """
    Steps:
    - Find the peak value post within a large window. For this, re-use the waveform peak-finder code,
    considering each trial as a waveform, i.e. N trace per waveform = 1.
    - Do this peak finding both at each trial, and for the average PSTH

    - Compute the time difference between the peak at each trial and the average peak ; return mean and STD

    - Take a number of sample around that peak avg-PSTH index for each trial
    - Average those samples over time
    - Take the samples in the pre-condition (note: different window size), and average over time
    - Compute signal variation compared to baseline around time of max peak found on average PSTH
    (abs difference and zscore)
    """
    # TODO assert if window[0] negative, cannot have abs value > post_w[0] ?

    psth_pre = psth(calcium, times, t_events, fs=fs, peri_event_window=pre_w)[0]
    psth_post = psth(calcium, times, t_events, fs=fs, peri_event_window=post_w)[0]

    # Find peak index in post for each trial and average PSTH
    df_trial, df_avg = peak_indx_post(psth_post)

    # Compute time difference between peak-average PSTH and peak at each trace; then take STD
    df_trial['time_diff'] = 1/fs * (df_trial.peak_time_idx - df_avg.peak_time_idx.values[0])
    std_peak_time_diff = df_trial['time_diff'].std()
    mean_peak_time_diff = df_trial['time_diff'].mean()

    # Find window around avg PSTH peak and average signal within it; then take STD
    window_idx = np.floor(wind_around * fs) + df_avg.peak_time_idx.values[0]
    window_idx = window_idx.astype(int)
    w_range = range(window_idx[0], window_idx[1])
    w_psth = psth_post[w_range, :]
    # Average over time
    avg_psth_post = np.median(w_psth, axis=0)
    # Compute mean and std
    mean_peak_amplitude = np.mean(avg_psth_post)
    std_peak_amplitude = np.std(avg_psth_post)

    # Compare peak values to baseline pre
    # Average pre over time
    avg_psth_pre = np.median(psth_pre, axis=0)
    std_pre = np.std(psth_pre)
    mean_pre = np.mean(psth_pre)
    # Mean and std of absolute difference pre/post
    absdiff_post = np.abs(avg_psth_post - avg_psth_pre)
    mean_absdiff = np.mean(absdiff_post)
    std_absdiff = np.std(absdiff_post)

    # Z score
    z_score_post = (avg_psth_post - mean_pre) / std_pre
    mean_z_score = np.mean(z_score_post)


    # Output variable containing metrics
    out_dict = {
        'mean_peak_time_diff': mean_peak_time_diff,
        'std_peak_time_diff' : std_peak_time_diff,
        'mean_peak_amplitude' : mean_peak_amplitude,
        'std_peak_amplitude' : std_peak_amplitude,
        'mean_absdiff' : mean_absdiff,
        'std_absdiff' : std_absdiff,
        'mean_z_score' : mean_z_score
    }
    return out_dict
