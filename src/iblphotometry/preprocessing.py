"""
This modules offers pre-processing for raw photometry data.
It implements different kinds of pre-processings depending on the preference of the user.
Where applicable, I have indicated a publication with the methods.
"""
import scipy.signal
import numpy as np

import ibldsp.utils
from iblutil.numerical import rcoeff


def preprocess_sliding_mad(raw_calcium, times, fs=None, wlen=120, overlap=90, returns_gain=False, **params):
    """
    Applies one pass of fiber photobleaching
    :param raw_calcium:
    :param times:
    :param fs:
    :param wlen:
    :param overlap:
    :param params:
    :return:
    """
    calcium = photobleaching_lowpass(raw_calcium, fs, **params)
    wg = ibldsp.utils.WindowGenerator(ns=calcium.size, nswin=int(wlen * fs), overlap=overlap)
    trms = np.array([first for first, last in wg.firstlast]) / fs
    rmswin = psth(calcium, times, t_events=trms, fs=fs, peri_event_window=[0, wlen])
    gain = np.median(np.abs(calcium)) / np.median(np.abs(rmswin), axis=0)
    gain = np.interp(times, trms, gain)
    if returns_gain:
        return calcium * gain, gain
    else:
        return calcium * gain


def jove2019(raw_calcium, raw_isosbestic, fs, **params):
    """
    Martianova, Ekaterina, Sage Aronson, and Christophe D. Proulx. "Multi-fiber photometry to record neural activity in freely-moving animals." JoVE (Journal of Visualized Experiments) 152 (2019): e60278.
    :param raw_calcium:
    :param raw_isosbestic:
    :param params:
    :return:
    """
    # the first step is to remove the photobleaching w
    sos = scipy.signal.butter(fs=fs, output='sos', **params.get('butterworth_lowpass', {'N': 3, 'Wn': 0.01, 'btype': 'lowpass'}))
    calcium = raw_calcium - scipy.signal.sosfiltfilt(sos, raw_calcium)
    isosbestic = raw_isosbestic - scipy.signal.sosfiltfilt(sos, raw_isosbestic)
    calcium = (calcium - np.median(calcium)) / np.std(calcium)
    isosbestic = (isosbestic - np.median(isosbestic)) / np.std(isosbestic)
    m = np.polyfit(isosbestic, calcium, 1)
    ref = isosbestic * m[0] + m[1]
    ph = (calcium - ref) / 100
    return ph


def photobleaching_lowpass(raw_calcium, fs, **params):
    """
    Here the isosbestic recording is ignored, and the reference is computed as the low-pass component of the
    calcium band signal.
    :param calcium:
    :param params: dictionary with parameters
    {
        'butterworth_lowpass': {'N': 3, 'Wn': 0.01, 'btype': 'lowpass'}
        }
        dictionary with parameters for the butterworth filter applied to the calcium band for the sole purpose of regression
    :return:
    iso (np.array): the corrected isosbestic signal to be used as control
    ph (np.array): the corrected calcium signal
    """
    params = {} if params is None else params
    sos = scipy.signal.butter(fs=fs, output='sos', **params.get('butterworth_lowpass', {'N': 3, 'Wn': 0.01, 'btype': 'lowpass'}))
    calcium_lp = scipy.signal.sosfiltfilt(sos, raw_calcium)
    calcium = (raw_calcium - calcium_lp) / calcium_lp
    return calcium


def isosbestic_regression(raw_isosbestic, raw_calcium, fs, **params):
    """
    Prototype of baseline correction for photometry data.
    Fits a low pass version of the isosbestic signal to the calcium signal. The baseline signal is
    the low pass isosbestic signal multiplied by the fit slope and added to the fit intercept.
    The corrected signal is the calcium signal minus the baseline signal divided by the baseline signal.
    We apply the same procedure to the full-band isosbestic signal to check for remaining correlations.
    :param raw_isosbestic:
    :param raw_calcium:
    :param params: dictionary with parameters
        butterworth_regression: dictionary with parameters for the butterworth filter applied to both isosbestic and
        calcium band for the sole purpose of regression {'N': 3, 'Wn': 0.01, 'btype': 'lowpass'}
        butterworth_signal: dictionary with parameters for the butterworth filter {'N': 3, 'Wn': 0.01, 'btype': 'lowpass'}
        applied to the outputs. Set to None to disable filtering
    :return:
    iso (np.array): the corrected isosbestic signal to be used as control
    ph (np.array): the corrected calcium signal
    """
    params = {} if params is None else params

    sos = scipy.signal.butter(**params.get('butterworth_regression', {'N': 3, 'Wn': 0.1, 'btype': 'lowpass', 'fs': fs}), output='sos')
    calcium_lp = scipy.signal.sosfiltfilt(sos, raw_calcium)
    isosbestic_lp = scipy.signal.sosfiltfilt(sos, raw_isosbestic)
    m = np.polyfit(isosbestic_lp, calcium_lp, 1)

    ref = isosbestic_lp * m[0] + m[1]
    ph = (raw_calcium - ref) / ref

    butterworth_signal = params.get('butterworth_signal', {'N': 3, 'Wn': 10, 'btype': 'lowpass', 'fs': fs})
    if butterworth_signal is not None:
        sosbp = scipy.signal.butter(**butterworth_signal, output='sos')
        ph = scipy.signal.sosfiltfilt(sosbp, ph)
    return ph


def isosbestic_correction_dataframe(df_photometry):
    """
    Wrapper around the baseline correction function to apply it to a dataframe with the raw signals
    `calcium` is the corrected calcium signal
    `isosbestic_control` is the isosbestic signal having gone through the same correction procedure
    :param df_photometry: should contain columns `raw_isosbestic' and `raw_calcium'
    :return: df_photometry with columns `calcium' and `isosbestic_control'
    """
    fs = 1 / np.median(np.diff(df_photometry['times'].values))
    ph = isosbestic_regression(df_photometry['raw_isosbestic'].values, df_photometry['raw_calcium'].values, fs=fs)
    iso = isosbestic_regression(df_photometry['raw_isosbestic'].values, df_photometry['raw_isosbestic'].values, fs=fs)
    df_photometry['isosbestic_control'] = iso
    df_photometry['calcium'] = ph
    return df_photometry


def psth(calcium, times, t_events, fs=None, peri_event_window=None):
    """
    Compute the peri-event time histogram of a calcium signal
    :param calcium:
    :param times:
    :param t_events:
    :param fs:
    :param peri_event_window:
    :return:
    """
    fs = 1 / np.median(np.diff(times)) if fs is None else fs
    peri_event_window = [-1, 2] if peri_event_window is None else peri_event_window
    # compute a vector of indices corresponding to the perievent window at the given sampling rate
    sample_window = np.arange(peri_event_window[0] * fs, peri_event_window[1] * fs + 1).astype(int)
    # we inflate this vector to a 2d array where each column corresponds to an event
    idx_psth = np.tile(sample_window[:, np.newaxis], (1, t_events.size))
    # we add the index of each event too their respective column
    idx_event = np.searchsorted(times, t_events)
    idx_psth += idx_event
    i_out_of_bounds = np.logical_or(idx_psth > (calcium.size - 1), idx_psth < 0)
    idx_psth[i_out_of_bounds] = -1
    psth = calcium[idx_psth]  # psth is a 2d array (ntimes, nevents)
    psth[i_out_of_bounds] = np.nan  # remove events that are out of bounds
    return psth


def sliding_rcoeff(signal_a, signal_b, nswin, overlap=0):
    """
    Computes the local correlation coefficient between two signals in sliding windows
    :param signal_a:
    :param signal_b:
    :param nswin: window size in samples
    :param overlap: overlap of successiv windows in samples
    :return: ix: indices of the center of the windows, r: correlation coefficients
    """
    wg = ibldsp.utils.WindowGenerator(ns=signal_a.size, nswin=nswin, overlap=overlap)
    first_samples = np.array([fl[0] for fl in wg.firstlast])
    iwin = np.zeros([wg.nwin, wg.nswin], dtype=np.int32) + np.arange(wg.nswin)
    iwin += first_samples[:, np.newaxis]
    iwin[iwin >= signal_a.size] = signal_a.size - 1
    r = rcoeff(signal_a[iwin], signal_b[iwin])
    ix = first_samples + nswin // 2
    return ix, r
