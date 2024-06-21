"""
This modules offers pre-processing for raw photometry data.
It implements different kinds of pre-processings depending on the preference of the user.
Where applicable, I have indicated a publication with the methods.
"""

import scipy.signal
import numpy as np


def photobleaching_lowpass(raw_calcium, **params):
    """
    Here the isosbestic recording is ignored, and the reference is computed as the low-pass component of the
    calcium band signal.
    :param calcium:
    :param params: dictionary with parameters
        butterworth_reference: dictionary with parameters for the butterworth filter applied to the
        calcium band for the sole purpose of regression {'N': 3, 'Wn': 0.01, 'btype': 'lowpass'}
    :return:
    iso (np.array): the corrected isosbestic signal to be used as control
    ph (np.array): the corrected calcium signal
    """
    params = {} if params is None else params

    sos = scipy.signal.butter(**params.get('butterworth_reference', {'N': 3, 'Wn': 0.01, 'btype': 'lowpass'}), output='sos')
    calcium_lp = scipy.signal.sosfiltfilt(sos, raw_calcium)
    calcium = (raw_calcium - calcium_lp) / calcium_lp
    return calcium


def isosbestic_regression(isosbestic, calcium, fs, **params):
    """
    Prototype of baseline correction for photometry data.
    Fits a low pass version of the isosbestic signal to the calcium signal. The baseline signal is
    the low pass isosbestic signal multiplied by the fit slope and added to the fit intercept.
    The corrected signal is the calcium signal minus the baseline signal divided by the baseline signal.
    We apply the same procedure to the full-band isosbestic signal to check for remaining correlations.
    :param isosbestic:
    :param calcium:
    :param calcium:
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
    calcium_lp = scipy.signal.sosfiltfilt(sos, calcium)
    isosbestic_lp = scipy.signal.sosfiltfilt(sos, isosbestic)
    m = np.polyfit(isosbestic_lp, calcium_lp, 1)

    ref = isosbestic_lp * m[0] + m[1]
    ph = (calcium - ref) / ref

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


def psth(calcium, times, t_events, fs, peri_event_window=None):
    """
    Compute the peri-event time histogram of a calcium signal
    :param calcium:
    :param times:
    :param t_events:
    :param fs:
    :param peri_event_window:
    :return:
    """
    peri_event_window = [-1, 2] if peri_event_window is None else peri_event_window
    # compute a vector of indices corresponding to the perievent window at the given sampling rate
    sample_window = np.arange(peri_event_window[0] * fs, peri_event_window[1] * fs + 1)
    # we inflate this vector to a 2d array where each column corresponds to an event
    idx_psth = np.tile(sample_window[:, np.newaxis], (1, t_events.size))
    # we add the index of each event too their respective column
    idx_event = np.searchsorted(times, t_events)
    idx_psth += idx_event
    psth = calcium[idx_psth]  # psth is a 2d array (ntimes, nevents)
    psth[idx_psth > (calcium.size - 1)] = np.nan  # remove events that are out of bounds
    return psth


# def sliding_rcoeff(signal_a, signal_b, wsize, overlap=0):
#     from ibldsp.utils import WindowGenerator
#     overlap = 0
#     wsize = 240
#     wg = WindowGenerator(ns=signal_a.size, nswin=wsize, overlap=overlap)
#     signal_a = df_photometry['calcium'].values
#     signal_b = df_photometry['isosbestic_control'].values
#
#     import numpy.lib
#
#     from iblutil.numerical import rcoeff
#     r = rcoeff(
#         numpy.lib.stride_tricks.sliding_window_view(signal_a, wsize),
#         numpy.lib.stride_tricks.sliding_window_view(signal_b, wsize))
#
#
#     pass