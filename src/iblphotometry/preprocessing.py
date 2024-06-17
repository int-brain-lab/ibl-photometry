"""
This modules offers pre-processing for raw photometry data.
It implements different kinds of pre-processings depending on the preference of the user.
Where applicable, I have indicated a publication with the methods.

"""


import scipy.signal
import numpy as np


def isosbestic_regression(isosbestic, calcium, fs=None, **params):
    """
    Prototype of baseline correction for photometry data.
    Fits a low pass version of the isosbestic signal to the calcium signal. The baseline signal is
    the low pass isosbestic signal multiplied by the fit slope and added to the fit intercept.
    The corrected signal is the calcium signal minus the baseline signal divided by the baseline signal.
    We apply the same procedure to the full-band isosbestic signal to check for remaining correlations.
    :param isosbestic:
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

    sos = scipy.signal.butter(**params.get('butterworth_regression', {'N': 3, 'Wn': 0.01, 'btype': 'lowpass'}), output='sos')
    calcium_lp = scipy.signal.sosfiltfilt(sos, calcium)
    isosbestic_lp = scipy.signal.sosfiltfilt(sos, isosbestic)
    m = np.polyfit(isosbestic_lp, calcium_lp, 1)
    ph = (calcium - (ref := isosbestic_lp * m[0] + m[1])) / ref

    butterworth_signal = params.get('butterworth_signal', {'N': 3, 'Wn': 0.01, 'btype': 'lowpass'})
    if butterworth_signal is not None:
        sosbp = scipy.signal.butter(**butterworth_signal, output='sos')
        ph = scipy.signal.sosfiltfilt(sosbp, ph)
        iso = scipy.signal.sosfiltfilt(sosbp, (isosbestic - ref) / ref)
    return iso, ph


def isosbestic_correction_dataframe(df_photometry):
    """
    Wrapper around the baseline correction function to apply it to a dataframe with the raw signals
    `calcium` is the corrected calcium signal
    `isosbestic_control` is the isosbestic signal having gone through the same correction procedure
    :param df_photometry: should contain columns `raw_isosbestic' and `raw_calcium'
    :return: df_photometry with columns `calcium' and `isosbestic_control'
    """
    iso, ph = isosbestic_regression(df_photometry['raw_isosbestic'].values, df_photometry['raw_calcium'].values)
    df_photometry['isosbestic_control'] = iso
    df_photometry['calcium'] = ph
    return df_photometry
