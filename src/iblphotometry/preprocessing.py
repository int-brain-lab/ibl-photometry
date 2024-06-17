"""
This modules offers pre-processing for raw photometry data.
It implements different kinds of pre-processings depending on the preference of the user.
Where applicable, I have indicated a publication with the methods.


"""
import abc
from dataclasses import dataclass


import scipy.signal
import numpy as np
import pandas as pd

@dataclass
@abc.ABC
class BaseProcessor():
    parameters: dict = None
    df_photometry: pd.DataFrame = None
    fs: float = 30

    @abc.abstractmethod
    def process(self):
        pass


class IsosesbesticRegression(BaseProcessor):

    def params(self):
        pass

    def process(self):
        self.isosbestic_control, self.calcium = baseline_correction(self.raw_isosbestic, self.raw_calcium)


def isosbestic_regression(isosbestic, calcium, fil):
    """
    Prototype of baseline correction for photometry data.
    Fits a low pass version of the isosbestic signal to the calcium signal. The baseline signal is
    the low pass isosbestic signal multiplied by the fit slope and added to the fit intercept.
    The corrected signal is the calcium signal minus the baseline signal divided by the baseline signal.
    We apply the same procedure to the full-band isosbestic signal to check for remaining correlations.
    :param isosbestic:
    :param calcium:
    :return:
    """
    sos = scipy.signal.butter(**{'N': 3, 'Wn': 0.01, 'btype': 'lowpass'}, output='sos')
    calcium_lp = scipy.signal.sosfiltfilt(sos, calcium)
    isosbestic_lp = scipy.signal.sosfiltfilt(sos, isosbestic)
    m = np.polyfit(isosbestic_lp, calcium_lp, 1)
    sosbp = scipy.signal.butter(**{'N': 3, 'Wn': [.001, 0.5], 'btype': 'bandpass'}, output='sos')
    ph = (calcium - (ref := isosbestic_lp * m[0] + m[1])) / ref
    ph = scipy.signal.sosfiltfilt(sosbp, ph)
    iso = scipy.signal.sosfiltfilt(sosbp, (isosbestic - ref) / ref)
    return iso, ph


def baseline_correction_dataframe(df_photometry):
    """
    Wrapper around the baseline correction function to apply it to a dataframe with the raw signals
    `calcium` is the corrected calcium signal
    `isosbestic_control` is the isosbestic signal having gone through the same correction procedure
    :param df_photometry: should contain columns `raw_isosbestic' and `raw_calcium'
    :return: df_photometry with columns `calcium' and `isosbestic_control'
    """
    _, ph = baseline_correction(df_photometry['raw_isosbestic'].values, df_photometry['raw_calcium'].values)
    _, ph_control = baseline_correction(df_photometry['raw_isosbestic'].values, df_photometry['raw_isosbestic'].values)
    df_photometry['isosbestic_control'] = ph_control
    df_photometry['calcium'] = ph
    return df_photometry
