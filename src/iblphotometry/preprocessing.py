"""
This modules offers pre-processing for raw photometry data.
It implements different kinds of pre-processings depending on the preference of the user.
Where applicable, I have indicated a publication with the methods.
"""

import scipy.signal
import numpy as np

import ibldsp.utils
from iblutil.numerical import rcoeff

import iblphotometry.helpers as helpers
import pynapple as nap


# def photobleaching_lowpass(raw_calcium: nap.Tsd, fs: float, **params):
#     """
#     Here the isosbestic recording is ignored, and the reference is computed as the low-pass component of the
#     calcium band signal.
#     :param calcium:
#     :param params: dictionary with parameters
#     {
#         'butterworth_lowpass': {'N': 3, 'Wn': 0.01, 'btype': 'lowpass'}
#         }
#         dictionary with parameters for the butterworth filter applied to the calcium band for the sole purpose of regression
#     :return:
#     iso (np.array): the corrected isosbestic signal to be used as control
#     ph (np.array): the corrected calcium signal
#     """
#     params = {} if params is None else params
#     calcium_lp = utils.filt(
#         raw_calcium,
#         **params.get("butterworth_lowpass", {"N": 3, "Wn": 0.01, "btype": "lowpass"}),
#     )
#     # sos = scipy.signal.butter(
#     #     fs=fs,
#     #     output="sos",
#     #     **params.get("butterworth_lowpass", {"N": 3, "Wn": 0.01, "btype": "lowpass"}),
#     # )
#     # calcium_lp = scipy.signal.sosfiltfilt(sos, raw_calcium)
#     calcium = (raw_calcium.values - calcium_lp.values) / calcium_lp.values
#     return nap.Tsd(t=raw_calcium.times(), d=calcium)


# def isosbestic_correction_dataframe(df_photometry):
#     """
#     Wrapper around the baseline correction function to apply it to a dataframe with the raw signals
#     `calcium` is the corrected calcium signal
#     `isosbestic_control` is the isosbestic signal having gone through the same correction procedure
#     :param df_photometry: should contain columns `raw_isosbestic' and `raw_calcium'
#     :return: df_photometry with columns `calcium' and `isosbestic_control'
#     """
#     fs = 1 / np.median(np.diff(df_photometry["times"].values))
#     ph = isosbestic_regression(
#         df_photometry["raw_isosbestic"].values,
#         df_photometry["raw_calcium"].values,
#         fs=fs,
#     )
#     iso = isosbestic_regression(
#         df_photometry["raw_isosbestic"].values,
#         df_photometry["raw_isosbestic"].values,
#         fs=fs,
#     )
#     df_photometry["isosbestic_control"] = iso
#     df_photometry["calcium"] = ph
#     return df_photometry
