"""this module holds a collection of processing pipelines for fiber photometry data"""

import numpy as np
import pandas as pd
import pynapple as nap
from utils import z, psth, filt
from ibldsp.utils import WindowGenerator
import sliding_operations
import bleach_corrections


def bc_lp_sliding_mad(
    F: nap.Tsd,
    w_len: float = 120,
    overlap: int = 90,
    butterworth_lowpass=dict(N=3, Wn=0.01, btype="lowpass"),
):
    """_summary_

    Args:
        F (nap.Tsd): _description_
        w_len (float, optional): _description_. Defaults to 120.
        overlap (int, optional): _description_. Defaults to 90.
        butterworth_lowpass (_type_, optional): _description_. Defaults to dict(N=3, Wn=0.01, btype="lowpass").

    Returns:
        _type_: _description_
    """
    bc = bleach_corrections.LowpassCorrection()
    F_bc = bc.bleach_correct(F)
    F_res = sliding_operations.sliding_mad(F_bc, w_len=w_len, overlap=overlap)
    return F_res


def jove2019(raw_calcium: nap.Tsd, raw_isosbestic: nap.Tsd, fs: float, **params):
    """
    Martianova, Ekaterina, Sage Aronson, and Christophe D. Proulx. "Multi-fiber photometry to record neural activity in freely-moving animals." JoVE (Journal of Visualized Experiments) 152 (2019): e60278.
    :param raw_calcium:
    :param raw_isosbestic:
    :param params:
    :return:
    """
    # DOCME more
    # the first step is to remove the photobleaching w
    calcium_lp = filt(
        raw_calcium,
        **params.get("butterworth_lowpass", {"N": 3, "Wn": 0.01, "btype": "lowpass"}),
    )
    calcium = raw_calcium.values - calcium_lp.values

    isosbestic_lp = filt(
        raw_isosbestic,
        **params.get("butterworth_lowpass", {"N": 3, "Wn": 0.01, "btype": "lowpass"}),
    )
    isosbestic = raw_isosbestic.values - isosbestic_lp.values

    # zscoring using median instead of mean
    # this is not the same as the modified zscore
    calcium = (calcium - np.median(calcium)) / np.std(calcium)
    isosbestic = (isosbestic - np.median(isosbestic)) / np.std(isosbestic)
    m = np.polyfit(isosbestic, calcium, 1)
    ref = isosbestic * m[0] + m[1]
    ph = (calcium - ref) / 100
    return nap.Tsd(t=raw_calcium.times(), d=ph)


def isosbestic_regression(
    raw_isosbestic: nap.Tsd, raw_calcium: nap.Tsd, fs: float, **params
):
    isosbestic_correction = bleach_corrections.IsosbesticCorrection(
        regressor="linear", correction="deltaF"
    )
    F_corr = isosbestic_correction.correct(
        raw_calcium,
        raw_isosbestic,
        lowpass_isosbestic=dict(N=3, Wn=0.01, btype="lowpass"),
    )

    butterworth_signal = params.get(
        "butterworth_signal", {"N": 3, "Wn": 10, "btype": "lowpass", "fs": fs}
    )

    F_corr = filt(F_corr, **butterworth_signal)
    return F_corr


## the originals

# def preprocess_sliding_mad(
#     raw_calcium: nap.Tsd, fs=None, wlen=120, overlap=90, returns_gain=False, **params
# ):
#     """
#     Applies one pass of fiber photobleaching
#     :param raw_calcium:
#     :param times:
#     :param fs:
#     :param wlen:
#     :param overlap:
#     :param params:
#     :return:
#     """
#     times = raw_calcium.times()
#     fs = 1 / np.median(np.diff(times)) if fs is None else fs

#     calcium = photobleaching_lowpass(raw_calcium, fs, **params).values

#     wg = ibldsp.utils.WindowGenerator(
#         ns=calcium.size, nswin=int(wlen * fs), overlap=overlap
#     )
#     trms = np.array([first for first, last in wg.firstlast]) / fs + times[0]
#     rmswin, _ = psth(calcium, times, t_events=trms, fs=fs, peri_event_window=[0, wlen])
#     gain = np.nanmedian(np.abs(calcium)) / np.nanmedian(np.abs(rmswin), axis=0)
#     gain = np.interp(times, trms, gain)
#     if returns_gain:
#         return nap.Tsd(t=times, d=calcium * gain), gain
#     else:
#         return nap.Tsd(t=times, d=calcium * gain)

# something like this would be nice
# pipeline = dict(
#     name="pipeline_name",
#     parts = [func, params]

#         )


# def isosbestic_regression(
#     raw_isosbestic: nap.Tsd, raw_calcium: nap.Tsd, fs: float, **params
# ):
#     """
#     Prototype of baseline correction for photometry data.
#     Fits a low pass version of the isosbestic signal to the calcium signal. The baseline signal is
#     the low pass isosbestic signal multiplied by the fit slope and added to the fit intercept.
#     The corrected signal is the calcium signal minus the baseline signal divided by the baseline signal.
#     We apply the same procedure to the full-band isosbestic signal to check for remaining correlations.
#     :param raw_isosbestic:
#     :param raw_calcium:
#     :param params: dictionary with parameters
#         butterworth_regression: dictionary with parameters for the butterworth filter applied to both isosbestic and
#         calcium band for the sole purpose of regression {'N': 3, 'Wn': 0.01, 'btype': 'lowpass'}
#         butterworth_signal: dictionary with parameters for the butterworth filter {'N': 3, 'Wn': 0.01, 'btype': 'lowpass'}
#         applied to the outputs. Set to None to disable filtering
#     :return:
#     iso (np.array): the corrected isosbestic signal to be used as control
#     ph (np.array): the corrected calcium signal
#     """
#     params = {} if params is None else params
#     times = raw_isosbestic.times()

#     # sos = scipy.signal.butter(
#     #     **params.get(
#     #         "butterworth_regression", {"N": 3, "Wn": 0.1, "btype": "lowpass", "fs": fs}
#     #     ),
#     #     output="sos",
#     # )
#     butterworth_regression = params.get(
#         "butterworth_regression", {"N": 3, "Wn": 0.1, "btype": "lowpass", "fs": fs}
#     )
#     calcium_lp = utils.filt(raw_calcium, **butterworth_regression)
#     # calcium_lp = scipy.signal.sosfiltfilt(sos, raw_calcium)
#     # isosbestic_lp = scipy.signal.sosfiltfilt(sos, raw_isosbestic)
#     isosbestic_lp = utils.filt(raw_isosbestic, **butterworth_regression)
#     m = np.polyfit(isosbestic_lp, calcium_lp, 1)

#     ref = isosbestic_lp * m[0] + m[1]
#     ph = nap.Tsd(t=times, d=(raw_calcium - ref) / ref)

#     butterworth_signal = params.get(
#         "butterworth_signal", {"N": 3, "Wn": 10, "btype": "lowpass", "fs": fs}
#     )
#     ph = utils.filt(ph, **butterworth_signal)
#     # if butterworth_signal is not None:
#     # sosbp = scipy.signal.butter(**butterworth_signal, output="sos")
#     # ph = scipy.signal.sosfiltfilt(sosbp, ph)
#     return ph
