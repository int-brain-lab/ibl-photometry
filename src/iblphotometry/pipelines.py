"""this module holds a collection of processing pipelines for fiber photometry data"""

import numpy as np
import pandas as pd
import pynapple as nap
from iblphotometry.helpers import z, psth, filt
from ibldsp.utils import WindowGenerator
from iblphotometry import sliding_operations
from iblphotometry import bleach_corrections

import logging

logger = logging.getLogger()


def bc_lp_sliding_mad(
    F: nap.Tsd | nap.TsdFrame,
    w_len: float = 120,
    overlap: int = 90,
    butterworth_lowpass=dict(N=3, Wn=0.01, btype='lowpass'),
    signal_name: str = 'raw_calcium',
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

    if isinstance(
        F, nap.TsdFrame
    ):  # if F is as TsdFrame, then use signal name to get the correct column - this is needed for the pipeline functionality in run_qc
        if signal_name is None:
            logger.critical('no signal name is provided for the pipeline')
        else:
            F = F[signal_name]

    bleach_correction = bleach_corrections.LowpassBleachCorrection(
        correction_method='subtract-divide',
        filter_params=butterworth_lowpass,
    )
    F_bc = bleach_correction.correct(F)
    F_res = sliding_operations.sliding_mad(F_bc, w_len=w_len, overlap=overlap)
    return F_res


def jove2019(
    F: nap.TsdFrame,
    ca_signal_name: str = 'raw_calcium',
    isosbestic_signal_name: str = 'raw_isosbestic',
    **params,
):
    """
    Martianova, Ekaterina, Sage Aronson, and Christophe D. Proulx. "Multi-fiber photometry to record neural activity in freely-moving animals." JoVE (Journal of Visualized Experiments) 152 (2019): e60278.
    :param raw_calcium:
    :param raw_isosbestic:
    :param params:
    :return:
    """
    raw_calcium = F[ca_signal_name]
    raw_isosbestic = F[isosbestic_signal_name]

    # replace this with a low pass corrector
    # remove photobleaching
    bleach_correction = bleach_corrections.LowpassBleachCorrection(
        correction_method='subtract-divide',
        filter_params=dict(N=3, Wn=0.01, btype='lowpass'),
    )
    calcium = bleach_correction.correct(
        raw_calcium,
        mode='subtract',
    ).values
    isosbestic = bleach_correction.correct(
        raw_isosbestic,
        mode='subtract',
    ).values

    # zscoring using median instead of mean
    calcium = z(calcium, mode='median')
    isosbestic = z(isosbestic, mode='median')

    # regular regression
    m = np.polyfit(isosbestic, calcium, 1)
    ref = isosbestic * m[0] + m[1]
    ph = (calcium - ref) / 100
    return nap.Tsd(t=raw_calcium.times(), d=ph)


def isosbestic_regression(
    F: nap.TsdFrame,
    ca_signal_name: str = 'raw_calcium',
    isosbestic_signal_name: str = 'raw_isosbestic',
    fs: float = None,
    regression_method: str = 'irls',
    correction_method: str = 'subtract-divide',
    **params,
):
    raw_calcium = F[ca_signal_name]
    raw_isosbestic = F[isosbestic_signal_name]

    t = F.times()
    fs = 1 / np.median(np.diff(t)) if fs is None else fs

    isosbestic_correction = bleach_corrections.IsosbesticCorrection(
        regression_method=regression_method,
        correction_method=correction_method,
        lowpass_isosbestic=dict(N=3, Wn=0.01, btype='lowpass'),
    )

    F_corr = isosbestic_correction.correct(
        raw_calcium,
        raw_isosbestic,
    )

    butterworth_signal = params.get(
        'butterworth_signal',
        dict(N=3, Wn=7, btype='lowpass', fs=fs),  # changed from 10 to 7
    )

    F_corr = filt(F_corr, **butterworth_signal)
    return F_corr
