"""this module holds a collection of processing pipelines for fiber photometry data"""

import numpy as np
import pandas as pd

# from ibldsp.utils import WindowGenerator
from iblphotometry.outlier_detection import remove_spikes
from iblphotometry.bleach_corrections import lowpass_bleachcorrect, isosbestic_correct
from iblphotometry.sliding_operations import sliding_mad

# TODO this will probably be refactored to to processing
from iblphotometry.helpers import zscore

import logging

logger = logging.getLogger()


def run_pipeline(
    pipeline,
    F_signal: pd.DataFrame,
    F_reference: pd.DataFrame = None,
) -> pd.DataFrame:
    # copy
    Fc = F_signal.copy()
    if F_reference is not None:
        Fc_ref = F_reference.copy()

    if isinstance(F_signal, pd.Series):
        raise TypeError(
            'F_signal can not be pd.Series, is now required to be pd.DataFrame'
        )

    # iterate over the individual processing steps of the pipeline
    for i, (pipe_func, pipe_args) in enumerate(pipeline):
        # if pipeline function is to be executed on columns of a TsdFrame
        if 'needs_reference' in pipe_args:
            _pipe_args = {k: v for k, v in pipe_args.items() if k != 'needs_reference'}
            # check if F_ref is not None
            _d = np.zeros_like(Fc.values)
            # _Fcd_ref = np.zeros_like(Fc_ref.d)
            for i, col in enumerate(Fc.columns):
                _d[:, i] = pipe_func(Fc[col], Fc_ref[col], **_pipe_args)
            # this step consumes the reference!
            Fc = pd.DataFrame(_d, index=Fc.index.values, columns=Fc.columns)
        else:
            _d = np.zeros_like(Fc.values)
            for i, col in enumerate(Fc.columns):
                _d[:, i] = pipe_func(Fc[col], **pipe_args)
            Fc = pd.DataFrame(_d, index=Fc.index.values, columns=Fc.columns)
    return Fc


# these are now pipelines
sliding_mad_pipeline = [
    (remove_spikes, dict(sd=5)),
    (
        lowpass_bleachcorrect,
        dict(
            correction_method='subtract-divide',
            filter_params=dict(N=3, Wn=0.01, btype='lowpass'),
        ),
    ),
    (sliding_mad, dict(w_len=120, overlap=90)),
    (zscore, dict(mode='median')),
]

isosbestic_correction_pipeline = [
    (remove_spikes, dict(sd=5)),
    (
        isosbestic_correct,
        dict(
            needs_reference=True,
            correction_method='subtract',
            regression_method='huber',
            lowpass_isosbestic=dict(N=3, Wn=0.01, btype='lowpass'),
        ),
    ),
    (zscore, dict(mode='median')),
]


# TODO this is not following the definition of a pipeline anymore
# to be reconstructed
# def bc_lp_sliding_mad(
#     F: nap.Tsd | nap.TsdFrame,
#     w_len: float = 120,
#     overlap: int = 90,
#     butterworth_lowpass=dict(N=3, Wn=0.01, btype='lowpass'),
#     signal_name: str = 'raw_calcium',
# ):
#     """_summary_

#     Args:
#         F (nap.Tsd): _description_
#         w_len (float, optional): _description_. Defaults to 120.
#         overlap (int, optional): _description_. Defaults to 90.
#         butterworth_lowpass (_type_, optional): _description_. Defaults to dict(N=3, Wn=0.01, btype="lowpass").

#     Returns:
#         _type_: _description_
#     """

#     if isinstance(
#         F, nap.TsdFrame
#     ):  # if F is as TsdFrame, then use signal name to get the correct column - this is needed for the pipeline functionality in run_qc
#         if signal_name is None:
#             logger.critical('no signal name is provided for the pipeline')
#         else:
#             F = F[signal_name]

#     bleach_correction = bleach_corrections.LowpassBleachCorrection(
#         correction_method='subtract-divide',
#         filter_params=butterworth_lowpass,
#     )
#     F_bc = bleach_correction.correct(F)
#     F_res = sliding_operations.sliding_mad(F_bc, w_len=w_len, overlap=overlap)
#     return F_res


# # TODO this is not following the definition of a pipeline anymore
# def jove2019(
#     F: nap.TsdFrame,
#     ca_signal_name: str = 'raw_calcium',
#     isosbestic_signal_name: str = 'raw_isosbestic',
#     **params,
# ):
#     """
#     Martianova, Ekaterina, Sage Aronson, and Christophe D. Proulx. "Multi-fiber photometry to record neural activity in freely-moving animals." JoVE (Journal of Visualized Experiments) 152 (2019): e60278.
#     :param raw_calcium:
#     :param raw_isosbestic:
#     :param params:
#     :return:
#     """
#     raw_calcium = F[ca_signal_name]
#     raw_isosbestic = F[isosbestic_signal_name]

#     # replace this with a low pass corrector
#     # remove photobleaching
#     bleach_correction = bleach_corrections.LowpassBleachCorrection(
#         correction_method='subtract-divide',
#         filter_params=dict(N=3, Wn=0.01, btype='lowpass'),
#     )
#     calcium = bleach_correction.correct(
#         raw_calcium,
#         mode='subtract',
#     ).values
#     isosbestic = bleach_correction.correct(
#         raw_isosbestic,
#         mode='subtract',
#     ).values

#     # zscoring using median instead of mean
#     calcium = z(calcium, mode='median')
#     isosbestic = z(isosbestic, mode='median')

#     # regular regression
#     m = np.polyfit(isosbestic, calcium, 1)
#     ref = isosbestic * m[0] + m[1]
#     ph = (calcium - ref) / 100
#     return nap.Tsd(t=raw_calcium.times(), d=ph)


# # TODO this is not following the definition of a pipeline anymore
# def isosbestic_regression(
#     F: nap.TsdFrame,
#     ca_signal_name: str = 'raw_calcium',
#     isosbestic_signal_name: str = 'raw_isosbestic',
#     fs: float = None,
#     regression_method: str = 'irls',
#     correction_method: str = 'subtract-divide',
#     **params,
# ):
#     raw_calcium = F[ca_signal_name]
#     raw_isosbestic = F[isosbestic_signal_name]

#     t = F.times()
#     fs = 1 / np.median(np.diff(t)) if fs is None else fs

#     isosbestic_correction = bleach_corrections.IsosbesticCorrection(
#         regression_method=regression_method,
#         correction_method=correction_method,
#         lowpass_isosbestic=dict(N=3, Wn=0.01, btype='lowpass'),
#     )

#     F_corr = isosbestic_correction.correct(
#         raw_calcium,
#         raw_isosbestic,
#     )

#     butterworth_signal = params.get(
#         'butterworth_signal',
#         dict(N=3, Wn=7, btype='lowpass', fs=fs),  # changed from 10 to 7
#     )

#     F_corr = filt(F_corr, **butterworth_signal)
#     return F_corr
