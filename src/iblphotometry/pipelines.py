# %%
import numpy as np
import pandas as pd
from iblphotometry import processing
import logging

logger = logging.getLogger()


def run_pipeline(
    pipeline,
    signal: pd.Series,
    reference: pd.Series | None = None,
    full_output: bool = False,
):
    """function to run a pipeline.

    Parameters
    ----------
    pipeline : _type_
        _description_
    signal : pd.Series
        _description_
    reference : pd.Series | None, optional
        _description_, by default None
    full_output : bool, optional
        _description_, by default False

    Returns
    -------
    _type_
        _description_
    """
    res = dict(signal=signal, reference=reference)
    for step in pipeline:
        # resolving inputs
        inputs = [res[inp] for inp in step['inputs']]
        # calling the steps sequentially
        res[step['output']] = step['function'](*inputs, **step['parameters'])
        # passing metadata through - taking the name of the first
        res[step['output']].name = inputs[0].name
        res[step['output']].index.name = inputs[0].index.name
    if not full_output:
        return res['result']
    else:
        return res


"""
Definition of a pipeline:

1. a list of individual (sequentially ordered) processing steps
2. each list entry is a dict with keys
    - function: a callable that takes one (or more) series as ordered inputs
    - parameters: a dict with all kwargs to be passed to the function
    - inputs: a tuple of str with: the name of the input (usually the previous processing step)
        in case of multiple inputs (such as for isosbestic correction), need to be ordered accordingly
        the first input of the first function is either called 'signal' or 'reference'
    - output: the name of the output of the processing step (usually the input for the next step)
        after all pipeline steps are done, 'result' will be returend
"""
sliding_mad_pipeline = [
    dict(
        function=processing.lowpass_bleachcorrect,
        parameters=dict(
            correction_method='subtract-divide',
            N=3,
            Wn=0.01,
        ),
        inputs=('signal',),
        output='bleach_corrected',
    ),
    dict(
        function=processing.sliding_mad,
        parameters=dict(
            w_len=120,
            overlap=90,
        ),
        inputs=('bleach_corrected',),
        output='mad',
    ),
    dict(
        function=processing.zscore,
        parameters=dict(mode='median'),
        inputs=('mad',),
        output='result',
    ),
]

isosbestic_correction_pipeline = [
    dict(
        function=processing.lowpass_bleachcorrect,
        parameters=dict(
            correction_method='subtract-divide',
            N=3,
            Wn=0.01,
        ),
        inputs=('signal',),
        output='signal_bleach_corrected',
    ),
    dict(
        function=processing.lowpass_bleachcorrect,
        parameters=dict(
            correction_method='subtract-divide',
            N=3,
            Wn=0.01,
        ),
        inputs=('reference',),
        output='reference_bleach_corrected',
    ),
    dict(
        function=processing.isosbestic_correct,
        parameters=dict(
            regression_method='mse',
            correction_method='subtract',
        ),
        inputs=('signal_bleach_corrected', 'reference_bleach_corrected'),
        output='result',
    ),
]

# # %%

# %% get example data
# from one.api import ONE
# from brainbox.io.one import PhotometrySessionLoader

# one = ONE()

# eid = '58861dac-4b4c-4f82-83fb-33d98d67df3a'
# eid = '34f55b3a-725e-4cc7-aed3-6e6338f573bf'
# psl = PhotometrySessionLoader(eid=eid, one=one)
# psl.load_photometry()
# brain_region = psl.photometry['GCaMP'].columns[0]
# signal = psl.photometry['GCaMP'][brain_region]

# signal_opt = run_pipeline(sliding_mad_pipeline, signal=signal)
# from iblphotometry.plotters import plot_photometry_trace

# plot_photometry_trace(signal_opt)
