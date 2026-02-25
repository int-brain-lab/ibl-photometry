from joblib import Parallel, delayed
import traceback
from tqdm import tqdm
from typing import List, Optional, Dict, Literal

import numpy as np
from scipy.stats import linregress
import pandas as pd

from one.api import ONE
from brainbox.io.one import PhotometrySessionLoader
from iblphotometry.pipelines import run_pipeline


def qc_signals(
    raw_dfs: Dict[str, pd.DataFrame],
    metrics: List[callable],
    metrics_kwargs: Dict = {},
    signal_band: Optional[str | List[str]] = None,
    brain_region: Optional[str | List[str]] = None,
    pipeline: Optional[List[Dict]] = None,
    sliding_kwargs: Optional[Dict] = None,
) -> pd.DataFrame:
    """runs a set of qc metrics on a given photometry dataset

    Parameters
    ----------
    raw_dfs : dict[pd.DataFrame]
        Photometry data in the format of a dictionary, where the keys are the individual signal bands, and their respective values are pandas DataFrames, one column per fiber
    metrics : List[callable]
        A List of metrics (= callable functions taking pd.Series as the first argument)
    metrics_kwargs : dict, optional
        additional optional kwargs as passed to the metrics. keys are the .__name__ of the metric, values are the kwargs, by default {}
    signal_band : str | List[str] | None, optional
        if provided, restrict evaluation to this signal band, by default None
    brain_region : str | List[str] | None, optional
        if provided, restrict evaluation to this brain region, by default None
    pipeline : List[dict] | None, optional
        if provided, apply this processing pipeline before evaluation, by default None
    sliding_kwargs : dict | None, optional
        if provided, apply metrics evaluation in a number of windows of specified length along the time course of the signal, by default None

    Returns
    -------
    pd.DataFrame
        the qc result in tidy data format
    """
    # which data to operate on
    if signal_band is None:
        signal_bands = raw_dfs.keys()
    else:
        if type(signal_band) is str:
            assert signal_band in raw_dfs.keys(), f'signal band {signal_band} not present in data'
            signal_bands = [signal_band]
    if brain_region is None:
        brain_regions = raw_dfs[list(signal_bands)[0]].columns
    else:
        if type(brain_region) is str:
            assert brain_region in raw_dfs[list(signal_bands)[0]].columns, f'brain region {brain_region} not present in data'
            brain_regions = [brain_region]

    # the main qc loop
    qc_result = []
    for band in signal_bands:
        for brain_region in brain_regions:
            signal = raw_dfs[band][brain_region]

            # if a pipeline is provided, run it here
            if pipeline is not None:
                # reference_band = reference_band or None
                # pipelines that consume a reference band, not supported by this function
                signal = run_pipeline(pipeline, signal)

            for metric in metrics:
                _metric_kwargs = metrics_kwargs.get(metric.__name__, {})
                qc_result.append({
                    'band': band,
                    'brain_region': brain_region,
                    'metric': metric.__name__,
                    'value': metric(signal, **_metric_kwargs),
                })
                # I don't think we can rely on using stride tricks as the input into the
                # metrics might be a restricted to a series
                if sliding_kwargs is not None:
                    # Using window length and step length (both in seconds)
                    w_len = sliding_kwargs['w_len']
                    step_len = sliding_kwargs['step_len']  # in seconds
                    # dt = np.median(np.diff(signal.index))
                    # w_size = int(w_len // dt)
                    # step_size = int(step_len // dt)  # compute step size from step length
                    t_start = signal.index[0]
                    t_stop = signal.index[-1] - w_len

                    # Generate window start times with step_size
                    w_start_times = np.arange(t_start, t_stop, step_len)

                    for w_start in w_start_times:
                        ix = np.logical_and(
                            signal.index.values >= w_start,
                            signal.index.values < w_start + w_len,
                        )
                        signal_ = signal.loc[ix]
                        if sliding_kwargs.get('detrend', False):
                            res = linregress(signal_.index, signal_.values)
                            signal_ -= signal_.index * res.slope + res.intercept

                        qc_result.append(
                            {
                                'band': band,
                                'brain_region': brain_region,
                                'metric': metric.__name__,
                                'value': metric(signal_),
                                'window': w_start + w_len / 2,
                            },
                        )

    return pd.DataFrame(qc_result)


def qc_eid(
    eid: str,
    one: ONE,
    metrics: List[callable],
    metrics_kwargs: Dict = {},
    signal_band: Optional[str | List[str]] = None,
    brain_region: Optional[str | List[str]] = None,
    pipeline: Optional[List[Dict]] = None,
    sliding_kwargs: Optional[Dict] = None,
    on_error: Literal['log', 'raise'] = 'log',
) -> pd.DataFrame:
    """
    Convenience function for running qc on a dataset as given by an eid. See qc_signals for a description of the individual parameters
    """
    try:
        psl = PhotometrySessionLoader(eid=eid, one=one)
        psl.load_photometry()
        qc_result = qc_signals(
            psl.photometry,
            metrics=metrics,
            metrics_kwargs=metrics_kwargs,
            signal_band=signal_band,
            brain_region=brain_region,
            pipeline=pipeline,
            sliding_kwargs=sliding_kwargs,
        )
        qc_result['eid'] = eid
    except Exception as e:
        if on_error == 'log':
            # Collect exception info
            qc_result = pd.DataFrame([
                {  # dataframe for downstream compatibility
                    'eid': eid,
                    'exception_type': type(e).__name__,
                    'exception_message': str(e),
                    'traceback': traceback.format_exc(),
                }
            ])
        else:
            raise e
    return qc_result


def run_qc(
    eids: List[str],
    one: ONE,
    metrics: List[callable],
    metrics_kwargs: dict = {},
    signal_band: Optional[str | List[str]] = None,
    brain_region: Optional[str | List[str]] = None,
    pipeline: Optional[List[dict]] = None,
    sliding_kwargs: Optional[Dict] = None,
    n_jobs: int = 1,
    on_error: Literal['log', 'raise'] = 'log',
) -> pd.DataFrame:
    """
    Conveninece function to run qc on many datasets given by a list of eids. See qc_signals for a description of the individual parameters

    Parameters
    ----------

    n_jobs : int, optional
        if larger than 1, use joblib to process sessions in parallel, by default 1

    Returns
    -------
    pd.DataFrame
        the qc result in tidy data format
    """
    if n_jobs == 1:
        qc_results = []
        for eid in tqdm(eids):
            qc_result_ = qc_eid(
                eid,
                one,
                metrics,
                metrics_kwargs=metrics_kwargs,
                signal_band=signal_band,
                brain_region=brain_region,
                pipeline=pipeline,
                sliding_kwargs=sliding_kwargs,
                on_error=on_error,
            )
            qc_results.append(qc_result_)
    else:
        qc_results = Parallel(n_jobs=n_jobs)(
            delayed(qc_eid)(
                eid,
                one,
                metrics,
                metrics_kwargs=metrics_kwargs,
                signal_band=signal_band,
                brain_region=brain_region,
                pipeline=pipeline,
                sliding_kwargs=sliding_kwargs,
                on_error=on_error,
            )
            for eid in eids
        )

    return pd.concat(qc_results)
