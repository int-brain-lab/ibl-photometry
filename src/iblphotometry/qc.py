import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from iblphotometry.pipelines import run_pipeline
from one.api import ONE
from brainbox.io.one import PhotometrySessionLoader
from tqdm import tqdm
from scipy.stats import linregress


def qc_signals(
    raw_dfs: dict[pd.DataFrame],
    metrics: list[callable],
    metrics_kwargs: dict = {},
    signal_band: str | list[str] | None = None,
    brain_region: str | list[str] | None = None,
    pipeline: list[dict] | None = None,
    sliding_kwargs: dict | None = None,
) -> pd.DataFrame:
    """runs a set of qc metrics on a given photometry dataset

    Parameters
    ----------
    raw_dfs : dict[pd.DataFrame]
        Photometry data in the format of a dictionary, where the keys are the individual signal bands, and their respective values are pandas DataFrames, one column per fiber
    metrics : list[callable]
        A list of metrics (= callable functions taking pd.Series as the first argument)
    metrics_kwargs : dict, optional
        additional optional kwargs as passed to the metrics. keys are the .__name__ of the metric, values are the kwargs, by default {}
    signal_band : str | list[str] | None, optional
        if provided, restrict evaluation to this signal band, by default None
    brain_region : str | list[str] | None, optional
        if provided, restrict evaluation to this brain region, by default None
    pipeline : list[dict] | None, optional
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
                qc_result.append(
                    {
                        'band': band,
                        'brain_region': brain_region,
                        'metric': metric.__name__,
                        'value': metric(signal, **_metric_kwargs),
                    }
                )
                # I don't think we can rely on using stride tricks as the input into the
                # metrics might be a restricted to a series
                if sliding_kwargs is not None:
                    # this is for creating n evently spaced windows of size w_len along the signal
                    w_len = sliding_kwargs['w_len']
                    dt = np.median(np.diff(signal.index))
                    w_size = int(w_len // dt)
                    n_windows = sliding_kwargs['n_windows']
                    t_start = signal.index[0]
                    t_stop = signal.index[-1] - w_size - dt  # one extra dt to be on the safe side
                    w_start_times = np.linspace(t_start, t_stop, n_windows)
                    for i in range(n_windows):
                        ix = np.logical_and(
                            signal.index.values > w_start_times[i],
                            signal.index.values < w_start_times[i] + w_size,
                        )
                        signal_ = signal.loc[ix]
                        if 'detrend' not in sliding_kwargs.keys():
                            sliding_kwargs['detrend'] = False
                        if sliding_kwargs['detrend']:
                            res = linregress(signal_.index, signal.values)
                            signal_.values -= signal_.index * res.slope + res.intercept

                        qc_result.append(
                            {
                                'band': band,
                                'brain_region': brain_region,
                                'metric': metric.__name__,
                                'value': metric(signal_),
                                'window': i,
                            },
                        )

    return pd.DataFrame(qc_result)


def qc_eid(
    eid: str,
    one: ONE,
    metrics: list[callable],
    metrics_kwargs: dict = {},
    signal_band: str | list[str] | None = None,
    brain_region: str | list[str] | None = None,
    pipeline: list[dict] | None = None,
    sliding_kwargs: dict | None = None,
) -> pd.DataFrame:
    """
    Convenience function for running qc on a dataset as given by an eid. See qc_signals for a description of the individual parameters
    """
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
    return qc_result


def run_qc(
    eids: list[str],
    one: ONE,
    metrics: list[callable],
    metrics_kwargs: dict = {},
    signal_band: str | list[str] | None = None,
    brain_region: str | list[str] | None = None,
    pipeline: list[dict] | None = None,
    sliding_kwargs: dict | None = None,
    n_jobs: int = 1,
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
            )
            for eid in eids
        )

    return pd.concat(qc_results)
