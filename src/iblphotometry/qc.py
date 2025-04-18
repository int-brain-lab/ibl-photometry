# %%
import gc
from collections.abc import Callable
from tqdm import tqdm
import logging

import numpy as np
from numpy.lib.stride_tricks import as_strided
import pandas as pd
from scipy.stats import linregress

from iblphotometry.processing import make_sliding_window
from iblphotometry.pipelines import run_pipeline
import iblphotometry.metrics as metrics

logger = logging.getLogger()


# %% # those could be in metrics
def sliding_metric(
    F: pd.Series,
    w_len: float,
    metric: Callable,
    fs: float | None = None,
    n_wins: int = -1,
    metric_kwargs: dict | None = None,
):
    """applies a metric along time.

    Args:
        F (nap.Tsd): _description_
        w_size (int): _description_
        metric (callable, optional): _description_. Defaults to None.
        n_wins (int, optional): _description_. Defaults to -1.

    Returns:
        _type_: _description_
    """
    y, t = F.values, F.index.values
    fs = 1 / np.median(np.diff(t)) if fs is None else fs
    w_size = int(w_len * fs)

    yw = make_sliding_window(y, w_size)
    if n_wins > 0:
        n_samples = y.shape[0]
        inds = np.linspace(0, n_samples - w_size, n_wins, dtype='int64')
        yw = yw[inds, :]
    else:
        inds = np.arange(yw.shape[0], dtype='int64')

    m = metric(yw, **metric_kwargs) if metric_kwargs is not None else metric(yw)

    return pd.Series(m, index=t[inds + int(w_size / 2)])


def _eval_metric_sliding(
    F: pd.Series,
    metric: Callable,
    w_len: float = 60,
    metric_kwargs: dict | None = None,
) -> pd.Series:
    metric_kwargs = {} if metric_kwargs is None else metric_kwargs
    dt = np.median(np.diff(F.index))
    w_size = int(w_len // dt)
    step_size = int(w_size // 2)
    n_windows = int((len(F) - w_size) // step_size + 1)
    if n_windows <= 2:
        return
    a = F.values
    windows = as_strided(
        a, shape=(n_windows, w_size), strides=(step_size * a.strides[0], a.strides[0])
    )
    S_values = np.apply_along_axis(
        lambda w: metric(w, **metric_kwargs), axis=1, arr=windows
    )
    S_times = F.index.values[
        np.linspace(step_size, n_windows * step_size, n_windows).astype(int)
    ]
    return pd.Series(S_values, index=S_times)


# eval pipleline will be here
def eval_metric(
    F: pd.Series,
    metric: Callable,
    metric_kwargs: dict | None = None,
    sliding_kwargs: dict | None = None,
    full_output=True,
):
    results_vals = ['value', 'sliding_values', 'sliding_timepoints', 'r', 'p']
    result = {k: np.nan for k in results_vals}
    metric_func = getattr(metrics, metric)
    result['value'] = (
        metric_func(F) if metric_kwargs is None else metric_func(F, **metric_kwargs)
    )
    sliding_kwargs = {} if sliding_kwargs is None else sliding_kwargs
    if sliding_kwargs:
        S = _eval_metric_sliding(F, metric_func, sliding_kwargs['w_len'], metric_kwargs)
        if S is None:
            pass
        else:
            result['r'], result['p'] = linregress(S.index.values, S.values)[2:4]
            if full_output:
                result['sliding_values'] = S.values
                result['sliding_timepoints'] = S.index.values
    return result


def qc_series(
    F: pd.Series,
    qc_metrics: dict,
    sliding_kwargs=None,  # if present, calculate everything in a sliding manner
    trials=None,  # if present, put trials into params
    eid: str = None,  # FIXME but left as is for now just to keep the logger happy
    brain_region: str = None,  # FIXME but left as is for now just to keep the logger happy
) -> dict:
    if isinstance(F, pd.DataFrame):
        raise TypeError('F cannot be a dataframe')

    # if sliding_kwargs is None:  # empty dicts indicate no sliding application
    #     sliding_kwargs = {metric:{} for metric in qc_metrics.keys()}
    # elif (
    #     isinstance(sliding_kwargs, dict) and
    #     not sorted(sliding_kwargs.keys()) == sorted(qc_metrics.keys())
    #     ):  # the same sliding kwargs will be applied to all metrics
    #     sliding_kwargs = {metric:sliding_kwargs for metric in qc_metrics.keys()}
    # elif (
    #     isinstance(sliding_kwargs, dict) and
    #     sorted(sliding_kwargs.keys()) == sorted(qc_metrics.keys())
    #     ):  # each metric has it's own sliding kwargs
    #     pass
    # else:  # is not None, a simple dict, or a nested dict
    #     raise TypeError(
    #         'sliding_kwargs must be None, dict of kwargs, or nested dict with same keys as qc_metrics'
    #     )
    sliding_kwargs = {} if sliding_kwargs is None else sliding_kwargs

    # should cover all cases
    qc_results = {}
    for metric, params in qc_metrics.items():
        # try:
        if trials is not None:  # if trials are passed
            params['trials'] = trials
        res = eval_metric(
            F, metric, metric_kwargs=params, sliding_kwargs=sliding_kwargs[metric]
        )
        qc_results[f'{metric}'] = res['value']
        if sliding_kwargs[metric]:
            qc_results[f'_{metric}_values'] = res['sliding_values']
            qc_results[f'_{metric}_times'] = res['sliding_timepoints']
            qc_results[f'_{metric}_r'] = res['r']
            qc_results[f'_{metric}_p'] = res['p']
        # except Exception as e:
        #     logger.warning(
        #         f'{eid}, {brain_region}: metric {metric.__name__} failure: {type(e).__name__}:{e}'
        #     )
    return qc_results


# %% main QC loop
def run_qc(
    data_loader,
    eids: list[str],
    pipelines_reg,  # registered pipelines
    qc_metrics: dict,  # metrics. keys: raw, processed, repsonse, sliding_kwargs
    sigref_mapping: dict = None,  # think about this one - the mapping of signal and reference # dict(signal=signal_band_name, reference=ref_band_name)
):
    qc_results = []
    for eid in tqdm(eids):
        print(eid)
        try:
            # get photometry data
            raw_dfs = data_loader.load_photometry_data(eid=eid)
            signal_bands = list(raw_dfs.keys())
            brain_regions = raw_dfs[signal_bands[0]]

            # get behavioral data
            # TODO this should be provided
            # sl = SessionLoader(eid=eid, one=data_loader.one)
            # for caroline
            # trials = sl.load_trials(
            #     collection='alf/task_00'
            # )  # this is necessary fo caroline
            # trials = sl.load_trials()  # should be good for all others

            # the old way
            trials = data_loader.one.load_dataset(eid, '*trials.table.pqt')

            for band in signal_bands:
                raw_tf = raw_dfs[band]
                for region in brain_regions:
                    qc_result = qc_series(
                        raw_tf[region], qc_metrics['raw'], sliding_kwargs=None, eid=eid
                    )
                    qc_results.append(
                        dict(
                            eid=eid,
                            pipeline='raw',
                            band=band,
                            region=region,
                            **qc_result,
                        )
                    )

            # run the pipelines and qc on the processed data
            # here it needs to be specified if one band is a reference of the other
            for pipeline_name, pipeline in pipelines_reg.items():
                if 'reference' in sigref_mapping:  # this is for isosbestic pipelines
                    proc_tf = run_pipeline(
                        pipeline,
                        raw_dfs[sigref_mapping['signal']],
                        raw_dfs[sigref_mapping['reference']],
                    )
                else:
                    # FIXME this fails for true-multiband
                    # this hack works for single-band
                    # possible fix could be that signal could be a list
                    proc_tf = run_pipeline(pipeline, raw_dfs[sigref_mapping['signal']])

                for region in brain_regions:
                    # sliding qc of the processed data
                    qc_proc = qc_series(
                        proc_tf[region],
                        qc_metrics=qc_metrics['processed'],
                        sliding_kwargs=qc_metrics['sliding_kwargs'],
                        eid=eid,
                        brain_region=region,
                    )

                    # qc with metrics that use behavior
                    qc_resp = qc_series(
                        proc_tf[region],
                        qc_metrics['response'],
                        trials=trials,
                        eid=eid,
                        brain_region=region,
                    )
                    qc_result = qc_proc | qc_resp
                    qc_results.append(
                        dict(
                            eid=eid,
                            pipeline=pipeline_name,
                            region=region,
                            **qc_result,
                        )
                    )
        except Exception as e:
            logger.warning(f'{eid}: failure: {type(e).__name__}:{e}')

        gc.collect()
    return qc_results
