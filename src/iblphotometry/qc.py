# %%
import gc
from collections.abc import Callable
from tqdm import tqdm
import logging

import numpy as np
import pandas as pd
from scipy.stats import linregress

from iblphotometry.processing import make_sliding_window
from iblphotometry.pipelines import run_pipeline
from brainbox.io.one import SessionLoader

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


# eval pipleline will be here
def eval_metric(
    F: pd.Series,
    metric: Callable,
    metric_kwargs: dict | None = None,
    sliding_kwargs: dict | None = None,
):
    m = metric(F, **metric_kwargs) if metric_kwargs is not None else metric(F)

    if sliding_kwargs is not None:
        S = sliding_metric(
            F, metric=metric, **sliding_kwargs, metric_kwargs=metric_kwargs
        )
        r, p = linregress(S.times(), S.values)[2:4]
    else:
        r = np.nan
        p = np.nan

    return dict(value=m, rval=r, pval=p)


def qc_series(
    F: pd.Series,
    qc_metrics: dict,
    sliding_kwargs=None,  # if present, calculate everything in a sliding manner
    trials=None,  # if present, put trials into params
    eid: str = None,  # FIXME but left as is for now just to keep the logger happy
    brain_region: str = None,  # FIXME but left as is for now just to keep the logger happy
) -> dict:
    if isinstance(F, pd.DataFrame):
        raise TypeError('F can not be a dataframe')

    # should cover all cases
    qc_results = {}
    for metric, params in qc_metrics:
        try:
            if trials is not None:  # if trials are passed
                params['trials'] = trials
            res = eval_metric(F, metric, params, sliding_kwargs)
            qc_results[f'{metric.__name__}'] = res['value']
            if sliding_kwargs:
                qc_results[f'{metric.__name__}_r'] = res['rval']
                qc_results[f'{metric.__name__}_p'] = res['pval']
        except Exception as e:
            logger.warning(
                f'{eid}, {brain_region}: metric {metric.__name__} failure: {type(e).__name__}:{e}'
            )
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
            sl = SessionLoader(eid=eid, one=data_loader.one)
            # for caroline
            # trials = sl.load_trials(
            #     collection='alf/task_00'
            # )  # this is necessary fo caroline
            trials = sl.load_trials()  # should be good for all others

            # the old way
            # trials = data_loader.one.load_dataset(eid, '*trials.table.pqt')

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
