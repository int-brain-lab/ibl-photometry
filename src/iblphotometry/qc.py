# %%
import numpy as np
from scipy.stats import linregress
import pandas as pd
import pynapple as nap

from tqdm import tqdm
import logging

from copy import copy
import gc

from iblphotometry.sliding_operations import make_sliding_window
from pipelines import run_pipeline
import warnings

logger = logging.getLogger()


# %% # those could be in metrics
def sliding_metric(
    F: nap.Tsd,
    w_len: float,
    fs: float = None,
    metric: callable = None,
    n_wins: int = -1,
    metric_kwargs: dict = None,
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
    y, t = F.values, F.times()
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

    return nap.Tsd(t=t[inds + int(w_size / 2)], d=m)


# eval pipleline will be here
def eval_metric(
    F: nap.Tsd,
    metric: callable = None,
    metric_kwargs: dict = None,
    sliding_kwargs: dict = None,
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


def qc_Tsd(
    raw_tf: nap.TsdFrame,
    qc_metrics: dict,
    sliding_kwargs=None,  # if present, calculate everything in a sliding manner
    trials=None,  # if present, put trials into params
    eid: str = None,  # just for logging purposes
) -> dict:
    # should cover all cases
    qc_results = {}
    for metric, params in qc_metrics:
        # iterate over brain regions
        for brain_region in raw_tf.columns:
            F = raw_tf[brain_region]
            try:
                if trials:
                    params['trials'] = trials
                res = eval_metric(F, metric, params, sliding_kwargs)
                qc_results[f'{metric.__name__}_{brain_region}'] = res['value']
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
    # HOW TO STORE THE RESULTS
    # how it was before seems reasonable. One .csv per pipeline, each row is eid, brain_region , metrics ...
    qc = {}
    for pipeline_name in pipelines_reg.keys():
        qc[pipeline_name] = {}
    # make a ginormous pandas dataframe

    # Creating dictionary of dictionary, with each key being the pipeline name
    # qc_dfs = dict((name, {}) for name in pipelines_reg.keys())
    qc_eid = {}
    for eid in tqdm(eids):
        qc_eid[eid] = {}

        # get data
        raw_tfs, brain_regions = data_loader.load_photometry_data(
            eid=eid, return_regions=True
        )
        # eid, pname = data_loader.pid2eid(pid)
        trials = data_loader.one.load_dataset(eid, '*trials.table.pqt')
        # trials = data_loader.load_trials_table(eid)

        # qc on the raw data
        qc_eid[eid]['raw'] = {}
        for band in raw_tfs.keys():
            qc_eid[eid]['raw'][band] = qc_Tsd(
                raw_tfs[band], qc_metrics['raw'], None, eid
            )

        # run the pipelines and qc on the processed data
        # here it needs to be specified if one band is a reference of the other
        qc_eid[eid]['processed'] = {}
        for pipeline_name, pipeline in pipelines_reg.items():
            if sigref_mapping:  # this is for isosbestic pipelines
                proc_tf = run_pipeline(
                    pipeline,
                    raw_tfs[sigref_mapping['signal']],
                    raw_tfs[sigref_mapping['reference']],
                )
            else:
                # FUCK this fails for true-multiband
                proc_tf = run_pipeline(pipeline, raw_tfs)

            qc_eid[eid]['processed'][pipeline_name] = {}
            for region in brain_regions:
                # sliding qc of the processed data
                qc_proc = qc_Tsd(
                    proc_tf[region],
                    qc_metrics=qc_metrics['processed'],
                    sliding_kwargs=qc_metrics['sliding_kwargs'],
                    eid=eid,
                )

                # qc with metrics that use behavior
                qc_resp = qc_Tsd(
                    proc_tf[region],
                    qc_metrics['response'],
                    trials=trials,
                    eid=eid,
                )

                qc_eid[eid]['processed'][pipeline_name][region] = qc_proc | qc_resp

    #     for pipe in pipelines_reg.keys():
    #         qc_res[pipe]['pname'] = pname
    #         qc_dfs[pipe][eid] = qc_res[pipe]

    #     gc.collect()
    # return qc_dfs
