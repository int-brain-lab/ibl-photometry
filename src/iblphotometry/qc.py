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

warnings.filterwarnings('once', category=DeprecationWarning, module='pynapple')

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
    F: nap.Tsd,
    qc_metrics: dict,
    sliding_kwargs=None,  # if present, calculate everything in a sliding manner
    trials=None,  # if present, put trials into params
    eid: str = None,  # FIXME but left as is for now just to keep the logger happy
    brain_region: str = None,  # FIXME but left as is for now just to keep the logger happy
) -> dict:
    if isinstance(F, nap.TsdFrame):
        raise TypeError('F can not be nap.TsdFrame')

    # should cover all cases
    qc_results = {}
    for metric, params in qc_metrics:
        try:
            if trials:  # if trials are passed
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
        # get data
        raw_tfs, brain_regions = data_loader.load_photometry_data(
            eid=eid, return_regions=True
        )
        # TODO this should be provided
        trials = data_loader.one.load_dataset(eid, '*trials.table.pqt')

        for band in raw_tfs.keys():  # TODO this should be bands
            raw_tf = raw_tfs[band]
            for region in brain_regions:
                qc_result = qc_Tsd(
                    raw_tf[region], qc_metrics['raw'], sliding_kwargs=None, eid=eid
                )
                qc_results.append(
                    dict(eid=eid, pipeline='raw', band=band, region=region, **qc_result)
                )

        # run the pipelines and qc on the processed data
        # here it needs to be specified if one band is a reference of the other
        for pipeline_name, pipeline in pipelines_reg.items():
            if 'reference' in sigref_mapping:  # this is for isosbestic pipelines
                proc_tf = run_pipeline(
                    pipeline,
                    raw_tfs[sigref_mapping['signal']],
                    raw_tfs[sigref_mapping['reference']],
                )
            else:
                # FIXME this fails for true-multiband
                # this hack works for single-band
                proc_tf = run_pipeline(pipeline, raw_tfs[sigref_mapping['signal']])

            for region in brain_regions:
                # sliding qc of the processed data
                qc_proc = qc_Tsd(
                    proc_tf[region],
                    qc_metrics=qc_metrics['processed'],
                    sliding_kwargs=qc_metrics['sliding_kwargs'],
                    eid=eid,
                    brain_region=region,
                )

                # qc with metrics that use behavior
                qc_resp = qc_Tsd(
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
                        band=band,
                        region=region,
                        **qc_result,
                    )
                )
    #     gc.collect()
    return qc_results
