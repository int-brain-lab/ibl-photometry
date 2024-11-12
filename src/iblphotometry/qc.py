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

import warnings

logger = logging.getLogger()


# %%
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


# %%
def qc_single(
    raw_photometry: nap.Tsd | nap.TsdFrame,
    trials: pd.DataFrame,
    pipelines_reg: dict,
    qc_metrics: dict,
    eid: str,  # <- should be pid
) -> dict:
    """run QC on a single experiment / photometry session.

    Args:
        raw_photometry (nap.TsdFrame): _description_
        trials (pd.DataFrame): _description_
        pipelines_reg (dict): _description_
        qc_metrics (dict): _description_

    Returns:
        dict: _description_
    """

    qc = {}  # dict of dicts - fist level keys are pipeline_names, second level keys are metrics
    for pipe_name in pipelines_reg.keys():
        qc[pipe_name] = {}

    # restricting the fluorescence data to the time within the task
    t_start = trials.iloc[0]['intervals_0'] - 10
    t_stop = trials.iloc[-1]['intervals_1'] + 10
    session_interval = nap.IntervalSet(t_start, t_stop)
    raw_photometry = raw_photometry.restrict(session_interval)

    # iterate over pipelines
    for pipe_name, pipeline in pipelines_reg.items():
        # run pipeline
        try:
            photometry = copy(raw_photometry)
            # run the entire pipeline function by funtion
            # for this to work, the output and input types of each
            # pipeline function have to be compatible!
            for i, (pipe_func, pipe_args) in enumerate(pipeline):
                photometry = pipe_func(photometry, **pipe_args)
        except Exception as e:
            logger.warning(
                f'{eid}: pipeline {pipe_name} fails with: {type(e).__name__}:{e}'
            )
            continue

        # raw metrics - a bit redundant but just to have everyting combined together
        # if multiple channels:
        for metric, params in qc_metrics['raw']:
            if isinstance(photometry, nap.TsdFrame):
                for ch in photometry.columns:
                    F = photometry[ch]
                    try:
                        res = eval_metric(F, metric, params)
                        qc[pipe_name][f'{metric.__name__}_{ch}'] = res['value']
                    except Exception as e:
                        logger.warning(
                            f'{eid}, {ch}: metric {metric.__name__} failure: {type(e).__name__}:{e}'
                        )

            else:  # is a nap.Tsd
                try:
                    res = eval_metric(photometry, metric, params)
                    qc[pipe_name][f'{metric.__name__}'] = res['value']
                except Exception as e:
                    logger.warning(
                        f'{eid}: metric {metric.__name__} failure: {type(e).__name__}:{e}'
                    )

        # metrics on the output of the pipeline
        # at this point, photometry is a nap.Tsd
        Fpp = photometry
        for metric, params in qc_metrics['processed']:
            try:
                res = eval_metric(Fpp, metric, params, qc_metrics['sliding_kwargs'])
                qc[pipe_name][f'{metric.__name__}'] = res['value']
                qc[pipe_name][f'{metric.__name__}_r'] = res['rval']
                qc[pipe_name][f'{metric.__name__}_p'] = res['pval']
            except Exception as e:
                logger.warning(
                    f'{eid}: metric {metric.__name__} failure: {type(e).__name__}:{e}'
                )

        # metrics that factor in behavior
        for metric, params in qc_metrics['response']:
            params['trials'] = trials
            try:
                res = eval_metric(Fpp, metric, params)
                qc[pipe_name][f'{metric.__name__}'] = res['value']
            except Exception as e:
                logger.warning(
                    f'{eid}: metric {metric.__name__} failure: {type(e).__name__}:{e}'
                )

    return qc


# %% main QC loop


def run_qc(data_loader, pids: list[str], pipelines_reg, qc_metrics):
    # Creating dictionary of dictionary, with each key being the pipeline name
    qc_dfs = dict((ikey, dict()) for ikey in pipelines_reg.keys())

    for pid in tqdm(pids):
        raw_photometry = data_loader.load_photometry_data(pid=pid)
        eid, pname = data_loader.pid2eid(pid)
        trials = data_loader.load_trials_table(eid)

        qc_res = qc_single(raw_photometry, trials, pipelines_reg, qc_metrics, pid)

        for pipe in pipelines_reg.keys():
            qc_res[pipe]['pname'] = pname
            qc_dfs[pipe][eid] = qc_res[pipe]

        gc.collect()
    return qc_dfs
