# %%
import pandas as pd
from pathlib import Path
import pynapple as nap

from iblphotometry import metrics, outlier_detection, pipelines
from iblphotometry.evaluation import eval_metric

from one.api import ONE

from tqdm import tqdm
import logging
from pprint import pprint

from copy import copy
import gc

import warnings

logger = logging.getLogger()


# %%
def qc_single(
    raw_photometry: nap.TsdFrame,
    trials: pd.DataFrame,
    pipelines_reg: dict,
    qc_metrics: dict,
    eid: str,
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

    # process all pipelines
    for pipe_name, pipe in pipelines_reg.items():
        # run pipeline
        try:
            photometry = copy(raw_photometry)
            for i, (pipe_func, pipe_args) in enumerate(pipe):
                photometry = pipe_func(photometry, **pipe_args)
        except Exception as e:
            logger.warning(
                f'{eid}: pipeline {pipe_name} fails with: {type(e).__name__}:{e}'
            )
            continue

        # raw metrics - a bit redundant but just to have everyting combined together
        for ch in raw_photometry.columns:
            F = raw_photometry[ch]

            # raw metrics
            for metric, params in qc_metrics['raw']:
                try:
                    res = eval_metric(F, metric, params)
                    qc[pipe_name][f'{metric.__name__}_{ch}'] = res['value']
                except Exception as e:
                    logger.warning(
                        f'{eid}: {metric.__name__} failure: {type(e).__name__}:{e}'
                    )

        # metrics on the output of the pipeline
        Fpp = photometry  # at this point, should be a nap.Tsd
        for metric, params in qc_metrics['processed']:
            try:
                res = eval_metric(Fpp, metric, params, qc_metrics['sliding_kwargs'])
                qc[pipe_name][f'{metric.__name__}'] = res['value']
                qc[pipe_name][f'{metric.__name__}_r'] = res['rval']
                qc[pipe_name][f'{metric.__name__}_p'] = res['pval']
            except Exception as e:
                logger.warning(
                    f'{eid}: {metric.__name__} failure: {type(e).__name__}:{e}'
                )

        # metrics that factor in behavior
        for metric, params in qc_metrics['response']:
            params['trials'] = trials
            try:
                res = eval_metric(Fpp, metric, params)
                qc[pipe_name][f'{metric.__name__}_{ch}'] = res['value']
            except Exception as e:
                logger.warning(
                    f'{eid}: {metric.__name__} failure: {type(e).__name__}:{e}'
                )

    return qc


# %% main QC loop


def run_qc(eids, one, pipelines_reg, qc_metrics, local=True):
    qc_dfs = {}
    for pipe in pipelines_reg.keys():
        qc_dfs[pipe] = {}  # pd.DataFrame(index=eids)

    # -> this part is specific to locally stored data (= kcenia)
    for i, eid in enumerate(tqdm(eids)):
        trials = one.load_dataset(eid, '*trials.table')
        session_path = one.eid2path(eid)
        if local:
            brain_regions = [
                reg.name for reg in session_path.joinpath('alf').glob('Region*')
            ]
        else:
            rois = one.load_dataset(eid, 'photometryROI.locations.pqt')
            brain_regions = list(rois.brain_region)

        for i, region in enumerate(brain_regions):
            # io related
            if local:
                pqt_path = session_path / 'alf' / region / 'raw_photometry.pqt'
                raw_photometry = pd.read_parquet(pqt_path)
                raw_photometry = nap.TsdFrame(raw_photometry.set_index('times'))
            else:
                photometry = one.load_dataset(eid, 'photometry.signal.pqt')
                photometry = photometry.groupby('name').get_group(
                    'GCaMP'
                )  # discard empty
                photometry = photometry.rename(columns=rois['brain_region'].to_dict())
                raw_photometry = nap.TsdFrame(
                    t=photometry['times'].values,
                    d=photometry[region].values,
                    columns=['raw_calcium'],
                )

            qc_res = qc_single(raw_photometry, trials, pipelines_reg, qc_metrics, eid)

            for pipe in pipelines_reg.keys():
                qc_res[pipe]['brain_region'] = region
                qc_dfs[pipe][eid] = qc_res[pipe]

        gc.collect()
    return qc_dfs
