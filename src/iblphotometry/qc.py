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

logger = logging.getLogger()


def qc_series(
    F: pd.Series,
    metrics: dict,
    sliding_kwargs=None,  # if present, calculate everything in a sliding manner
    trials=None,  # if present, put trials into params
    eid: str = None,  # FIXME but left as is for now just to keep the logger happy
    brain_region: str = None,  # FIXME but left as is for now just to keep the logger happy
) -> dict:
    if isinstance(F, pd.DataFrame):
        raise TypeError('F can not be a dataframe')

    qc_results = {}
    for metric, params in metrics:
        try:
            if trials is not None:  # if trials are passed
                params['trials'] = trials
            qc_results[metric] = eval_metric(F, metric, params, sliding_kwargs)
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
