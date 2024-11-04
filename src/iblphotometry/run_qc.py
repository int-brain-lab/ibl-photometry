# %%

import pandas as pd
from pathlib import Path
import pynapple as nap

from iblphotometry import metrics, outlier_detection, pipelines
from iblphotometry.evaluation import eval_metric

from one.api import ONE

from tqdm import tqdm
import logging

from copy import copy
import gc

# %%
run_name = 'test_3'
debug = True
dataset = 'kcenia'  # or 'alejandro'

output_folder = Path('/home/georg/code/ibl-photometry/qc_results')
output_folder.mkdir(parents=True, exist_ok=True)

# %%
# logging related
logger = logging.getLogger()
filemode = 'w'  # append 'w' is overwrite
filename = output_folder / f'fphot_qc_{run_name}.log'
file_handler = logging.FileHandler(filename=filename, mode=filemode)
log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
date_fmt = '%Y-%m-%d %H:%M:%S'
formatter = logging.Formatter(log_fmt, datefmt=date_fmt)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# %%
if dataset == 'kcenia':
    # one related - for xcenias dataset
    one_dir = Path('/mnt/h0/kb/data/one')
    one = ONE(cache_dir=one_dir)

    df = pd.read_csv('/home/georg/code/ibl-photometry/src/iblphotometry/website.csv')
    eids = list(df['eid'])

if dataset == 'alejandro':
    # one - for alejandros dataset
    one = ONE(base_url='https://alyx.internationalbrainlab.org', mode='remote')
    eids = one.search(dataset='photometry.signal.pqt')


# %%
if debug:
    eids = eids[:10]

# %% setup metrics

# to be applied on the raw signal
raw_metrics = [
    [metrics.n_unique_samples, None],
    [metrics.n_outliers, dict(w_size=1000, alpha=0.000005)],
    [metrics.n_spikes, dict(sd=5)],
    [metrics.bleaching_tau, None],
]

# only apply after pipelines are run
processed_metrics = [
    [metrics.signal_asymmetry, dict(pc_comp=95)],
    [metrics.percentile_dist, dict(pc=(5, 95))],
    [metrics.signal_skew, None],
]

# apply after providing trial information
BEHAV_EVENTS = [
    'stimOn_times',
    'goCue_times',
    'response_times',
    'feedback_times',
    'firstMovement_times',
    'intervals_0',
    'intervals_1',
]

response_metrics = [
    [metrics.ttest_pre_post, dict(event_name='feedback_times')],
    [metrics.has_responses, dict(event_names=BEHAV_EVENTS)],  # <- to be included
]

sliding_kwargs = dict(w_len=10, n_wins=15)  # 10 seconds

# %% pipeline definition / registrations

# note care has to be taken that all the output and input of consecutive pipeline funcs are compatible

if dataset == 'kcenia':
    pipelines_reg = dict(
        sliding_mad=(
            (outlier_detection.remove_spikes_, dict(sd=5)),
            (pipelines.bc_lp_sliding_mad, dict(signal_name='raw_calcium')),
        ),
        isosbestic=(
            (outlier_detection.remove_spikes_, dict(sd=5)),
            (pipelines.isosbestic_regression, dict(regression_method='irls')),
        ),
        jove2019=((pipelines.jove2019, dict()),),
    )

if dataset == 'alejandro':
    pipelines_reg = dict(
        sliding_mad=(
            (outlier_detection.remove_spikes_, dict(sd=5)),
            (pipelines.bc_lp_sliding_mad, dict(signal_name='GCaMP')),
        ),
        jove2019=((pipelines.jove2019, dict()),),
    )

# %% main QC loop
qc_dfs = {}
for pipe in pipelines_reg.keys():
    qc_dfs[pipe] = pd.DataFrame(index=eids)

for i, eid in enumerate(tqdm(eids)):
    # trials = one.load_dataset(eid, "_ibl_trials.table.pqt", collection='alf')
    trials = one.load_dataset(eid, '*trials.table')
    session_path = one.eid2path(eid)
    regions = [reg.name for reg in session_path.joinpath('alf').glob('Region*')]

    for i, region in enumerate(regions):
        # io related
        pqt_path = session_path / 'alf' / region / 'raw_photometry.pqt'
        raw_photometry = pd.read_parquet(pqt_path)
        raw_photometry = nap.TsdFrame(raw_photometry.set_index('times'))

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
                for metric, params in raw_metrics:
                    try:
                        res = eval_metric(F, metric, params)
                        qc_dfs[pipe_name].loc[eid, f'{metric.__name__}_{ch}'] = res[
                            'value'
                        ]
                    except Exception as e:
                        logger.warning(
                            f'{eid}: {metric.__name__} failure: {type(e).__name__}:{e}'
                        )

            # metrics on the output of the pipeline
            Fpp = photometry  # at this point, should be a nap.Tsd
            for metric, params in processed_metrics:
                try:
                    res = eval_metric(Fpp, metric, params, sliding_kwargs)
                    qc_dfs[pipe_name].loc[eid, f'{metric.__name__}'] = res['value']
                    qc_dfs[pipe_name].loc[eid, f'{metric.__name__}_r'] = res['rval']
                    qc_dfs[pipe_name].loc[eid, f'{metric.__name__}_p'] = res['pval']
                except Exception as e:
                    logger.warning(
                        f'{eid}: {metric.__name__} failure: {type(e).__name__}:{e}'
                    )

            # metrics that factor in behavior
            for metric, params in response_metrics:
                params['trials'] = trials
                try:
                    res = eval_metric(Fpp, metric, params)
                    qc_dfs[pipe_name].loc[eid, f'{metric.__name__}_{ch}'] = res['value']
                except Exception as e:
                    logger.warning(
                        f'{eid}: {metric.__name__} failure: {type(e).__name__}:{e}'
                    )
    gc.collect()

# %%
# storing all the qc
for pipe_name in pipelines_reg.keys():
    qc_dfs[pipe_name].to_csv(output_folder / f'qc_{run_name}_{pipe_name}.csv')
