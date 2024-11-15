# %%
import pandas as pd
from iblphotometry import metrics, outlier_detection, pipelines
from one.api import ONE
import logging
import iblphotometry.qc as qc
import iblphotometry.loaders as loaders
from itertools import chain

##
# User case specific variable
path_user = loaders.user_config('georg')

##
run_name = 'test_post_merge'
debug = True

output_folder = path_user['dir_results'].joinpath('Kcenia')
output_folder.mkdir(parents=True, exist_ok=True)

one = ONE(cache_dir=path_user['dir_one'])

path_websitecsv = path_user['file_websheet']


# %% get all eids in the correct order

data_loader = loaders.KceniaLoader(one)
df = pd.read_csv(path_websitecsv)
eids = list(df['eid'])[:5]  # <- debug


# %%
# logging related
logger = logging.getLogger()
filemode = 'w'  # append 'w' is overwrite
filename = output_folder / f'fphot_qc_{run_name}.log'
file_handler = logging.FileHandler(filename=filename, mode=filemode)
log_fmt = '%(asctime)s - %(filename)s: - %(lineno)d] - %(levelname)s - %(message)s'
date_fmt = '%Y-%m-%d %H:%M:%S'
formatter = logging.Formatter(log_fmt, datefmt=date_fmt)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)


# %% setup metrics
qc_metrics = {}
# to be applied on the raw signal
qc_metrics['raw'] = [
    [metrics.n_unique_samples, None],
    [metrics.n_outliers, dict(w_size=1000, alpha=0.000005)],
    [metrics.n_spikes, dict(sd=5)],
    [metrics.bleaching_tau, None],
]

# only apply after pipelines are run
qc_metrics['processed'] = [
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

qc_metrics['response'] = [
    [metrics.ttest_pre_post, dict(event_name='feedback_times')],
    [metrics.has_responses, dict(event_names=BEHAV_EVENTS)],
]

qc_metrics['sliding_kwargs'] = dict(w_len=10, n_wins=15)  # 10 seconds

# %% pipeline definition / registrations

# note care has to be taken that all the output and input of consecutive pipeline funcs are compatible
# pipelines_reg = dict(
#     sliding_mad=(
#         (outlier_detection.remove_spikes, dict(sd=5)),
#         (pipelines.bc_lp_sliding_mad, dict(signal_name='raw_calcium')),
#     ),
#     isosbestic=(
#         (outlier_detection.remove_spikes, dict(sd=5)),
#         (pipelines.isosbestic_regression, dict(regression_method='irls')),
#     ),
#     jove2019=((pipelines.jove2019, dict()),),
# )
from iblphotometry.outlier_detection import remove_spikes
from iblphotometry.bleach_corrections import (
    isosbestic_correct,
    exponential_bleachcorrect,
    lowpass_bleachcorrect,
)
from iblphotometry.sliding_operations import sliding_mad
from iblphotometry.helpers import zscore
from iblphotometry.pipelines import run_pipeline

pipeline = [
    (remove_spikes, dict(sd=5)),
    (
        exponential_bleachcorrect,
        dict(regression_method='huber', correction_method='subtract'),
    ),
    (
        isosbestic_correct,
        dict(
            needs_reference=True,
            correction_method='subtract',
            regression_method='huber',
            lowpass_isosbestic=dict(N=3, Wn=0.01, btype='lowpass'),
        ),
    ),
    #   (sliding_mad, dict(w_len=120,overlap=90)),
    (zscore, dict(mode='median')),
]

pipelines_reg = dict(test=pipeline)

# %% run qc

qc_dfs = qc.run_qc(
    data_loader,
    eids,
    pipelines_reg,
    qc_metrics,
    sigref_mapping=dict(signal='raw_calcium', reference='raw_isosbestic'),
)

# storing all the qc
# for pipe_name in pipelines_reg.keys():
#     df = pd.DataFrame(qc_dfs[pipe_name]).T
#     df.to_csv(output_folder / f'qc_{run_name}_{pipe_name}.csv')
