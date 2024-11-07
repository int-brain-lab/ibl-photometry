# %%
import pandas as pd
from iblphotometry import metrics, outlier_detection, pipelines
from one.api import ONE
import logging
import qc
import iblphotometry.loaders as ffld

# %%
# User case specific variable
path_user = ffld.user_config('gaelle')

output_folder = path_user['dir_results']
output_folder.mkdir(parents=True, exist_ok=True)

one = ONE(mode='remote')

##
run_name = 'test_debug'
debug = True

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

# %% get all eids in the correct order
eids = one.search(dataset='photometry.signal.pqt')

if debug:
    eids = eids[:3]

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
    [metrics.has_responses, dict(event_names=BEHAV_EVENTS)],  # <- to be included
]

qc_metrics['sliding_kwargs'] = dict(w_len=10, n_wins=15)  # 10 seconds

# %% pipeline definition / registrations

# note care has to be taken that all the output and input of consecutive pipeline funcs are compatible
pipelines_reg = dict(
    sliding_mad=(
        (outlier_detection.remove_spikes_, dict(sd=5)),
        (pipelines.bc_lp_sliding_mad, dict(signal_name='raw_calcium')),
    )
)

# %% run qc
data_loader = ffld.AlexLoader(one)

qc_dfs = qc.run_qc(
    data_loader,
    pipelines_reg,
    qc_metrics,
    debug=True,
)

# storing all the qc
for pipe_name in pipelines_reg.keys():
    df = pd.DataFrame(qc_dfs[pipe_name]).T
    df.to_csv(output_folder / f'qc_{run_name}_{pipe_name}.csv')
