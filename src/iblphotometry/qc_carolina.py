# %% just here
# %load_ext autoreload
# %autoreload 2

# %%
import pandas as pd
from iblphotometry import metrics, outlier_detection, pipelines
from one.api import ONE
import logging
import iblphotometry.qc as qc
import iblphotometry.loaders as loaders

import warnings

warnings.simplefilter('ignore', category=DeprecationWarning)

# %% config

# User case specific variable
path_user = loaders.user_config('georg')
output_folder = path_user['dir_results'].joinpath('Carolina')
output_folder.mkdir(parents=True, exist_ok=True)

## params
run_name = 'caro_debug'
debug = False

# %% ONE related
one = ONE(mode='remote')
eids = list(one.search(dataset='photometry.signal.pqt', lab='cortexlab'))
if debug:
    eids = eids[:5]
data_loader = loaders.PhotometryLoader(one, verbose=False)

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
    [metrics.has_responses, dict(event_names=['feedback_times'])],
]

qc_metrics['sliding_kwargs'] = dict(w_len=10, n_wins=15)  # 10 seconds

# %% pipeline definition / registrations
from iblphotometry.outlier_detection import remove_spikes
from iblphotometry.bleach_corrections import lowpass_bleachcorrect
from iblphotometry.sliding_operations import sliding_mad
from iblphotometry.pipelines import run_pipeline
from iblphotometry.helpers import zscore

pipeline = [
    (remove_spikes, dict(sd=5)),
    (
        lowpass_bleachcorrect,
        dict(
            correction_method='subtract-divide',
            filter_params=dict(N=3, Wn=0.01, btype='lowpass'),
        ),
    ),
    (sliding_mad, dict(w_len=120, overlap=90)),
    (zscore, dict(mode='median')),
]

pipelines_reg = dict(sliding_mad=pipeline)

# %% run qc
qc_result = qc.run_qc(
    data_loader, eids, pipelines_reg, qc_metrics, sigref_mapping=dict(signal='GCaMP')
)
qc_df = pd.DataFrame(qc_result)
qc_df.to_csv(output_folder / f'qc_caroline_{run_name}.csv')
# %%
