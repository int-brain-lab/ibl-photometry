# %%
import pandas as pd
from pathlib import Path
from iblphotometry import metrics, outlier_detection, pipelines
from one.api import ONE
import logging
import qc
import pynapple as nap


# %%
class alex_data_loader:
    def __init__(self, one, eids: list[str] = None):
        self.i = 0
        self.j = 0
        self.eids = (
            one.search(dataset='photometry.signal.pqt') if eids is None else eids
        )
        self.one = one
        pass

    def eid2regions(self, eid):
        rois = one.load_dataset(eid, 'photometryROI.locations.pqt')
        brain_regions = list(rois.brain_region)
        return brain_regions

    def get_data(self, eid, brain_region):
        # self.one.eid2pid(eid)
        photometry = one.load_dataset(eid, 'photometry.signal.pqt')
        photometry = photometry.groupby('name').get_group('GCaMP')  # discard empty
        trials = self.one.load_dataset(eid, '*trials.table')
        rois = one.load_dataset(eid, 'photometryROI.locations.pqt')
        photometry = photometry.rename(columns=rois['brain_region'].to_dict())
        raw_photometry = nap.TsdFrame(
            t=photometry['times'].values,
            d=photometry[brain_region].values,
            columns=['raw_calcium'],
        )
        pid = None  # FIXME
        return raw_photometry, trials, eid, pid, brain_region

    def __next__(self):
        # check if i is valid
        # if not, end iteration
        if self.i == len(self.eids):
            raise StopIteration
        eid = self.eids[self.i]

        # if i is valid, get brain regions
        brain_regions = self.eid2regions(eid)

        # check if j is valid
        if self.j < len(brain_regions):
            brain_region = brain_regions[self.j]
            self.j += 1
            return self.get_data(eid, brain_region)
        else:
            self.j = 0
            self.i += 1
            self.__next__()


# %%
run_name = 'test_debug'
debug = True

output_folder = Path('/home/georg/code/ibl-photometry/qc_results_alex/')
output_folder.mkdir(parents=True, exist_ok=True)

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

# %% one related
one = ONE(mode='remote')

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
data_loader = alex_data_loader(one)

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
