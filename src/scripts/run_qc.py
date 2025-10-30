# %%
from pathlib import Path
from iblphotometry import metrics
from one.api import ONE
from iblphotometry.qc import run_qc
import sys
from tqdm import tqdm
from brainbox.io.one import PhotometrySessionLoader

# %%
one = ONE()
django = [
    'users__username,laura.silva',
    'lab__name,mainenlab',
    'projects__name__icontains,ibl_fibrephotometry',
    'start_time__lte,2025-10-15',
    'start_time__gte,2025-09-15',
]

sessions = one.alyx.rest('sessions', 'list', django=django)
eids = [session['id'] for session in sessions]

# %% download / filter data
bad_eids = []
for eid in tqdm(eids):
    try:
        psl = PhotometrySessionLoader(eid=eid, one=one)
        psl.load_photometry()
    except:
        bad_eids.append(eid)

eids = set(eids) - set(bad_eids)

print(len(eids))

# store?
# with open(Path(__file__).parent / 'eids.txt', 'w') as fH:
#     fH.writelines([eid + '\n' for eid in eids])

# with open(Path(__file__).parent / 'eids.txt', 'r') as fH:
#     lines = fH.readlines()
# eids = [line.strip() for line in lines]


# %% raw qc example
raw_metrics = [metrics.n_early_samples, metrics.n_edges]
raw_qc = run_qc(eids, one, metrics=raw_metrics, sliding_kwargs=None, n_jobs=16)
raw_qc.to_csv(Path(__file__).parent / 'raw_qc.csv')


# %% processed qc
from iblphotometry.pipelines import sliding_mad_pipeline


processed_metrics = [
    metrics.percentile_distance,
    metrics.percentile_asymmetry,
    metrics.signal_skew,
    metrics.ar_score,
    metrics.median_absolute_deviance,
]

# some custom settings for some metrics
metrics_kwargs = {'percentile_asymmetry': {'pc_comp': 75}}

# 10 windows of 5 seconds evently spread throughout the recording
sliding_kwargs = dict(
    n_windows=10,
    w_len=5,
)

processed_qc = run_qc(
    eids,
    one,
    # signal_band='GCaMP',
    metrics=processed_metrics,
    metrics_kwargs=metrics_kwargs,
    sliding_kwargs=sliding_kwargs,
    pipeline=sliding_mad_pipeline,
    n_jobs=16,
)

processed_qc.to_csv(Path(__file__).parent / 'processed_qc.csv')
