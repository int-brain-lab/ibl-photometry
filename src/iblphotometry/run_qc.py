# %%
import sys

import numpy as np
import pandas as pd
from pathlib import Path
import pynapple as nap

from utils import *  # don't
import metrics
import outlier_detection
import pipelines
from evaluation import eval_metric

from one.api import ONE
from tqdm import tqdm

import logging

# logging related
logger = logging.get_logger()
filemode = "a"  # append 'w' is overwrite
filename = Path("fphot_qc.log")
file_handler = logging.FileHandler(filename=filename, mode=filemode)
log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
date_fmt = "%Y-%m-%d %H:%M:%S"
formatter = logging.Formatter(log_fmt, datefmt=date_fmt)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# one = ONE(base_url="https://alyx.internationalbrainlab.org")
one_dir = Path("/mnt/h0/kb/data/one")
one = ONE(cache_dir=one_dir)

# %% setup metrics

# to be applied on the raw signal
raw_metrics = [
    [metrics.n_unique_samples, None],
    [metrics.n_outliers, dict(w_size=1000, alpha=0.000005)],
    [metrics.n_spikes, dict(sd=5)],
    [metrics.bleaching_tau, None],
]

# only apply after some form of processing
processed_metrics = [
    [metrics.signal_asymmetry, dict(pc_comp=95)],
    [metrics.percentile_dist, dict(pc=(5, 95))],
    [metrics.signal_skew, None],
]

# apply after providing trial information
response_metrics = [
    [metrics.ttest_pre_post, dict(event_name="feedback_time")]  # <- to be included
]

sliding_kwargs = dict(w_len=10, n_wins=15)  # 10 seconds
# %% get all eids in the correct order
df = pd.read_csv("website.csv")
eids = list(df["eid"])

# %%
qc_df = pd.DataFrame(index=eids)

for i, eid in enumerate(tqdm(eids)):
    # trials = one.load_dataset(eid, "_ibl_trials.table.pqt", collection='alf')
    trials = one.load_dataset(eid, "*trials.table")
    session_path = one.eid2path(eid)
    regions = [reg.name for reg in session_path.joinpath("alf").glob("Region*")]

    for i, region in enumerate(regions):
        # io related
        pqt_path = session_path / "alf" / region / "raw_photometry.pqt"
        raw_photometry = pd.read_parquet(pqt_path)
        raw_photometry = nap.TsdFrame(raw_photometry.set_index("times"))

        # restricting the fluorescence data to the time within the task
        t_start = trials.iloc[0]["intervals_0"] - 10
        t_stop = trials.iloc[-1]["intervals_1"] + 10
        session_interval = nap.IntervalSet(t_start, t_stop)
        raw_photometry = raw_photometry.restrict(session_interval)

        # metrics on processed
        for ch in ["calcium", "isosbestic"]:
            F = raw_photometry[f"raw_{ch}"]

            # raw metrics
            for metric, params in raw_metrics:
                try:
                    res = eval_metric(F, metric, params)
                    qc_df.loc[eid, f"{metric.__name__}_{ch}"] = res["value"]
                except Exception as e:
                    logger.warning(f"{eid}: {metric.__name__} failure: {e}")

            # metrics on preprocessed data
            try:  # this is essentially the pipeline
                Fpp = outlier_detection.remove_spikes(F)
                Fc = pipelines.bc_lp_sliding_mad(Fpp)
            except Exception as e:
                logger.warning(f"{eid}: preproccessing failure: {e}")
                continue

            for metric, params in processed_metrics:
                try:
                    res = eval_metric(Fc, metric, params, sliding_kwargs)
                    qc_df.loc[eid, f"{metric.__name__}_{ch}"] = res["value"]
                    qc_df.loc[eid, f"{metric.__name__}_{ch}_r"] = res["r"]
                    qc_df.loc[eid, f"{metric.__name__}_{ch}_p"] = res["p"]
                except Exception as e:
                    logger.warning(f"{eid}: {metric.__name__} failure: {e}")

            # metrics that factor in behavior
            for metric, params in response_metrics:
                try:
                    res = eval_metric(Fc, metric, params)
                    qc_df.loc[eid, f"{metric.__name__}_{ch}"] = res["value"]
                except Exception as e:
                    logger.warning(f"{eid}: {metric.__name__} failure: {e}")

# %%
qc_df.to_csv("qc.csv")
