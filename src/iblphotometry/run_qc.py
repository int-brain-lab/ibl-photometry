# %%
import sys

import numpy as np
import pandas as pd
from pathlib import Path
import pynapple as nap

from utils import *  # don't

from one.api import ONE
from tqdm import tqdm

# one = ONE(base_url="https://alyx.internationalbrainlab.org")
one_dir = Path("/mnt/h0/kb/data/one")
one = ONE(cache_dir=one_dir)

# %% setup metrics
import metrics

# can be applied at all times
raw_metrics = [
    [metrics.n_unique_samples, None, None],
    [metrics.n_outliers, dict(w_size=1000, alpha=0.000005), None],
    [metrics.n_spikes, dict(sd=5), None],
    [
        metrics.bleaching_tau,
        None,
        None,
    ],  # <- problem: this one has a different call signature
]

# only apply after some form of processing
processed_metrics = [
    [metrics.signal_asymmetry, dict(pc_comp=95), None],
    [metrics.percentile_dist, dict(pc=(5, 95)), None],
    [metrics.signal_skew, None, None],
]

# apply after providing trial information
signal_metrics = [
    # [metrics.ttest_pre_post, dict()] # <- to be included
]


# %% get all eids in the correct order
df = pd.read_csv("website.csv")
eids = list(df["eid"])

# %%
import outlier_detection
import pipelines

qc_df = pd.DataFrame(index=eids)
problems = []

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

        # raw metrics
        for metric, params, _ in raw_metrics:
            for ch in ["calcium", "isosbestic"]:
                try:
                    F = raw_photometry[f"raw_{ch}"]
                    if params is not None:
                        qc_df.loc[eid, f"{metric.__name__}_{ch}"] = metric(
                            F.values, **params
                        )
                    else:
                        qc_df.loc[eid, f"{metric.__name__}_{ch}"] = metric(F.values)
                except Exception as e:
                    problems.append(f"{eid}_{metric.__name__}_{e}")

        # metrics on processed
        for ch in ["calcium", "isosbestic"]:
            F = raw_photometry[f"raw_{ch}"]

            # package this into a pipeline
            try:
                Fpp = outlier_detection.remove_spikes(F)
                Fc = pipelines.bc_lp_sliding_mad(Fpp)
            except Exception as e:
                problems.append(f"{eid}_{'preproccessing'}_{e}")
                continue

            for metric, params, _ in processed_metrics:
                try:
                    if params is not None:
                        qc_df.loc[eid, f"{metric.__name__}_{ch}"] = metric(
                            Fc.values, **params
                        )
                    else:
                        qc_df.loc[eid, f"{metric.__name__}_{ch}"] = metric(Fc.values)
                except Exception as e:
                    problems.append(f"{eid}_{metric.__name__}_{e}")


# %%
qc_df.to_csv("qc.csv")
