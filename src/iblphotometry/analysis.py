import numpy as np
import pandas as pd
import pynapple as nap


def psth_np(
    signal: np.ndarray,
    times: np.ndarray,
    trials_df: pd.DataFrame,
    align_on: str = 'feedback_times',
    pre: float = -2.0,
    post: float = 2.0,
    split_by: str | None = 'feedbackType',
): ...


def psth_nap(
    signal: nap.Tsd,
    trials_df: pd.DataFrame,
    align_on: str = 'feedback_times',
    pre: float = -2.0,
    post: float = 2.0,
    split_by: str | None = 'feedbackType',
):
    psths = {}
    for outcome, group in trials_df.groupby(split_by):
        tstamps = nap.Ts(group[align_on].values)
        tstamps = tstamps.get(signal.t[0] + pre, signal.t[-1] + post)
        psth = nap.compute_perievent_continuous(signal, tstamps, (pre, post))
        psths[outcome] = psth
    return psths
