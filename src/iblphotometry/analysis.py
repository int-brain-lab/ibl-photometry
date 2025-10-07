"""
a collection of helpers for analysis
"""

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


# TODO change this to match the above call signature
def psth(signal, times, t_events, fs=None, event_window=np.array([-1, 2])):
    """
    Compute the peri-event time histogram of a calcium signal
    :param signal:
    :param times:
    :param t_events:
    :param fs:
    :param event_window:
    :return:
    """
    if fs is None:
        fs = 1 / np.nanmedian(np.diff(times))
    # compute a vector of indices corresponding to the perievent window at the given sampling rate
    sample_window = np.round(np.arange(event_window[0] * fs, event_window[1] * fs + 1)).astype(int)
    # we inflate this vector to a 2d array where each column corresponds to an event
    idx_psth = np.tile(sample_window[:, np.newaxis], (1, t_events.size))
    # we add the index of each event too their respective column
    idx_event = np.searchsorted(times, t_events)
    idx_psth += idx_event
    i_out_of_bounds = np.logical_or(idx_psth > (signal.size - 1), idx_psth < 0)
    idx_psth[i_out_of_bounds] = -1
    psth = signal[idx_psth]  # psth is a 2d array (ntimes, nevents)
    psth[i_out_of_bounds] = np.nan  # remove events that are out of bounds

    return psth, idx_psth
