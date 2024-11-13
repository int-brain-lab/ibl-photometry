import numpy as np
import copy
from iblutil.util import Bunch


def psth(calcium, times, t_events, fs=None, peri_event_window=None):
    """
    Compute the peri-event time histogram of a calcium signal
    :param calcium:
    :param times:
    :param t_events:
    :param fs:
    :param peri_event_window:
    :return:
    """
    fs = 1 / np.median(np.diff(times)) if fs is None else fs
    peri_event_window = [-1, 2] if peri_event_window is None else peri_event_window
    # compute a vector of indices corresponding to the perievent window at the given sampling rate
    sample_window = np.round(
        np.arange(peri_event_window[0] * fs, peri_event_window[1] * fs + 1)
    ).astype(int)
    # we inflate this vector to a 2d array where each column corresponds to an event
    idx_psth = np.tile(sample_window[:, np.newaxis], (1, t_events.size))
    # we add the index of each event too their respective column
    idx_event = np.searchsorted(times, t_events)
    idx_psth += idx_event
    i_out_of_bounds = np.logical_or(idx_psth > (calcium.size - 1), idx_psth < 0)
    idx_psth[i_out_of_bounds] = -1
    psth = calcium[idx_psth]  # psth is a 2d array (ntimes, nevents)
    psth[i_out_of_bounds] = np.nan  # remove events that are out of bounds
    return psth, idx_psth


# -------------------------------------------------------------------------------------------------
# Filtering of trials
# -------------------------------------------------------------------------------------------------
def _filter(obj, idx):
    obj = Bunch(copy.deepcopy(obj))
    for key in obj.keys():
        obj[key] = obj[key][idx]

    return obj


def filter_trials_by_trial_idx(trials, trial_idx):
    return _filter(trials, trial_idx)
