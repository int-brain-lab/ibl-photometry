import numpy as np
import copy
from iblutil.util import Bunch


def psth_times(fs, event_window):
    psth_times = np.arange(event_window[0] * fs, event_window[1] * fs + 1)
    return psth_times

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
    sample_window = np.round(
        np.arange(event_window[0] * fs, event_window[1] * fs + 1)
    ).astype(int)
    # we inflate this vector to a 2d array where each column corresponds to an event
    idx_psth = np.tile(sample_window[:, np.newaxis], (1, t_events.size))
    # we add the index of each event too their respective column
    idx_event = np.searchsorted(times, t_events)
    idx_psth += idx_event
    i_out_of_bounds = np.logical_or(idx_psth > (signal.size - 1), idx_psth < 0)
    idx_psth[i_out_of_bounds] = -1
    psth = signal[idx_psth]  # psth is a 2d array (ntimes, nevents)
    psth[i_out_of_bounds] = np.nan  # remove events that are out of bounds

    return psth, idx_psth  #TODO transpose PSTH before return


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
