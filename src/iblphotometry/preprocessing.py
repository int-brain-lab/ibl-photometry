"""
This modules offers pre-processing for raw photometry data.
"""
from typing import Optional
import numpy as np
import pandas as pd
import scipy.signal


def find_early_samples(
    A: pd.Series, dt: Optional[float] = None, dt_tol: float = 0.001
) -> np.ndarray:
    """
    Find instances where the dt between samples is smaller than expected, given
    a certain tolerance. The expected dt is calculated as the median dt.

    Parameters
    ----------
    A :
        the Pandas series containing the photometry signal (the index must be
        sample times)
    dt_tol :
        the maximum difference allowed between the expected and observed sample
        time

    Returns
    -------
    out :
        boolean array indexing samples that occurred before the expected
        sampling rate
    """
    if dt is None:
        dt = np.median(np.diff(A.index))
    return dt - A.index.diff() > dt_tol


def _fill_missing_channel_names(A: np.ndarray) -> np.ndarray:
    """
    Fill missing channel names assuming a certain sampling sequence (defined by
    the name_alternator variable).

    Parameters
    ----------
    A :
        an array containng channel names with missing entries (empty strings)

    Returns
    -------
    A :
        an array containing channel names with no missing entries
    """
    missing_inds = np.where(A == '')[0]
    ## FIXME: ideally this name alternator is defined using the excitation sequence
    name_alternator = {'GCaMP': 'Isosbestic', 'Isosbestic': 'GCaMP', '': ''}
    for i in missing_inds:
        if i == 0:  # handle case where first sample is missing a name
            ## FIXME: this only works in the case of 2 channels!!!
            A[i] = name_alternator[A[i + 1]]
        else:  # use the previous sample to get next channel name
            A[i] = name_alternator[A[i - 1]]
    return A


def find_repeated_samples(
    A: pd.DataFrame,
    dt: Optional[float] = None,
    dt_tol: float = 0.001,
) -> int:
    if any(A['name'] == ''):
        A['name'] = _fill_missing_channel_names(A['name'].values)
    repeated_sample_mask = A['name'].iloc[1:].values == A['name'].iloc[:-1].values
    repeated_samples = A.iloc[1:][repeated_sample_mask]
    early_samples = A[find_early_samples(A, dt=dt, dt_tol=dt_tol)]
    if not all([idx in early_samples.index for idx in repeated_samples.index]):
        print('WARNING: repeated samples found without early sampling')
    return repeated_sample_mask


def fix_repeated_sampling(
    A: pd.DataFrame, dt_tol: float = 0.001, w_size: int = 10, roi: str | None = None
) -> int:
    ## TODO: this only works if gcamp and iso are still interleaved
    ## TODO: avoid this by explicitly handling multiple channels
    assert roi is not None
    # Drop first samples if channel labels are missing
    A.loc[A['name'].replace({'': np.nan}).first_valid_index() :]
    # Fix remaining missing channel labels
    if any(A['name'] == ''):
        A['name'] = _fill_missing_channel_names(A['name'].values)
    repeated_sample_mask = find_repeated_samples(A, dt_tol=dt_tol)
    name_alternator = {'GCaMP': 'Isosbestic', 'Isosbestic': 'GCaMP'}
    for i in np.where(repeated_sample_mask)[0] + 1:
        name = A.iloc[i]['name']
        value = A.iloc[i][roi]
        i0, i1 = A.index[i - w_size], A.index[i]
        same = A.loc[i0:i1].query('name == @name')[roi].mean()
        other_name = name_alternator[name]
        other = A.loc[i0:i1].query('name == @other_name')[roi].mean()
        assert np.abs(value - same) > np.abs(value - other)
        A.loc[A.index[i] :, 'name'] = [
            name_alternator[name] for name in A.loc[A.index[i] :, 'name']
        ]
    return A

def low_pass_filter(raw_signal, fs):
    params = {}
    sos = scipy.signal.butter(
        fs=fs,
        output='sos',
        **params.get('butterworth_lowpass', {'N': 3, 'Wn': 0.01, 'btype': 'lowpass'}),
    )
    signal_lp = scipy.signal.sosfiltfilt(sos, raw_signal)
    return signal_lp


def mad_raw_signal(raw_signal, fs):
    # This is a convenience function to get going whilst the preprocessing refactoring is being done
    # TODO delete this function once processing can be applied
    signal_lp = low_pass_filter(raw_signal, fs)
    signal_processed = (raw_signal - signal_lp) / signal_lp
    return signal_processed
