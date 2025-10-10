"""definition processing / preprocessing:
preprocessing operates on the raw data files (pd.DataFrame, as returned by fpio.from_neurophotometrics_file_to_photometry_df)
processing operates on the dict[pd.DataFrame] format (split by signal band) as returned by fpio.from_photometry_df"""

import pandas as pd
import numpy as np
from typing import Optional


def has_gaps(photometry_df: pd.DataFrame) -> bool:
    # some neurophotometrics files come with "gaps" in the band information
    return '' in photometry_df['color'].unique()


def find_gaps(photometry_df: pd.DataFrame) -> list[np.ndarray]:
    # looks for gaps and returns a list of gap indices
    gap_ix = np.where(photometry_df['color'] == '')[0]

    # split into consecutive segments
    gap_splits = np.where(np.diff(gap_ix) > 1)[0]
    gaps = [gap_ix[0 : gap_splits[0] + 1]]
    for i in range(gap_splits.shape[0] - 1):
        gaps.append(gap_ix[gap_splits[i] + 1 : gap_splits[i + 1] + 1])
    gaps.append(gap_ix[gap_splits[-1] :])

    return gaps


def fill_gaps(photometry_df: pd.DataFrame, gaps: list[np.ndarray]) -> pd.DataFrame:
    photometry_df['dt'] = photometry_df['times'].diff()
    mu = photometry_df['dt'].mean()

    # fills the gaps
    for gap in gaps:
        if photometry_df.iloc[gap]['dt'].mean() > (mu - mu * 0.1) and photometry_df.iloc[gap]['dt'].mean() < (mu + mu * 0.1):
            # fill
            # TODO deal with these hardcodes
            prev_color = photometry_df.iloc[gap[0] - 1]['color']
            if prev_color == 'Blue':
                fill_colors = np.array(['Violet', 'Blue'] * gap.shape[0])[: gap.shape[0]]
            if prev_color == 'Violet':
                fill_colors = np.array(['Blue', 'Violet'] * gap.shape[0])[: gap.shape[0]]
            photometry_df.loc[gap, 'color'] = fill_colors
        else:
            # TODO should have a logger warning or similar
            print('bad gap')

    return photometry_df


def has_band_inversion(raw_df, check_col='color'):
    # this checks for strictly alternating
    # valid only if measurement is dual band
    # first frame is not all led on
    col = raw_df[check_col]
    return np.any(col.values == np.roll(col.values, 1))


## FIXME: we now have two ways to check for the same NPH sampling error!!
def find_early_samples(A: pd.Series, dt: Optional[float] = None, dt_tol: float = 0.001) -> np.ndarray:
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


def fix_repeated_sampling(A: pd.DataFrame, dt_tol: float = 0.001, w_size: int = 10, roi: str | None = None) -> int:
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
        # other = A.loc[i0:i1].query('name == @other_name')[roi].mean()
        # alternative syntax just to keep ruff happy
        other = A.loc[i0:i1].groupby('name').get_group(other_name)[roi].mean()
        assert np.abs(value - same) > np.abs(value - other)
        A.loc[A.index[i] :, 'name'] = [name_alternator[name] for name in A.loc[A.index[i] :, 'name']]
    return A
