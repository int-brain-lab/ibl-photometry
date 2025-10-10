"""definition processing / preprocessing:
preprocessing operates on the raw data files (pd.DataFrame, as returned by fpio.from_neurophotometrics_file_to_photometry_df)
processing operates on the dict[pd.DataFrame] format (split by signal band) as returned by fpio.from_photometry_df"""

import pandas as pd
import numpy as np


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
