"""definition processing / preprocessing:
preprocessing operates on the raw data files (pd.DataFrame, as returned by fpio.from_neurophotometrics_file_to_photometry_df)
processing operates on the dict[pd.DataFrame] format (split by signal band) as returned by fpio.from_photometry_df"""

# %%
import pandas as pd
from one.api import ONE
from pathlib import Path
from iblphotometry import fpio
import matplotlib.pyplot as plt
import numpy as np

# one = ONE()


# %%
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


# %% get example data
from one.api import ONE
from brainbox.io.one import PhotometrySessionLoader

one = ONE()

eid = '58861dac-4b4c-4f82-83fb-33d98d67df3a'
eid = '34f55b3a-725e-4cc7-aed3-6e6338f573bf'
psl = PhotometrySessionLoader(eid=eid, one=one)
psl.load_photometry()
brain_region = psl.photometry['GCaMP'].columns[0]
signal = psl.photometry['GCaMP'][brain_region]


psl.photometry['GCaMP'].shape
psl.photometry['Isosbestic'].shape

# %%
base_folder = Path('/mnt/s0/Data/Subjects')
session_folder = base_folder / one.eid2path(eid).session_path_short()
photometry_df = fpio.from_neurophotometrics_file_to_photometry_df(
    session_folder / 'raw_photometry_data' / '_neurophotometrics_fpData.raw.pqt', drop_first=True
)
photometry_df.shape[0] / 2
fpio.from_photometry_df(photometry_df, drop_first=False)

(photometry_df['name'] == '').sum()

# %%


# # %%
# eids = pd.read_csv('/home/georg/code/ibldevtools/georg/photometry/extraction/kcenia/kb_photometry_sessions_table.csv')[
#     'eid'
# ].unique()
# BASE_FOLDER = Path('/mnt/s0/Data/Subjects')
# eid = eids[0]
# for eid in eids:
#     session_folder = BASE_FOLDER / one.eid2path(eid).session_path_short()
#     raw_df = fpio.from_neurophotometrics_file_to_photometry_df(
#         session_folder / 'raw_photometry_data' / '_neurophotometrics_fpData.raw.pqt'
#     )
#     raw_df['dt'] = raw_df['times'].diff()
#     mu = raw_df['dt'].dropna().mean()
#     tol = 0.01
#     violations = (raw_df['dt'] < (mu - mu * tol)).values
#     if violations.sum() > 0:
#         print(eid)


# # %%
# # eid = "ecb124ad-ff23-497b-9143-8383f5dc55dd"
# # eid = "00575787-7b8a-431a-a70e-1df857871a64"
# eid = '23f10df6-fa11-44d1-8f71-07145f3c65a1'
# # eid = "12b299c1-69de-4c40-a8a3-c83d13d77a31"

# eid = '00575787-7b8a-431a-a70e-1df857871a64'
# eid = '23f10df6-fa11-44d1-8f71-07145f3c65a1'
# eid = '12b299c1-69de-4c40-a8a3-c83d13d77a31'
# eid = 'ecb124ad-ff23-497b-9143-8383f5dc55dd'
# eid = 'b7b186ee-1b36-4b1a-ae15-70078c472f97'
# eid = 'c5cdd496-7400-4c0f-b326-55016bce4fcf'
# eid = '27f7dd36-9a7c-4bbb-98ba-7d0c9efc9c6e'
# eid = '386e3195-48c0-4e5a-8e87-63482a68b8d1'
# eid = '40909756-d0ce-4146-9588-249bf97f074b'
# eid = '7c67fbd4-18c1-42f2-b989-8cbfde0d2374'
# eid = 'f7bc369d-2408-4d23-a586-c58363b2a02b'
# eid = '32eaad6d-fd35-46e9-b7cb-22656d8e3375'
# # eid = "8d1f87d1-58bf-4ac9-bc3b-9291ccc17535"
# # eid = "a44f72af-755f-4d58-b9bb-c5b345bcd788"

# session_folder = BASE_FOLDER / one.eid2path(eid).session_path_short()
# raw_df = fpio.from_neurophotometrics_file_to_photometry_df(
#     session_folder / 'raw_photometry_data' / '_neurophotometrics_fpData.raw.pqt'
# )
# raw_df['dt'] = raw_df['times'].diff()
# mu = raw_df['dt'].mean()
# tol = 0.01
# raw_df['dt_viol'] = raw_df['dt'] < (mu - mu * tol)

# fig, axes = plt.subplots()
# data_cols = [col for col in raw_df.columns if 'Region' in col]
# df_ = raw_df.query('color == "Blue"')
# for col in data_cols:
#     axes.plot(df_['times'].values, df_[col].values, label=col)
# axes.legend()

# for t in raw_df.groupby('dt_viol').get_group(True)['times'].values:
#     axes.axvline(t, lw=1, c='r', alpha=0.5)

# # %% find gaps
# gap_ix = np.where(raw_df['color'] == '')[0]

# # split into consecutive segments
# gap_splits = np.where(np.diff(gap_ix) > 1)[0]
# gaps = [gap_ix[0 : gap_splits[0] + 1]]
# for i in range(gap_splits.shape[0] - 1):
#     gaps.append(gap_ix[gap_splits[i] + 1 : gap_splits[i + 1] + 1])
# gaps.append(gap_ix[gap_splits[-1] :])

# print(len(gaps))

# # %% fill all gaps
# mu = raw_df['dt'].mean()
# for gap in gaps:
#     if raw_df.iloc[gap]['dt'].mean() > (mu - mu * 0.1) and raw_df.iloc[gap]['dt'].mean() < (mu + mu * 0.1):
#         # fill
#         prev_color = raw_df.iloc[gap[0] - 1]['color']
#         if prev_color == 'Blue':
#             fill_colors = np.array(['Violet', 'Blue'] * gap.shape[0])[: gap.shape[0]]
#         if prev_color == 'Violet':
#             fill_colors = np.array(['Blue', 'Violet'] * gap.shape[0])[: gap.shape[0]]
#         raw_df.loc[gap, 'color'] = fill_colors
#     else:
#         print('bad gap')

# # %%
# has_gaps(raw_df)

# # %% find the first sample that has bad dt and kick it
# mu = raw_df['dt'].mean()
# tol = 0.01
# violations = (raw_df['dt'] < (mu - mu * tol)).values
# while violations.sum() > 0:
#     print(violations.sum())
#     i = np.argmax(violations)
#     raw_df = raw_df.drop(i, axis=0)
#     raw_df.reset_index(inplace=True, drop=True)
#     raw_df['dt'] = raw_df['times'].diff()
#     violations = (raw_df['dt'] < (mu - mu * tol)).values

# # %%
# has_band_inversion(raw_df)


# # %% remove duplicates
# duplicate_ix = np.where(raw_df['color'].values == np.roll(raw_df['color'].values, -1))[0]
# while duplicate_ix.shape[0] > 0:
#     print(duplicate_ix.shape[0])
#     i = duplicate_ix[0]
#     color_violated = raw_df.iloc[i]['color'] == raw_df.iloc[i + 1]['color']
#     dt_violated = raw_df.iloc[i + 1]['dt'] < (mu - mu * 0.05)
#     if color_violated:
#         raw_df = raw_df.drop(i + 1, axis=0)
#         raw_df.reset_index(inplace=True, drop=True)
#         duplicate_ix = np.where(raw_df['color'].values == np.roll(raw_df['color'].values, -1))[0]
#         continue

#     # if dt_violated:
#     #     raw_df = raw_df.drop(i+1, axis=0)
#     #     raw_df.reset_index(inplace=True, drop=True)
#     #     duplicate_ix = np.where(raw_df['color'].values == np.roll(raw_df['color'].values,-1))[0]
#     #     continue

# # %%
# fig, axes = plt.subplots()
# data_cols = [col for col in raw_df.columns if 'Region' in col]
# df_ = raw_df.query('color == "Blue"')
# for col in data_cols:
#     axes.plot(df_['times'].values, df_[col].values, label=col)
# axes.legend()

# for t in raw_df.groupby('dt_viol').get_group(True)['times'].values:
#     axes.axvline(t, lw=1, c='r', alpha=0.5)

# %%
