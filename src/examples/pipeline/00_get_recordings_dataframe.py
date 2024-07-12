import numpy as np
import pandas as pd

from pathlib import Path

from one.api import ONE  # always after the imports

PATH_RAW_PHOTOMETRY = Path("/mnt/h0/kb/data/external_drive")
PATH_WORKDIR = Path("/mnt/h0/kb/data/staged_data/")
# one = ONE()
one = ONE(cache_dir="/mnt/h0/kb/data/one")

dfxls = pd.read_excel('/mnt/h0/kb/Mice performance tables 100.xlsx', 'A4_2024', dtype={'nph_file': int, 'nph_bnc': int, 'region': int})
# dfxls.shape
# dfxls['region2'].unique()
assert np.isnan(dfxls['region2'].unique())
dfxls = dfxls.drop(columns=['region2', 'Good?'])
dfxls['date'] = pd.to_datetime(dfxls['date'])
for f in ['eid', 'file_raw_photometry', 'file_digital_input']:
    dfxls[f] = None

for i, rec in dfxls.iterrows():
    eid = one.search(subject=rec.mouse, date=rec.date)[0]
    dfxls.loc[i, 'eid'] = eid
    dfxls.loc[i, 'region'] = f'Region{rec.region}G'

    source_folder = PATH_RAW_PHOTOMETRY.joinpath(str(rec.date.date()))
    if not source_folder.exists():
        print(i, f"Folder {source_folder} does not exist")
        continue
    file_raw_photometry = source_folder.joinpath(f"raw_photometry{rec.nph_file}.csv")
    file_digital_input = source_folder.joinpath(f"bonsai_DI{rec.nph_bnc}{rec.nph_file}.csv")
    if not file_raw_photometry.exists() or not file_digital_input.exists():
        print(i, f"Files {file_raw_photometry} or {file_digital_input} do not exist")
        continue
    dfxls.loc[i, 'file_raw_photometry'] = file_raw_photometry
    dfxls.loc[i, 'file_digital_input'] = file_digital_input

dfxls = dfxls.loc[~dfxls['file_raw_photometry'].apply(lambda x: x is None), :].copy()
dfxls['file_raw_photometry'] = dfxls['file_raw_photometry'].apply(str)
dfxls['file_digital_input'] = dfxls['file_digital_input'].apply(str)
dfxls.to_parquet(PATH_WORKDIR.joinpath('00_recordings.pqt'))


# TODO Kcenia: fix those non-existing folders / files
# 376 Folder /mnt/h0/kb/data/external_drive/2022-08-04 does not exist
# 377 Folder /mnt/h0/kb/data/external_drive/2022-08-03 does not exist
# 378 Folder /mnt/h0/kb/data/external_drive/2022-08-02 does not exist
# 379 Folder /mnt/h0/kb/data/external_drive/2022-07-06 does not exist
# 380 Folder /mnt/h0/kb/data/external_drive/2022-07-05 does not exist
# 381 Folder /mnt/h0/kb/data/external_drive/2022-06-30 does not exist
# 382 Folder /mnt/h0/kb/data/external_drive/2022-06-29 does not exist
# 383 Folder /mnt/h0/kb/data/external_drive/2022-06-28 does not exist
# 384 Folder /mnt/h0/kb/data/external_drive/2022-06-27 does not exist
# 385 Folder /mnt/h0/kb/data/external_drive/2022-06-22 does not exist
# 386 Folder /mnt/h0/kb/data/external_drive/2022-06-21 does not exist
# 392 Files /mnt/h0/kb/data/external_drive/2022-12-24/raw_photometry3.csv or /mnt/h0/kb/data/external_drive/2022-12-24/bonsai_DI03.csv do not exist
# 453 Files /mnt/h0/kb/data/external_drive/2022-11-22/raw_photometry1.csv or /mnt/h0/kb/data/external_drive/2022-11-22/bonsai_DI01.csv do not exist
# 499 Files /mnt/h0/kb/data/external_drive/2022-12-24/raw_photometry3.csv or /mnt/h0/kb/data/external_drive/2022-12-24/bonsai_DI13.csv do not exist
# 504 Files /mnt/h0/kb/data/external_drive/2022-11-22/raw_photometry1.csv or /mnt/h0/kb/data/external_drive/2022-11-22/bonsai_DI11.csv do not exist


# a9c3231c-f747-4a1b-80e0-bbcd5c6df16b example of a session that needs re-extraction because the sound BNC is off