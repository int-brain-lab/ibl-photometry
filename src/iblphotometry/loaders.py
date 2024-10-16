from one.api import ONE
import glob
from pathlib import Path
import pandas as pd
from iblphotometry.preprocessing import jove2019, psth, preprocess_sliding_mad, photobleaching_lowpass



def get_pname_ccu(path_eid):
    path_patter = path_eid.joinpath('alf/Region*/raw_photometry.pqt')
    path_ffpqt = glob.glob(path_patter.as_posix())
    pname_list = list()
    if len(path_ffpqt) > 1:
        for path in path_ffpqt:
            pname_list.append(Path(path).parent.parts[-1])
    else:
        pname_list.append(Path(path_ffpqt[0]).parent.parts[-1])

    return pname_list, path_ffpqt


def get_df_nph_ccu(eid, pname=None, one=None):
    if one is None:
        one = ONE()
    path_eid = one.eid2path(eid)

    if pname is None:
        pname_list, path_ffpqt = get_pname_ccu(path_eid)
        # Forcing the first region to be taken
        path_ffpqt = path_ffpqt[0]
        pname = pname_list[0]
    else:
        path_ffpqt = path_eid.joinpath(f'alf/{pname}/raw_photometry.pqt')

    df_nph = pd.read_parquet(path_ffpqt)

    return df_nph, pname


def get_df_nph_princeton(eid, one, pname='Region0G'):
    # Load the photometry signal dataset
    photometry = one.load_dataset(eid, 'photometry.signal.pqt')
    # Take only the wavelength for the signal CA
    # There is no ISO for Alejandro's data
    photometry = photometry[photometry['wavelength'] == 470]
    # Create dataframe for internal representation
    df_nph = pd.DataFrame()
    if pname not in photometry.columns:
        # Forcing the first region to be taken
        cols = photometry.filter(regex='^Region').columns
        pname = cols[0]
    df_nph["raw_calcium"] = photometry[pname]
    df_nph["times"] = photometry['times']
    return df_nph, pname


# Load trial from ONE
def load_trial(eid, one):
    a = one.load_object(eid, 'trials')
    df_trials = a.to_df()
    return df_trials


def prep_df_nph(df_nph, df_trials):
    # Ugly way to get sampling frequency
    time_diffs = df_nph["times"].diff().dropna()
    fs = 1 / time_diffs.median()

    # Crop the photometry data around the behavior
    inc_crop = int(round(100 / fs))  # Crop 100 second at start/end of recording
    session_start = df_trials.intervals_0.values[0] - inc_crop  # Crop before the first tph value
    session_end = df_trials.intervals_1.values[-1] + inc_crop  # Crop after the last tph value

    # Select data within the specified time range
    selected_data = df_nph[(df_nph['times'] >= session_start) & (df_nph['times'] <= session_end)]

    df_nph = selected_data.reset_index(drop=True)

    # Processing
    df_nph['calcium_photobleach'] = photobleaching_lowpass(df_nph["raw_calcium"].values, fs=fs)  # KB
    df_nph['calcium_mad'] = preprocess_sliding_mad(df_nph["raw_calcium"].values, df_nph["times"].values, fs=fs)
    # TODO add alejandro preprocessing

    return df_nph, fs