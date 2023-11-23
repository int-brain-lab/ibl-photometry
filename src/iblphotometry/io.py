import pandas as pd
import numpy as np

import neurodsp.utils
from brainbox.io.one import SessionLoader


def read_digital_input_bonsai(csv_file):
    """
    input = raw DI data (TTL)
    output = "Timestamp"; DI TTL times, only True values, this means only ups

     > columns:
         * old ones = "Seconds"
         * new ones = "Timestamp"
    """
    df_DI0 = pd.read_csv(csv_file)
    if 'Value.Value' in df_DI0.columns: #for the new ones
        df_DI0 = df_DI0.rename(columns={"Value.Seconds": "Seconds", "Value.Value": "Value"})
    else:
        df_DI0["Timestamp"] = df_DI0["Seconds"]  # for the old ones
    # use Timestamp from this part on, for any of the files
    raw_phdata_DI0_true = df_DI0[df_DI0.Value == True]
    df_raw_phdata_DI0_T_timestamp = pd.DataFrame(raw_phdata_DI0_true, columns=["Timestamp"])
    df_raw_phdata_DI0_T_timestamp = df_raw_phdata_DI0_T_timestamp.reset_index(drop=True)
    return df_raw_phdata_DI0_T_timestamp.squeeze()


def sync_photometry(file_photometry, file_digital_input, trials=None, region=None, **kwargs):
    """
    Synchronizes photometry data to bpod events and get the relevant column.
    Splits the calcium dependent and isosbestic signal
    :param file_photometry:
    :param file_digital_input:
    :param trials: pd.Dataframe
    :param region:
    :return: dataframe with fields: times, times_isosbestic, isosbestic, calcium
    """
    # first load photometry
    df_photometry_raw = pd.read_csv(file_photometry)
    tph = read_digital_input_bonsai(file_digital_input).values
    assert region in df_photometry_raw.columns, f"region {region} not found in {df_photometry_raw.columns}"
    # we get the events that correspond to the photometry TTLs
    ntrials = trials.shape[0]
    if (tph.size / ntrials) > 1.5:
        tbpod = np.sort(np.r_[trials['intervals_0'].values, trials['intervals_1'].values -1 , trials.loc[trials['feedbackType'] == 1, 'feedback_times'].values])
    else:
        tbpod = trials['stimOn_times'].values
    tbpod = tbpod[~np.isnan(tbpod)]  # sometimes the stimon time is NaN
    # synchronize photometry to bpod using events TODO: output a qc dictionary
    fcn_nph_to_bpod_times, drift_ppm, iph, ibpod = neurodsp.utils.sync_timestamps(tph, tbpod, return_indices=True)
    print(f"{ntrials} trials, {tph.size} photometry TTLs, {tbpod.size} bpod TTLs, {iph.size} matched TTLs")
    print('max deviation:', np.max(np.abs(fcn_nph_to_bpod_times(tph[iph]) - tbpod[ibpod]) * 1e6), 'drift: ', drift_ppm, 'ppm')

    # import matplotlib.pyplot as plt
    # fig, ax = plt.subplots(1, 1)
    # ax.plot(np.diff(tbpod), label='diff bpod')
    # ax.plot(np.diff(tph), label='diff photometry')
    # ax.legend()

    # loads the photometry data
    i_isosbestic = np.where(df_photometry_raw['LedState'] == 2)[0]
    i_calcium = np.where(df_photometry_raw['LedState'] == 1)[0]
    assert np.abs(i_isosbestic.size - i_calcium.size) <= 1, "number of isosbestic and calcium samples differ by more than 1"
    i_isosbestic = i_isosbestic[:np.minimum(i_calcium.size, i_isosbestic.size)]
    i_calcium = i_calcium[:np.minimum(i_calcium.size, i_isosbestic.size)]
    df_photometry = pd.DataFrame({
        'times': fcn_nph_to_bpod_times(df_photometry_raw.loc[i_calcium, 'Timestamp']),
        'times_isosbestic': fcn_nph_to_bpod_times(df_photometry_raw.loc[i_isosbestic, 'Timestamp']),
        'isosbestic': df_photometry_raw.loc[i_isosbestic, region].values,
        'calcium': df_photometry_raw.loc[i_calcium, region].values,
    })
    # restricts the photometry dataset to the session timings
    tmin, tmax = (trials['intervals_0'][0] - 1, trials['intervals_1'].values[-1] + 1)
    i = np.logical_and(df_photometry.times > tmin, df_photometry.times < tmax)
    return df_photometry.loc[i].reindex()
