import pandas as pd
import numpy as np

from iblutil.util import setup_logger
import ibldsp.utils


logger = setup_logger(__name__)


def read_digital_input_bonsai(csv_file):
    """
    :param csv_file: digital input file from bonsai
    :return:
        numpy array of timestamps
        qc_timestamp: boolean: False means the timestamps were corrected
    """
    qc_timestamp = True
    df_di = pd.read_csv(csv_file)
    if 'Value.Value' in df_di.columns: #for the new ones
        df_di = df_di.rename(columns={"Value.Seconds": "Seconds", "Value.Value": "Value"})
    else:
        df_di["Timestamp"] = df_di["Seconds"]  # for the old ones
    # use Timestamp from this part on, for any of the files
    df_di = df_di.loc[df_di['Value'] == True, :].reset_index()
    # the issue is that when the timestamp columns starts repeating itself, we use the Seconds columns to recover the missing timestamps
    ibad = np.diff(df_di['Timestamp'].values) == 0
    if np.any(ibad):
        ibad = np.r_[ibad, ibad[-1]]
        slope, intercept = np.polyfit(df_di.loc[~ibad, 'Seconds'].values, df_di.loc[~ibad, 'Timestamp'].values, deg=1)
        df_di.loc[ibad, 'Timestamp'] = np.polyval([slope, intercept], df_di['Seconds'].values[ibad])
        qc_timestamp = False
    if df_di.shape[0] == 0:
        qc_timestamp = False
    return df_di['Timestamp'].values, qc_timestamp


def sync_photometry(df_photometry, digital_input, trials=None, region=None, **kwargs):
    """
    Synchronizes photometry data to bpod events and get the relevant column.
    Splits the calcium dependent and isosbestic signal
    :param df_photometry:
    :param file_digital_input:
    :param trials: pd.Dataframe
    :param region:
    :return: dataframe with fields: times, times_isosbestic, isosbestic, calcium
    """
    # first load photometry
    assert region in df_photometry_raw.columns, f"region {region} not found in {df_photometry_raw.columns}"
    # we get the events that correspond to the photometry TTLs
    ntrials = trials.shape[0]
    if (tph.size / ntrials) > 1.5:
        tbpod = np.sort(np.r_[trials['intervals_0'].values, trials['intervals_1'].values -1 , trials.loc[trials['feedbackType'] == 1, 'feedback_times'].values])
    else:
        tbpod = trials['stimOn_times'].values
    tbpod = tbpod[~np.isnan(tbpod)]  # sometimes the stimon time is NaN
    # synchronize photometry to bpod using events TODO: output a qc dictionary
    fcn_nph_to_bpod_times, drift_ppm, iph, ibpod = ibldsp.utils.sync_timestamps(tph, tbpod, return_indices=True, linear=True)
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
        'raw_isosbestic': df_photometry_raw.loc[i_isosbestic, region].values,
        'raw_calcium': df_photometry_raw.loc[i_calcium, region].values,
    })
    # automatically swap isosbestic signal if there is a labeling issue
    if np.mean(df_photometry['raw_calcium']) < np.mean(df_photometry['raw_isosbestic']) :
        df_photometry = df_photometry.rename(
            columns={'times_isosbestic': 'times', 'times': 'times_isosbestic', 'raw_isosbestic': 'raw_calcium', 'raw_calcium': 'raw_isosbestic'})
        logger.warning("isosbestic and calcium signals were swapped")
    # restricts the photometry dataset to the session timings
    tmin, tmax = (trials['intervals_0'][0] - 1, trials['intervals_1'].values[-1] + 1)
    i = np.logical_and(df_photometry.times > tmin, df_photometry.times < tmax)
    return df_photometry.loc[i].reindex()
