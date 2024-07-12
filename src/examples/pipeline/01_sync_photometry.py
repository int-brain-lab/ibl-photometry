from pathlib import Path
import tqdm
import pandas as pd
import numpy as np
import shutil
import ibldsp.utils
import iblphotometry.io
from brainbox.io.one import SessionLoader
from one.api import ONE

PATH_WORKDIR = Path("/mnt/h0/kb/data/staged_data/")
df_recordings = pd.read_parquet(PATH_WORKDIR.joinpath('00_recordings.pqt'))

one = ONE(base_url='https://alyx.internationalbrainlab.org', cache_dir='/mnt/h0/kb/data/one')

new_fields = [
    'eid',
    'region',
    'n_trials',
    'n_trials_correct',
    'n_photometry_timestamps',
    'qc_digital_input_file',   # If True, the digital input file is not corrupt
    'qc_n_matching_timestamps',
    'qc_std_sync',
    'sync_complete',
]
df_sync = pd.DataFrame(columns=new_fields)
df_recordings = df_recordings.merge(df_sync, how="left", left_on=['eid', 'region'], right_on=['eid', 'region'])

# %%

IMIN = 0
for i, rec in tqdm.tqdm(df_recordings.iterrows(), total=df_recordings.shape[0]):
    if i < IMIN:
        continue
    # loads the photometry timestamps
    sl = SessionLoader(eid=rec.eid, one=one)
    file_photometry_pqt = sl.session_path.joinpath('alf', rec.region, 'raw_photometry.pqt')

    df_photometry_raw = pd.read_csv(rec['file_raw_photometry'])
    tph, qc_digital_input = iblphotometry.io.read_digital_input_bonsai(rec['file_digital_input'])
    df_recordings.at[i, 'qc_digital_input_file'] = qc_digital_input
    # # there are cases where the digital input is corrupt: revert to the dataframe tagged frames
    if qc_digital_input is False:
        iup = ibldsp.utils.rises(df_photometry_raw[f'Input{rec.nph_bnc}'].values)
        # we need to make sure that the digital input is not corrupt in the photometry file, in which case we keep interpoalted tph from above
        if len(iup) == 0:
            print(i, rec.mouse, rec.date, rec.eid, f"No digital input: n events {tph.size}", rec.eid)
        if len(tph) == 0 or len(iup) / len(tph) >= 0.75:
            tph = (df_photometry_raw['Timestamp'].values[iup] + df_photometry_raw['Timestamp'].values[iup - 1]) / 2
    # loads the trials
    sl.load_trials()
    rig_settings = sl.load_rig_settings()  # this is needed to get the ITI delay secs
    df_recordings.at[i, 'n_photometry_timestamps'] = tph.size
    df_recordings.at[i, 'n_trials'] = nt = sl.trials.shape[0]
    df_recordings.at[i, 'n_trials_correct'] = nt_correct = np.sum(sl.trials['feedbackType'] == 1)

    # get the behaviour events
    ireward = (sl.trials['feedbackType'] == 1).values
    sync_signals = np.argmin(np.abs(1 - tph.size / np.array([nt, nt + nt_correct, nt * 2 + nt_correct])))
    iti_delay_secs = rig_settings.get('ITI_DELAY_SECS', 0.5)
    match sync_signals:
        case 0:  # this is a single pulse per trial: stimOnTrigger time
            tbpod = sl.trials['stimOnTrigger_times'].values
        case 1:  # reward + intervals_1 - ITI (ITI should always be 1)
            tbpod = np.sort(np.r_[sl.trials['intervals_1'].values - iti_delay_secs, sl.trials['feedback_times'].values[ireward]])
        case 2:  # reward + intervals_0 + intervals_1 - ITI (ITI should always be 1)
            tbpod = np.sort(np.r_[
                                sl.trials['intervals_0'].values,
                                sl.trials['intervals_1'].values - iti_delay_secs,
                                sl.trials['feedback_times'].values[ireward]]
                            )

    # sync the behaviour events to the photometry timestamps
    fcn_nph_to_bpod_times, drift_ppm, iph, ibpod = ibldsp.utils.sync_timestamps(tph, tbpod, return_indices=True, linear=True)
    # then we check the alignment, should be less than the screen refresh rate
    tcheck = fcn_nph_to_bpod_times(tph[iph]) - tbpod[ibpod]
    df_recordings.at[i, 'qc_std_sync'] = np.std(tcheck)
    df_recordings.at[i, 'qc_n_matching_timestamps'] = tcheck.size

    if np.any(np.abs(tcheck) > 1 / 60):
        print(i, rec.mouse, rec.date, rec.eid, f'sync issue n trials {nt}, n bpod sync {len(tbpod)}, n photometry {len(tph)}, n match {len(iph)}, sync events: {sync_signals}')

        # import matplotlib.pyplot as plt
        # plt.hist(tcheck * 1e3)
        # todo put assertion here

    led_state = df_photometry_raw['LedState'].values
    assert np.all(np.abs(np.diff(led_state[1:])) == 1)
    df_photometry = {
        'times': fcn_nph_to_bpod_times(df_photometry_raw.loc[i_cal := np.isin(led_state, [2, 18]), 'Timestamp'].values),
        'raw_isosbestic': df_photometry_raw.loc[i_iso := np.isin(led_state, [1, 17]), rec.region].values,
        'raw_calcium': df_photometry_raw.loc[i_cal, rec.region].values,
    }
    ns = np.minimum(np.sum(i_cal), np.sum(i_iso))
    df_photometry = pd.DataFrame({k: v[:ns] for k, v in df_photometry.items()})
    file_photometry_pqt.parent.mkdir(exist_ok=True, parents=True)
    sl.session_path.joinpath('raw_photometry_data').mkdir(exist_ok=True, parents=True)
    df_photometry.to_parquet(file_photometry_pqt)
    shutil.copy(rec['file_raw_photometry'], sl.session_path.joinpath('raw_photometry_data', 'raw_photometry.csv'))
    shutil.copy(rec['file_digital_input'], sl.session_path.joinpath('raw_photometry_data', 'raw_digital_input.csv'))
    sl.session_path.joinpath('alf', rec.region, 'raw_photometry.pqt')
    df_recordings.loc[i, 'sync_complete'] = True


df_recordings.to_parquet(PATH_WORKDIR.joinpath('01_recordings_sync.pqt'))
#
# import matplotlib.pyplot as plt
# # plt.plot(interp0(tph), tph * 0 + 1, '*')
# # plt.plot(tbpod, tbpod * 0 + 1, '.')
# fig, ax = plt.subplots(figsize=(16, 8))
# ax.plot(sl.trials['feedback_times'].values[ireward] - sl.trials['intervals_0'][ireward], np.arange(sl.trials.shape[0])[ireward], 'g|', label='reward')
# ax.plot(sl.trials['intervals_1'].values - 1 - sl.trials['intervals_0'], np.arange(sl.trials.shape[0]), 'b|', label='trial offset')
#
#
# # remove the trial onset time from tph
# it = np.minimum(np.searchsorted(sl.trials['intervals_0'], fcn_nph_to_bpod_times(tph)) - 1, sl.trials.shape[0] - 1)
# ax.plot(fcn_nph_to_bpod_times(tph[iph]) - sl.trials['intervals_0'].values[it[iph]], it[iph], 'm|', label='photometry')
#
# first = np.where(np.diff(it, prepend=min(it) - 1) == 1)[0]
# last = np.r_[first[1:]  - 1, len(it) - 1]
#
# # ax.plot(interp0(tph) - sl.trials['intervals_0'].values[it], it, 'm|', label='photometry')
# ax.set(xlim=[0, 60], xlabel='Time (s)', ylabel='Trial number')
# ax.legend()
