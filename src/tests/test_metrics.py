# import pandas as pd
# from pathlib import Path
# from one.api import ONE
# from iblphotometry.preprocessing import jove2019
# from iblphotometry.metrics import ttest_pre_post


# def test_ttest_pre_post():
#     # Get data
#     one = ONE()
#     eid = '77a6741c-81cc-475f-9454-a9b997be02a4'

#     # Load NP file locally - TODO: this is local directory to Github
#     # nph_path = Path(f'../src/tests/data/{eid}')
#     nph_path = Path(f'/Users/gaellechapuis/Desktop/FiberPhotometry/{eid}')
#     df_nph = pd.read_parquet(nph_path.joinpath(f'raw_photometry.pqt'))

#     # Load trial from ONE
#     a = one.load_object(eid, 'trials')
#     df_trials = a.to_df()

#     # Ugly way to get sampling frequency
#     time_diffs = df_nph["times"].diff().dropna()
#     fs = 1 / time_diffs.median()

#     # Process signal
#     df_nph['calcium_jove2019'] = jove2019(df_nph["raw_calcium"], df_nph["raw_isosbestic"], fs=fs)

#     # Get event
#     event = 'feedback_times'
#     calcium = df_nph['calcium_jove2019'].values
#     times = df_nph['times'].values
#     t_events = df_trials[event]

#     pre_w = [-1, -0.2]  # seconds
#     post_w = [0.2, 1]  # seconds

#     pass_test = ttest_pre_post(calcium, times, t_events, fs, pre_w=pre_w, post_w=post_w)
#     assert pass_test == True
