import pandas as pd
from pathlib import Path
from one.api import ONE
from iblphotometry.preprocessing import jove2019
from iblphotometry.metrics import ttest_pre_post
import numpy as np

# Set the seed
np.random.seed(seed=0)

DATA_PATH = Path(__file__).parent / 'data'

def load_data_run_ttest(eid, nph_path, one=None, event='feedback_times',
                        pre_w=np.array([-1, -0.2]), post_w=np.array([0.2, 1])):
    # Get data
    if one is None:
        one = ONE()

    # Load NP file locally
    df_nph = pd.read_parquet(nph_path.joinpath(f'raw_photometry.pqt'))

    # Load trial from ONE
    a = one.load_object(eid, 'trials')
    df_trials = a.to_df()

    # Ugly way to get sampling frequency
    time_diffs = df_nph["times"].diff().dropna()
    fs = 1 / time_diffs.median()

    # Process signal
    df_nph['calcium_jove2019'] = jove2019(df_nph["raw_calcium"], df_nph["raw_isosbestic"], fs=fs)

    # Get event
    calcium = df_nph['calcium_jove2019'].values
    times = df_nph['times'].values
    t_events = df_trials[event]

    # T-test if responsive to event
    pass_test = ttest_pre_post(calcium, times, t_events, fs, pre_w=pre_w, post_w=post_w)

    # Check that if we input random time point for event, the T-test fails
    # Take first/last times as anchor, same N times as t_events
    t_random = np.sort(times[0] + np.random.sample(len(t_events)) * (times[-1] - times[0]))
    random_test = ttest_pre_post(calcium, times, t_random, fs, pre_w=pre_w, post_w=post_w)

    return pass_test, random_test


def test_ttest_pre_post():
    one = ONE()

    # Get data
    eid = '77a6741c-81cc-475f-9454-a9b997be02a4'  # Good response to feedback times
    pname = 'Region3G'
    nph_path = DATA_PATH.joinpath(Path(f'{eid}/{pname}'))
    pass_test, random_test = load_data_run_ttest(eid, nph_path, one=one)
    assert pass_test == True
    assert random_test == False

    # Check the bad session with no response does not pass the pass_test
    eid = '3e675f34-9b6f-47b4-9064-e9c9c5e17872' # Bad response, flat
    pname = 'Region1G'
    nph_path = DATA_PATH.joinpath(Path(f'{eid}/{pname}'))
    pass_test, random_test = load_data_run_ttest(eid, nph_path, one=one)
    assert pass_test == False
    assert random_test == False


# Remove, written here to check rapidly
test_ttest_pre_post()
