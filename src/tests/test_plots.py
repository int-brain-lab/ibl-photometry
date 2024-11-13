import pandas as pd
from pathlib import Path
from one.api import ONE
# from iblphotometry.preprocessing import jove2019
import numpy as np

# Set the seed
np.random.seed(seed=0)
one = ONE()
DATA_PATH = Path(__file__).parent / 'data'

def get_test_data(eid, nph_path, event, one=None):
    '''
    This is a throw-away loader function to help testing the plotting functions
    '''
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
    # df_nph['calcium_jove2019'] = jove2019(df_nph["raw_calcium"], df_nph["raw_isosbestic"], fs=fs)

    # Get event
    signal = df_nph['raw_calcium'].values  # TODO replace with processed signal once module is working again
    times = df_nph['times'].values
    t_events = df_trials[event]

    return signal, times, t_events, fs


def test_plot_psth():
    signal, times, t_events, fs = get_test_data(eid)
    psth_mat = psth(signal, times, t_events, fs=fs, peri_event_window=peri_event_window)
