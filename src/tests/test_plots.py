import pandas as pd
from pathlib import Path
from one.api import ONE
import numpy as np
from iblphotometry.behavior import psth
import iblphotometry.plots as plots
from iblphotometry.synthetic import synthetic101
import matplotlib.pyplot as plt

# TODO fix import once processing settled
# from iblphotometry.preprocessing import jove2019
import scipy.signal

# Set the seed
np.random.seed(seed=0)
one = ONE()
DATA_PATH = Path(__file__).parent / 'data'

def _preprocessing(raw_signal, fs):
    # This is a convenience function to get going whilst the preprocessing refactoring is being done
    # TODO delete this function once processing can be applied
    params = {}
    sos = scipy.signal.butter(fs=fs, output='sos', **params.get('butterworth_lowpass', {'N': 3, 'Wn': 0.01, 'btype': 'lowpass'}))
    signal_lp = scipy.signal.sosfiltfilt(sos, raw_signal)
    signal_processed = (raw_signal - signal_lp) / signal_lp
    return signal_processed


def get_synthetic_data():
    fs = 50
    df_nph, t_events = synthetic101(fs=50)
    # Get event
    signal = df_nph['raw_calcium'].values  # TODO replace with processed signal once module is working again
    signal_processed = _preprocessing(signal, fs=fs)
    times = df_nph['times'].values
    return signal_processed, times, t_events, fs

def get_test_data():
    '''
    This is a throw-away loader function to help testing the plotting functions
    '''
    # --- Use real data for test ---
    event = 'feedback_times'
    eid = '77a6741c-81cc-475f-9454-a9b997be02a4'  # Good response to feedback times
    pname = 'Region3G'
    nph_path = DATA_PATH.joinpath(Path(f'{eid}/{pname}'))
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
    signal_processed = _preprocessing(signal, fs=fs)
    times = df_nph['times'].values
    t_events = df_trials[event]

    return signal_processed, times, t_events, fs


def test_plot_psth():
    peri_event_window = [-1.5, 2.75]

    for test_case in ['synt', 'real']:

        match test_case:
            case 'synt':
                # --- Use real data for test ---
                signal, times, t_events, fs = get_test_data()
            case 'real':
                # --- Use synthetic data for test ---
                signal, times, t_events, fs = get_synthetic_data()

        # Compute PSTH
        psth_mat, _ = psth(signal, times, t_events, fs=fs, peri_event_window=peri_event_window)
        # Plot PSTH
        plots.plot_psth(psth_mat, fs)
        plt.show()
        plt.close()
