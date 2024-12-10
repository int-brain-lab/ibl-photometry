import unittest
import pandas as pd
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from iblphotometry.behavior import psth, psth_times
import iblphotometry.plots as plots
# from gui.rawdata_visualizer import BehaviorVisualizerGUI
from iblphotometry.synthetic import synthetic101
import iblphotometry.preprocessing as ffpr
from iblphotometry_tests.base_tests import PhotometryDataTestCase

# Set the seed
np.random.seed(seed=0)
# one = ONE()
DATA_PATH = Path(__file__).parent / 'data'

"""
------------------------------------------------
Functions to get test data
------------------------------------------------
"""


class TestPlotters(PhotometryDataTestCase):
    def get_synthetic_data(self):
        fs = 50
        df_nph, t_events = synthetic101(fs=50)
        # Get signal and process it
        df_nph['signal_processed'] = ffpr.mad_raw_signal(
            df_nph['raw_calcium'].values, fs=fs
        )
        return df_nph, t_events, fs

    def get_test_data(self):
        """
        This is a throw-away loader function to help testing the plotting functions
        """
        self.set_paths('kcenia')
        # --- Use real data for test ---
        event = 'feedback_times'
        # eid = '77a6741c-81cc-475f-9454-a9b997be02a4'  # Good response to feedback times
        # pname = 'Region3G'
        # nph_path = DATA_PATH.joinpath(Path(f'{eid}/{pname}'))
        # one = ONE()

        # Load NP file locally
        # df_nph = pd.read_parquet(nph_path.joinpath('raw_photometry.pqt'))
        df_nph = pd.read_parquet(self.paths['raw_kcenia_pqt'])
        # Load trial from ONE
        # a = one.load_object(eid, 'trials')
        # df_trials = a.to_df()
        df_trials = pd.read_parquet(self.paths['trials_table_kcenia_pqt'])
        # Get event
        t_events = df_trials[event]

        # Ugly way to get sampling frequency
        time_diffs = df_nph['times'].diff().dropna()
        fs = 1 / time_diffs.median()

        # Get signal and process it
        df_nph['signal_processed'] = ffpr.mad_raw_signal(
            df_nph['raw_calcium'].values, fs=fs
        )

        return df_nph, t_events, fs

    """
    ------------------------------------------------
    TEST: Loader objects for plotting
    ------------------------------------------------
    """

    def test_class_plotsignal(self):
        # --- Use real data for test ---
        df_nph, _, fs = self.get_test_data()

        raw_signal = df_nph['raw_calcium'].values
        raw_isosbestic = df_nph['raw_isosbestic'].values
        processed_signal = df_nph['signal_processed'].values
        times = df_nph['times'].values

        plotobj = plots.PlotSignal()
        plotobj.set_data(raw_signal, times, raw_isosbestic, processed_signal)
        plotobj.raw_processed_figure()
        plt.close('all')

    def test_class_plotsignalresponse(self):
        # --- Use real data for test ---
        df_nph, _, fs = self.get_test_data()
        processed_signal = df_nph['signal_processed'].values
        times = df_nph['times'].values
        # Load trial from ONE
        # eid = '77a6741c-81cc-475f-9454-a9b997be02a4'
        # trials = one.load_object(eid, 'trials')
        trials = pd.read_parquet(self.paths['trials_table_kcenia_pqt'])
        plotobj = plots.PlotSignalResponse()
        plotobj.set_data(trials, processed_signal, times)
        _, axs = plotobj.set_fig_layout()
        plotobj.plot_trialsort_psth(axs)
        _, ax = plt.subplots(1, 1)
        plotobj.plot_processed_trialtick(ax)
        # plt.show()
        plt.close('all')

    """
    ------------------------------------------------
    TEST: Plotting functions requiring FF signals only
    ------------------------------------------------
    """

    def test_plot_raw_signals(self):
        for test_case in ['synt', 'real']:
            match test_case:
                case 'synt':
                    # --- Use real data for test ---
                    df_nph, _, fs = self.get_test_data()
                case 'real':
                    # --- Use synthetic data for test ---
                    df_nph, _, fs = self.get_synthetic_data()

            raw_signal = df_nph['raw_calcium'].values
            raw_isosbestic = df_nph['raw_isosbestic'].values
            times = df_nph['times'].values

            plots.plot_raw_signals(raw_signal, times, raw_isosbestic)
            plt.close('all')

    def test_plot_processed_signal(self):
        # --- Use synthetic data for test ---
        df_nph, _, fs = self.get_synthetic_data()

        signal = df_nph['signal_processed'].values
        times = df_nph['times'].values
        plots.plot_processed_signal(signal, times)
        plt.close('all')

    def test_plot_photometry_correlation(self):
        # --- Use real data for test ---
        df_nph, _, fs = self.get_test_data()

        signal_lp = ffpr.low_pass_filter(df_nph['raw_calcium'].values, fs)
        isosbestic_lp = ffpr.low_pass_filter(df_nph['raw_isosbestic'].values, fs)
        times = df_nph['times'].values
        plots.plot_photometry_correlation(signal_lp, isosbestic_lp, times)
        plt.close('all')

    def test_plot_psd(self):
        # --- Use synthetic data for test ---
        df_nph, _, fs = self.get_synthetic_data()

        signal = df_nph['signal_processed'].values
        times = df_nph['times'].values
        fs = 1 / np.median(np.diff(times))
        plots.plot_psd(signal, fs)
        plt.close('all')

    """
    ------------------------------------------------
    TEST: Plotting functions requiring behavioral events
    ------------------------------------------------
    """

    def test_plot_psth(self):
        event_window = [-1.5, 2.75]

        for test_case in ['synt', 'real']:
            match test_case:
                case 'synt':
                    # --- Use real data for test ---
                    df_nph, t_events, fs = self.get_test_data()
                case 'real':
                    # --- Use synthetic data for test ---
                    df_nph, t_events, fs = self.get_synthetic_data()

            signal = df_nph['signal_processed'].values
            times = df_nph['times'].values

            # Compute PSTH
            psth_mat, _ = psth(
                signal, times, t_events, fs=fs, event_window=event_window
            )
            times = psth_times(fs, event_window)
            # Plot PSTH
            plots.plot_psth(psth_mat, times)
            plt.close('all')

    def test_plot_event_tick(self):
        # --- Use synthetic data for test ---
        df_nph, t_events, fs = self.get_synthetic_data()
        plots.plot_event_tick(t_events)
        plt.close('all')

    # def test_gui(self):
    #     df_nph, _, fs = self.get_test_data()
    #     processed_signal = df_nph['signal_processed'].values
    #     times = df_nph['times'].values
    #     trials = pd.read_parquet(self.paths['trials_table_kcenia_pqt'])

    #     from PyQt5.QtWidgets import QApplication
    #     app = QApplication(sys.argv)
    #     window = BehaviorVisualizerGUI()
    #     window.set_data(processed_signal, times)
    #     window.load_trials(trials)
    #     window.show()
    #     # Uncomment to debug
    #     app.exec_()


if __name__ == '__main__':
    unittest.main()
    # suite = unittest.TestSuite()
    # suite.addTest(TestPlotters())
    # runner = unittest.TextTestRunner()
    # runner.run(suite)
