from iblphotometry import metrics, fpio
import pandas as pd

from iblphotometry_tests.base_tests import PhotometryDataTestCase


class TestMetrics(PhotometryDataTestCase):
    def setUp(self):
        super().setUp()
        path = self.versions_path / 'version_5' / '_neurophotometrics_fpData.raw.pqt'
        self.photometry_df = fpio.from_neurophotometrics_file_to_photometry_df(path)
        self.signals_dfs = fpio.from_photometry_df(self.photometry_df)

    def test_raw_signal_metrics(self):
        # have an automated way of inferring
        metrics_to_test = [
            # metrics.bleaching_tau,
            metrics.n_spikes,
            metrics.detect_spikes,
            # metrics.n_outliers,
            metrics.n_unique_samples,
            metrics.signal_asymmetry,
            metrics.signal_skew,
            metrics.percentile_dist,
        ]

        for metric_ in metrics_to_test:
            # passing a series
            metric_(self.signals_dfs['GCaMP']['G0'])
            # or passing an array
            metric_(self.signals_dfs['GCaMP']['G0'].values)

        # BEHAV_EVENTS = [
        #     # 'stimOn_times',
        #     # 'goCue_times',
        #     # 'response_times',
        #     'feedback_times',
        #     # 'firstMovement_times',
        #     # 'intervals_0',
        #     # 'intervals_1',
        # ]
        # for event_name in BEHAV_EVENTS:
        #     metrics.ttest_pre_post(raw_tsd, trials, event_name)
        #     metrics.has_responses(raw_tsd, trials, BEHAV_EVENTS)
