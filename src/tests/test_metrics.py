import iblphotometry.io as fio
import iblphotometry.metrics as metrics
import pandas as pd

import tests.base_tests


class TestMetrics(tests.base_tests.PhotometryDataTestCase):
    # think here about the possible use cases

    def test_metrics(self):
        # get data
        raw_tfs = fio.from_pqt(
            self.paths['photometry_signal_pqt'],
            self.paths['photometryROI_locations_pqt'],
        )
        trials = pd.read_parquet(self.paths['trials_table_pqt'])

        # testing metrics with nap.Tsd
        raw_tsd = raw_tfs['GCaMP']['DMS']

        metrics.bleaching_tau(raw_tsd)
        metrics.n_spikes(raw_tsd)
        metrics.detect_spikes(raw_tsd)
        metrics.n_outliers(raw_tsd)
        metrics.n_unique_samples(raw_tsd)
        metrics.signal_asymmetry(raw_tsd)
        metrics.signal_skew(raw_tsd)
        metrics.percentile_dist(raw_tsd)

        BEHAV_EVENTS = [
            # 'stimOn_times',
            # 'goCue_times',
            # 'response_times',
            'feedback_times',
            # 'firstMovement_times',
            # 'intervals_0',
            # 'intervals_1',
        ]
        for event_name in BEHAV_EVENTS:
            metrics.ttest_pre_post(raw_tsd, trials, event_name)
            metrics.has_responses(raw_tsd, trials, BEHAV_EVENTS)

        # testing metrics with np.array
        raw_array = raw_tfs['GCaMP']['DMS'].d
        # metrics.bleaching_tau(raw_tsd)
        metrics.n_spikes(raw_array)
        metrics.detect_spikes(raw_array)
        metrics.n_outliers(raw_array)
        metrics.n_unique_samples(raw_array)
        metrics.signal_asymmetry(raw_array)
        metrics.signal_skew(raw_array)
        metrics.percentile_dist(raw_array)
