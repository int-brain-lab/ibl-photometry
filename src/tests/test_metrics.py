from pathlib import Path
import unittest
import iblphotometry.io as fio
import iblphotometry.metrics as metrics
import pandas as pd

data_folder = Path(__file__).parent / 'data'

# TODO make this a list and have a few
# this is currently ony alejandro
session_folder = Path('wittenlab/Subjects/fip_40/2023-05-18/001')
signal_pqt_path_rel = Path('alf/photometry/photometry.signal.pqt')
photometryROI_locations_pqt_path_rel = Path(
    'alf/photometry/photometryROI.locations.pqt'
)
raw_neurophotometrics_file_path_rel = Path(
    'raw_photometry_data/_neurophotometrics_fpData.raw.pqt'
)

signal_pqt_path = data_folder / session_folder / signal_pqt_path_rel
photometryROI_locations_pqt_path = (
    data_folder / session_folder / photometryROI_locations_pqt_path_rel
)
raw_neurophotometrics_file_path = (
    data_folder / session_folder / raw_neurophotometrics_file_path_rel
)

trials_table_file_path = data_folder / session_folder / 'alf/_ibl_trials.table.pqt'


class TestMetrics(unittest.TestCase):
    # think here about the possible use cases

    def test_metrics(self):
        # get data
        raw_tfs = fio.from_pqt(signal_pqt_path, photometryROI_locations_pqt_path)
        trials = pd.read_parquet(trials_table_file_path)

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
