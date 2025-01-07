import iblphotometry.io as fio
import iblphotometry.processing as processing
import pandas as pd

from iblphotometry_tests.base_tests import PhotometryDataTestCase


class TestProcessing(PhotometryDataTestCase):
    # think here about the possible use cases

    def test_processing(self):
        self.set_paths('alejandro')
        # get data
        raw_dfs = fio.from_ibl_pqt(
            self.paths['photometry_signal_pqt'],
            self.paths['photometryROI_locations_pqt'],
        )
        # trials = pd.read_parquet(self.paths['trials_table_pqt'])
        raw_df = raw_dfs['GCaMP']['DMS']

        # bleach corrections
        processing.lowpass_bleachcorrect(raw_df)
        processing.exponential_bleachcorrect(raw_df)

        # outlier removal
        processing.remove_outliers(raw_df)
        processing.remove_spikes(raw_df)

        # other functions
        processing.make_sliding_window(raw_df.values, 100, method='stride_tricks')
        processing.make_sliding_window(raw_df.values, 100, method='window_generator')
        processing.sliding_dFF(raw_df, w_len=60)
        processing.sliding_z(raw_df, w_len=60)
        processing.sliding_mad(raw_df, w_len=60)
