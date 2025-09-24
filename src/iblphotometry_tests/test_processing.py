from iblphotometry import fpio, processing
from iblphotometry_tests.base_tests import PhotometryDataTestCase


class TestProcessing(PhotometryDataTestCase):
    def setUp(self):
        super().setUp()
        path = self.versions_path / 'version_5' / '_neurophotometrics_fpData.raw.pqt'
        self.photometry_df = fpio.from_neurophotometrics_file_to_photometry_df(path)
        self.signals_dfs = fpio.from_photometry_df(self.photometry_df)

    def test_processing(self):
        # trials = pd.read_parquet(self.paths['trials_table_pqt'])
        raw_df = self.signals_dfs['GCaMP']['G0']

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
