from iblphotometry import fpio, analysis, processing
from tests.base_tests import PhotometryDataTestCase
import pandas as pd
import pynapple as nap


class TestAnalysis(PhotometryDataTestCase):
    def setUp(self):
        super().setUp()
        path = self.versions_path / 'version_5' / '_neurophotometrics_fpData.raw.pqt'
        self.photometry_df = fpio.from_neurophotometrics_file_to_photometry_df(path)
        self.signals_dfs = fpio.from_photometry_df(self.photometry_df)
        self.trials_table = pd.read_parquet(self.versions_path / 'version_5' / '_ibl_trials.table.pqt')

    def test_processing(self):
        raw_df = self.signals_dfs['GCaMP']
        df = processing.resample_signal(raw_df)
        signal = nap.Tsd(df['G0'])
        analysis.psth_nap(signal, self.trials_table)
