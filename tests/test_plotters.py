from iblphotometry import fpio, plotters
from tests.base_tests import PhotometryDataTestCase
import pandas as pd
import pynapple as nap


class TestPlotters(PhotometryDataTestCase):
    def setUp(self):
        super().setUp()
        path = self.versions_path / 'version_5' / '_neurophotometrics_fpData.raw.pqt'
        self.photometry_df = fpio.from_neurophotometrics_file_to_photometry_df(path)
        self.signals_dfs = fpio.from_photometry_df(self.photometry_df)
        self.trials_table = pd.read_parquet(self.versions_path / 'version_5' / '_ibl_trials.table.pqt')

    def test_trace_plotters(self):
        plotters.plot_photometry_traces(self.signals_dfs['GCaMP'])
        plotters.plot_photometry_bands(self.signals_dfs)

    def test_plot_psths(self):
        plotters.plot_psths_from_trace(self.signals_dfs['GCaMP']['G0'], self.trials_table)
