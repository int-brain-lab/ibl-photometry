import iblphotometry.io as fpio
import numpy as np
from iblphotometry_tests.base_tests import PhotometryDataTestCase
import pandas as pd


class TestLoaders(PhotometryDataTestCase):
    # think here about the possible use cases

    # to read from a .csv from disk
    def test_from_array(self):
        n_samples = 1000
        n_channels = 3
        times = np.linspace(0, 100, n_samples)
        data = np.random.randn(n_samples, n_channels)
        names = ['a', 'b', 'c']
        fpio.from_array(times, data, names)

    # for neurophotometrics hardware
    def test_from_raw_neurophotometrics_file(self):
        # the single direct version
        raw_dfs_a = fpio.from_raw_neurophotometrics_file(self.paths['raw_neurophotometrics_csv'])
        
        # the chained version
        df = fpio.from_raw_neurophotometrics_file_to_ibl_df(self.paths['raw_neurophotometrics_csv'])
        raw_dfs_b = fpio.from_ibl_dataframe(df)

        # check if they are the same
        assert raw_dfs_a.keys() == raw_dfs_b.keys()
        for key in raw_dfs_a.keys():
            pd.testing.assert_frame_equal(raw_dfs_a[key], raw_dfs_b[key])
        

    # from pqt files as they are returned from ONE by .load_dataset()
    def test_from_ibl_pqt(self):
        fpio.from_ibl_pqt(self.paths['photometry_signal_pqt'])
        raw_dfs_a = fpio.from_ibl_pqt(
            self.paths['photometry_signal_pqt'],
            self.paths['photometryROI_locations_pqt'],
        )
