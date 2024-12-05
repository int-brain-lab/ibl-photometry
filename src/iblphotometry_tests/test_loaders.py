import iblphotometry.io as fpio
from iblphotometry_tests.base_tests import PhotometryDataTestCase
import pandas as pd


class TestLoaders(PhotometryDataTestCase):
    # think here about the possible use cases

    # to read from a .csv from disk
    # def test_from_array(self):
    #     n_samples = 1000
    #     n_channels = 3
    #     times = np.linspace(0, 100, n_samples)
    #     data = np.random.randn(n_samples, n_channels)
    #     names = ['a', 'b', 'c']
    #     fpio.from_array(times, data, names)

    # for neurophotometrics hardware
    def test_from_raw_neurophotometrics_file(self):
        self.set_paths('carolina')
        # 1) validation reading a raw photometrics file
        # unfortunately I don't have the corresponding pqt files. TODO change this
        # fpio.from_raw_neurophotometrics_file_to_raw_df(self.paths['raw_neurophotometrics_csv'])

        # 2) read a pqt file, compare
        raw_df = fpio.from_raw_neurophotometrics_file_to_raw_df(
            self.paths['raw_neurophotometrics_pqt']
        )
        ibl_df_a = fpio.from_raw_neurophotometrics_file_to_ibl_df(
            self.paths['raw_neurophotometrics_pqt']
        )

        ibl_df_b = fpio.from_raw_neurophotometrics_df_to_ibl_df(raw_df)
        pd.testing.assert_frame_equal(ibl_df_a, ibl_df_b)

        # 2) converting from ibl format to final
        dfs_a = fpio.from_ibl_dataframe(ibl_df_a)
        dfs_b = fpio.from_raw_neurophotometrics_file(
            self.paths['raw_neurophotometrics_pqt']
        )

        # check if they are the same
        assert dfs_a.keys() == dfs_b.keys()
        for key in dfs_a.keys():
            pd.testing.assert_frame_equal(dfs_a[key], dfs_b[key])

    # from pqt files as they are returned from ONE by .load_dataset()
    # def test_from_ibl_pqt(self):
    #     fpio.from_ibl_pqt(self.paths['photometry_signal_pqt'])
    #     fpio.from_ibl_pqt(
    #         self.paths['photometry_signal_pqt'],
    #         self.paths['photometryROI_locations_pqt'],
    #     )
