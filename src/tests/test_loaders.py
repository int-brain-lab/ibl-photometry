import iblphotometry.io as fio
import numpy as np
import tests.base_tests


class TestLoaders(tests.base_tests.PhotometryDataTestCase):
    # think here about the possible use cases

    # to read from a .csv from disk
    def test_from_array(self):
        n_samples = 1000
        n_channels = 3
        times = np.linspace(0, 100, n_samples)
        data = np.random.randn(n_samples, n_channels)
        names = ['a', 'b', 'c']
        fio.from_array(times, data, names)

    # for neurophotometrics hardware
    def test_from_raw_neurophotometrics_output(self):
        fio.from_raw_neurophotometrics(self.paths['raw_neurophotometrics_csv'])

    # from pqt files as they are returned from ONE by .load_dataset()
    def test_from_pqt(self):
        fio.from_pqt(self.paths['photometry_signal_pqt'])
        fio.from_pqt(
            self.paths['photometry_signal_pqt'],
            self.paths['photometryROI_locations_pqt'],
        )
