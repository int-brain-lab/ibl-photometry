import unittest
import iblphotometry.io as fio
import numpy as np
import data_paths


class TestLoaders(unittest.TestCase):
    # think here about the possible use cases

    # to read from a .csv from disk
    def test_from_array(self):
        n_samples = 1000
        n_channels = 3
        times = np.linspace(0, 100, n_samples)
        data = np.random.randn(n_samples, n_channels)
        names = ['a', 'b', 'c']
        fio.from_array(times, data, names)

    def test_from_csv(self):
        # TODO get files
        ...

    # for doric based hardware
    def test_from_doric(self):
        # TODO get files
        ...

    # for neurophotometrics hardware
    def test_from_raw_neurophotometrics_output(self):
        fio.from_raw_neurophotometrics(data_paths.raw_neurophotometrics_csv)

    # from pqt files as they are returned from ONE by .load_dataset()
    def test_from_pqt(self):
        fio.from_pqt(data_paths.signal_pqt)
        fio.from_pqt(data_paths.signal_pqt, data_paths.photometryROI_locations_pqt)

    # read from file (can be either csv or pqt)
    # get extention
