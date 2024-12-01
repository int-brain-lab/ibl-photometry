from pathlib import Path
import unittest
import iblphotometry.io as fio
import numpy as np

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
        neurophotometrics_csv_path = data_folder / 'raw_photometry.csv'
        fio.from_raw_neurophotometrics(neurophotometrics_csv_path)

    # from pqt files as they are returned from ONE by .load_dataset()
    def test_from_pqt(self):
        fio.from_pqt(signal_pqt_path, photometryROI_locations_pqt_path)

    # read from file (can be either csv or pqt)
    # get extention
