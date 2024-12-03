from pathlib import Path
import unittest


def get_fixtures() -> dict:
    data_folder = Path(__file__).parent / 'data'
    # this is currently ony alejandro
    session_folder = Path('wittenlab/Subjects/fip_40/2023-05-18/001')
    return {
        'photometry_signal_pqt': data_folder
        / session_folder
        / Path('alf/photometry/photometry.signal.pqt'),
        'photometryROI_locations_pqt': data_folder
        / session_folder
        / Path('alf/photometry/photometryROI.locations.pqt'),
        'raw_neurophotometrics_pqt': data_folder
        / session_folder
        / Path('raw_photometry_data/_neurophotometrics_fpData.raw.pqt'),
        'raw_neurophotometrics_csv': data_folder / 'raw_photometry.csv',
        'trials_table_pqt': data_folder / session_folder / 'alf/_ibl_trials.table.pqt',
    }


class PhotometryDataTestCase(unittest.TestCase):
    def setUp(self):
        self.paths = get_fixtures()
