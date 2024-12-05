from pathlib import Path
import unittest


# # this is currently ony alejandro
# return {
#     'photometry_signal_pqt': data_folder
#     / session_folder
#     / Path('alf/photometry/photometry.signal.pqt'),
#     'photometryROI_locations_pqt': data_folder
#     / session_folder
#     / Path('alf/photometry/photometryROI.locations.pqt'),
#     'raw_neurophotometrics_pqt': data_folder
#     / session_folder
#     / Path('raw_photometry_data/_neurophotometrics_fpData.raw.pqt'),
#     'raw_neurophotometrics_csv': data_folder / 'raw_photometry.csv',
#     'raw_kcenia_pqt': data_folder / 'raw_photometry.pqt',
#     'trials_table_kcenia_pqt': data_folder / '_ibl_trials.table.pqt',
#     'trials_table_pqt': data_folder / session_folder / 'alf/_ibl_trials.table.pqt',
# }

# def get_fixtures_2() -> dict:
#     data_folder = Path(__file__).parent / 'data'
#     # /home/georg/code/ibl-photometry/src/iblphotometry_tests/data/src/iblphotometry_tests/data/cortexlab/Subjects/CQ002/2024-11-05/001/raw_photometry_data
#     return paths


class PhotometryDataTestCase(unittest.TestCase):
    def setUp(self):
        self.paths = self.set_paths()

    def set_paths(self, dataset: str = 'carolina') -> dict:
        data_folder = Path(__file__).parent / 'data'
        match dataset:
            case 'kcenia':
                paths = dict(
                    raw_kcenia_pqt=data_folder / 'raw_photometry.pqt',
                    trials_table_kcenia_pqt=data_folder / '_ibl_trials.table.pqt',
                )
            case 'alejandro':
                session_folder = Path('wittenlab/Subjects/fip_40/2023-05-18/001')
                paths = dict(
                    photometry_signal_pqt=data_folder
                    / session_folder
                    / 'alf/photometry/photometry.signal.pqt',
                    photometryROI_locations_pqt=data_folder
                    / session_folder
                    / 'alf/photometry/photometryROI.locations.pqt',
                    raw_neurophotometrics_pqt=data_folder
                    / session_folder
                    / 'raw_photometry_data/_neurophotometrics_fpData.raw.pqt',
                    trials_table_pqt=data_folder
                    / session_folder
                    / 'alf/_ibl_trials.table.pqt',
                )
            case 'carolina':
                session_folder = Path('cortexlab/Subjects/CQ002/2024-11-05/001/')
                paths = dict(
                    photometry_signal_pqt=data_folder
                    / session_folder
                    / 'alf/photometry/photometry.signal.pqt',
                    photometryROI_locations_pqt=data_folder
                    / session_folder
                    / 'alf/photometry/photometryROI.locations.pqt',
                    raw_neurophotometrics_pqt=data_folder
                    / session_folder
                    / 'raw_photometry_data/_neurophotometrics_fpData.raw.pqt',
                    trials_table_pqt=data_folder
                    / session_folder
                    / 'alf/task_00/_ibl_trials.table.pqt',
                )
        self.paths = paths
