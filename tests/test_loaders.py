from iblphotometry import fpio, neurophotometrics
from tests.base_tests import PhotometryDataTestCase
import pandas as pd


class TestLoaders(PhotometryDataTestCase):
    def test_infer_version(self):
        versions = ['version_1', 'version_2', 'version_5']  # ...
        for version in versions:
            path = self.versions_path / version / '_neurophotometrics_fpData.raw.pqt'
            df = neurophotometrics.read_neurophotometrics_file(path)
            version_inferred = fpio.infer_neurophotometrics_version_from_data(df)
            assert version == version_inferred

        for version in versions:
            path = self.versions_path / version / '_neurophotometrics_fpData.digitalInputs.pqt'
            df = pd.read_parquet(path)  # shortcut to be able to test just infer version, see test below
            version_inferred = fpio.infer_neurophotometrics_version_from_digital_inputs(df)
            assert version == version_inferred

    def test_read_digital_inputs(self):
        versions = ['version_1', 'version_2', 'version_5']  # ...
        for version in versions:
            path = self.versions_path / version / '_neurophotometrics_fpData.digitalInputs.pqt'
            channel = 0 if version == 'version_1' or version == 'version_2' else None
            timestamps_colname = 'Timestamp' if version == 'version_2' else None
            neurophotometrics.read_digital_inputs_file(path, channel=channel, timestamps_colname=timestamps_colname)

    def test_read_neurophotometrics_file(self):
        versions = ['version_1', 'version_2', 'version_5']  # ...
        for version in versions:
            path = self.versions_path / version / '_neurophotometrics_fpData.raw.pqt'
            raw_df = fpio.read_neurophotometrics_file(path)
            neurophotometrics.from_neurophotometrics_df_to_photometry_df(raw_df)
            # the chained version
            neurophotometrics.from_neurophotometrics_file(path)

    # def test_PhotometrySessionLoader(self):
    #     for version in ['version_1', 'version_2', 'version_5']:
    #         path = self.versions_path / version
    #         loader = fpio.PhotometrySessionLoader(path)
    #         assert loader.version == version
    #         assert loader.raw_data_path == path / '_neurophotometrics_fpData.raw.pqt'
    #         assert loader.digital_inputs_path == path / '_neurophotometrics_fpData.digitalInputs.pqt'


class TestFromPhotometryDf(PhotometryDataTestCase):
    """Column selection and renaming in `fpio.from_photometry_df`."""

    @staticmethod
    def _photometry_df():
        """Two ROIs, two bands, four frames. Values identify their ROI."""
        return pd.DataFrame(
            {
                'times': [0.0, 1.0, 2.0, 3.0],
                'valid': [True] * 4,
                'wavelength': [470.0, 415.0, 470.0, 415.0],
                'name': ['GCaMP', 'Isosbestic', 'GCaMP', 'Isosbestic'],
                'color': ['green', 'violet', 'green', 'violet'],
                'Region3G': [1.0, 2.0, 3.0, 4.0],
                'Region5G': [10.0, 20.0, 30.0, 40.0],
            }
        )

    def test_rename_to_distinct_regions(self):
        """One column per ROI, carrying that ROI's values."""
        signal_dfs = fpio.from_photometry_df(
            self._photometry_df(),
            rename={'Region3G': 'VTA', 'Region5G': 'SNc'},
            drop_first=False,
        )
        gcamp = signal_dfs['GCaMP']
        assert list(gcamp.columns) == ['VTA', 'SNc']
        assert list(gcamp['VTA']) == [1.0, 3.0]
        assert list(gcamp['SNc']) == [10.0, 30.0]

    def test_rename_to_repeated_region(self):
        """Two fibers in one region: one column per ROI, not one per match.

        The region name cannot tell the two fibers apart, so both columns carry
        it. Selecting by that name is the caller's problem; what this must not
        do is return a column per (ROI x matching name) pair, which silently
        doubles the data.
        """
        signal_dfs = fpio.from_photometry_df(
            self._photometry_df(),
            rename={'Region3G': 'LC', 'Region5G': 'LC'},
            drop_first=False,
        )
        gcamp = signal_dfs['GCaMP']
        assert list(gcamp.columns) == ['LC', 'LC']
        assert gcamp.to_numpy().tolist() == [[1.0, 10.0], [3.0, 30.0]]
