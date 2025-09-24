import iblphotometry.fpio as fpio
from iblphotometry_tests.base_tests import PhotometryDataTestCase
import pandas as pd
from pathlib import Path


class TestLoaders(PhotometryDataTestCase):
    def test_infer_version(self):
        versions = ['version_1', 'version_2', 'version_5']  # ...
        for version in versions:
            path = self.versions_path / version / '_neurophotometrics_fpData.raw.pqt'
            df = fpio.read_neurophotometrics_file(path)
            version_inferred = fpio.infer_neurophotometrics_version_from_data(df)
            assert version == version_inferred

        for version in versions:
            path = (
                self.versions_path
                / version
                / '_neurophotometrics_fpData.digitalInputs.pqt'
            )
            df = pd.read_parquet(
                path
            )  # shortcut to be able to test just infer version, see test below
            version_inferred = fpio.infer_neurophotometrics_version_from_digital_inputs(
                df
            )
            assert version == version_inferred

    def test_read_digital_inputs(self):
        versions = ['version_1', 'version_2', 'version_5']  # ...
        for version in versions:
            path = (
                self.versions_path
                / version
                / '_neurophotometrics_fpData.digitalInputs.pqt'
            )
            channel = 0 if version == 'version_1' or version == 'version_2' else None
            timestamps_colname = 'Timestamp' if version == 'version_2' else None
            df = fpio.read_digital_inputs_file(
                path, channel=channel, timestamps_colname=timestamps_colname
            )

    def test_read_neurophotometrics_file(self):
        versions = ['version_1', 'version_2', 'version_5']  # ...
        for version in versions:
            path = self.versions_path / version / '_neurophotometrics_fpData.raw.pqt'
            raw_df = fpio.read_neurophotometrics_file(path)
            fpio.from_neurophotometrics_df_to_photometry_df(raw_df)
            # the chained version
            fpio.from_neurophotometrics_file(path)
