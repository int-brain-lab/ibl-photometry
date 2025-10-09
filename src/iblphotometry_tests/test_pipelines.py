from iblphotometry.pipelines import (
    run_pipeline,
    sliding_mad_pipeline,
    isosbestic_correction_pipeline,
)
from iblphotometry.synthetic import generate_dataframe
from iblphotometry_tests.base_tests import PhotometryDataTestCase


class TestPipelines(PhotometryDataTestCase):
    def test_single_band_pipeline(self):
        # on synthetic data
        raw_dfs = generate_dataframe()
        brain_regions = raw_dfs['raw_calcium'].columns
        run_pipeline(sliding_mad_pipeline, signal=raw_dfs['raw_calcium'][brain_regions[0]])

        signal_bands = list(raw_dfs.keys())
        brain_regions = raw_dfs['raw_calcium'].columns
        run_pipeline(sliding_mad_pipeline, signal=raw_dfs[signal_bands[0]][brain_regions[0]])

    def test_isosbestic_pipeline(self):
        # on synthetic data
        raw_dfs = generate_dataframe()
        brain_regions = raw_dfs['raw_calcium'].columns

        # run pipeline
        run_pipeline(
            isosbestic_correction_pipeline,
            signal=raw_dfs['raw_calcium'][brain_regions[0]],
            reference=raw_dfs['raw_isosbestic'][brain_regions[0]],
        )
