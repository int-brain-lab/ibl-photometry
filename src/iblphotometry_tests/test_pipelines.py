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
        run_pipeline(sliding_mad_pipeline, raw_dfs['raw_calcium'])

        signal_bands = list(raw_dfs.keys())
        run_pipeline(sliding_mad_pipeline, raw_dfs[signal_bands[0]])

    def test_isosbestic_pipeline(self):
        # on synthetic data
        raw_dfs = generate_dataframe()

        # run pipeline
        run_pipeline(
            isosbestic_correction_pipeline,
            raw_dfs['raw_calcium'],
            raw_dfs['raw_isosbestic'],
        )
