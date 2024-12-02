from pathlib import Path
import iblphotometry.io as fio
from iblphotometry.pipelines import (
    run_pipeline,
    sliding_mad_pipeline,
    isosbestic_correction_pipeline,
)
from iblphotometry.synthetic import generate_TsdFrame
import tests.base_tests


class TestPipelines(tests.base_tests.PhotometryDataTestCase):
    def test_single_band_pipeline(self):
        # on synthetic data
        raw_tfs = generate_TsdFrame()
        run_pipeline(sliding_mad_pipeline, raw_tfs['signal'])

        Path(__file__).parent.joinpath()
        # on real data
        raw_tfs = fio.from_pqt(
            self.paths['photometry_signal_pqt'],
            self.paths['photometryROI_locations_pqt'],
        )
        signal_bands = list(raw_tfs.keys())
        run_pipeline(sliding_mad_pipeline, raw_tfs[signal_bands[0]])

    def test_isosbestic_pipeline(self):
        # on synthetic data
        raw_tfs = generate_TsdFrame()

        # run pipeline
        run_pipeline(
            isosbestic_correction_pipeline, raw_tfs['signal'], raw_tfs['reference']
        )
