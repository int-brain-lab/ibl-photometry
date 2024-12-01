import unittest
import iblphotometry.io as fio
from iblphotometry.synthetic import synthetic101
from iblphotometry.pipelines import (
    run_pipeline,
    sliding_mad_pipeline,
    isosbestic_correction_pipeline,
)
import data_paths
import numpy as np
import pynapple as nap
from iblphotometry.synthetic import generate_TsdFrame


class TestPipelines(unittest.TestCase):
    def test_single_band_pipeline(self):
        # on synthetic data
        raw_tfs = generate_TsdFrame()
        run_pipeline(sliding_mad_pipeline, raw_tfs['signal'])

        # on real data
        raw_tfs = fio.from_pqt(
            data_paths.signal_pqt, data_paths.photometryROI_locations_pqt
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
