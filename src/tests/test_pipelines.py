import numpy as np
from pathlib import Path
import unittest
import iblphotometry.io as fio
from iblphotometry.pipelines import run_pipeline, sliding_mad_pipeline


data_folder = Path(__file__).parent / 'data'

# TODO make this a list and have a few
session_folder = Path('wittenlab/Subjects/fip_40/2023-05-18/001')

signal_pqt_path_rel = Path('alf/photometry/photometry.signal.pqt')

photometryROI_locations_pqt_path_rel = Path(
    'alf/photometry/photometryROI.locations.pqt'
)

signal_pqt_path = data_folder / session_folder / signal_pqt_path_rel

photometryROI_locations_pqt_path = (
    data_folder / session_folder / photometryROI_locations_pqt_path_rel
)


class TestPipelines(unittest.TestCase):
    def test_single_band_pipeline(self):
        raw_tfs = fio.from_pqt(signal_pqt_path, photometryROI_locations_pqt_path)
        signal_bands = list(raw_tfs.keys())
        run_pipeline(sliding_mad_pipeline, raw_tfs[signal_bands[0]])

    def test_isosbestic_pipeline(self): ...
