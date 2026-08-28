import numpy as np

from iblphotometry import fpio, metrics, neurophotometrics, qc

from tests.base_tests import PhotometryDataTestCase


class TestQCSignals(PhotometryDataTestCase):
    def setUp(self):
        super().setUp()
        path = self.versions_path / 'version_5' / '_neurophotometrics_fpData.raw.pqt'
        photometry_df = neurophotometrics.from_neurophotometrics_file_to_photometry_df(path)
        self.signals_dfs = fpio.from_photometry_df(photometry_df)

    def test_runs_over_every_band_and_region_by_default(self):
        """signal_band=None and brain_region=None means all of both."""
        qc_result = qc.qc_signals(self.signals_dfs, [metrics.signal_skew])

        self.assertEqual(set(qc_result['band']), set(self.signals_dfs))
        self.assertEqual(set(qc_result['brain_region']), {'G0', 'G1'})

    def test_restricts_to_the_named_band_and_region(self):
        qc_result = qc.qc_signals(
            self.signals_dfs,
            [metrics.signal_skew],
            signal_band='GCaMP',
            brain_region='G0',
        )

        self.assertEqual(set(qc_result['band']), {'GCaMP'})
        self.assertEqual(set(qc_result['brain_region']), {'G0'})
