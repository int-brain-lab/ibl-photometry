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

    def test_metrics_kwargs_reach_the_sliding_windows(self):
        """percentile_asymmetry defaults to pc_comp=95, so asking for 75 has to
        move the per-window values, not only the whole-signal one.

        Flat windows make the denominator zero, so the comparison is restricted
        to the windows where both runs are finite; np.allclose would pass on the
        NaNs alone and hide the very thing under test.
        """

        def run(metrics_kwargs):
            return qc.qc_signals(
                self.signals_dfs,
                [metrics.percentile_asymmetry],
                metrics_kwargs=metrics_kwargs,
                signal_band='GCaMP',
                brain_region='G0',
                sliding_kwargs={'w_len': 120, 'step_len': 60},
            )

        default = run({})
        with_kwargs = run({'percentile_asymmetry': {'pc_comp': 75}})
        windowed = default['window'].notna()

        # Control: the whole-signal row has always honoured metrics_kwargs, so
        # this fixture is sensitive to pc_comp.
        self.assertNotAlmostEqual(
            default.loc[~windowed, 'value'].iloc[0],
            with_kwargs.loc[~windowed, 'value'].iloc[0],
        )

        a = default.loc[windowed, 'value'].values
        b = with_kwargs.loc[windowed, 'value'].values
        finite = np.isfinite(a) & np.isfinite(b)
        self.assertTrue(finite.any(), 'no finite sliding windows were produced')
        self.assertFalse(np.array_equal(a[finite], b[finite]))
