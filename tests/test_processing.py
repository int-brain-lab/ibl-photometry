import unittest

import numpy as np
import pandas as pd

from iblphotometry import fpio, processing, neurophotometrics
from tests.base_tests import PhotometryDataTestCase


class TestProcessing(PhotometryDataTestCase):
    def setUp(self):
        super().setUp()
        path = self.versions_path / 'version_5' / '_neurophotometrics_fpData.raw.pqt'
        self.photometry_df = neurophotometrics.from_neurophotometrics_file_to_photometry_df(path)
        self.signals_dfs = fpio.from_photometry_df(self.photometry_df)

    def test_processing(self):
        # trials = pd.read_parquet(self.paths['trials_table_pqt'])
        raw_df = self.signals_dfs['GCaMP']['G0']

        # bleach corrections
        processing.lowpass_bleachcorrect(raw_df)
        processing.exponential_bleachcorrect(raw_df, 2)

        # outlier removal
        processing.remove_outliers(raw_df)
        processing.remove_spikes(raw_df)

        # other functions
        processing.make_sliding_window(raw_df.values, 100, method='stride_tricks', on_error='expand')
        processing.make_sliding_window(raw_df.values, 100, method='window_generator', on_error='expand')
        processing.sliding_dFF(raw_df, w_len=60, on_error='expand')
        processing.sliding_z(raw_df, w_len=60, on_error='expand')
        processing.sliding_mad(raw_df, w_len=60)

    def test_resample_signal_defaults_to_the_signals_own_rate(self):
        """No fs means the median sample spacing, so the rate is preserved."""
        signal = self.signals_dfs['GCaMP']['G0']
        resampled = processing.resample_signal(signal)

        dt = np.median(np.diff(signal.index))
        np.testing.assert_allclose(np.diff(resampled.index), dt)

    def test_resample_signal_honours_an_explicit_rate(self):
        """fs puts the signal on a 1/fs grid regardless of its own rate."""
        signal = self.signals_dfs['GCaMP']['G0']
        resampled = processing.resample_signal(signal, fs=30)

        np.testing.assert_allclose(np.diff(resampled.index), 1 / 30)
        self.assertGreaterEqual(resampled.index[0], signal.index[0])
        self.assertLessEqual(resampled.index[-1], signal.index[-1])

    def test_resample_signal_pchip_differs_from_linear_on_a_curve(self):
        """PCHIP follows curvature; linear chords cut under a convex signal."""
        times = np.arange(0, 10, 0.1)
        curved = pd.Series(np.sin(times), index=times)

        linear = processing.resample_signal(curved, fs=7, method='linear')
        pchip = processing.resample_signal(curved, fs=7, method='pchip')

        np.testing.assert_allclose(linear.index, pchip.index)
        self.assertFalse(np.allclose(linear.values, pchip.values))
        # Both track the underlying function; PCHIP does it more closely.
        truth = np.sin(pchip.index)
        self.assertLess(np.abs(pchip.values - truth).max(), np.abs(linear.values - truth).max())

    def test_resample_signal_rejects_an_unknown_method(self):
        signal = self.signals_dfs['GCaMP']['G0']
        with self.assertRaises(ValueError):
            processing.resample_signal(signal, method='cubic')


class TestRegressionPredict(unittest.TestCase):
    """Predictions must stay matched to the samples they were requested for."""

    def setUp(self):
        rng = np.random.default_rng(42)
        self.x = rng.permutation(np.arange(16, dtype=float))
        self.m, self.b = 3.0, 1.0
        self.reg = processing.Regression(model=processing.LinearModel())
        self.reg.fit(self.x, self.m * self.x + self.b)

    def test_predict_preserves_sample_order(self):
        for return_type in ('numpy', 'pandas'):
            with self.subTest(return_type=return_type):
                y_hat = self.reg.predict(self.x, return_type=return_type)
                np.testing.assert_allclose(np.asarray(y_hat), self.m * self.x + self.b)

    def test_isosbestic_correct_on_exactly_linear_signal(self):
        # signal is an exact linear function of the reference, so the fit is
        # exact and subtracting it must leave a flat, zero residual
        t = np.arange(600) / 30.0
        rng = np.random.default_rng(0)
        reference = pd.Series(1.0 + np.cumsum(rng.normal(0, 0.05, t.size)), index=t)
        signal = pd.Series(2.0 * reference.values + 0.5, index=t)

        corrected = processing.isosbestic_correct(signal, reference, correction_method='subtract')

        np.testing.assert_allclose(corrected.values, 0.0, atol=1e-6)
        np.testing.assert_array_equal(corrected.index.values, t)
