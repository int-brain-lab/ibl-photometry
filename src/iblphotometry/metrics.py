from itertools import groupby
import numpy as np
import pandas as pd
from scipy import stats, signal
from iblphotometry.preprocessing import find_early_samples
from iblphotometry.processing import z, Regression, ExponDecay, detect_outliers, sobel


def n_early_samples(A: pd.DataFrame | pd.Series, dt_tol: float = 0.001) -> int:
    """
    Number of samples that occur before the expected (median) dt.

    Args:
        A (pd.Series): the input data, needs to be series with sample timing in
        index

        dt_tol (float, optional): acceptable deviance from the median dt

    Returns:
        int: the number of early samples
    """
    return find_early_samples(A, dt_tol=dt_tol).sum()


def n_unique_samples(A: pd.Series | np.ndarray) -> int:
    """number of unique samples in the signal. Low values indicate that the
    signal was not within the range of the digitizer during acquisition.

    Args:

    """
    a = A.values if isinstance(A, pd.Series) else A
    return np.unique(a).shape[0]


def median_absolute_deviance(A: pd.Series | np.ndarray) -> float:
    """median absolute distance from the signal median. Low values indicate a
    low overall signal amplitude.

    Args:
        A (pd.Series | np.ndarray): the input data

    Returns:
        float: median absolute deviance

    """
    a = A.values if isinstance(A, pd.Series) else A
    return np.median(np.abs(a - np.median(a)))


def percentile_distance(A: pd.Series | np.ndarray, pc: tuple = (50, 95), axis=-1) -> float:
    """the distance between two percentiles in units of z.

    Args:
        A (pd.Series | np.ndarray): the input data, np.ndarray for stride tricks sliding windows
        pc (tuple, optional): percentiles to be computed. Defaults to (50, 95).
        axis (int, optional): only for arrays, the axis to be computed. Defaults to -1.

    Returns:
        float: the value of the metric

    Notes:
        - if pc is set to (50, 95), the metric capture the magnitude of positive
        transients in the signal
    """
    A = A.values if isinstance(A, pd.Series) else A
    P = np.percentile(z(A), pc, axis=axis)
    return P[1] - P[0]


def percentile_asymmetry(A: pd.Series | np.ndarray, pc_comp: int = 95, axis=-1) -> float:
    """the ratio between the distance of two percentiles to the median. High
    values indicate large positive deflections in the signal.

    FIXME: Proportional to the the signal to noise.

    Args:
        A (pd.Series | np.ndarray): the input data
        pc_comp (int, optional): the percentiles to compare to the median
        (pc_comp, 100 - pc_comp). Defaults to 95.
        axis (int, optional): the axis over which to take the percentiles.
        Defaults to -1.

    Returns:
        float: the ratio of positive and negative percentile distances
    """
    # TODO embrace pydantic
    if not (isinstance(A, pd.Series) or isinstance(A, np.ndarray)):
        raise TypeError('A must be pd.Series or np.ndarray.')

    a = np.absolute(percentile_distance(A, (50, pc_comp), axis=axis))
    b = np.absolute(percentile_distance(A, (100 - pc_comp, 50), axis=axis))
    return a / b


def signal_skew(A: pd.Series | np.ndarray, axis=-1) -> float:
    A = A.values if isinstance(A, pd.Series) else A
    P = stats.skew(A, axis=axis)
    return P


def n_edges(A: pd.Series | np.ndarray, sd: float = 5, k: int = 2, uniform=True):
    """counts the number of abrupt jumps in a signal.

    Args:
        A (pd.Series | np.ndarray): the input data
        sd (float, optional): the number of standard deviations to use as a
        threshold, defaults to 5
        k (int, optional): the half-length of the Sobel filter applied to the
        signal, larger values will consider more points to estimate the signal
        gradient
        uniform (bool, optional): whether the Sobel filter should uniformly
        weight all points or give a higher weight to more distant points,
        defaults to True

    Returns:
        n_edges (int): the number of abrupt jumps in the signal
    """
    a = A.values if isinstance(A, pd.Series) else A

    # Detect edges using a sobel filter
    a_sobel = sobel(a, k, uniform)
    median = np.median(a_sobel)
    mad = np.median(np.abs(median - a_sobel))
    dy_threshold = sd * mad / 0.67  # mad / 0.67 approximates 1 s.d. in a normal distribution
    jumps = (a_sobel > (median + dy_threshold)) | (a_sobel < (median - dy_threshold))

    # Detect outliers based on local median and global deviance
    mad = np.median(np.abs(np.median(a) - a))
    y_threshold = sd * mad / 0.67  # mad / 0.67 approximates 1 s.d. in a normal distribution
    # a_avg = np.convolve(a, np.ones(2 * k + 1) / (2 * k + 1), mode='same')
    a_median = np.roll(signal.medfilt(a, (2 * k + 1)), k)
    outliers = (a > (a_median + y_threshold)) | (a < (a_median - y_threshold))

    # Define edges as large jumps to values that deviate from the local median
    edges = (jumps & outliers)[k:-k]
    # N is the number of contiguous stretches of True
    n_edges = sum([1 for val, group in groupby(edges) if val])

    return n_edges


def n_outliers(A: pd.Series | np.ndarray, w_size: int = 1000, alpha: float = 0.0005) -> int:
    """counts the number of outliers as detected by grubbs test for outliers.
    int: _description_
    """
    a = A.values if isinstance(A, pd.Series) else A
    return detect_outliers(a, w_size=w_size, alpha=alpha).shape[0]


def _expected_max_gauss(x):
    """
    https://math.stackexchange.com/questions/89030/expectation-of-the-maximum-of-gaussian-random-variables
    """
    return np.mean(x) + np.std(x) * np.sqrt(2 * np.log(len(x)))


def n_expmax_violations(A: pd.Series | np.ndarray) -> int:
    a = A.values if isinstance(A, pd.Series) else A
    exp_max = _expected_max_gauss(a)
    return sum(np.abs(a) > exp_max)


def expmax_violation(A: pd.Series | np.ndarray) -> float:
    a = A.values if isinstance(A, pd.Series) else A
    exp_max = _expected_max_gauss(a)
    n_violations = sum(np.abs(a) > exp_max)
    if n_violations == 0:
        return 0.0
    else:
        return np.sum(np.abs(a[np.abs(a) > exp_max]) - exp_max) / n_violations


def bleaching_tau(A: pd.Series) -> float:
    """overall tau of bleaching."""
    y, t = A.values, A.index.values
    reg = Regression(model=ExponDecay())
    reg.fit(y, t)
    return reg.popt[1]


def low_freq_power_ratio(A: pd.Series, f_cutoff: float = 3.18) -> float:
    """
    Fraction of the total signal power contained below a given cutoff frequency.

    Parameters
    ----------
    A :
        the signal time series with signal values in the columns and sample
        times in the index
    f_cutoff :
        cutoff frequency, default value of 3.18 esitmated using the formula
        1 / (2 * pi * tau) and an approximate tau_rise for GCaMP6f of 0.05s.
    """
    a = A.copy()
    assert a.ndim == 1  # only 1D for now
    # Get frequency bins
    tpts = a.index.values
    dt = np.median(np.diff(tpts))
    n_pts = len(a)
    freqs = np.fft.rfftfreq(n_pts, dt)
    # Compute power spectral density
    psd = np.abs(np.fft.rfft(a - a.mean())) ** 2
    # Return the ratio of power contained in low freqs
    return psd[freqs <= f_cutoff].sum() / psd.sum()


def spectral_entropy(A: pd.Series, eps: float = np.finfo('float').eps) -> float:
    """
    Compute the normalized entropy of the signal power spectral density and
    return a metric (1 - entropy) that is low (0) if all frequency components
    have equal power, as for noise, and high (1) if all the power is
    concentrated in a single component.

    Parameters
    ----------
    A :
        the signal time series with signal values in the columns and sample
        times in the index
    eps :
        small number added to the PSD for numerical stability
    """
    a = A.copy()
    assert a.ndim == 1  # only 1D for now
    # Compute power spectral density
    psd = np.abs(np.fft.rfft(a - a.mean())) ** 2
    psd_norm = psd / np.sum(psd)
    # Compute spectral entropy in bits
    spectral_entropy = -1 * np.sum(psd_norm * np.log2(psd_norm + eps))
    # Normalize by the maximum entropy (bits)
    n_bins = len(psd)
    max_entropy = np.log2(n_bins)
    norm_entropy = spectral_entropy / max_entropy
    return 1 - norm_entropy


def ar_score(A: pd.Series | np.ndarray, order: int = 2) -> float:
    """
    R-squared from an AR(n) model fit to the signal as a measure of the temporal
    structure present in the signal.

    Parameters
    ----------
    A : pd.Series or np.ndarray
        The signal time series with signal values in the columns and sample
        times in the index.
    order : int, optional
        The order of the AR model. Default is 2.

    Returns
    -------
    float
        The R-squared value indicating the variance explained by the AR model.
        Returns NaN if the signal is constant.
    """
    # Pull signal out of pandas Series if needed
    a = A.values if isinstance(A, pd.Series) else A
    assert a.ndim == 1, 'Signal must be 1-dimensional.'

    # Handle constant signal case
    if len(np.unique(a)) == 1:
        return np.nan

    # Create design matrix X and target vector y based on AR order
    X = np.column_stack([a[i : len(a) - order + i] for i in range(order)])
    y = a[order:]

    try:
        # Fit linear regression using least squares
        _, residual, _, _ = np.linalg.lstsq(X, y)
    except np.linalg.LinAlgError:
        return np.nan

    if residual:
        # Calculate R-squared using residuals
        ss_residual = residual[0]
        ss_total = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ss_residual / ss_total)
    else:
        r_squared = np.nan

    return r_squared


def noise_simulation(A: pd.Series, metric: callable, noise_sd: np.ndarray = np.logspace(-2, 1)) -> np.ndarray:
    """
    See how a quality metric changes when adding Gaussian noise to a signal.
    The signal will be z-scored before noise is added, so noise_sd should be
    scaled appropriately.

    Parameters
    ----------
    A :
        a signal time series with signal values in the columns and sample
        times in the index
    metric :
        a function that computes a metric on the signal
    noise_sd :
        array of noise levels to add to the z-scored signal before computing the
        metric
    """
    A_z = z(A)
    scores = np.full(len(noise_sd), np.nan)
    for i, sd in enumerate(noise_sd):
        a = A_z + np.random.normal(scale=sd, size=len(A))
        scores[i] = metric(a)
    return scores
