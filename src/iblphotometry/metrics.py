from typing import Optional
from itertools import groupby
import numpy as np
import pandas as pd
from scipy import stats, signal
from numpy.lib.stride_tricks import as_strided


from iblphotometry.preprocessing import (
    find_early_samples
)
from iblphotometry.processing import (
    z,
    sobel,
    Regression,
    ExponDecay,
    detect_outliers,
)
from iblphotometry.analysis import psth


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


def median_absolute_deviance(
    A: pd.Series | np.ndarray,
    normalize: bool = False
    ) -> float:
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
    if isinstance(A, pd.Series):  # "overloading"
        P = np.percentile(z(A.values), pc, axis=axis)
    elif isinstance(A, np.ndarray):
        P = np.percentile(z(A), pc, axis=axis)
    else:
        raise TypeError('A must be pd.Series or np.ndarray.')
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
    if not (isinstance(A, pd.Series) or isinstance(A, np.ndarray)):
        raise TypeError('A must be pd.Series or np.ndarray.')

    a = np.absolute(percentile_distance(A, (50, pc_comp), axis=axis))
    b = np.absolute(percentile_distance(A, (100 - pc_comp, 50), axis=axis))
    return a / b


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


def n_outliers(
    A: pd.Series | np.ndarray, w_size: int = 1000, alpha: float = 0.0005
) -> int:
    """counts the number of outliers as detected by grubbs test for outliers.
    int: _description_
    """
    a = A.values if isinstance(A, pd.Series) else A
    return detect_outliers(a, w_size=w_size, alpha=alpha).shape[0]


def bleaching_tau(A: pd.Series) -> float:
    """overall tau of bleaching."""
    y, t = A.values, A.index.values
    reg = Regression(model=ExponDecay())
    reg.fit(y, t)
    return reg.popt[1]


def ttest_pre_post(
    A: pd.Series,
    trials: pd.DataFrame,
    event_name: str,
    fs=None,
    pre_w=[-1, -0.2],
    post_w=[0.2, 1],
    alpha=0.001,
) -> bool:
    """
    :param calcium: np array, trace of the signal to be used
    :param times: np array, times of the signal to be used
    :param t_events: np array, times of the events to align to
    :param fs: float, sampling frequency of the signal
    :param pre_w: list of floats, pre window sizes in seconds
    :param post_w: list of floats, post window sizes in seconds
    :param confid: float, confidence level (alpha)
    :return: boolean, True if metric passes
    """
    y, t = A.values, A.index.values
    fs = 1 / np.median(np.diff(t)) if fs is None else fs

    t_events = trials[event_name].values

    psth_pre = psth(y, t, t_events, fs=fs, event_window=pre_w)[0]
    psth_post = psth(y, t, t_events, fs=fs, event_window=post_w)[0]

    # Take median value of signal over time
    pre = np.median(psth_pre, axis=0)
    post = np.median(psth_post, axis=0)
    # Paired t-test
    ttest = stats.ttest_rel(pre, post)
    passed_confg = ttest.pvalue < alpha
    return passed_confg


def has_response_to_event(
    A: pd.Series,
    event_times: np.ndarray,
    fs: Optional[float] = None,
    window: tuple = (-1, 1),
    alpha: float = 0.005,
    mode='peak',
) -> bool:
    # checks if there is a significant response to an event

    # ibldsb way
    y, t = A.values, A.index.values
    fs = 1 / np.median(np.diff(t)) if fs is None else fs
    P, psth_ix = psth(y, t, event_times, fs=fs, event_window=window)

    # temporally averages the samples in the window. Sensitive to window size!
    if mode == 'mean':
        sig_samples = np.average(P, axis=0)
    # takes the peak sample, minus one sd
    if mode == 'peak':
        sig_samples = np.max(P, axis=0) - np.std(y)

    # baseline is all samples that are not part of the response
    base_ix = np.setdiff1d(np.arange(y.shape[0]), psth_ix.flatten())
    base_samples = y[base_ix]

    res = stats.ttest_ind(sig_samples, base_samples)
    return res.pvalue < alpha


def has_responses(
    A: pd.Series,
    trials: pd.DataFrame,
    event_names: list,
    fs: Optional[float] = None,
    window: tuple = (-1, 1),
    alpha: float = 0.005,
) -> bool:
    t = A.index.values
    fs = 1 / np.median(np.diff(t)) if fs is None else fs

    res = []
    for event_name in event_names:
        event_times = trials[event_name]
        res.append(
            has_response_to_event(A, event_times, fs=fs, window=window, alpha=alpha)
        )

    return np.any(res)


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


def eval_metric(
    F: pd.Series,
    metric: callable,
    metric_kwargs: dict = {},
    sliding_kwargs: dict = {},
    detrend: bool = True
) -> dict:
    """
    Evaluate a metric on a time series, optionally with sliding window analysis.

    Parameters:
    -----------
    F : pd.Series
        Input time series data
    metric : Callable
        Metric function to evaluate
    metric_kwargs : dict, optional
        Arguments to pass to the metric function
    sliding_kwargs : dict, optional
        Sliding window parameters. If None or empty, evaluates on full signal only.
        Expected keys: 'w_len' (window length)
    full_output : bool
        Whether to include sliding values and timepoints in output

    Returns:
    --------
    dict : Results dictionary with keys:
        - 'value': metric evaluated on full signal
        - 'sliding_values': metric values for each window (if full_output=True)
        - 'sliding_timepoints': timepoints for each window (if full_output=True)
        - 'r': correlation coefficient of sliding values vs time
        - 'p': p-value for the correlation
    """

    results_vals = ['value', 'sliding_values', 'sliding_timepoints', 'r', 'p']
    result = {k: np.nan for k in results_vals}

    # Always calculate the full signal metric
    result['value'] = metric(F.values, **metric_kwargs)

    # Determine windowing parameters
    if sliding_kwargs and 'w_len' in sliding_kwargs:
        # Sliding window case
        dt = np.median(np.diff(F.index))
        w_len = sliding_kwargs['w_len']
        w_size = int(w_len // dt)
        step_size = int(w_size // 2)
        n_windows = int((len(F) - w_size) // step_size + 1)

        # Check if we have enough data for meaningful sliding analysis
        if n_windows <= 2:
            return result

        # Create time indices for sliding windows
        S_times = F.index.values[
            np.linspace(step_size, n_windows * step_size, n_windows).astype(int)
        ]

        # Create windowed view into array
        a = F.values
        windows = as_strided(
            a,
            shape=(n_windows, w_size),
            strides=(step_size * a.strides[0], a.strides[0])
        )

        if detrend:
            def _metric(w, **metric_kwargs):
                x = np.arange(len(w))
                slope, intercept = stats.linregress(x, w)[:2]
                x_detrended = x - (slope * x + intercept)
                return metric(w, **metric_kwargs)
        else:
            _metric = metric
        # Apply metric to each window
        S_values = np.apply_along_axis(
            lambda w: _metric(w, **metric_kwargs), axis=1, arr=windows
        )

        result['sliding_values'] = S_values
        result['sliding_timepoints'] = S_times

        # Calculate trend statistics
        if n_windows > 1:
            result['r'], result['p'] = stats.linregress(S_times, S_values)[2:4]

    return result


def noise_simulation(
    A: pd.Series, metric: callable, noise_sd: np.ndarray = np.logspace(-2, 1)
) -> np.ndarray:
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
