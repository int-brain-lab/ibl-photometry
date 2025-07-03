from typing import Optional
import numpy as np
import pandas as pd
from scipy import stats

from iblphotometry.processing import (
    z,
    Regression,
    ExponDecay,
    detect_spikes,
    detect_outliers,
)
from iblphotometry.behavior import psth


def n_early_samples(A: pd.DataFrame | pd.Series, dt_tol: float = 0.001) -> int:
    return find_early_samples(A, dt_tol=dt_tol).sum()


def n_unique_samples(A: pd.Series | np.ndarray) -> int:
    """
    Number of unique samples in the signal. Low values indicate that the signal
    was not within the range of the digitizer during acquisition.
    """
    a = A.values if isinstance(A, pd.Series) else A
    return np.unique(a).shape[0]


def percentile_dist(A: pd.Series | np.ndarray, pc: tuple = (50, 95), axis=-1) -> float:
    """the distance between two percentiles in units of z. Captures the magnitude of transients.

    Args:
        A (pd.Series | np.ndarray): the input data, np.ndarray for stride tricks sliding windows
        pc (tuple, optional): percentiles to be computed. Defaults to (50, 95).
        axis (int, optional): only for arrays, the axis to be computed. Defaults to -1.

    Returns:
        float: the value of the metric
    """
    if isinstance(A, pd.Series):  # "overloading"
        P = np.percentile(z(A.values), pc)
    elif isinstance(A, np.ndarray):
        P = np.percentile(z(A), pc, axis=axis)
    else:
        raise TypeError('A must be pd.Series or np.ndarray.')
    return P[1] - P[0]


def signal_asymmetry(A: pd.Series | np.ndarray, pc_comp: int = 95, axis=-1) -> float:
    """the ratio between the distance of two percentiles to the median. Proportional to the the signal to noise.

    Args:
        A (pd.Series | np.ndarray): _description_
        pc_comp (int, optional): _description_. Defaults to 95.
        axis (int, optional): _description_. Defaults to -1.

    Returns:
        float: _description_
    """
    if not (isinstance(A, pd.Series) or isinstance(A, np.ndarray)):
        raise TypeError('A must be pd.Series or np.ndarray.')

    a = np.absolute(percentile_dist(A, (50, pc_comp), axis=axis))
    b = np.absolute(percentile_dist(A, (100 - pc_comp, 50), axis=axis))
    return a / b


def signal_skew(A: pd.Series | np.ndarray, axis=-1) -> float:
    if isinstance(A, pd.Series):
        P = stats.skew(A.values)
    elif isinstance(A, np.ndarray):
        P = stats.skew(A, axis=axis)
    else:
        raise TypeError('A must be pd.Series or np.ndarray.')
    return P


def n_unique_samples(A: pd.Series | np.ndarray) -> int:
    """number of unique samples in the signal. Low values indicate that the signal during acquisition was not within the range of the digitizer."""
    a = A.values if isinstance(A, pd.Series) else A
    return np.unique(a).shape[0]


def n_spikes(A: pd.Series | np.ndarray, sd: int = 5):
    """count the number of spike artifacts in the recording."""
    a = A.values if isinstance(A, pd.Series) else A
    return detect_spikes(a, sd=sd).shape[0]


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
    signal = A.copy()
    assert signal.ndim == 1  # only 1D for now
    # Get frequency bins
    tpts = signal.index.values
    dt = np.median(np.diff(tpts))
    n_pts = len(signal)
    freqs = np.fft.rfftfreq(n_pts, dt)
    # Compute power spectral density
    psd = np.abs(np.fft.rfft(signal - signal.mean())) ** 2
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
    signal = A.copy()
    assert signal.ndim == 1  # only 1D for now
    # Compute power spectral density
    psd = np.abs(np.fft.rfft(signal - signal.mean())) ** 2
    psd_norm = psd / np.sum(psd)
    # Compute spectral entropy in bits
    spectral_entropy = -1 * np.sum(psd_norm * np.log2(psd_norm + eps))
    # Normalize by the maximum entropy (bits)
    n_bins = len(psd)
    max_entropy = np.log2(n_bins)
    norm_entropy = spectral_entropy / max_entropy
    return 1 - norm_entropy


def ar_score(A: pd.Series) -> float:
    """
    R-squared from an AR(1) model fit to the signal as a measure of the temporal
    structure present in the signal.

    Parameters
    ----------
    A :
        the signal time series with signal values in the columns and sample
        times in the index
    """
    # Pull signal out of pandas series
    signal = A.values
    assert signal.ndim == 1  # only 1D for now
    X = signal[:-1]
    y = signal[1:]
    res = stats.linregress(X, y)
    return res.rvalue**2


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
        signal = A_z + np.random.normal(scale=sd, size=len(A))
        scores[i] = metric(signal)
    return scores
