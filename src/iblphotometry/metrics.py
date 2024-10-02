import numpy as np
import pynapple as nap
from scipy.stats import linregress, skew
from utils import make_sliding_window, z
import outlier_detection


# metrics: - common definition: all receive a np.array as first argument, all other must be keyword arguments
def percentile_dist(A: np.array, pc: tuple = (50, 95), axis=-1):
    """the distance between two percentiles in units of z
    should be proportional to SNR, assuming the signal is
    in the positive 5th percentile

    Args:
        A (np.array): _description_
        pc (tuple, optional): _description_. Defaults to (50, 95).
        axis (int, optional): _description_. Defaults to -1.

    Returns:
        _type_: _description_
    """

    P = np.percentile(z(A), pc, axis=axis)
    return P[1] - P[0]


def signal_asymmetry(A: np.array, pc_comp: int = 95, axis=-1):
    """_summary_

    Args:
        A (np.array): _description_
        pc_comp (int, optional): _description_. Defaults to 95.
        axis (int, optional): _description_. Defaults to -1.

    Returns:
        _type_: _description_
    """
    a = percentile_dist(A, (50, pc_comp), axis=axis)
    b = percentile_dist(A, (100 - pc_comp, 50), axis=axis)
    return a - b


def number_unique_samples(A: np.array):
    """_summary_

    Args:
        A (np.array): _description_

    Returns:
        _type_: _description_
    """
    return np.unique(A).shape[0]


def number_of_outliers(A: np.array, w_size: int = 1000, alpha: float = 0.0005):
    """implements a sliding version of using grubbs test to detect outliers.

    Args:
        A (np.array): _description_
        w_size (int, optional): _description_. Defaults to 1000.
        alpha (float, optional): _description_. Defaults to 0.0005.

    Returns:
        _type_: _description_
    """
    return outlier_detection.grubbs_sliding(A, w_size=w_size, alpha=alpha).shape[0]


def signal_skew(A: np.array):
    return skew(A)


# funcs to run
def sliding_metric(
    F: nap.Tsd, w_size: int, metric: callable = None, n_wins: int = -1, **metric_kwargs
):
    y, t = F.values, F.times()
    yw = make_sliding_window(y, w_size)
    if n_wins > 0:
        n_samples = y.shape[0]
        inds = np.linspace(0, n_samples - w_size, n_wins, dtype="int64")
        yw = yw[inds, :]
    else:
        inds = np.arange(yw.shape[0], dtype="int64")

    if metric_kwargs is not None:
        m = metric(yw, **metric_kwargs)
    else:
        m = metric(yw)

    # return m, inds
    return nap.Tsd(t=t[inds + int(w_size / 2)], d=m)


# eval pipleline will be here
def eval_metric(
    F: nap.Tsd,
    metric: callable = None,
    metric_kwargs=None,
    sliding_kwargs=None,
):
    if metric_kwargs is not None:
        m = metric(F, **metric_kwargs)
    else:
        m = metric(F)

    if sliding_kwargs is not None:
        S = sliding_metric(F, metric=metric, **sliding_kwargs, **metric_kwargs)
        r, p = linregress(S.times(), S.values)[2:4]
    else:
        r = np.NaN
        p = np.NaN

    return dict(value=m, rval=r, pval=p)


def eval_pipeline(F_processed: nap.Tsd):
    sliding_kwargs = dict(w_size=300, n_wins=15)
    # the way how they are placed here - just for convenience
    # metric name, metric_kwargs, sliding_kwargs
    eval_metrics = [
        [percentile_dist, dict(pc=(50, 99)), sliding_kwargs],
        [signal_asymmetry, dict(pc_comp=95), sliding_kwargs],
        [number_unique_samples, None, None],
        [number_of_outliers, dict(w_size=1000, alpha=0.005), None],
    ]

    res = {}
    for metric, metric_kwargs, sliding_kwargs in eval_metrics:
        res[metric.__name__] = eval_metric(
            F_processed,
            metric,
            metric_kwargs=metric_kwargs,
            sliding_kwargs=sliding_kwargs,
        )

    return res
