"""module summary"""

import numpy as np
from scipy import stats
import pynapple as nap
from sliding_operations import make_sliding_window


# funcs to run
def sliding_metric(
    F: nap.Tsd,
    w_len: float,
    fs: float = None,
    metric: callable = None,
    n_wins: int = -1,
    **metric_kwargs,
):
    """applies a metric along time.

    Args:
        F (nap.Tsd): _description_
        w_size (int): _description_
        metric (callable, optional): _description_. Defaults to None.
        n_wins (int, optional): _description_. Defaults to -1.

    Returns:
        _type_: _description_
    """
    y, t = F.values, F.times()
    fs = 1 / np.median(np.diff(t)) if fs is None else fs
    w_size = int(w_len * fs)

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
    metric_kwargs: dict = None,
    sliding_kwargs: dict = None,
):
    m = metric(F, **metric_kwargs) if metric_kwargs is not None else metric(F)

    if sliding_kwargs is not None:
        S = sliding_metric(F, metric=metric, **sliding_kwargs, **metric_kwargs)
        r, p = stats.linregress(S.times(), S.values)[2:4]
    else:
        r = np.NaN
        p = np.NaN

    return dict(value=m, rval=r, pval=p)


def eval_metrics_given_pipeline(
    F: nap.Tsd, pipeline: callable, metrics: list[callable]
):
    pass


# def eval_pipeline(F_processed: nap.Tsd):
#     sliding_kwargs = dict(w_size=300, n_wins=15)
#     # the way how they are placed here - just for convenience
#     # metric name, metric_kwargs, sliding_kwargs
#     eval_metrics = [
#         [percentile_dist, dict(pc=(50, 99)), sliding_kwargs],
#         [signal_asymmetry, dict(pc_comp=95), sliding_kwargs],
#         [n_unique_samples, None, None],
#         [n_outliers, dict(w_size=1000, alpha=0.005), None],
#         [bleaching_tau, None, None],
#     ]

#     res = {}
#     for metric, metric_kwargs, sliding_kwargs in eval_metrics:
#         res[metric.__name__] = eval_metric(
#             F_processed,
#             metric,
#             metric_kwargs=metric_kwargs,
#             sliding_kwargs=sliding_kwargs,
#         )

#     return res
