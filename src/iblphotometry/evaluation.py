"""module summary"""

import numpy as np
from scipy import stats
import pynapple as nap
from iblphotometry.sliding_operations import make_sliding_window


# funcs to run
def sliding_metric(
    F: nap.Tsd,
    w_len: float,
    fs: float = None,
    metric: callable = None,
    n_wins: int = -1,
    metric_kwargs: dict = None,
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
        inds = np.linspace(0, n_samples - w_size, n_wins, dtype='int64')
        yw = yw[inds, :]
    else:
        inds = np.arange(yw.shape[0], dtype='int64')

    m = metric(yw, **metric_kwargs) if metric_kwargs is not None else metric(yw)

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
        S = sliding_metric(
            F, metric=metric, **sliding_kwargs, metric_kwargs=metric_kwargs
        )
        r, p = stats.linregress(S.times(), S.values)[2:4]
    else:
        r = np.nan
        p = np.nan

    return dict(value=m, rval=r, pval=p)