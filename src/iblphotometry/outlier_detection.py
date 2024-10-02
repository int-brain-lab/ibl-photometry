import numpy as np
from scipy.stats import t
import pynapple as nap
from ibldsp.utils import WindowGenerator
from copy import copy
import pandas as pd
from scipy.stats import gaussian_kde as kde


def grubbs_single(y, alpha=0.005, mode='median'):
    # to apply a single pass of grubbs outlier detection
    # see https://en.wikipedia.org/wiki/Grubbs%27s_test

    N = y.shape[0]
    if mode == 'classic':
        # this is the formulation as from wikipedia
        G = np.max(np.absolute(y - np.average(y))) / np.std(y)
    if mode == 'median':
        # this should be better in our case
        G = np.max(np.absolute(y - np.median(y))) / np.std(y)
    tsq = t.ppf(1 - alpha / (2 * N), df=N - 2) ** 2
    g = (N - 1) / np.sqrt(N) * np.sqrt(tsq / (N - 2 + tsq))

    if G > g:  # if G > g, reject null hypothesis (of no outliers)
        return True
    else:
        return False


def grubbs_it(y, alpha=0.05, mode='median'):
    # apply grubbs test iteratively until no more outliers are found
    outliers = []
    j = 0
    while grubbs_single(y, alpha=alpha):
        # get the outlier index
        if mode == 'classic':
            ix = np.argmax(np.absolute(y - np.average(y)))  
        if mode == 'median':
            ix = np.argmax(np.absolute(y - np.median(y)))
        outliers.append(ix + j)
        j += 1
        y = np.delete(y, ix)
    return np.sort(outliers)


def grubbs_sliding(y: np.array, w_size: int, alpha: float):
    # sliding grubbs test for a np.array
    n_samples = y.shape[0]
    wg = WindowGenerator(n_samples - (n_samples % w_size), w_size, 0)
    dtype = np.dtype((np.float64, w_size))
    B = np.fromiter(wg.slice_array(y), dtype=dtype)

    # sliding window outlier detection
    outlier_ix = []
    for i in range(B.shape[0]):
        outlier_ix.append((grubbs_it(B[i, :], alpha=alpha) + i * w_size))

    # last window TODO FIXME - this gave an index error, but as it is right now,
    # it leaves the remainder unchecked
    # outlier_ix.append(grubbs_it(y[-(n_samples % w_size):], alpha = alpha)+B.shape[0]*w_size)

    outlier_ix = np.concatenate(outlier_ix).astype("int64")
    return np.unique(outlier_ix)


def fillnan_kde(y: np.array, w: int = 25):
    # fill nans with KDE from edges
    inds = np.where(pd.isna(y))[0]
    if inds.shape[0] > 0:
        if inds[0] < w or inds[-1] > y.shape[0] - w:
            # first ix
            y_ = y[inds[0]: inds[0] + 2*w]
            y_ = y_[~pd.isna(y_)]  # nanfilter
            y[inds[0]] = kde(y_).resample(1)[0][0]
            
            # all middle inds
            for ix in inds[1:-1]:
                y_ = y[ix - w : ix + w]
                y_ = y_[~pd.isna(y_)]  # nanfilter
                y[ix] = kde(y_).resample(1)[0][0]
                
            # last ix
            y_ = y[inds[-1] - 2*w:inds[-1]]
            y_ = y_[~pd.isna(y_)]  # nanfilter
            y[inds[-1]] = kde(y_).resample(1)[0][0]

        else: # all fine, just iterate over all inds
            for ix in inds:
                y_ = y[ix - w : ix + w]
                y_ = y_[~pd.isna(y_)]  # nanfilter
                y[ix] = kde(y_).resample(1)[0][0]
                
        return y
    else:
        return y

def remove_outliers(F: nap.Tsd, w_size: int, alpha: float=0.005, w: int= 25):
    y, t = F.values, F.times()
    y = copy(y)
    outliers = grubbs_sliding(y, w_size=w_size, alpha=alpha)
    while len(outliers) > 0:
        y[outliers] = np.NaN
        y = fillnan_kde(y, w=w)
        outliers = grubbs_sliding(y, w_size=w_size, alpha=alpha)
    return nap.Tsd(t=t, d=y)
    