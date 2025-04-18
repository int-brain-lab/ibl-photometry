from abc import ABC, abstractmethod
import warnings

import numpy as np
from scipy import signal
import pandas as pd

from iblutil.numerical import rcoeff
from ibldsp.utils import WindowGenerator
from numpy.lib.stride_tricks import as_strided

from scipy.optimize import minimize
from scipy.stats.distributions import norm
from scipy.stats import gaussian_kde, t
from scipy.special import pseudo_huber

from iblphotometry.behavior import psth
from inspect import signature
from copy import copy


# machine resolution
eps = np.finfo(np.float64).eps


"""
######## ##     ## ##    ##  ######  ######## ####  #######  ##    ##  ######
##       ##     ## ###   ## ##    ##    ##     ##  ##     ## ###   ## ##    ##
##       ##     ## ####  ## ##          ##     ##  ##     ## ####  ## ##
######   ##     ## ## ## ## ##          ##     ##  ##     ## ## ## ##  ######
##       ##     ## ##  #### ##          ##     ##  ##     ## ##  ####       ##
##       ##     ## ##   ### ##    ##    ##     ##  ##     ## ##   ### ##    ##
##        #######  ##    ##  ######     ##    ####  #######  ##    ##  ######
"""


def z(A: np.ndarray, mode='classic'):
    """classic z-score. Deviation from sample mean in units of sd

    :param A: _description_
    :type A: np.ndarray
    :param mode: _description_, defaults to 'classic'
    :type mode: str, optional
    :return: _description_
    :rtype: _type_
    """
    if mode == 'classic':
        return (A - np.average(A)) / np.std(A)
    if mode == 'median':
        return (A - np.median(A)) / np.std(A)


def mad(A: np.ndarray):
    """the MAD is defined as the median of the absolute deviations from the data's median
    see https://en.wikipedia.org/wiki/Median_absolute_deviation

    :param A: _description_
    :type A: np.ndarray
    :return: _description_
    :rtype: _type_
    """
    return np.median(np.absolute(A - np.median(A)), axis=-1)


def madscore(F: pd.Series):
    # TODO overloading of mad?
    y, t = F.values, F.index.values
    return pd.Series(mad(y), index=t)


def zscore(F: pd.Series, mode='classic'):
    y, t = F.values, F.index.values
    # mu, sig = np.average(y), np.std(y)
    return pd.Series(z(y, mode=mode), index=t)


def filt(F: pd.Series, N: int, Wn: float, fs: float | None = None, btype='low'):
    """a wrapper for scipy.signal.butter and sosfiltfilt"""
    y, t = F.values, F.index.values
    if fs is None:
        fs = 1 / np.median(np.diff(t))
    sos = signal.butter(N, Wn, btype, fs=fs, output='sos')
    y_filt = signal.sosfiltfilt(sos, y)
    return pd.Series(y_filt, index=t)


def sliding_rcoeff(signal_a, signal_b, nswin, overlap=0):
    """
    Computes the local correlation coefficient between two signals in sliding windows
    :param signal_a:
    :param signal_b:
    :param nswin: window size in samples
    :param overlap: overlap of successiv windows in samples
    :return: ix: indices of the center of the windows, r: correlation coefficients
    """
    wg = WindowGenerator(ns=signal_a.size, nswin=nswin, overlap=overlap)
    first_samples = np.array([fl[0] for fl in wg.firstlast])
    iwin = np.zeros([wg.nwin, wg.nswin], dtype=np.int32) + np.arange(wg.nswin)
    iwin += first_samples[:, np.newaxis]
    iwin[iwin >= signal_a.size] = signal_a.size - 1
    r = rcoeff(signal_a[iwin], signal_b[iwin])
    ix = first_samples + nswin // 2
    return ix, r


"""
##        #######   ######   ######     ######## ##     ## ##    ##  ######  ######## ####  #######  ##    ##  ######
##       ##     ## ##    ## ##    ##    ##       ##     ## ###   ## ##    ##    ##     ##  ##     ## ###   ## ##    ##
##       ##     ## ##       ##          ##       ##     ## ####  ## ##          ##     ##  ##     ## ####  ## ##
##       ##     ##  ######   ######     ######   ##     ## ## ## ## ##          ##     ##  ##     ## ## ## ##  ######
##       ##     ##       ##       ##    ##       ##     ## ##  #### ##          ##     ##  ##     ## ##  ####       ##
##       ##     ## ##    ## ##    ##    ##       ##     ## ##   ### ##    ##    ##     ##  ##     ## ##   ### ##    ##
########  #######   ######   ######     ##        #######  ##    ##  ######     ##    ####  #######  ##    ##  ######
"""


def mse_loss(p, x, y, fun):
    # mean squared error
    y_hat = fun(x, *p)
    return np.sum((y - y_hat) ** 2) / y.shape[0]


def mae_loss(p, x, y, fun):
    # mean absolute error
    y_hat = fun(x, *p)
    return np.sum(np.absolute(y - y_hat)) / y.shape[0]


def huber_loss(p, x, y, fun, d):
    # huber loss function
    y_hat = fun(x, *p)
    r = y - y_hat
    return np.sum(pseudo_huber(d, r)) / y.shape[0]


def irls_loss(p, x, y, fun, d=1e-7):
    # iteratively reweighted least squares - loss function
    y_hat = fun(x, *p)
    a = y - y_hat
    # irls weight
    f = np.max(np.stack([np.abs(a), np.ones(a.shape[0]) * d]), axis=0)
    w = 1 / f
    return np.sum(w * np.abs(a) ** 2) / y.shape[0]


"""
########  ########  ######   ########  ########  ######   ######  ####  #######  ##    ##
##     ## ##       ##    ##  ##     ## ##       ##    ## ##    ##  ##  ##     ## ###   ##
##     ## ##       ##        ##     ## ##       ##       ##        ##  ##     ## ####  ##
########  ######   ##   #### ########  ######    ######   ######   ##  ##     ## ## ## ##
##   ##   ##       ##    ##  ##   ##   ##             ##       ##  ##  ##     ## ##  ####
##    ##  ##       ##    ##  ##    ##  ##       ##    ## ##    ##  ##  ##     ## ##   ###
##     ## ########  ######   ##     ## ########  ######   ######  ####  #######  ##    ##
"""

# wrapper class for regressions with different loss functions
# following a sklearn style of .fit() and .predict()


class Regression:
    def __init__(self, model=None, method: str = 'mse', method_params=None):
        self.model = model
        self.method = method
        self.params = method_params if method_params is not None else {}

    def fit(self, x: np.ndarray, y: np.ndarray, algorithm='L-BFGS-B') -> tuple:
        p0 = self.model.est_p0(x, y)
        bounds = self.model.bounds if hasattr(self.model, 'bounds') else None
        if self.method == 'mse':
            minimize_result = minimize(
                mse_loss,
                p0,
                args=(x, y, self.model.eq),
                bounds=bounds,
                method=algorithm,
            )
        if self.method == 'mae':
            minimize_result = minimize(
                mae_loss,
                p0,
                args=(x, y, self.model.eq),
                bounds=bounds,
                method=algorithm,
            )
        if self.method == 'huber':
            d = self.params.get('d', 3)
            minimize_result = minimize(
                huber_loss,
                p0,
                args=(x, y, self.model.eq, d),
                bounds=bounds,
                method=algorithm,
            )
        if self.method == 'irls':
            d = self.params.get('d', 1e-7)
            minimize_result = minimize(
                irls_loss,
                p0,
                args=(x, y, self.model.eq, d),
                bounds=bounds,
                method=algorithm,
            )
        if not minimize_result.success:
            raise Exception(f'Fitting failed. {minimize_result.message}')
        else:
            self.popt = minimize_result.x

        # if self.method == 'sklearn-RANSAC':
        #     self.reg = RANSACRegressor(random_state=42)
        #     xr = x.reshape(-1, 1)
        #     self.reg.fit(xr, y)
        #     self.popt = (self.reg.estimator_.coef_, self.reg.estimator_.intercept_)

        # if self.method == 'sklearn-linear':
        #     self.reg = LinearRegression()
        #     xr = x.reshape(-1, 1)
        #     self.reg.fit(xr, y)
        #     self.popt = (self.reg.coef_, self.reg.intercept_)
        #     print(self.popt)

    def predict(self, x: np.ndarray, return_type='numpy'):
        x = np.sort(x)  # just in case
        y_hat = self.model.eq(x, *self.popt)
        if return_type == 'numpy':
            return y_hat
        if return_type == 'pandas':
            return pd.Series(y_hat, index=x)


"""
########  ##       ########    ###     ######  ##     ##     ######   #######  ########  ########  ########  ######  ######## ####  #######  ##    ##
##     ## ##       ##         ## ##   ##    ## ##     ##    ##    ## ##     ## ##     ## ##     ## ##       ##    ##    ##     ##  ##     ## ###   ##
##     ## ##       ##        ##   ##  ##       ##     ##    ##       ##     ## ##     ## ##     ## ##       ##          ##     ##  ##     ## ####  ##
########  ##       ######   ##     ## ##       #########    ##       ##     ## ########  ########  ######   ##          ##     ##  ##     ## ## ## ##
##     ## ##       ##       ######### ##       ##     ##    ##       ##     ## ##   ##   ##   ##   ##       ##          ##     ##  ##     ## ##  ####
##     ## ##       ##       ##     ## ##    ## ##     ##    ##    ## ##     ## ##    ##  ##    ##  ##       ##    ##    ##     ##  ##     ## ##   ###
########  ######## ######## ##     ##  ######  ##     ##     ######   #######  ##     ## ##     ## ########  ######     ##    ####  #######  ##    ##
"""


class BleachCorrection:
    def __init__(
        self,
        model=None,  # TODO bring back type checking
        regression_method: str = 'mse',
        regression_params: dict = None,
        correction_method: str = 'subtract',
    ):
        self.model = model
        self.regression = Regression(
            model=model, method=regression_method, method_params=regression_params
        )
        self.correction_method = correction_method

    def correct(self, F: pd.Series):
        self.regression.fit(F.index.values, F.values)
        ref = self.regression.predict(F.index.values, return_type='pandas')
        return correct(F, ref, mode=self.correction_method)


class LowpassBleachCorrection:
    def __init__(
        self,
        filter_params=dict(N=3, Wn=0.01, btype='lowpass'),
        correction_method='subtract-divide',
    ):
        self.filter_params = filter_params
        self.correction_method = correction_method

    def correct(self, F: pd.Series):
        F_filt = filt(F, **self.filter_params)
        return correct(F, F_filt, mode=self.correction_method)


class IsosbesticCorrection:
    def __init__(
        self,
        regression_method: str = 'mse',
        regression_params: dict | None = None,
        correction_method: str = 'subtract-divide',
        lowpass_isosbestic: dict | None = None,
    ):
        self.reg = Regression(
            model=LinearModel(),
            method=regression_method,
            method_params=regression_params,
        )
        self.lowpass_isosbestic = lowpass_isosbestic
        self.correction_method = correction_method

    def correct(
        self,
        F_ca: pd.Series,
        F_iso: pd.Series,
    ):
        if self.lowpass_isosbestic is not None:
            F_iso = filt(F_iso, **self.lowpass_isosbestic)

        self.reg.fit(F_iso.values, F_ca.values)
        F_iso_fit = self.reg.predict(F_iso.values, return_type='pandas')

        return correct(F_ca, F_iso_fit, mode=self.correction_method)


def correct(
    signal: pd.Series, reference: pd.Series, mode: str = 'subtract'
) -> pd.Series:
    """the main function that applies the correction of a signal with a reference. Correcions can be applied in 3 principle ways:
    - The reference can be subtracted from the signal
    - the signal can be divided by the reference
    - both of the above (first subtraction, then division) - this is akin to df/f

    :param signal: _description_
    :type signal: pd.Series
    :param reference: _description_
    :type reference: pd.Series
    :param mode: _description_, defaults to 'subtract'
    :type mode: str, optional
    :return: _description_
    :rtype: pd.Series
    """
    if mode == 'subtract':
        signal_corrected = signal.values - reference.values
    if mode == 'divide':
        signal_corrected = signal.values / reference.values
    if mode == 'subtract-divide':
        signal_corrected = (signal.values - reference.values) / reference.values
    return pd.Series(signal_corrected, index=signal.index.values)


"""
##     ##  #######  ########  ######## ##        ######
###   ### ##     ## ##     ## ##       ##       ##    ##
#### #### ##     ## ##     ## ##       ##       ##
## ### ## ##     ## ##     ## ######   ##        ######
##     ## ##     ## ##     ## ##       ##             ##
##     ## ##     ## ##     ## ##       ##       ##    ##
##     ##  #######  ########  ######## ########  ######
"""


class AbstractModel(ABC):
    @abstractmethod
    def eq():
        # the model equation
        ...

    @abstractmethod
    def est_p0():
        # given data, estimate model parameters
        ...

    def _calc_r_squared(self, y, y_hat):
        r = 1 - np.sum((y - y_hat) ** 2) / np.sum((y - np.average(y)) ** 2)
        return r

    def _calc_likelihood(self, y, y_hat, n_samples=-1, use_kde=False):
        rs = y - y_hat
        if n_samples > 0:
            inds = np.random.randint(0, y.shape[0], size=n_samples)
            rs = rs[inds]
        if use_kde is True:
            # explicit estimation of the distribution of residuals
            if n_samples == -1:
                warnings.warn(
                    f'calculating KDE on {y.values.shape[0]} samples. This might be slow'
                )
            dist = gaussian_kde(rs)
        else:
            # using RSME
            sig = np.sqrt(np.average((y - y_hat) ** 2))
            dist = norm(0, sig)
        ll = np.sum(np.log(dist.pdf(rs)))
        return ll

    def _calc_aic(self, k, ll):
        aic = 2 * k - 2 * ll  # np.log(ll)
        return aic

    def calc_model_stats(self, y, y_hat, n_samples: int = -1, use_kde: bool = False):
        r_sq = self._calc_r_squared(y, y_hat)
        ll = self._calc_likelihood(y, y_hat, n_samples, use_kde)
        k = len(signature(self.eq).parameters) - 1
        aic = self._calc_aic(ll, k)
        return dict(r_sq=r_sq, ll=ll, aic=aic)


# the actual models


class LinearModel(AbstractModel):
    def eq(self, x, m, b):
        return x * m + b

    def est_p0(self, x: np.ndarray, y: np.ndarray):
        return tuple(np.polyfit(x, y, 1))
        # x, y = np.sort(x), np.sort(y)
        # dx = x[-1] - x[0]
        # dy = y[-1] - y[0]
        # m = dy / dx
        # b = np.average(y - (m * x))
        # return (m, b)


class ExponDecay(AbstractModel):
    bounds = ((0, np.inf), (eps, np.inf), (-np.inf, np.inf))

    def eq(self, t, A, tau, b):
        return A * np.exp(-t / tau) + b

    def est_p0(self, t: np.ndarray, y: np.ndarray):
        return (y[0], t[int(t.shape[0] / 3)], y[-1])


class DoubleExponDecay(AbstractModel):
    bounds = (
        (0, np.inf),
        (eps, np.inf),
        (0, np.inf),
        (eps, np.inf),
        (-np.inf, np.inf),
    )

    def eq(self, t, A1, tau1, A2, tau2, b):
        return A1 * np.exp(-t / tau1) + A2 * np.exp(-t / tau2) + b

    def est_p0(self, t: np.ndarray, y: np.ndarray):
        A_est = y[0]
        tau_est = t[int(t.shape[0] / 3)]
        b_est = y[-1]
        return (A_est, tau_est, A_est / 2, tau_est / 2, b_est)


class TripleExponDecay(AbstractModel):
    bounds = (
        (0, np.inf),
        (eps, np.inf),
        (0, np.inf),
        (eps, np.inf),
        (0, np.inf),
        (eps, np.inf),
        (-np.inf, np.inf),
    )

    def eq(self, t, A1, tau1, A2, tau2, A3, tau3, b):
        return (
            A1 * np.exp(-t / tau1) + A2 * np.exp(-t / tau2) + A3 * np.exp(-t / tau3) + b
        )

    def est_p0(self, t: np.ndarray, y: np.ndarray):
        A_est = y[0]
        tau_est = t[int(t.shape[0] / 3)]
        b_est = y[-1]
        return (
            A_est,
            tau_est,
            A_est * 0.66,
            tau_est * 0.66,
            A_est * 0.33,
            tau_est * 0.33,
            b_est,
        )


"""
 ######   #######  ########  ########  ########  ######  ######## ####  #######  ##    ##    ######## ##     ## ##    ##  ######  ######## ####  #######  ##    ##  ######
##    ## ##     ## ##     ## ##     ## ##       ##    ##    ##     ##  ##     ## ###   ##    ##       ##     ## ###   ## ##    ##    ##     ##  ##     ## ###   ## ##    ##
##       ##     ## ##     ## ##     ## ##       ##          ##     ##  ##     ## ####  ##    ##       ##     ## ####  ## ##          ##     ##  ##     ## ####  ## ##
##       ##     ## ########  ########  ######   ##          ##     ##  ##     ## ## ## ##    ######   ##     ## ## ## ## ##          ##     ##  ##     ## ## ## ##  ######
##       ##     ## ##   ##   ##   ##   ##       ##          ##     ##  ##     ## ##  ####    ##       ##     ## ##  #### ##          ##     ##  ##     ## ##  ####       ##
##    ## ##     ## ##    ##  ##    ##  ##       ##    ##    ##     ##  ##     ## ##   ###    ##       ##     ## ##   ### ##    ##    ##     ##  ##     ## ##   ### ##    ##
 ######   #######  ##     ## ##     ## ########  ######     ##    ####  #######  ##    ##    ##        #######  ##    ##  ######     ##    ####  #######  ##    ##  ######
"""

# these are the convenience functions that are called in pipelines


def lowpass_bleachcorrect(F: pd.Series, **kwargs):
    bc = LowpassBleachCorrection(**kwargs)
    return bc.correct(F)


def exponential_bleachcorrect(F: pd.Series, **kwargs):
    model = DoubleExponDecay()
    ec = BleachCorrection(model, **kwargs)
    return ec.correct(F)


def isosbestic_correct(F_sig: pd.DataFrame, F_ref: pd.DataFrame, **kwargs):
    ic = IsosbesticCorrection(**kwargs)
    return ic.correct(F_sig, F_ref)


"""
 #######  ##     ## ######## ##       #### ######## ########     ########  ######## ######## ########  ######  ######## ####  #######  ##    ##
##     ## ##     ##    ##    ##        ##  ##       ##     ##    ##     ## ##          ##    ##       ##    ##    ##     ##  ##     ## ###   ##
##     ## ##     ##    ##    ##        ##  ##       ##     ##    ##     ## ##          ##    ##       ##          ##     ##  ##     ## ####  ##
##     ## ##     ##    ##    ##        ##  ######   ########     ##     ## ######      ##    ######   ##          ##     ##  ##     ## ## ## ##
##     ## ##     ##    ##    ##        ##  ##       ##   ##      ##     ## ##          ##    ##       ##          ##     ##  ##     ## ##  ####
##     ## ##     ##    ##    ##        ##  ##       ##    ##     ##     ## ##          ##    ##       ##    ##    ##     ##  ##     ## ##   ###
 #######   #######     ##    ######## #### ######## ##     ##    ########  ########    ##    ########  ######     ##    ####  #######  ##    ##
"""


def _grubbs_single(y: np.ndarray, alpha: float = 0.005, mode: str = 'median') -> bool:
    # to apply a single pass of grubbs outlier detection
    # see https://en.wikipedia.org/wiki/Grubbs%27s_test

    N = y.shape[0]
    if mode == 'classic':
        # this is the formulation as from wikipedia
        G = np.max(np.absolute(y - np.average(y))) / np.std(y)
    if mode == 'median':
        # this is more robust against outliers
        G = np.max(np.absolute(y - np.median(y))) / np.std(y)
    tsq = t.ppf(1 - alpha / (2 * N), df=N - 2) ** 2
    g = (N - 1) / np.sqrt(N) * np.sqrt(tsq / (N - 2 + tsq))

    if G > g:  # if G > g, reject null hypothesis (of no outliers)
        return True
    else:
        return False


def grubbs_test(y: np.ndarray, alpha: float = 0.005, mode: str = 'median'):
    # apply grubbs test iteratively until no more outliers are found
    outliers = []
    while _grubbs_single(y, alpha=alpha):
        # get the outlier index
        if mode == 'classic':
            ix = np.argmax(np.absolute(y - np.average(y)))
        if mode == 'median':
            ix = np.argmax(np.absolute(y - np.median(y)))
        outliers.append(ix)
        y[ix] = np.median(y)
    return np.sort(outliers)


def detect_outliers(y: np.ndarray, w_size: int = 1000, alpha: float = 0.005):
    # sliding grubbs test for a np.ndarray
    n_samples = y.shape[0]
    wg = WindowGenerator(n_samples - (n_samples % w_size), w_size, 0)
    dtype = np.dtype((np.float64, w_size))
    B = np.fromiter(wg.slice_array(y), dtype=dtype)

    # sliding window outlier detection
    outlier_ix = []
    for i in range(B.shape[0]):
        outlier_ix.append(grubbs_test(B[i, :], alpha=alpha) + i * w_size)

    # explicitly taking care of the remaining samples if they exist
    if y.shape[0] > np.prod(B.shape):
        outlier_ix.append(
            grubbs_test(y[np.prod(B.shape) :], alpha=alpha) + B.shape[0] * w_size
        )

    outlier_ix = np.concatenate(outlier_ix).astype('int64')
    return np.unique(outlier_ix)


def remove_nans(y: np.ndarray):
    y = y[~pd.isna(y)]  # nanfilter
    if y.shape[0] == 0:
        warnings.warn('y was all NaN and is now empty')
    return y


def fillnan_kde(y: np.ndarray, w: int = 25):
    # fill nans with KDE from edges
    inds = np.where(pd.isna(y))[0]
    if inds.shape[0] > 0:
        for ix in inds:
            ix_start = np.clip(ix - w, 0, y.shape[0] - 1)
            ix_stop = np.clip(ix + w, 0, y.shape[0] - 1)
            y_ = y[ix_start:ix_stop]
            y_ = remove_nans(y_)
            y[ix] = gaussian_kde(y_).resample(1)[0][0]

        return y
    else:
        # no NaNs present, doing nothing
        return y


def remove_outliers(
    F: pd.Series, w_size: int = 1000, alpha: float = 0.005, w: int = 25
):
    y, t = F.values, F.index.values
    y = copy(y)
    outliers = detect_outliers(y, w_size=w_size, alpha=alpha)
    while len(outliers) > 0:
        y[outliers] = np.nan
        y = fillnan_kde(y, w=w)
        outliers = detect_outliers(y, w_size=w_size, alpha=alpha)
    return pd.Series(y, index=t)


def detect_spikes_dt(t: np.ndarray, atol: float = 0.001):
    dts = np.diff(t)
    dt = np.median(dts)
    # bad_inds = dt < np.average(dt) - sd * np.std(dt)
    # return np.where(bad_inds)[0]
    return np.where(np.abs(dts - dt) > atol)[0]


def detect_spikes_dy(y: np.ndarray, sd: float = 5.0):
    dy = np.abs(np.diff(y))
    # bad_inds = dt < np.average(dt) - sd * np.std(dt)
    return np.where(dy > np.average(dy) + sd * np.std(dy))[0]


def remove_spikes(F: pd.Series, delta: str = 't', sd: int = 5, w: int = 25):
    y, t = F.values, F.index.values
    y = copy(y)
    if delta == 't':
        outliers = detect_spikes_dt(t, atol=0.001)
    elif delta == 'y':
        outliers = detect_spikes_dy(y, sd=sd)
    else:
        raise ValueError('delta must be "t" or "y"')
    y[outliers] = np.nan
    try:
        y = fillnan_kde(y, w=w)
    # except np.linalg.LinAlgError:
    except:
        i0s = (outliers - w).clip(0)
        i1s = outliers + w
        y[outliers] = [np.nanmedian(y[i0:i1]) for i0, i1 in zip(i0s, i1s)]
        warnings.warn('KDE fillnan failed, using local median')  # TODO logger
    return pd.Series(y, index=t)


## TODO: consider this simple interpolation method that uses the local median
# def remove_spikes(F: pd.Series, sd: int = 5, w: int = 5):
#     f = F.copy()
#     y, t = f.values, f.index.values
#     outliers = detect_spikes(y, sd=sd)
#     outliers = np.unique(np.concatenate([outliers - 1, outliers]))
#     i0s = (outliers - w).clip(0)
#     i1s = outliers + w
#     y[outliers] = [np.nanmedian(y[i0:i1]) for i0, i1 in zip(i0s, i1s)]
#     return pd.Series(y, index=t)


def find_early_samples(
    A: pd.DataFrame | pd.Series, dt_tol: float = 0.001
) -> np.ndarray:
    dt = np.median(np.diff(A.index))
    return dt - A.index.diff() > dt_tol


def _fill_missing_channel_names(A: np.ndarray) -> np.ndarray:
    missing_inds = np.where(A == '')[0]
    name_alternator = {'GCaMP': 'Isosbestic', 'Isosbestic': 'GCaMP', '': ''}
    for i in missing_inds:
        if i == 0:
            A[i] = name_alternator[A[i + 1]]
        else:
            A[i] = name_alternator[A[i - 1]]
    return A


def find_repeated_samples(
    A: pd.DataFrame,
    dt_tol: float = 0.001,
) -> int:
    if any(A['name'] == ''):
        A['name'] = _fill_missing_channel_names(A['name'].values)
    else:
        A
    repeated_sample_mask = A['name'].iloc[1:].values == A['name'].iloc[:-1].values
    repeated_samples = A.iloc[1:][repeated_sample_mask]
    dt = np.median(np.diff(A.index))
    early_samples = A[find_early_samples(A, dt_tol=dt_tol)]
    if not all([idx in early_samples.index for idx in repeated_samples.index]):
        print('WARNING: repeated samples found without early sampling')
    return repeated_sample_mask


def fix_repeated_sampling(
    A: pd.DataFrame, dt_tol: float = 0.001, w_size: int = 10, roi: str | None = None
) -> int:
    ## TODO: avoid this by explicitly handling multiple channels
    assert roi is not None
    # Drop first samples if channel labels are missing
    A.loc[A['name'].replace({'': np.nan}).first_valid_index() :]
    # Fix remaining missing channel labels
    if any(A['name'] == ''):
        A['name'] = _fill_missing_channel_names(A['name'].values)
    repeated_sample_mask = find_repeated_samples(A, dt_tol=dt_tol)
    name_alternator = {'GCaMP': 'Isosbestic', 'Isosbestic': 'GCaMP'}
    for i in np.where(repeated_sample_mask)[0] + 1:
        name = A.iloc[i]['name']
        value = A.iloc[i][roi]
        i0, i1 = A.index[i - w_size], A.index[i]
        same = A.loc[i0:i1].query('name == @name')[roi].mean()
        other_name = name_alternator[name]
        other = A.loc[i0:i1].query('name == @other_name')[roi].mean()
        assert np.abs(value - same) > np.abs(value - other)
        A.loc[A.index[i] :, 'name'] = [
            name_alternator[name] for name in A.loc[A.index[i] :, 'name']
        ]
    return A


"""
 ######  ##       #### ########  #### ##    ##  ######       #######  ########  ######## ########     ###    ######## ####  #######  ##    ##  ######
##    ## ##        ##  ##     ##  ##  ###   ## ##    ##     ##     ## ##     ## ##       ##     ##   ## ##      ##     ##  ##     ## ###   ## ##    ##
##       ##        ##  ##     ##  ##  ####  ## ##           ##     ## ##     ## ##       ##     ##  ##   ##     ##     ##  ##     ## ####  ## ##
 ######  ##        ##  ##     ##  ##  ## ## ## ##   ####    ##     ## ########  ######   ########  ##     ##    ##     ##  ##     ## ## ## ##  ######
      ## ##        ##  ##     ##  ##  ##  #### ##    ##     ##     ## ##        ##       ##   ##   #########    ##     ##  ##     ## ##  ####       ##
##    ## ##        ##  ##     ##  ##  ##   ### ##    ##     ##     ## ##        ##       ##    ##  ##     ##    ##     ##  ##     ## ##   ### ##    ##
 ######  ######## #### ########  #### ##    ##  ######       #######  ##        ######## ##     ## ##     ##    ##    ####  #######  ##    ##  ######
"""

"""this module holds a collection of processing pipelines for fiber photometry data"""


def make_sliding_window(
    A: np.ndarray,
    w_size: int,
    pad_mode='edge',
    method='stride_tricks',
    warning=None,
):
    """use np.stride_tricks to make a sliding window view of a 1-d np.ndarray A
    full overlap, step size 1
    assumes 8 byte numbers (to be exposed? but needs to be tested first)
    pads beginning and end of array with edge values

    Args:
        A (np.ndarray): Array to be windowed
        w_size (int): window size in samples

    Raises:
        ValueError: is raised if w_size is not an even number

    Returns:
        _type_: The view on the array, with shape (A.shape[0], w_size)
    """
    n_samples = A.shape[0]

    if method == 'stride_tricks':
        if w_size % 2 != 0:
            if warning == 'raise_exception':
                raise ValueError('w_size needs to be an even number')
            else:
                w_size += 1  # dangerous
        B = np.lib.stride_tricks.as_strided(A, ((n_samples - w_size), w_size), (8, 8))

    if method == 'window_generator':
        wg = WindowGenerator(n_samples - 1, w_size, w_size - 1)
        dtype = np.dtype((np.float64, w_size))
        B = np.fromiter(wg.slice_array(A), dtype=dtype)

    if pad_mode is not None:
        B = np.pad(B, ((int(w_size / 2), int(w_size / 2)), (0, 0)), mode=pad_mode)
    return B


def sliding_dFF(F: pd.Series, w_len: float, fs=None, weights=None):
    y, t = F.values, F.index.values
    fs = 1 / np.median(np.diff(t)) if fs is None else fs
    w_size = int(w_len * fs)

    def _dFF(A: np.ndarray):
        return (A - np.average(A)) / np.average(A)

    if weights is not None:
        # note: passing weights makes the stride trick not possible, or only with allocating a matrix of shape (n_samples * w_size)
        # true sliding operation implemented, much slower
        weights /= weights.sum()
        n_samples = y.shape[0]
        wg = WindowGenerator(n_samples - 1, w_size, w_size - 1)
        d = [
            np.sum(_dFF(F[first:last].values) * weights)
            for (first, last) in wg.firstlast
        ]
    else:
        B = make_sliding_window(y, w_size)
        mus = np.average(B, axis=1)
        d = (y - mus) / mus
    return pd.Series(d, index=t)


def sliding_z(F: pd.Series, w_len: float, fs=None, weights=None):
    """sliding window z-score of a pynapple time series with data with optional weighting

    Args:
        F (nap.Tsd): Signal to be zscored
        w_size (int): window size
        w_weights (_type_, optional): weights of the window . Defaults to None.

    Returns:
        _type_: _description_
    """
    y, t = F.values, F.index.values
    fs = 1 / np.median(np.diff(t)) if fs is None else fs
    w_size = int(w_len * fs)

    if weights is not None:
        # note: passing weights makes the stride trick not possible, or only with allocating a matrix of shape (n_samples * w_size)
        # true sliding operation implemented, much slower
        weights /= weights.sum()
        n_samples = y.shape[0]
        wg = WindowGenerator(n_samples - 1, w_size, w_size - 1)
        d = [
            np.sum(z(F[first:last].values) * weights) for (first, last) in wg.firstlast
        ]
    else:
        B = make_sliding_window(y, w_size)
        mus, sds = np.average(B, axis=1), np.std(B, axis=1)
        d = (y - mus) / sds
    return pd.Series(d, index=t)


def sliding_mad(F: pd.DataFrame, w_len: float = None, fs=None, overlap=90):
    y, t = F.values, F.index.values
    fs = 1 / np.median(np.diff(t)) if fs is None else fs
    w_size = int(w_len * fs)

    n_samples = y.shape[0]
    wg = WindowGenerator(ns=n_samples, nswin=w_size, overlap=overlap)
    trms = np.array([first for first, last in wg.firstlast]) / fs + t[0]

    rmswin, _ = psth(y, t, t_events=trms, fs=fs, event_window=[0, w_len])
    gain = np.nanmedian(np.abs(y)) / np.nanmedian(np.abs(rmswin), axis=0)
    gain = np.interp(t, trms, gain)
    return pd.Series(y * gain, index=t)


def sliding_robust_zscore(F: pd.Series, w_len: float, scale: bool = True) -> pd.Series:
    """
    Compute a robust z-score for each data point in a pandas Series using a sliding window.

    For each data point at which a full window (centered around that point)
    is available, compute the robust z-score:

         z = (x - median(window)) / MAD(window)

    where MAD is the median absolute deviation of the window. If scale=True,
    the MAD is multiplied by 1.4826 to approximate the standard deviation under normality.

    Parameters
    ----------
    F : pd.Series
        Input time-series with a numeric index (time) and signal values.
    w_len : float
        Window length in seconds.
    scale : bool, optional
        Whether to scale the MAD by 1.4826. Default is True.

    Returns
    -------
    pd.Series
        A new Series of the same length as F containing the robust z-scores.
        Data points near the boundaries without a full window are NaN.
    """
    # Ensure the index is numeric (time in seconds)
    times = F.index.values.astype(float)
    dt = np.median(np.diff(times))

    # Number of samples corresponding to w_len in seconds.
    w_size = int(w_len // dt)
    # Make window size odd so that a window can be centered
    if w_size % 2 == 0:
        w_size += 1
    half_win = w_size // 2

    a = F.values  # Underlying data
    n = len(a)

    # We can only compute a full (centered) window where there's enough data on both sides.
    # Valid center positions are indices half_win to n - half_win - 1.
    n_valid = n - 2 * half_win
    if n_valid <= 0:
        raise ValueError('Window length is too long for the given series.')

    # Using step size of 1: each valid index gets its own window.
    # Create a 2D view of the signal:
    # windows shape: (n_valid, w_size)
    windows = as_strided(
        a[half_win : n - half_win],
        shape=(n_valid, w_size),
        strides=(a.strides[0], a.strides[0]),
    )
    # However, the above would take contiguous blocks from a[half_win: n - half_win] only.
    # To get a sliding window centered at each valid index, we need a trick:
    # We'll use as_strided on the full array, starting at index 0, then select the valid windows:
    windows_full = as_strided(
        a, shape=(n - w_size + 1, w_size), strides=(a.strides[0], a.strides[0])
    )
    # The center of the k-th window in windows_full is at index k + half_win.
    # We want windows centered at indices half_win, half_win+1, ..., n - half_win - 1.
    # Thus, we select:
    windows = windows_full[0 + 0 : 0 + n_valid]  # shape (n_valid, w_size)

    # Compute the median for each window (row-wise).
    medians = np.median(windows, axis=1)
    # Compute the MAD for each window.
    mads = np.median(np.abs(windows - medians[:, None]), axis=1)
    if scale:
        mads *= 1.4826  # Scale MAD to approximate standard deviation under a normal distribution.

    # Avoid division by zero: if MAD is zero, set those z-scores to 0.
    safe_mads = np.where(mads == 0, np.nan, mads)

    # Compute robust z-scores for the center value of each window.
    # The center value for the k-th window is at index: k + half_win in the original array.
    centers = a[half_win : n - half_win]
    z_scores_valid = (centers - medians) / safe_mads

    # Pre-allocate result (all values NaN)
    robust_z = np.full(n, np.nan)
    # Fill in the computed z-scores at valid indices.
    valid_idx = np.arange(half_win, n - half_win)
    robust_z[valid_idx] = z_scores_valid

    # Return as a Series with the original index
    return pd.Series(robust_z, index=F.index)


def sliding_robust_zscore_rolling(
    F: pd.Series, w_len: float, scale: bool = True
) -> pd.Series:
    """
    Compute a robust z-score for each data point using a sliding window via pandas’ rolling().

    For each point where a full, centered window is available, compute:
          z = (x_center - median(window)) / MAD(window)
    where MAD is the median absolute deviation and, if scale=True, MAD is scaled by 1.4826.

    Parameters
    ----------
    F : pd.Series
        Input time-series with a numeric index (time in seconds) and signal values.
    w_len : float
        The window length in seconds.
    scale : bool, optional
        If True, multiply MAD by 1.4826 (default is True).

    Returns
    -------
    pd.Series
        A Series containing the robust z-scores at the center of each window.
        Points for which a full window cannot be computed will be NaN.
    """
    # Get the sample interval from the index
    times = F.index.values.astype(float)
    dt = np.median(np.diff(times))
    # Compute window size in samples and ensure it is odd (so there's a unique center)
    w_size = int(w_len / dt)
    if w_size % 2 == 0:
        w_size += 1

    def robust_zscore(window):
        # window is passed as a NumPy array (raw=True)
        center = window[len(window) // 2]
        med = np.median(window)
        mad = np.median(np.abs(window - med))
        if mad == 0:
            return np.nan
        if scale:
            mad *= 1.4826
        return (center - med) / mad

    # Use rolling window with center=True so that the result corresponds to the window center.
    F_proc = F.rolling(window=w_size, center=True).apply(robust_zscore, raw=True)
    # Return only valid (non-NaN) portions of the transformed signal
    return F_proc.loc[F_proc.first_valid_index() : F_proc.last_valid_index()]
