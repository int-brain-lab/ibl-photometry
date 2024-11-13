from abc import ABC, abstractmethod
import warnings
from tqdm import tqdm
import numpy as np
import pandas as pd
import pynapple as nap
from scipy.optimize import minimize
from scipy.stats.distributions import norm
from scipy.stats import gaussian_kde
from scipy.special import pseudo_huber

from iblphotometry.helpers import filt
from inspect import signature


def correct(signal: nap.Tsd, reference: nap.Tsd, mode: str = 'subtract') -> nap.Tsd:
    if mode == 'subtract':
        signal_corrected = signal.values - reference.values
    if mode == 'divide':
        signal_corrected = signal.values / reference.values
    if mode == 'subtract-divide':
        signal_corrected = (signal.values - reference.values) / reference.values
    return nap.Tsd(t=signal.times(), d=signal_corrected)


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


### MODELS
eps = np.finfo(np.float64).eps


class LinearModel(AbstractModel):
    def eq(self, x, m, b):
        return x * m + b

    def est_p0(self, x: np.array, y: np.array):
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

    def est_p0(self, t: np.array, y: np.array):
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

    def est_p0(self, t: np.array, y: np.array):
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

    def est_p0(self, t: np.array, y: np.array):
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
        if return_type == 'pynapple':
            return nap.Tsd(t=x, d=y_hat)


class BleachCorrection:
    def __init__(
        self,
        model: AbstractModel = None,
        regression_method: str = 'mse',
        regression_params: dict = None,
        correction_method: str = 'subtract',
    ):
        self.model = model
        self.regression = Regression(
            model=model, method=regression_method, method_params=regression_params
        )
        self.correction_method = correction_method

    def correct(self, F: nap.Tsd):
        self.regression.fit(F.times(), F.values)
        ref = self.regression.predict(F.times(), return_type='pynapple')
        return correct(F, ref, mode=self.correction_method)


class IsosbesticCorrection:
    def __init__(
        self,
        regression_method: str = 'mse',
        regression_params: dict = None,
        correction_method: str = 'subtract-divide',
        lowpass_isosbestic: dict = None,
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
        F_ca: nap.Tsd,
        F_iso: nap.Tsd,
    ):
        if self.lowpass_isosbestic is not None:
            F_iso = filt(F_iso, **self.lowpass_isosbestic)

        self.reg.fit(F_iso.values, F_ca.values)
        F_iso_fit = self.reg.predict(F_iso.values, return_type='pynapple')

        return correct(F_ca, F_iso_fit, mode=self.correction_method)


class LowpassBleachCorrection:
    def __init__(
        self,
        filter_params=dict(N=3, Wn=0.01, btype='lowpass'),
        correction_method='subtract-divide',
    ):
        self.filter_params = filter_params
        self.correction_method = correction_method

    def correct(self, F: nap.Tsd):
        F_filt = filt(F, **self.filter_params)
        return correct(F, F_filt, mode=self.correction_method)


# convenience functions for pipelines
def lowpass_bleachcorrect(F: nap.Tsd, filter_params, correction_method):
    bc = LowpassBleachCorrection(filter_params, correction_method)
    return bc.correct(F)


def isosbestic_correct(
    F: nap.TsdFrame, signal_name=None, reference_name=None, **kwargs
):
    ic = IsosbesticCorrection(**kwargs)
    return ic.correct(F[signal_name], F[reference_name])
