import numpy as np
import pynapple as nap
from scipy.optimize import minimize, curve_fit, least_squares
from abc import ABC, abstractmethod
from tqdm import tqdm
from scipy.stats.distributions import norm
from scipy.stats import gaussian_kde
from sklearn.linear_model import RANSACRegressor, LinearRegression, TheilSenRegressor
from utils import filt


class AbstractBleachingModel(ABC):
    def __init__(self, method="L-BFGS-B", **method_kwargs):
        self.p0 = None
        self.popt = None
        self.perr = None
        self.method = method
        self.method_kwargs = method_kwargs

    @abstractmethod
    def model(self, t: np.ndarray, *args): ...

    @abstractmethod
    def estimate_p0(self, y: np.array, t: np.array): ...

    def _obj_fun(self, p, t, y):
        y_hat = self.model(t, *p)
        return np.sum((y - y_hat) ** 2)

    def _fit(self, y: np.array, t: np.array, calc_err: bool = True):
        if self.p0 is None:
            self.p0 = self.estimate_p0(y, t)

        # if self.method == "L-BFGS-B":
        self.minimize_result = minimize(
            self._obj_fun,
            self.p0,
            args=(t, y),
            bounds=self.bounds,
            method=self.method,
        )

        # if self.method == "least_squares":
        #     bounds = np.array(self.bounds)
        #     bounds = (bounds[:, 0], bounds[:, 1])
        #     self.minimize_result = least_squares(
        #         self._obj_fun,
        #         self.p0,
        #         loss=self.method_kwargs.get('loss', 'soft_l1'),
        #         f_scale=self.method_kwargs.get('f_scale', 1.0),
        #         bounds=bounds,
        #         args=(t, y),
        #     )

        # print(f"fitting using {self.method}")
        if not self.minimize_result.success:
            raise Exception(f"Fitting failed. {self.minimize_result.message}")

        self.popt = self.minimize_result.x

        if calc_err:
            self.perr = self._calc_perr(y, t)

    def _predict(self, t: np.array):
        return self.model(t, *self.popt)

    def _calc_perr(self, y: np.array, t: np.array):
        bounds = np.array(self.bounds)
        pcov = curve_fit(
            self.model, t, y, p0=self.popt, bounds=(bounds[:, 0], bounds[:, 1])
        )[1]
        return np.sqrt(np.diag(pcov))

    def _calc_r_squared(self, F):
        y, t = F.values, F.times()
        y_hat = self._predict(t)
        r = 1 - np.sum((y - y_hat) ** 2) / np.sum((y - np.average(y)) ** 2)
        return r

    def _calc_likelihood(self, F, n_samples=-1, use_kde=False):
        y, t = F.values, F.times()
        y_hat = self._predict(t)
        rs = y - y_hat
        if n_samples > 0:
            inds = np.random.randint(0, t.shape[0], size=n_samples)
            rs = rs[inds]
        if use_kde is True:
            # explicit estimation of the distribution of residuals
            if n_samples == -1:
                # raise a warning? KDE estimation on many samples is slow
                ...
            dist = gaussian_kde(rs)
        else:
            # using RSME
            sig = np.sqrt(np.average((y - y_hat) ** 2))
            dist = norm(0, sig)
        ll = np.sum(np.log(dist.pdf(rs)))
        return ll

    def _calc_aic(self, ll):
        aic = 2 * self.popt.shape[0] - 2 * np.log(ll)
        return aic

    def calc_model_stats(self, F: nap.Tsd, n_samples: int = -1, use_kde: bool = False):
        if self.popt is None:
            # have it like this for now
            raise NameError("model has not yet been fitted.")
            # self._fit(y, t)
        r_sq = self._calc_r_squared(F)
        ll = self._calc_likelihood(F, n_samples, use_kde)
        aic = self._calc_aic(ll)
        return dict(r_sq=r_sq, ll=ll, aic=aic)

    def calc_model_conf_ints(self, t, ci=(5, 95), n_samples=100):
        Y = np.zeros((t.shape[0], n_samples))
        for i in tqdm(range(n_samples)):
            p = self.popt + self.perr * np.random.randn(self.popt.shape[0])
            Y[:, i] = self.model(t, *p)
        self.model_ci = np.percentile(Y, ci, axis=1)

    def bleach_correct(self, F: nap.Tsd, refit=True):
        # method will have to be changed into a class attribute
        y, t = F.values, F.times()
        # if self.popt is None and refit:
        self._fit(y, t)
        y_hat = self._predict(t)
        return nap.Tsd(t=t, d=y - y_hat)


# bleach correction models
class ExponDecayBleachingModel(AbstractBleachingModel):
    bounds = ((0, np.Inf), (0, np.Inf), (-np.Inf, np.Inf))

    def model(self, t, A, tau, b):
        return A * np.exp(-t / tau) + b

    def estimate_p0(self, y: np.array, t: np.array):
        return (y[0], t[int(t.shape[0] / 3)], y[-1])


class DoubleExponDecayBleachingModel(AbstractBleachingModel):
    bounds = ((0, np.Inf), (0, np.Inf), (0, np.Inf), (0, np.Inf), (-np.Inf, np.Inf))

    def model(self, t, A1, tau1, A2, tau2, b):
        return A1 * np.exp(-t / tau1) + A2 * np.exp(-t / tau2) + b

    def estimate_p0(self, y: np.array, t: np.array):
        A_est = y[0]
        tau_est = t[int(t.shape[0] / 3)]
        b_est = y[-1]
        return (A_est, tau_est, A_est / 2, tau_est / 2, b_est)


class IsosbesticCorrection:
    def __init__(self, regressor="RANSAC", correction="deltaF"):
        self.regressor = regressor
        self.correction = correction

    def _fit(self, F_ca: nap.Tsd, F_iso: nap.Tsd):
        # this will allow for easy drop in replacement
        if self.regressor == "RANSAC":
            reg = RANSACRegressor(random_state=42)
        if self.regressor == "linear":
            reg = LinearRegression()

        # fit
        ca = F_ca.values[:, np.newaxis]
        iso = F_iso.values[:, np.newaxis]

        reg.fit(iso, ca)
        iso_fit = reg.predict(iso)
        return nap.Tsd(t=F_ca.times(), d=iso_fit.flatten())

    def correct(
        self,
        F_ca: nap.Tsd,
        F_iso: nap.Tsd,
        lowpass_isosbestic=dict(N=3, Wn=0.01, btype="lowpass"),
    ):
        if lowpass_isosbestic is not None:
            F_iso = filt(F_iso, **lowpass_isosbestic)
            iso_fit = self._fit(F_ca, F_iso)

        if self.correction == "deltaF":
            F_corr = (F_ca.values - iso_fit.values) / iso_fit.values

        if self.correction == "subtract":
            F_corr = F_ca.values - iso_fit.values

        if self.correction == "divide":
            F_corr = F_ca.values / iso_fit.values

        return nap.Tsd(t=F_ca.times(), d=F_corr)
