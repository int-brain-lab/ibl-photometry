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

    def __init__(self, method='L-BFGS-B', **method_kwargs):
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

        if self.method == "L-BFGS-B":
            self.minimize_result = minimize(
                self._obj_fun,
                self.p0,
                args=(t, y),
                bounds=self.bounds,
                method="L-BFGS-B",
            )

        if self.method == "least_squares":
            bounds = np.array(self.bounds)
            bounds = (bounds[:, 0], bounds[:, 1])
            self.minimize_result = least_squares(
                self._obj_fun,
                self.p0,
                loss=self.method_kwargs.get('loss', 'soft_l1'),
                f_scale=self.method_kwargs.get('f_scale', 1.0),
                bounds=bounds,
                args=(t, y),
            )
            
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

# turn this into a class
# other
def isosbestic_correct(
    F_ca: nap.Tsd,
    F_iso: nap.Tsd,
    regressor="RANSAC",
    correction="deltaF",
    butter_params=dict(order=3, fc=0.1),
):
    """preprocessing using isosbestic correction
    allows to choose regression type
    and correction type


    Args:
        F_ca (nap.Tsd): _description_
        F_iso (nap.Tsd): _description_
        regressor (str, optional): _description_. Defaults to "RANSAC".
        correction (str, optional): _description_. Defaults to "deltaF".
        butter_params (_type_, optional): _description_. Defaults to dict(order=3, fc=0.1).

    Returns:
        _type_: _description_
    """

    # of different variants
    if regressor == "RANSAC":
        reg = RANSACRegressor(random_state=42)
        reg_type = "sklearn"
    if regressor == "linear":
        reg = LinearRegression()
        reg_type = "sklearn"
    if regressor == "Theil-Sen":  # very slow ...
        reg = TheilSenRegressor(random_state=42)
        reg_type = "sklearn"
    if regressor == "soft_l1":
        reg_type = "least_squares"

    ca = F_ca.values[:, np.newaxis]
    iso = F_iso.values[:, np.newaxis]

    if reg_type == "sklearn":
        reg.fit(iso, ca)
        iso_fit = reg.predict(iso)
        iso_fit = nap.Tsd(t=F_ca.times(), d=iso_fit.flatten())

    if reg_type == "least_squares":
        # res_robust = least_squares(fun, x0, loss='soft_l1', loss_scale=0.1, args=(t_train, y_train))
        ...

    iso_fit_filt = filt(iso_fit, butter_params["order"], butter_params["fc"])

    if correction == "deltaF":
        F_corr = (F_ca.values - iso_fit_filt.values) / iso_fit_filt.values

    if correction == "subtract":
        F_corr = F_ca.values - iso_fit_filt.values

    if correction == "divide":
        F_corr = F_ca.values / iso_fit_filt.values

    return nap.Tsd(t=F_ca.times(), d=F_corr)
