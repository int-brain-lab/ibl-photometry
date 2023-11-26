import scipy.signal
import numpy as np


def baseline_correction(isosbestic, calcium):
    """
    Prototype of baseline correction for photometry data.
    Fits a low pass version of the isosbestic signal to the calcium signal. The baseline signal is
    the low pass isosbestic signal multiplied by the fit slope and added to the fit intercept.
    The corrected signal is the calcium signal minus the baseline signal divided by the baseline signal.
    We apply the same procedure to the full-band isosbestic signal to check for remaining correlations.
    :param isosbestic:
    :param calcium:
    :return:
    """
    sos = scipy.signal.butter(**{'N': 3, 'Wn': 0.01, 'btype': 'lowpass'}, output='sos')
    calcium_lp = scipy.signal.sosfiltfilt(sos, calcium)
    isosbestic_lp = scipy.signal.sosfiltfilt(sos, isosbestic)
    m = np.polyfit(isosbestic_lp, calcium_lp, 1)
    sosbp = scipy.signal.butter(**{'N': 3, 'Wn': [.001, 0.2], 'btype': 'bandpass'}, output='sos')
    ph = (calcium - (ref := isosbestic_lp * m[0] + m[1])) / ref
    ph = scipy.signal.sosfiltfilt(sosbp, ph)
    iso = scipy.signal.sosfiltfilt(sosbp,(isosbestic - ref) / ref)
    return ph, iso
