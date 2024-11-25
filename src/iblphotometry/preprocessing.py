"""
This modules offers pre-processing for raw photometry data.
"""

import scipy.signal


def low_pass_filter(raw_signal, fs):
    params = {}
    sos = scipy.signal.butter(
        fs=fs,
        output='sos',
        **params.get('butterworth_lowpass', {'N': 3, 'Wn': 0.01, 'btype': 'lowpass'}),
    )
    signal_lp = scipy.signal.sosfiltfilt(sos, raw_signal)
    return signal_lp


def mad_raw_signal(raw_signal, fs):
    # This is a convenience function to get going whilst the preprocessing refactoring is being done
    # TODO delete this function once processing can be applied
    signal_lp = low_pass_filter(raw_signal, fs)
    signal_processed = (raw_signal - signal_lp) / signal_lp
    return signal_processed
