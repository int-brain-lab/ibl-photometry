import scipy.signal
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import ibllib.plots


def plot_raw_data_df(df_photometry, **kwargs):
    """
    Creates a two rows supblot with isosbestic and calcium raw traces as a function of time on top
    And a PSD and a cross-plot of the two signals on the bottom row. Optionally saves the figure to a file.
    :param df_photometry: dataframe with columns times, calcium, isosbestic
    :param event_times:
    :param suptitle:
    :param output_file:
    :return:
    """
    sns.set_style("whitegrid")
    times = df_photometry['times'].values
    calcium = df_photometry['raw_calcium'].values
    isosbestic = df_photometry['raw_isosbestic'].values
    return plot_photometry_traces(times, isosbestic, calcium, **kwargs)


def plot_photometry_traces(times, isosbestic, calcium, event_times=None, suptitle=None, output_file=None, low_pass_cross_plot=0.01):
    if low_pass_cross_plot:
        sos = scipy.signal.butter(**{'N': 3, 'Wn': 0.01, 'btype': 'lowpass'}, output='sos')
        calcium_lp = scipy.signal.sosfiltfilt(sos, calcium)
        isosbestic_lp = scipy.signal.sosfiltfilt(sos, isosbestic)
    else:
        calcium_lp, isosbestic_lp = (calcium, isosbestic)
    # start the plotting functions, first the raw signals in time domain
    fig, axd = plt.subplot_mosaic([['top', 'top'], ['left', 'right']], constrained_layout=True, figsize=(14, 8))
    axd['top'].plot(times, isosbestic, color='#803896', linewidth=.5, label='isosbestic')
    axd['top'].plot(times, calcium, color="#279F95", linewidth=.5, label='calcium dependent')
    if event_times is not None:
        ibllib.plots.vertical_lines(
            event_times, ymin=np.min(isosbestic), ymax=np.max(calcium), ax=axd['top'], alpha=.1, color='red')
    axd['top'].set(xlabel='time (s)', ylabel='photometry raw trace', title='raw photometry signal')
    axd['top'].legend()
    # lower left plot is the PSD of the two signals
    axd['left'].psd(calcium, Fs=1 / np.median(np.diff(times)), color="#279F95", linewidth=2, label='calcium dependent')
    axd['left'].psd(isosbestic, Fs=1 / np.median(np.diff(times)), color='#803896', linewidth=2, label='isosbestic')
    # lower right plot is the cross plot of the two signals to see if a regression makes sense
    scatter = axd['right'].scatter(isosbestic_lp, calcium_lp, s=None, c=times, cmap='magma', alpha=.8)
    axd['right'].set(xlabel='isosbestic F', ylabel='calcium dependent F', title='Cross-plot')
    fig.colorbar(scatter, ax=axd['right'], label='time (s)')
    if suptitle is not None:
        fig.suptitle(suptitle, fontsize=16)
    if output_file is not None:
        fig.savefig(output_file)
    return fig, axd
