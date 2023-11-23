import scipy.signal
import numpy as np
import matplotlib.pyplot as plt

import ibllib.plots


def plot_raw_data(df_photometry, event_times=None, suptitle=None, output_file=None):
    """
    Creates a two columns supblot with isosbestic and calcium raw traces as a function of time on the left
    and a cross-plot of the two signals on the right. Optionally saves the figure to a file
    :param df_photometry: dataframe with columns times, calcium, isosbestic
    :param event_times:
    :param suptitle:
    :param output_file:
    :return:
    """
    times = df_photometry['times'].values
    calcium = df_photometry['calcium'].values
    isosbestic = df_photometry['isosbestic'].values
    sos = scipy.signal.butter(**{'N': 3, 'Wn': 0.01, 'btype': 'lowpass'}, output='sos')
    calcium_lp = scipy.signal.sosfiltfilt(sos, calcium)
    isosbestic_lp = scipy.signal.sosfiltfilt(sos, isosbestic)
    fig, ax = plt.subplots(1, 2, figsize=(15, 6), gridspec_kw={'width_ratios': [2, 1]})
    plt.title("Entire signal, raw data")
    ax[0].plot(times, calcium, color="#279F95", linewidth=.5, label='calcium dependent')
    ax[0].plot(times, isosbestic, color='#803896', linewidth=.5, label='isosbestic')
    if event_times is not None:
        ibllib.plots.vertical_lines(
            event_times, ymin=np.min(isosbestic), ymax=np.max(calcium), ax=ax[0], alpha=.1, color='red')
    ax[0].set(xlabel='time (s)', ylabel='photometry raw trace', title='raw photometry signal')
    ax[0].plot(times, calcium_lp, color="#279F95", linewidth=2, label='calcium dependent low-pass')
    ax[0].plot(times, isosbestic_lp, color='#803896', linewidth=2, label='isosbestic low-pass')
    ax[0].legend()
    scatter = ax[1].scatter(isosbestic_lp, calcium_lp, s=None, c=times, cmap='magma', alpha=.8)
    ax[1].set(xlabel='isosbestic F', ylabel='calcium dependent F', title='Cross-plot')
    fig.colorbar(scatter, ax=ax[1], label='time (s)')
    if suptitle is not None:
        fig.suptitle(suptitle, fontsize=16)
    if output_file is not None:
        fig.savefig(output_file)
    return fig, ax