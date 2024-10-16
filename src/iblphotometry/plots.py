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
    if np.min(isosbestic) < np.min(calcium): 
        minimum_event = np.min(isosbestic)
    else: 
        minimum_event = np.min(calcium) 
    if np.max(isosbestic) < np.min(calcium): 
        maximum_event = np.max(calcium)
    else: 
        maximum_event = np.max(isosbestic)
    #TO DO REFRACTOR WITH NP.MINIMUM
    if event_times is not None:
        ibllib.plots.vertical_lines(
            # event_times, ymin=np.min(isosbestic), ymax=np.max(calcium), ax=axd['top'], alpha=.1, color='red')
            event_times, ymin=minimum_event, ymax=maximum_event, ax=axd['top'], alpha=.1, color='red')
    axd['top'].set(xlabel='time (s)', ylabel='photometry trace', title='photometry signal')
    axd['top'].legend()
    # lower left plot is the PSD of the two signals
    axd['left'].psd(calcium, Fs=1 / np.median(np.diff(times)), color="#279F95", linewidth=2, label='calcium dependent')
    axd['left'].psd(isosbestic, Fs=1 / np.median(np.diff(times)), color='#803896', linewidth=2, label='isosbestic')
    # lower right plot is the cross plot of the two signals to see if a regression makes sense
    scatter = axd['right'].scatter(isosbestic_lp, calcium_lp, s=1, c=times, cmap='magma', alpha=.8)
    axd['right'].set(xlabel='isosbestic F', ylabel='calcium dependent F', title='Cross-plot')
    fig.colorbar(scatter, ax=axd['right'], label='time (s)')
    if suptitle is not None:
        fig.suptitle(suptitle, fontsize=16)
    if output_file is not None:
        fig.savefig(output_file)
    plt.show()
    return fig, axd


def plot_psth(psth_mat, fs, axs=None, vmin=-0.01, vmax=0.01, cmap='PuOr'):
    time = np.arange(0, psth_mat.shape[0]) / fs
    if axs is None:
        _, axs = plt.subplots(2, 1)

    sns.heatmap(psth_mat.T, cbar=False, ax=axs[0], cmap=cmap, vmin=vmin, vmax=vmax)

    mean_psth = np.nanmean(psth_mat, axis=1)
    std_psth = np.nanstd(psth_mat, axis=1)
    axs[1].plot(time, mean_psth, 'k')
    axs[1].plot(time, mean_psth + std_psth, 'k--')
    axs[1].plot(time, mean_psth - std_psth, 'k--')

    return axs


def plot_psth_summary(all_psth, psth_pre, fs, df_mi, eid, pname, event, preproc_key):
    fig, axs = plt.subplots(2, 2)

    axs_in = [axs[0, 0], axs[1, 0]]
    plot_psth(all_psth, fs, axs=axs_in)

    # Compute deviation z-score
    # Average pre over time
    avg_psth_pre = np.nanmedian(psth_pre, axis=0)
    std_psth_pre = np.nanstd(psth_pre, axis=0)
    c = (all_psth - avg_psth_pre) / std_psth_pre

    axs_in = [axs[0, 1], axs[1, 1]]
    plot_psth(c, fs, vmin=-1.5, vmax=1.5, axs=axs_in)

    fig.suptitle(f'{eid[0:7]} {pname}: {df_mi["pass_tests"].values[0]}   => '
                 f'peak zscore: {df_mi["test__peak_point_zscore"].values[0]} , '
                 f'ttest: {df_mi["test__ttest_peak"].values[0]} \n'
                 f'{event}, {preproc_key}')
    return fig, axs