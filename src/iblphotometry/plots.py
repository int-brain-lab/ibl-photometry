import numpy as np
import pandas as pd
import pynapple as nap
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from matplotlib import ticker

import ibllib.plots
from iblphotometry.helpers import filt

# TODO decorators for saving figures
# TODO decorators for generating axes or getting axes passed as a kw


# def saveable(func):
#     def wrapper(*args, **kwargs):
#         if 'output_file' in kwargs:
#             axes = func(*args, **kwargs)
#             plt.gcf().savefig(kwargs['output_file'])
#             # plt.close(fig)
#         else:
#             axes = func(*args, **kwargs)
#         return axes

#     return wrapper


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
    # todo deprecate, this is not in line with the current data model
    sns.set_style('whitegrid')
    tf_photometry = nap.TsdFrame(df_photometry).set_index('times')
    return plot_raw_data_tf(tf_photometry, **kwargs)


def plot_raw_data_tf(tf_photometry, **kwargs):
    sns.set_style('whitegrid')
    raw_isosbestic = (
        tf_photometry['raw_isosbestic']
        if 'raw_isosbestic' in tf_photometry.columns
        else None
    )
    return plot_photometry_traces(
        times=tf_photometry.times(),
        calcium=tf_photometry['raw_calcium'],
        isosbestic=raw_isosbestic,
        **kwargs,
    )


# @saveable
def plot_Tsd(signal: nap.Tsd, axes=None, **line_kwargs):
    if axes is None:
        _, axes = plt.subplots()

    line_kwargs.setdefault('linewidth', 0.5)
    axes.plot(signal, **line_kwargs)
    axes.set_xlabel('time (s)')
    axes.set_ylabel('signal (au)')

    return axes


def plot_TsdFrame(signal: nap.TsdFrame, axes=None):
    if axes is None:
        _, axes = plt.subplots()

    for col in signal.columns:
        plot_Tsd(signal[col], axes=axes, label=col)
    axes.legend()

    return axes


def plot_psd_Tsd(signal: nap.Tsd, fs=None, axes=None, **line_kwargs):
    if axes is None:
        _, axes = plt.subplots()

    if fs is None:
        fs = 1 / np.median(np.diff(signal.t))

    line_kwargs.setdefault('linewidth', 2)
    axes.psd(signal.values, **line_kwargs)

    return axes
    # color = ('#279F95',)


def plot_isosbestic_overview(
    calcium: nap.Tsd | nap.TsdFrame,
    isosbestic: nap.Tsd | nap.TsdFrame,
    low_pass_cross_plot=0.01,
    suptitle=None,
    output_file=None,
):
    fig, axd = plt.subplot_mosaic(
        [['top', 'top'], ['left', 'right']], constrained_layout=True, figsize=(14, 8)
    )
    # traces
    plot_Tsd(calcium, axes=axd['top'], color='#279F95', label='calcium')
    plot_Tsd(isosbestic, axes=axd['top'], color='#803896', label='isosbestic')

    axd['top'].set(
        xlabel='time (s)', ylabel='photometry trace', title='photometry signal'
    )
    axd['top'].legend()

    # PSDs
    plot_psd_Tsd(calcium, axes=axd['left'], color='#279F95', label='calcium')
    plot_psd_Tsd(isosbestic, axes=axd['left'], color='#803896', label='isosbestic')
    axd['left'].legend()

    # scatter
    if low_pass_cross_plot:
        filter_params = dict(N=3, Wn=low_pass_cross_plot, btype='lowpass')
        calcium_lp = filt(calcium, **filter_params)
        isosbestic_lp = filt(isosbestic, **filter_params)

    # lower right plot is the cross plot of the two signals to see if a regression makes sense
    scatter = axd['right'].scatter(
        isosbestic_lp.values,
        calcium_lp.values,
        s=1,
        c=calcium_lp.times(),
        cmap='magma',
        alpha=0.8,
    )
    axd['right'].set(
        xlabel='isosbestic signal',
        ylabel='calcium dependent signal',
        title='Cross-plot',
    )
    fig.colorbar(scatter, ax=axd['right'], label='time (s)')

    if suptitle is not None:
        fig.suptitle(suptitle, fontsize=16)
    if output_file is not None:
        fig.savefig(output_file)
    # plt.show()
    return fig, axd


def plot_photometry_traces(
    times: np.ndarray,
    signal: np.ndarray,
    reference_signal=None,
    event_times=None,
    suptitle=None,
    output_file=None,
    low_pass_cross_plot=0.01,
):
    reference_signal = signal * np.nan if reference_signal is None else reference_signal
    if low_pass_cross_plot:
        filter_params = dict(N=3, Wn=0.01, btype='lowpass')
        calcium_lp = filt(signal, **filter_params)
        isosbestic_lp = filt(reference_signal, **filter_params)
    else:
        calcium_lp, isosbestic_lp = (signal, reference_signal)
    # start the plotting functions, first the raw signals in time domain
    fig, axd = plt.subplot_mosaic(
        [['top', 'top'], ['left', 'right']], constrained_layout=True, figsize=(14, 8)
    )
    axd['top'].plot(
        times,
        reference_signal,
        color='#803896',
        linewidth=0.5,
        label='reference_signal',
    )
    axd['top'].plot(
        times, signal, color='#279F95', linewidth=0.5, label='calcium dependent'
    )
    if np.min(reference_signal) < np.min(signal):
        minimum_event = np.min(reference_signal)
    else:
        minimum_event = np.min(signal)
    if np.max(reference_signal) < np.min(signal):
        maximum_event = np.max(signal)
    else:
        maximum_event = np.max(reference_signal)
    # TO DO REFRACTOR WITH NP.MINIMUM
    if event_times is not None:
        ibllib.plots.vertical_lines(
            # event_times, ymin=np.min(reference_signal), ymax=np.max(signal), ax=axd['top'], alpha=.1, color='red')
            event_times,
            ymin=minimum_event,
            ymax=maximum_event,
            ax=axd['top'],
            alpha=0.1,
            color='red',
        )
    axd['top'].set(
        xlabel='time (s)', ylabel='photometry trace', title='photometry signal'
    )
    axd['top'].legend()
    # lower left plot is the PSD of the two signals
    axd['left'].psd(
        signal,
        Fs=1 / np.median(np.diff(times)),
        color='#279F95',
        linewidth=2,
        label='signal',
    )
    axd['left'].psd(
        reference_signal,
        Fs=1 / np.median(np.diff(times)),
        color='#803896',
        linewidth=2,
        label='reference_signal',
    )
    # lower right plot is the cross plot of the two signals to see if a regression makes sense
    scatter = axd['right'].scatter(
        isosbestic_lp, calcium_lp, s=1, c=times, cmap='magma', alpha=0.8
    )
    axd['right'].set(xlabel='reference signal', ylabel='signal', title='Cross-plot')
    fig.colorbar(scatter, ax=axd['right'], label='time (s)')
    if suptitle is not None:
        fig.suptitle(suptitle, fontsize=16)
    if output_file is not None:
        fig.savefig(output_file)
    plt.show()
    return fig, axd


# plot psths
from iblphotometry.helpers import psth

# event = 'feedback_times'
# split_by = 'feedbackType'
# split_by = 'choice'
# region = 'Region0G'


def plot_raster(
    F: nap.Tsd, trials: pd.DataFrame, event: str = None, split_by: str = None
):
    splits = {}
    for i, times in trials.groupby(split_by)[event]:
        splits[i] = times

    n_per_split = [v.shape[0] for _, v in splits.items()]
    vmin, vmax = np.percentile(F, (1, 99))
    w_start, w_stop = -2, 2

    fig, axes = plt.subplots(
        nrows=len(splits), gridspec_kw=dict(height_ratios=n_per_split)
    )
    for i, (label, times) in enumerate(splits.items()):
        p, ix = psth(
            F.values,
            F.times(),
            times,
            peri_event_window=(w_start, w_stop),
        )
        axes[i].matshow(
            p.T,
            origin='lower',
            extent=(w_start, w_stop, 0, p.shape[1]),
            vmin=vmin,
            vmax=vmax,
        )
        axes[i].set_aspect('auto')
        axes[i].axvline(0, lw=1, color='w')
        axes[i].set_ylabel(f'{split_by}={label}')
        axes[i].xaxis.set_ticks_position('bottom')
        # axes[i].yaxis.set_major_locator(ticker.MultipleLocator(20))
        if i < len(splits) - 1:
            axes[i].set_xticklabels('')
    axes[-1].set_xlabel('time (s)')
    fig.subplots_adjust(hspace=0.1)
    return axes
