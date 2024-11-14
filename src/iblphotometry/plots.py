import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import iblphotometry.preprocessing as ffpr
# from iblphotometry.behavior import filter_trials_by_trial_idx
from brainbox.task.trials import find_trial_ids
from iblphotometry.behavior import psth, psth_times

LINE_COLOURS = {
    'raw_isosbestic': '#9d4edd', #purple
    'raw_signal': '#43aa8b', #teal
    'processed_signal': '#0081a7'
    # 'moving_avg': '#f4a261'
}


PSTH_EVENTS = {
    'feedback_times': 'T from Feedback (s)',
    'stimOnTrigger_times': 'T from Stim on (s)',
    'firstMovement_times': 'T from First move (s)'
}

def set_axis_style(ax, fontsize=10, **kwargs):
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    ax.set_xlabel(kwargs.get('xlabel', None), fontsize=fontsize)
    ax.set_ylabel(kwargs.get('ylabel', None), fontsize=fontsize)
    ax.set_title(kwargs.get('title', None), fontsize=fontsize+2)

    return ax

"""
------------------------------------------------
Loader objects for plotting
------------------------------------------------
"""

class PlotSignal:
    def __init__(self, raw_signal, times, raw_isosbestic=None,
                 processed_signal=None, fs=None):

        # TODO this init could change, pass in a dataframe with specific keys and LP processing done earlier
        self.raw_signal = raw_signal
        self.raw_isosbestic = raw_isosbestic
        self.processed_signal = processed_signal
        self.times = times

        # Low pass filter for plotting
        if fs is None:
            self.fs = 1 / np.nanmedian(np.diff(times))
        else:
            self.fs = fs
        self.lp_signal = ffpr.low_pass_filter(self.raw_signal, self.fs)
        if self.raw_isosbestic is not None:
            self.lp_isosbestic = ffpr.low_pass_filter(self.raw_isosbestic, self.fs)
        else:
            self.lp_isosbestic = None


    def raw_processed_figure(self):
        fig, axs = plt.subplots(3, 2)
        axs[2, 1].axis('off')

        # --- Column 0
        plot_raw_signals(self.raw_signal, self.times, self.raw_isosbestic, ax=axs[0, 0], title='Raw')
        if self.processed_signal is not None:
            plot_processed_signal(self.processed_signal, self.times, ax=axs[1, 0], title='Processed Signal')
        plot_psd(self.processed_signal, ax=axs[2, 0], title='Processed Signal PSD')
        #--- Column 1
        plot_raw_signals(self.lp_signal, self.times, self.lp_isosbestic, ax=axs[0, 1], title='Low Pass')
        if self.raw_isosbestic is not None:
            plot_photometry_correlation(self.lp_signal, self.lp_isosbestic, self.times, ax=axs[1, 1])
        fig.tight_layout()
        return fig, axs

class PlotSignalResponse():

    def __init__(self, trials, processed_signal, times, fs=None):
        self.trials = trials
        self.times = times
        self.processed_signal = processed_signal
        if fs is None:
            self.fs = 1 / np.nanmedian(np.diff(times))
        else:
            self.fs = fs
        self.psth_dict = self.compute_events_psth()

    def compute_events_psth(self, event_window=np.array([-1, 2])):

        psth_dict = dict()
        for event in PSTH_EVENTS.keys():
            psth_dict[event], _ = psth(self.processed_signal, self.times, self.trials[event],
                                                   self.fs, event_window=event_window)
            # psth_dict[event] = psth_dict[event].T

        psth_dict['times'] = psth_times(self.fs, event_window)
        return psth_dict

    def plot_trialsort_psth(self):
        fig, axs = plt.subplots(2, len(PSTH_EVENTS.keys()))

        for iaxs, event in enumerate(PSTH_EVENTS.keys()):
            axs_plt = [axs[0, iaxs],
                       axs[1, iaxs]]
            plot_psth(self.psth_dict[event], self.psth_dict['times'], axs=axs_plt, title=event)

            if iaxs == 0:
                axs[0, iaxs].set_xlabel('Frames')
                axs[0, iaxs].set_ylabel('Trials')
                axs[1, iaxs].set_xlabel('Time (s)')
            if iaxs > 0:
                axs[0, iaxs].axis('off')
                axs[1, iaxs].set_yticks([])
        fig.tight_layout()
        return fig, axs
"""
------------------------------------------------
Plotting functions requiring FF signals only
------------------------------------------------
"""

def plot_raw_signals(raw_signal, times, raw_isosbestic=None,
                     ax=None, xlim=None, ylim=None, xlabel='Time', ylabel=None, title=None):

        if ax is None:
            fig, ax = plt.subplots(1, 1)
        else:
            fig = ax.get_figure()

        linewidth = 0.1 if xlim is None else 1
        # Plot signal
        ax.plot(times, raw_signal, linewidth=linewidth,
                c=LINE_COLOURS['raw_signal'], label='signal')
        # Plot isosbestic if passed in
        if raw_isosbestic is not None:
            ax.plot(times, raw_isosbestic, linewidth=linewidth,
                    c=LINE_COLOURS['raw_isosbestic'], label='isosbestic')
        ax.legend(fontsize=6)

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        set_axis_style(ax, xlabel=xlabel, ylabel=ylabel, title=title)
        ax.tick_params(axis='both', which='major')
        return fig, ax



def plot_processed_signal(signal, times, ax=None, xlim=None, ylim=None,
                          xlabel='Time', ylabel=None, title=None):

    if ax is None:
        fig, ax = plt.subplots(1, 1)
    else:
        fig = ax.get_figure()

    linewidth = 0.1 if xlim is None else 1
    col = LINE_COLOURS['processed_signal']
    # Plot signal over time
    ax.plot(times, signal, linewidth=linewidth, c=col)
    set_axis_style(ax, xlabel=xlabel, ylabel=ylabel, title=title)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.tick_params(axis='both', which='major')

    return fig, ax



def plot_photometry_correlation(signal_lp, isosbestic_lp, times, ax=None, ax_cbar=None, title=None):
    # Requires the Low pass filtered signals at minima
    if ax is None:
        fig, ax = plt.subplots(1, 1)
    else:
        fig = ax.get_figure()

    if ax_cbar is None:
        ax_cbar = ax

    scat = ax.scatter(isosbestic_lp, signal_lp, s=1, c=times,
                      cmap='magma', alpha=.8)
    set_axis_style(ax, xlabel='isobestic', ylabel='signal', title=title)
    fig.colorbar(scat, ax=ax_cbar, orientation='horizontal', label='Time in session (s)',
                 shrink=0.3, anchor=(0.0, 1.0), location='top')

    return fig, ax


def plot_psd(signal, ax=None, title=None, **line_kwargs):
    if ax is None:
        fig, ax = plt.subplots(1, 1)
    else:
        fig = ax.get_figure()
    line_kwargs.setdefault('linewidth', 2)
    ax.psd(signal, **line_kwargs)
    # TODO the freq x-axis is currently not informative
    ax.set_title(title)

    return fig, ax

"""
------------------------------------------------
Plotting functions requiring behavioral events
------------------------------------------------
"""

def plot_psth(psth_mat, time, axs=None, vmin=-0.01, vmax=0.01, cmap='PuOr', title=None):
    # if time is None:
    #     time = np.arange(0, psth_mat.shape[0]) / fs
    if axs is None:
        fig, axs = plt.subplots(2, 1)
    else:
        fig = axs[0].get_figure()

    sns.heatmap(psth_mat.T, cbar=False, ax=axs[0], cmap=cmap, vmin=vmin, vmax=vmax)
    axs[0].set_title(title)

    mean_psth = np.nanmean(psth_mat, axis=1)
    std_psth = np.nanstd(psth_mat, axis=1)
    axs[1].plot(time, mean_psth, 'k')
    axs[1].plot(time, mean_psth + std_psth, 'k--')
    axs[1].plot(time, mean_psth - std_psth, 'k--')

    return fig, axs


def plot_event_tick(events, ax=None, labels=None, color=None, ls='--'):
    if ax is None:
        fig, ax = plt.subplots(1, 1)
    else:
        fig = ax.get_figure()

    if color is None:
        color = 'k'

    ax.vlines(events, *ax.get_ylim(), color=color, ls=ls, zorder=ax.get_zorder() + 1)

    # TODO this part does not work
    # if labels is not None:
    #     ax.text(events, 1.01, labels, c=color, rotation=45,
    #             rotation_mode='anchor', ha='left', transform=ax.get_xaxis_transform())

    return fig, ax

def plot_iblevents_tick(ax, trials):
    '''

    Parameters
    ----------
    ax
    trials: Bunch object in ALF format
    text

    Returns
    -------

    '''

    events = ['stimOnTrigger_times', 'firstMovement_times', 'feedback_times']
    colors = ['b', 'g', 'r']
    labels = ['Stim On', 'First Move', 'Feedback']

    for e, c, l in zip(events, colors, labels):
        plot_event_tick(events=trials[e], ax=ax, labels=l, color=c)

    return ax



# from brainbox.task.trials import find_trial_ids
# trial_idx, dividers = find_trial_ids(trials, sort='choice')

'''
def plot_left_right_psth(psth, trials, ax=None, xlabel='T from Feedback (s)',
                           ylabel0='Signal', ylabel1='Sorted Trial Number',
                           order='trial num'):

    trial_idx, dividers = find_trial_ids(trials, sort='side', order=order)
    colours = ['g', 'y']
    labels = ['left', 'right']

    fig, ax = plot_processed_psth(psth, trial_idx, dividers, colours, labels,
                                   ax=ax)
    set_axis_style(ax,  xlabel=xlabel, ylabel=ylabel0)

    set_axis_style(axs[1], xlabel=xlabel, ylabel=ylabel1)
    set_axis_style(axs[0], ylabel=ylabel0)

    return fig


def plot_processed_psth(psth, psth_times, trial_idx, dividers, colors, labels, ax=None):

    dividers = [0] + dividers + [len(trial_idx)]

    if ax is None:
        fig, ax = plt.subplots(1, 1)
    else:
        fig = ax.get_figure()

    label, lidx = np.unique(labels, return_index=True)
    for lab, lid in zip(label, lidx):
        idx = np.where(np.array(labels) == lab)[0]
        for iD in range(len(idx)):
            if iD == 0:
                t_ids = trial_idx[dividers[idx[iD]] + 1:dividers[idx[iD] + 1] + 1]
                t_ints = dividers[idx[iD] + 1] - dividers[idx[iD]]
            else:
                t_ids = np.r_[t_ids, trial_idx[dividers[idx[iD]] + 1:dividers[idx[iD] + 1] + 1]]
                t_ints = np.r_[t_ints, dividers[idx[iD] + 1] - dividers[idx[iD]]]

        psth_div = np.nanmean(psth[t_ids], axis=0)

        ax.plot(psth_times, psth[t_ids].T, alpha=0.01, color=colors[lid])
        ax.plot(psth_times, psth_div, alpha=1, color=colors[lid], zorder=t_ids.size + 10)

    remove_spines(ax, spines=['right', 'top'])

    return fig, ax


def remove_spines(ax, spines=('left', 'right', 'top', 'bottom')):
    for sp in spines:
        ax.spines[sp].set_visible(False)

    return ax
'''
