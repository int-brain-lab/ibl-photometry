# %%
import numpy as np
import pandas as pd

from one.api import ONE
from iblphotometry import fpio, preprocessing, analysis, processing, pipelines,

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.axes import Axes

import pynapple as nap
from pynapple import TsdFrame

from brainbox.io.one import PhotometrySessionLoader


# plotting helpers
def _time_base_to_div(time_base: str):
    match time_base:
        case 'h':
            time_div = 3600
        case 'min':
            time_div = 60
        case 's':
            time_div = 1
    return time_div


# plotting primitives
def plot_photometry_trace(
    signal: pd.Series,
    time_base: str = 'min',
    axes: Axes | None = None,
    **plot_kwargs,
):
    if axes is None:
        fig, axes = plt.subplots()
        sns.despine(fig)

    time_div = _time_base_to_div(time_base)

    plot_kwargs.setdefault('lw', 1)
    axes.plot(
        signal.index.values / time_div,
        signal.values,
        **plot_kwargs,
    )
    axes.set_xlabel(f'time ({time_base})')
    axes.set_ylabel('fluorescence (au)')
    return axes


def plot_photometry_traces(
    signals: pd.Series | pd.DataFrame,
    time_base: str = 'min',
    axes: Axes | None = None,
    legend: bool = True,
    **plot_kwargs,
) -> Axes:
    if axes is None:
        fig, axes = plt.subplots()
        sns.despine(fig)

    brain_regions = signals.columns
    for brain_region in brain_regions:
        axes = plot_photometry_trace(
            signals[brain_region],
            time_base=time_base,
            axes=axes,
            **plot_kwargs,
        )

    if legend:
        axes.legend()
    return axes


def plot_photometry_bands(
    raw_dfs: dict,
    time_base: str = 'min',
):
    fig, axes = plt.subplots(nrows=len(raw_dfs), sharex=True)
    sns.despine(fig)

    for i, band in enumerate(raw_dfs.keys()):
        plot_photometry_traces(
            raw_dfs[band],
            time_base=time_base,
            axes=axes[i],
        )
        axes[i].set_ylabel(f'{band}\nfluorescence (au)')
    return axes


def add_session_time_to_axes(
    axes: Axes,
    trials_df: pd.DataFrame,
    event: str = 'feedback_times',
    time_base: str = 'min',
):
    time_div = _time_base_to_div(time_base)

    # the session
    t_start = trials_df.iloc[0]['intervals_0'] / time_div
    t_stop = trials_df.iloc[-1]['intervals_1'] / time_div
    axes.axvspan(
        t_start,
        t_stop,
        color='gray',
        alpha=0.2,
        zorder=-1,
    )

    # the events
    for t in trials_df[event].values:
        axes.axvline(t / time_div, color='k', lw=0.5)

    return axes


def plot_psth(
    psth: TsdFrame,
    axes: Axes | None = None,
    **matshow_kwargs,
):
    if axes is None:
        _, axes = plt.subplots()

    values = psth.values.flatten()
    matshow_kwargs.setdefault('vmin', np.percentile(values, 0.05))
    matshow_kwargs.setdefault('vmax', np.percentile(values, 99.5))
    matshow_kwargs.setdefault('cmap', 'magma')

    n_trials = psth.shape[1]
    axes.matshow(psth.values.T, extent=(psth.t[0], psth.t[-1], n_trials, 0), **matshow_kwargs)
    axes.set_aspect('auto')
    axes.grid(False)
    axes.axvline(0, color='w', lw=0.5)
    axes.set_xlabel('time (s)')
    axes.set_ylabel('trials')

    return axes


def plot_psths(
    psths: dict,
    split_by: str,
    align_on: str | None = None,
    **matshow_kwargs,
):
    outcomes = psths.keys()
    # outcomes = np.sort(trials_df[split_by].unique())
    # n_trials_per_group = dict([(i, t.shape[0]) for i, t in trials_df.groupby(split_by)])

    fig, axes = plt.subplots(
        nrows=len(outcomes),
        gridspec_kw=dict(height_ratios=[psths[o].shape[1] for o in outcomes]),
        sharex=True,
    )
    values = np.concatenate([psths[o].values.flatten() for o in outcomes])
    matshow_kwargs.setdefault('vmin', np.percentile(values, 0.05))
    matshow_kwargs.setdefault('vmax', np.percentile(values, 99.5))
    matshow_kwargs.setdefault('cmap', 'magma')

    for i, outcome in enumerate(outcomes):
        psth = psths[outcome]
        axes[i] = plot_psth(psth, axes=axes[i], **matshow_kwargs)
        axes[i].set_ylabel(f'{split_by}={outcome}')
        axes[i].grid(False)

    if align_on is not None:
        fig.suptitle(f'aligned on {align_on}')

    return axes


# complete plotters
def plot_photometry_df_from_eid(
    eid: str,
    one: ONE,
    channels: list[str] = ['GCaMP'],
    preprocess: bool = True,
):
    raw_df = one.load_dataset(eid, 'raw_photometry_data/_neurophotometrics_fpData.raw.pqt')
    photometry_df = fpio.from_neurophotometrics_df_to_photometry_df(raw_df)

    if preprocess:
        gaps = preprocessing.find_gaps(photometry_df)
        photometry_df = preprocessing.fill_gaps(photometry_df, gaps)

    if channels is None:
        channels = photometry_df['name'].unique()

    data_columns = fpio._infer_data_columns(photometry_df)
    for channel in channels:
        df = photometry_df.groupby('name').get_group(channel)
        fig, axes = plt.subplots()
        for col in data_columns:
            axes.plot(df['times'].values, df[col].values, label=col)
        axes.set_title(channel)
        axes.legend()
        axes.set_xlabel('time (s)')
        axes.set_ylabel('fluorescence (au)')
    sns.despine(fig)


def plot_photometry_traces_from_eid(
    eid: str,
    one: ONE,
):
    psl = PhotometrySessionLoader(eid=eid, one=one)
    psl.load_photometry()

    raw_dfs = psl.photometry
    trials_df = psl.trials

    axes = plot_photometry_bands(raw_dfs)
    for ax in axes:
        ax = add_session_time_to_axes(ax, trials_df)

    return axes


def plot_psths_from_trace(
    signal: pd.Series,
    trials_df: pd.DataFrame,
    split_by: str = 'feedbackType',
    align_on: str = 'feedback_times',
    axes: Axes | None = None,
):
    # if axes is None:
    # _, axes = plt.subplots()

    # cast to pynapple
    signal = nap.Tsd(signal.index, signal.values)
    psths = analysis.psth_nap(
        signal,
        trials_df,
        split_by=split_by,
        align_on=align_on,
    )
    axes = plot_psths(psths, split_by=split_by, align_on=align_on)
    return axes


def plot_psths_from_eid(
    eid: str,
    one: ONE,
    channel: str = 'GCaMP',
    split_by: str = 'feedbackType',
    align_on: str = 'feedback_times',
    pipeline: dict = pipelines.sliding_mad_pipeline,
):
    psl = PhotometrySessionLoader(eid=eid, one=one)
    psl.load_photometry()

    raw_df = psl.photometry[channel]
    trials_df = psl.trials

    brain_regions = raw_df.columns
    for brain_region in brain_regions:
        # run pipeline
        signal = pipelines.run_pipeline(pipeline, raw_df[[brain_region]])
        # and plot
        axes = plot_psths_from_trace(signal, trials_df, split_by=split_by, align_on=align_on)
        # add the brain region to the title
        fig = axes[0].figure
        fig.suptitle(f'{brain_region}: {fig.get_suptitle()}')
    return axes


# eid = '7c67fbd4-18c1-42f2-b989-8cbfde0d2374'  # looks highly problematic
# eid = '40909756-d0ce-4146-9588-249bf97f074b'
# eid = '58861dac-4b4c-4f82-83fb-33d98d67df3a'
# one = ONE()
# axes = plot_photometry_traces_from_eid(eid, one)
