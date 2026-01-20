#!/usr/bin/env python3
"""
Photometry Dataset QC Viewer

This script provides an interactive viewer for photometry dataset quality control (QC).
It loads a QC overview CSV file (containing eids and QC labels), visualizes photometry traces,
and allows the user to label datasets as "PASS", "FAIL", "CRITICAL", or unset via keyboard shortcuts.

Usage:
    python QC_manual_gui.py --qc-csv /path/to/qc_overview_table.csv [--index 0] [--restrict-to LABEL]

Arguments:
    --qc-csv        Path to the QC overview CSV file containing eids and QC labels.
    --index         (Optional) Integer index to start viewing from. Default is 0.
    --restrict-to   (Optional) Restrict navigation to datasets with a specific label (e.g., "FAIL", "PASS", etc.).

Example:
    python 01_qc_app_dev.py --qc-csv qc_overview_table_merged2.csv --index 10 --restrict-to PASS

Keyboard Shortcuts:
    Right Arrow   : Next dataset
    Left Arrow    : Previous dataset
    0             : Unset label
    1             : Set label to CRITICAL
    2             : Set label to FAIL
    3             : Set label to PASS
    e             : Mark as extraction_issue
    x             : Save current progress to CSV
    n             : Enter a comment for the current dataset

"""

import argparse
import enum
from pathlib import Path

import numpy as np
import pandas as pd
import pynapple as nap
import matplotlib.pyplot as plt

from iblphotometry import fpio, pipelines, processing, analysis, plotters
from iblphotometry.validation import PhotometryDataValidator

from one.api import ONE

on_error = 'raise'  # or 'log'


# class QCLabel(enum.Enum):
# UNSET = 0
# CRITICAL = 1
# FAIL = 2
# PASS = 3


QC_COLORS = {
    '1': '#EF6F6C',
    '2': '#fcf15d',
    '3': '#56E370',
    '0': '#FFFFFF',
}


class PhotometryDatasetQCViewer:
    def __init__(
        self,
        eids_file: str,
        index: int | None = None,
        restrict_to_label: str | None = None,
        one=None,
    ):
        self.one = ONE() if one is None else one

        # load the eids from an eid file, and generate the output csv
        self.QC_df = self._load_or_generate_QC_df(eids_file)

        #
        self.restrict_to_label = restrict_to_label
        self.current_index = index
        self.one = ONE()

        # Initial plot
        self.load_data()
        self.init_plot()
        self.update_plot()

    def _load_or_generate_QC_df(self, eids_file: Path):
        # get eids from file
        self.qc_file = eids_file.with_suffix('.qc.csv')
        if not self.qc_file.exists():
            with open(eids_file, 'r') as fH:
                eids = [eid.strip() for eid in fH.readlines()]
            validator = PhotometryDataValidator()

            df = pd.DataFrame(index=eids)
            df.index.name = 'eid'
            df = validator.validate_dataframe(df)
            QC_df = df.reset_index()
            QC_df = self.expand_eids_brain_regions(QC_df)
            QC_df['label'] = '0'
            QC_df.to_csv(self.qc_file)
        else:
            QC_df = pd.read_csv(self.qc_file, index_col=0, na_filter=False)
            QC_df['label'] = QC_df['label'].astype(str)

        return QC_df

    def expand_eids_brain_regions(self, df: pd.DataFrame) -> pd.DataFrame:
        # add brain region to the rows, duplicate eids when necessary
        df_ = []
        for i, row in df.iterrows():
            locations = self.one.load_dataset(row['eid'], '*photometryROI.locations')
            for brain_region in locations['brain_region'].values:
                row['brain_region'] = brain_region
                df_.append(row)
        return pd.DataFrame(df_)

    def load_data(self):
        # load data based on current index
        eid = self.QC_df.loc[self.current_index, 'eid']
        self.data = fpio.from_eid(eid, one=self.one)
        self.trials_table = self.one.load_dataset(eid, '*trials.table')
        self.data = fpio.restrict_to_session(self.data, self.trials_table)

        # resample
        self.data = processing.resample(self.data)

    def init_plot(self):
        self.fig, self.axes = plt.subplots(ncols=3, nrows=2, figsize=(10, 6))
        self.fig.canvas.mpl_connect('key_press_event', self._on_key_press)

    def update_figure_title(self):
        current = self.QC_df.loc[self.current_index]
        self.fig.suptitle(f'{self.current_index} - {current["eid"]} - {current["brain_region"]} - {current["label"]}')

    def update_plot(self):
        try:
            for ax in self.axes.flatten():
                ax.clear()

            self.load_data()

            brain_region = self.QC_df.loc[self.current_index, 'brain_region']
            plotters.plot_photometry_trace(self.data['GCaMP'][brain_region], axes=self.axes[0, 0])
            plotters.plot_photometry_trace(self.data['Isosbestic'][brain_region], axes=self.axes[0, 1])

            # minimal processing
            signal_processed_gc = pipelines.run_pipeline(pipelines.sliding_mad_pipeline, self.data['GCaMP'][brain_region])
            plotters.plot_photometry_trace(signal_processed_gc, axes=self.axes[1, 0])

            signal_processed_iso = pipelines.run_pipeline(pipelines.sliding_mad_pipeline, self.data['Isosbestic'][brain_region])
            plotters.plot_photometry_trace(signal_processed_iso, axes=self.axes[1, 1])

            # draw trials
            for t in self.trials_table['intervals_0']:
                self.axes[0, 0].axvline(t / 60, zorder=-1, color='k', alpha=0.2)
                self.axes[1, 0].axvline(t / 60, zorder=-1, color='k', alpha=0.2)
                self.axes[0, 1].axvline(t / 60, zorder=-1, color='k', alpha=0.2)
                self.axes[1, 1].axvline(t / 60, zorder=-1, color='k', alpha=0.2)

            split_by = 'feedbackType'
            align_on = 'feedback_times'

            # cast to pynapple
            signal = signal_processed_gc

            signal = nap.Tsd(signal.index, signal.values)
            psths = analysis.psth_nap(
                signal,
                self.trials_table,
                split_by=split_by,
                align_on=align_on,
            )

            outcome_axes_map = {1: 0, -1: 1}
            for outcome in [1, -1]:
                psth = psths[outcome]
                if psth.shape[1] > 0:
                    plotters.plot_psth(psth, axes=self.axes[outcome_axes_map[outcome], 2])

            self.update_figure_title()
            self.update_plot_label()
            plt.draw()

        except Exception as e:
            if on_error == 'raise':
                raise e
            self.QC_df.loc[self.current_index, 'error'] = f'{type(e).__name__}:{e}'

    def update_plot_label(self):
        # set the background color
        label = self.QC_df.loc[self.current_index, 'label']
        self.fig.patch.set_facecolor(QC_COLORS[label])
        self.update_figure_title()
        plt.draw()

    def get_next_index(
        self,
        restrict_to_label: str | None = None,
        skip_invalid: bool = True,
    ) -> int:
        """
        Load the next dataset. restrict_to_label kw restricts
        """
        if skip_invalid:
            df = self.QC_df.loc[self.QC_df['validation'] == '']
        else:
            df = self.QC_df
        if restrict_to_label is not None:
            df = df.groupby('label').get_group(restrict_to_label)

        # get the first index that is larger than the current
        return df.index[np.argmax(df.index > self.current_index)]

    def get_previous_index(self) -> int:
        # TODO implement the same logic as above
        return self.current_index - 1

    def _on_key_press(self, event):
        """Handle keyboard events."""
        if event.key == 'right':
            self.current_index = self.get_next_index(restrict_to_label=self.restrict_to_label)
            print(self.current_index)
            self.update_plot()

        elif event.key == 'left':
            self.current_index = self.get_previous_index()
            print(self.current_index)
            self.update_plot()

        if event.key.isdigit():
            # label the current row
            # label = QCLabel(event.key)
            label = event.key
            self.QC_df.loc[self.current_index, 'label'] = label
            print(f'label set to {label}')
            self.update_plot_label()

        elif event.key == 'x':
            print('saving current progress')
            self.QC_df.to_csv(self.qc_file, index=True)

        elif event.key == 'n':
            print('enter comment')
            self.QC_df.loc[self.current_index, 'comment'] = input()

    def show(self):
        """Display the plot window."""
        plt.show()


def main():
    parser = argparse.ArgumentParser(description='Interactive photometry dataset QC viewer.')
    parser.add_argument('--eids-file', type=str, required=True, help='Path to file containing line-separated eids to QC.')
    parser.add_argument('--index', type=int, default=0, required=False, help='Start index.')
    parser.add_argument(
        '--restrict-to-label',
        type=str,
        default=None,
        required=False,
        help="Restrict navigation to datasets with a specific label (e.g., 'FAIL', 'PASS').",
    )
    args = parser.parse_args()

    path = Path(args.eids_file)
    index = args.index
    restrict_to_label = args.restrict_to_label

    viewer = PhotometryDatasetQCViewer(path, index=index, restrict_to_label=restrict_to_label)
    viewer.show()


if __name__ == '__main__':
    main()
