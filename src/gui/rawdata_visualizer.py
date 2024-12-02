import sys
import pandas as pd
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QFileDialog, QTableWidget, \
    QTableWidgetItem, QComboBox, QGridLayout, QLineEdit
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5 import NavigationToolbar2QT as NavigationToolbar
from pydantic.v1 import NoneStr

from iblphotometry.io import from_raw_neurophotometrics
import iblphotometry.plots as plots

import iblphotometry.preprocessing as ffpr
import numpy as np
from matplotlib.colorbar import Colorbar


class DataFrameVisualizerApp(QWidget):
    def __init__(self):
        super().__init__()

        self.df = None  # Original DataFrame
        self.times = None
        self.dfiso = None  # Isosbestic Dataframe

        self.plot_time_index = None

        self.filtered_df = None  # Filtered DataFrame used for plotting only
        self.init_ui()

    def init_ui(self):
        # Create layout
        main_layout = QVBoxLayout()

        # Layout for file loading and selection
        file_layout = QHBoxLayout()
        self.load_button = QPushButton('Load File', self)
        self.load_button.clicked.connect(self.load_file)
        file_layout.addWidget(self.load_button)

        self.column_selector = QComboBox(self)
        self.column_selector.currentIndexChanged.connect(self.update_plots)
        file_layout.addWidget(self.column_selector)

        # Add filter dropdown menu
        self.filter_selector = QComboBox(self)
        self.filter_selector.addItem('Select Filter')
        self.filter_selector.addItem('Filter MAD')
        # self.filter_selector.addItem("Filter CAD")
        # self.filter_selector.addItem("Filter JOVE")
        self.filter_selector.currentIndexChanged.connect(self.apply_filter)
        file_layout.addWidget(self.filter_selector)

        # # Table widget to display DataFrame
        # self.table = QTableWidget(self)
        # self.table.setSelectionMode(QTableWidget.SingleSelection)
        # self.table.setSelectionBehavior(QTableWidget.SelectColumns)
        # self.table.horizontalHeader().sectionClicked.connect(self.on_column_header_clicked)
        #
        main_layout.addLayout(file_layout)
        # main_layout.addWidget(self.table)

        # Input boxes for time range
        time_layout = QHBoxLayout()
        self.start_time_edit = QLineEdit(self)
        self.start_time_edit.setPlaceholderText('Start Time (float)')
        self.end_time_edit = QLineEdit(self)
        self.end_time_edit.setPlaceholderText('End Time (float)')
        time_layout.addWidget(self.start_time_edit)
        time_layout.addWidget(self.end_time_edit)
        # Button to apply time range filter
        self.apply_button = QPushButton('Apply Time Range', self)
        self.apply_button.clicked.connect(self.apply_time_range)
        main_layout.addLayout(time_layout)
        main_layout.addWidget(self.apply_button)


        # Set up plots layout
        self.plot_layout = QGridLayout()
        self.plotobj = plots.PlotSignal()
        self.figure, self.axes = self.plotobj.set_fig_layout2()
        # self.figure, self.axes = plt.subplots(1, 3, figsize=(15, 5))
        self.canvas = FigureCanvas(self.figure)
        self.plot_layout.addWidget(self.canvas, 0, 0, 1, 3)
        # Create a NavigationToolbar
        self.toolbar = NavigationToolbar(self.canvas, self)

        main_layout.addLayout(self.plot_layout)
        self.setLayout(main_layout)

        self.setWindowTitle('DataFrame Plotter')
        self.setGeometry(300, 100, 800, 600)

    def load_file(self):
        # Open a file dialog to choose the CSV or PQT file
        file_path, _ = QFileDialog.getOpenFileName(
            self, 'Open File', '', 'CSV and PQT Files (*.csv *.pqt);;All Files (*)'
        )
        if file_path:
            # Load the file into a DataFrame based on its extension
            try:
                if (
                    file_path.endswith('.csv')
                    or file_path.endswith('.pqt')
                    or file_path.endswith('.parquet')
                ):
                    self.td = from_raw_neurophotometrics(file_path)
                else:
                    raise ValueError('Unsupported file format')

                if 'GCaMP' in self.td.keys():
                    self.df = self.td['GCaMP'].as_dataframe()
                    self.times = self.td['GCaMP'].t
                    self.plot_time_index = np.arange(0, len(self.times))
                    self.filtered_df = None
                else:
                    raise ValueError('No GCaMP found')

                if 'Isosbestic' in self.td.keys():
                    self.dfiso = self.td['Isosbestic'].as_dataframe()

                # Display the dataframe in the table
                # self.display_dataframe()
                # Update the column selector
                self.update_column_selector()

                # Load into Pynapple dataframe
                self.td = from_raw_neurophotometrics(file_path)

                # Set filter combo box
                self.filter_selector.setCurrentIndex(0)  # Reset to "Select Filter"

            except Exception as e:
                print(f'Error loading file: {e}')

    # TODO this does not work with pynapple as format, convert back to pandas DF
    # def display_dataframe(self):
    #     if self.df is not None:
    #         # Update the table to display the original dataframe
    #         self.table.setRowCount(len(self.df))
    #         self.table.setColumnCount(len(self.df.columns))
    #         self.table.setHorizontalHeaderLabels(self.df.columns)

    # for row in range(len(self.df)):
    #     for col in range(len(self.df.columns)):
    #         self.table.setItem(row, col, QTableWidgetItem(str(self.df.iloc[row, col])))

    def apply_time_range(self):
        """Apply the time range filter and update the plot."""
        start_time_str = self.start_time_edit.text()
        end_time_str = self.end_time_edit.text()
        try:
            # Convert the start and end time inputs to numbers (float or int)

            if start_time_str == '':
                start_time = 0
            else:
                start_time = float(start_time_str)

            if end_time_str == '':
                end_time = self.times[len(self.times)-1]
            else:
                end_time = float(end_time_str)
            # Filter dataframe based on user input
            indx_time = (self.times >= start_time) & (self.times <= end_time)


            if len(indx_time) == 0:
                print("No data in the specified range.")
            else:
                self.plot_time_index = indx_time
                self.update_plots()
        except ValueError:
            print("Invalid time format. Please enter a valid time point in the format of a float.")

    def update_column_selector(self):
        if self.df is not None:
            # Populate the column selector with column names from the original dataframe
            self.column_selector.clear()
            self.column_selector.addItems(self.df.columns)

    def update_plots(self):
        # Get the selected column from the column selector
        selected_column = self.column_selector.currentText()

        if selected_column and self.df is not None:

            raw_signal = self.df[selected_column].values[self.plot_time_index]
            times = self.times[self.plot_time_index]
            if self.dfiso is not None:
                raw_isosbestic = self.dfiso[selected_column].values[self.plot_time_index]
            else:
                raw_isosbestic = None

            # Clear previous plots
            self.clear_plots()

            if self.filtered_df is None:
                processed_signal = None
            else:
                processed_signal = self.filtered_df[selected_column].values[self.plot_time_index]

            self.plotobj.set_data(raw_signal=raw_signal,
                                  times=times,
                                  raw_isosbestic=raw_isosbestic,
                                  processed_signal=processed_signal)
            self.plotobj.raw_processed_figure2(self.axes)

            # Redraw the canvas
            self.canvas.draw()

    def clear_plots(self):
        self.figure.clear()
        _, self.axes = self.plotobj.set_fig_layout2(figure=self.figure)

        self.canvas.draw()

    def on_column_header_clicked(self, logical_index):
        # Get the clicked column's name based on the index
        column_name = self.df.columns[logical_index]

        # Select the column in the column selector
        self.column_selector.setCurrentText(column_name)

        # Update the plots based on the selected column
        self.update_plots()

    def apply_filter(self):
        # Get the selected filter option from the filter dropdown
        filter_option = self.filter_selector.currentText()

        if filter_option == "Select Filter":
            self.filtered_df = None
            # After applying the filter, update the plots
            self.update_plots()


        # Apply the appropriate filter to the dataframe and get the modified data
        if filter_option == 'Filter MAD':
            self.filtered_df = self.filter_mad(self.df)
        # elif filter_option == "Filter CAD":
        #     self.filtered_df = self.filter_cad(self.df)
        # elif filter_option == "Filter JOVE":
        #     self.filtered_df = self.filter_jove(self.df)

        # After applying the filter, update the plots
        self.update_plots()

    def filter_mad(self, df):
        # Example filter for MAD (Median Absolute Deviation)
        fs = 1 / np.nanmedian(np.diff(self.times))

        filtered_df = df.copy()
        for col in filtered_df.columns:
            filtered_df[col] = ffpr.mad_raw_signal(df[col], fs)

        return filtered_df

    # def filter_cad(self, df):
    #     # Example filter for CAD (Coefficient of Variation)
    #     coeff_of_variation = df.std() / df.mean()  # Example: Coefficient of Variation
    #     filtered_df = df.copy()  # Make a copy of the original DataFrame
    #     for col in filtered_df.columns:
    #         if coeff_of_variation[col] > 2:  # Apply filtering based on coefficient of variation
    #             filtered_df[col] = None  # Set high CV columns to None (or any other filtering logic)
    #     return filtered_df
    #
    # def filter_jove(self, df):
    #     # Example filter for JOVE (custom filtering logic)
    #     filtered_df = df.copy()  # Make a copy of the original DataFrame
    #     for col in filtered_df.columns:
    #         # Set values greater than 100 to NaN (this is just an example logic for "JOVE")
    #         filtered_df[col] = filtered_df[col].apply(lambda x: x if x <= 100 else None)
    #     return filtered_df


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = DataFrameVisualizerApp()
    window.show()
    sys.exit(app.exec_())
