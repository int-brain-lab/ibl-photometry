from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

import iblphotometry.plots
import iblphotometry.dsp

# example with raw Doric photometry files
file_csv = Path("/datadisk/photometry/solene/20231211_3M_4O_165ma_0001.csv")
mappings = [
    {'times': 0, 'raw_isosbestic': 2, 'raw_calcium': 1},
    {'times': 0, 'raw_isosbestic': 5, 'raw_calcium': 4}
]
header_csv = 1

# example with raw Doric photometry files
file_csv = Path("/datadisk/photometry/stefan/table_name_3_1.csv")
header_csv = 0
mappings = [
    {'times': 0, 'raw_isosbestic': 2, 'raw_calcium': 1},
]

for mapping in mappings:
    df_photometry = pd.read_csv(file_csv, header=header_csv)

    # reads in the photometry signals: we want a dataframe with columns 'times', 'raw_isosbestic', 'raw_calcium'
    df_photometry = df_photometry.rename(columns={df_photometry.columns[cindex]: cname for cname, cindex in mapping.items()})
    df_photometry = df_photometry.iloc[:, list(mapping.values())]
    df_photometry = df_photometry.dropna()

    # apply the baseline correction
    df_photometry = iblphotometry.dsp.baseline_correction_dataframe(df_photometry)

    # plots the raw data and save to file
    fig, ax = iblphotometry.plots.plot_raw_data_df(
        df_photometry,
        suptitle=f"{file_csv.name} channel {mapping['raw_calcium']}",
        output_file=file_csv.parent.joinpath(f"{file_csv.stem}_channel_{mapping['raw_calcium']}.png"),
    )

    # plots the processed data and save to file
    fig, ax = iblphotometry.plots.plot_photometry_traces(
        df_photometry['times'].values, df_photometry['raw_isosbestic'].values, df_photometry['calcium'].values,
        suptitle=f"{file_csv.name} channel {mapping['raw_calcium']} processed",
        output_file=file_csv.parent.joinpath(f"{file_csv.stem}_channel_{mapping['raw_calcium']}_processed.png"),
    )
    plt.close('all')

# could transients be detected using the difference between the calcium trace and the isosbestic control ?
# plt.plot(df_photometry['times'].values, df_photometry['isosbestic_control'].values - df_photometry['calcium'].values)


