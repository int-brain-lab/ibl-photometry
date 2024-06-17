from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import iblphotometry.plots
import iblphotometry.dsp
# """1 and 17 are isosbestic; 2 and 18 are GCaMP"""
# File 00
# L470 power 1.87, L415 2.57
# G3, R7
# File 01
# L470 power 1.87, L415 2.57
# G3, R7, G8, R9
# G8 and R9 are on a dark part of the FOV

folder_data = Path("/Users/olivier/Library/CloudStorage/GoogleDrive-olivier.winter@internationalbrainlab.org/Shared drives/WG-Neuromodulators/Tests/photobleaching/2024-05-20")
folder_proc = Path("/Users/olivier/Library/CloudStorage/GoogleDrive-olivier.winter@internationalbrainlab.org/Shared drives/WG-Neuromodulators/Tests/photobleaching/2024-05-20_processed")
folder_proc.mkdir(exist_ok=True)

for fname in ["raw_photometry0.csv", "raw_photometry1.csv"]:
    file_csv = folder_data.joinpath(fname)
    df_raw_photometry = pd.read_csv(file_csv)

    for region in ['Region3G', 'Region7R', 'Region8G', 'Region9R']:
        if region not in df_raw_photometry.columns:
            continue

        tag = file_csv.stem + "_" + region
        i_iso = np.where(np.mod(df_raw_photometry['LedState'].values, 16) == 1)[0]
        i_cal = np.where(np.mod(df_raw_photometry['LedState'].values, 16) == 2)[0]
        ns = np.minimum(len(i_iso), len(i_cal))
        i_iso = i_iso[300:ns]
        i_cal = i_cal[300:ns]
        df_photometry = pd.DataFrame({
            'times': (df_raw_photometry['Timestamp'].values[i_iso] + df_raw_photometry['Timestamp'].values[i_cal]) / 2,
            'raw_isosbestic': df_raw_photometry[region].values[i_iso],
            'raw_calcium': df_raw_photometry[region].values[i_cal],
        })
        # apply the baseline correction
        df_photometry = iblphotometry.dsp.baseline_correction_dataframe(df_photometry)
        df_photometry.to_parquet(folder_proc.joinpath(f"{tag}.pqt"))
        # plots the raw data and save to file
        fig, ax = iblphotometry.plots.plot_raw_data_df(
            df_photometry,
            suptitle=tag,
            output_file=folder_proc.joinpath(f"{tag}.png"),
            )

        # plots the processed data and save to file
        fig, ax = iblphotometry.plots.plot_photometry_traces(
            df_photometry['times'].values, df_photometry['isosbestic_control'].values, df_photometry['calcium'].values,
            suptitle=f"{tag} processed",
            output_file=folder_proc.joinpath(f"{tag}_processed.png"),
        )
        plt.close('all')
