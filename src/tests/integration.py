


from pathlib import Path




from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import iblphotometry.plots
import iblphotometry.preprocessing

path_integration_data = Path("/home/olivier/Documents/photometry/integration")
path_figures = path_integration_data.joinpath('figures')


for parquet_file in path_integration_data.glob('*.pqt'):
    df_photometry = pd.read_parquet(parquet_file)
    raw_calcium = df_photometry['raw_calcium'].values
    raw_isosbestic = df_photometry['raw_isosbestic'].values
    times = df_photometry['times'].values
    fs = 1 / np.median(np.diff(times))
    suptitle = parquet_file.stem

    # plot raw data
    iblphotometry.plots.plot_raw_data_df(
        df_photometry,
        suptitle=suptitle,
        output_file=path_figures.joinpath(f"{parquet_file.stem}_00.png")
    )

    # photobleach
    calcium = iblphotometry.preprocessing.photobleaching_lowpass(raw_calcium, fs=fs)
    isosbestic_control = iblphotometry.preprocessing.photobleaching_lowpass(raw_isosbestic, fs=fs)
    iblphotometry.plots.plot_photometry_traces(
        times,
        isosbestic_control,
        calcium,
        suptitle=parquet_file.stem,
        output_file=path_figures.joinpath(f"{parquet_file.stem}_01_photobleach.png")
    )

    # jove_2019
    calcium = iblphotometry.preprocessing.jove2019(raw_calcium, raw_isosbestic, fs=fs)
    isosbestic_control = iblphotometry.preprocessing.jove2019(raw_isosbestic, raw_isosbestic, fs=fs)
    iblphotometry.plots.plot_photometry_traces(
        times,
        isosbestic_control,
        calcium,
        suptitle=parquet_file.stem,
        output_file=path_figures.joinpath(f"{parquet_file.stem}_02_jove2019.png")
    )

    # sliding mad
    calcium = iblphotometry.preprocessing.preprocess_sliding_mad(raw_calcium, times, fs=fs)
    isosbestic_control = iblphotometry.preprocessing.preprocess_sliding_mad(raw_isosbestic, times, fs=fs)
    iblphotometry.plots.plot_photometry_traces(
        times,
        isosbestic_control,
        calcium,
        suptitle=parquet_file.stem,
        output_file=path_figures.joinpath(f"{parquet_file.stem}_03_sliding_mad.png")
    )

    print(parquet_file)
    # if suptitle == 'ZFM-04022_2023-01-12_003_alf_Region3G':
    #     break
    # break
    plt.close('all')



