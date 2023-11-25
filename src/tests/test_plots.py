import pandas as pd
import iblphotometry.plots

import importlib

importlib.reload(iblphotometry.plots)
df_photometry = pd.read_csv("/home/olivier/Downloads/Fig4_signals.csv").rename(columns={'Time (sec)': 'times', 'filt465': 'calcium', 'filt405': 'isosbestic'})
fig, ax = iblphotometry.plots.plot_raw_data_df(df_photometry, suptitle="Paper data")
