from iblphotometry.io import from_raw_neurophotometrics
import iblphotometry.plots as plots
from pathlib import Path
import matplotlib.pyplot as plt

subject_path = Path('/Users/gaellechapuis/Downloads/')

##
pqt_file_g = subject_path.joinpath('_neurophotometrics_fpData.raw.pqt')
td = from_raw_neurophotometrics(pqt_file_g)

##
column = 'G0'
raw_signal = td['GCaMP'][column].d
times = td['GCaMP'].t
raw_isosbestic = td['Isosbestic'][column].d

# Preprocess signal TODO change to use the proper pipeline
import iblphotometry.preprocessing as ffpr
import numpy as np

fs = 1 / np.nanmedian(np.diff(times))
processed_signal = ffpr.mad_raw_signal(raw_signal, fs)

plotobj = plots.PlotSignal(raw_signal, times, raw_isosbestic, processed_signal)

# Use this magic command in the python terminal
# %matplotlib qt5

plotobj.raw_processed_figure2()
plt.show()
