import matplotlib.pyplot as plt

from iblphotometry.synthetic import synthetic101
import iblphotometry.preprocessing
import iblphotometry.plots


df_photometry, t_events = synthetic101()
ca_reg = iblphotometry.preprocessing.isosbestic_regression(
    df_photometry['raw_isosbestic'],
    df_photometry['raw_calcium'],
)
ca_lpref = iblphotometry.preprocessing.photobleaching_lowpass(df_photometry['raw_calcium'])


plt.plot(ca_reg)
plt.plot(ca_lpref)


## %% try and compute PSTH


import numpy as np
calcium = ca_reg
times = df_photometry['times'].values
ne = t_events.size






    import matplotlib.pyplot as plt
    plt.matshow(psth)

