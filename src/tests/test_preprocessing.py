from iblphotometry.synthetic import synthetic101
import iblphotometry.preprocessing
import iblphotometry.plots

import matplotlib.pyplot as plt


def test_preproc_isosbestic_regression():
    df_photometry, _ = synthetic101()
    df_photometry = iblphotometry.preprocessing.isosbestic_correction_dataframe(df_photometry)
    # iblphotometry.plots.plot_raw_data_df(df_photometry)


def test_preproc_photobleaching_lowpass():
    df_photometry, _ = synthetic101()
    df_photometry['calcium'] = iblphotometry.preprocessing.photobleaching_lowpass(df_photometry['raw_calcium'])
    # plt.plot(df_photometry['calcium'])