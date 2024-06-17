from iblphotometry.synthetic import synthetic101
import iblphotometry.preprocessing
import iblphotometry.plots


def test_preproc_isosbestic_regression():
    df_photometry = synthetic101()
    iblphotometry.preprocessing.isosbestic_correction_dataframe(df_photometry)
    iblphotometry.plots.plot_raw_data_df(df_photometry)
