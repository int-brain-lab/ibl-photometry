import pandas as pd
from pathlib import Path
from iblphotometry import io
# from brainbox.io.one import SessionLoader


class PhotometryLoader:
    # TODO make this class a subclass of SessionLoader
    # TODO move this class to brainbox.io

    def __init__(self, one, verbose=False):
        self.one = one
        self.verbose = verbose

    def load_photometry_data(self, eid=None, pid=None, rename=True) -> nap.TsdFrame:
        if pid is not None:
            raise NotImplementedError
            # return self._load_data_from_pid(pid)

        if eid is not None:
            return self._load_data_from_eid(eid, rename=rename)

    def _load_data_from_eid(self, eid, rename=True) -> nap.TsdFrame:
        raw_photometry_df = self.one.load_dataset(eid, 'photometry.signal.pqt')
        locations_df = self.one.load_dataset(eid, 'photometryROI.locations.pqt')
        read_config = dict(
            data_columns=list(locations_df.index),
            rename=locations_df['brain_region'].to_dict() if rename else None,
        )
        raw_dfs = io.from_dataframe(raw_photometry_df, **read_config)

        signal_band_names = list(raw_dfs.keys())
        col_names = list(raw_dfs[signal_band_names[0]].columns)
        if self.verbose:
            print(f'available signal bands: {signal_band_names}')
            print(f'available brain regions: {col_names}')

        return raw_dfs


class KceniaLoader(PhotometryLoader):
    # soon do be OBSOLETE
    def _load_data_from_eid(self, eid: str, rename=True):
        session_path = self.one.eid2path(eid)
        pnames = self._eid2pnames(eid)

        raw_dfs = {}
        for pname in pnames:
            pqt_path = session_path / 'alf' / pname / 'raw_photometry.pqt'
            raw_dfs[pname] = pd.read_parquet(pqt_path).set_index('times')

        signal_bands = ['raw_calcium', 'raw_isosbestic']  # HARDCODED but fine

        # flipping the data representation
        raw_dfs = {}
        for band in signal_bands:
            df = pd.DataFrame([raw_dfs[pname][band].values for pname in pnames]).T
            df.columns = pnames
            df.index = raw_dfs[pname][band].index
            raw_dfs[band] = df

        if self.verbose:
            print(f'available signal bands: {list(raw_dfs.keys())}')
            cols = list(raw_dfs[list(raw_dfs.keys())[0]].columns)
            print(f'available brain regions: {cols}')

        return raw_dfs


    def _eid2pnames(self, eid: str):
        session_path = self.one.eid2path(eid)
        pnames = [reg.name for reg in session_path.joinpath('alf').glob('Region*')]
        return pnames


# TODO delete this once analysis settled
def user_config(user):
    path_users = dict()

    match user:
        case 'georg':
            path_users = {
                'dir_results': Path('/home/georg/code/ibl-photometry/qc_results/'),
                'file_websheet': Path(
                    '/home/georg/code/ibl-photometry/src/local/website.csv'
                ),
                'dir_one': Path('/mnt/h0/kb/data/one'),
            }
        case 'gaelle':
            path_users = {
                'dir_results': Path(
                    '/Users/gaellechapuis/Desktop/FiberPhotometry/Pipeline_GR'
                ),
                'file_websheet': Path(
                    '/Users/gaellechapuis/Desktop/FiberPhotometry/QC_Sheets/'
                    'website_overview - website_overview.csv'
                ),
                'dir_one': Path(
                    '/Users/gaellechapuis/Downloads/ONE/alyx.internationalbrainlab.org/'
                ),
            }

    return path_users
