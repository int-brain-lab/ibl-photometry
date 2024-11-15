import numpy as np
import pynapple as nap
import pandas as pd
from pathlib import Path
from iblphotometry import io
from brainbox.io.one import SessionLoader


class PhotometryLoader:
    # TODO move this class to brainbox.io
    def __init__(self, one, verbose=False):
        self.one = one
        self.verbose = verbose

    def load_photometry_data(
        self, eid=None, pid=None, return_regions=False
    ) -> nap.TsdFrame:
        # if eid is not None and pid is not None:
        #     if pid not in self.eid2pid(eid)[0]:
        #         raise ValueError(
        #             'both pid and eid are provided, however, the pid does not belong to the eid'
        #         )

        if pid is not None:
            # raise NotImplementedError
            return self._load_data_from_pid(pid)

        if eid is not None:
            return self._load_data_from_eid(eid, return_regions=return_regions)

    # def load_trials_table(self, eid):
    #     return self.one.load_dataset(eid, '*trials.table')

    # def get_mappable(self, eid):
    #     return self._load_locations(eid).reset_index().columns

    # def get_mapping(self, eid, key=None, value=None):
    #     locations = self._load_locations(eid)
    #     return locations.reset_index().set_index(key)[value].to_dict()

    def _load_data_from_eid(self, eid, return_regions=False) -> nap.TsdFrame:
        raw_photometry_df = self.one.load_dataset(eid, 'photometry.signal.pqt')
        locations_df = self.one.load_dataset(eid, 'photometryROI.locations.pqt')
        read_config = dict(data_columns=list(locations_df.index))
        raw_tfs = io.from_dataframe(raw_photometry_df, **read_config)

        cols = list(raw_tfs[list(raw_tfs.keys())[0]].columns)
        if self.verbose:
            print(f'available signal bands: {list(raw_tfs.keys())}')
            print(f'available brain regions: {cols}')

        if return_regions:
            return raw_tfs, cols
        else:
            return raw_tfs

    def _load_data_from_pid(self, pid=None, signal=None) -> nap.Tsd:
        eid, pname = self.one.pid2eid(pid)
        locations = self._load_locations(eid)
        roi_name = dict(zip(locations['fiber'], locations.index))[pname]
        return self._load_data_from_eid(eid, signal=signal)[roi_name]

    # def pid2eid(self, pid: str) -> tuple[str, str]:
    #     return self.one.pid2eid(pid)

    # def eid2pid(self, eid: str):
    #     return self.one.eid2pid(eid)


class KceniaLoader(PhotometryLoader):
    # soon do be OBSOLETE
    def _load_data_from_eid(self, eid: str, return_regions=False):
        session_path = self.one.eid2path(eid)
        pnames = self._eid2pnames(eid)

        raw_dfs = {}
        for pname in pnames:
            pqt_path = session_path / 'alf' / pname / 'raw_photometry.pqt'
            raw_dfs[pname] = pd.read_parquet(pqt_path).set_index('times')

        signal_bands = ['raw_calcium', 'raw_isosbestic']  # HARDCODED but fine

        raw_tfs = {}
        for band in signal_bands:
            df = pd.DataFrame([raw_dfs[pname][band].values for pname in pnames]).T
            df.columns = pnames
            df.index = raw_dfs[pname][band].index
            raw_tfs[band] = nap.TsdFrame(df)

        if self.verbose:
            print(f'available signal bands: {list(raw_tfs.keys())}')
            cols = list(raw_tfs[list(raw_tfs.keys())[0]].columns)
            print(f'available brain regions: {cols}')

        if return_regions:
            return raw_tfs, pnames
        else:
            return raw_tfs

    # def _load_data_from_eid(self, eid, signal=None):
    #     raise NotImplementedError

    # def get_mappable(self, eid):
    #     raise NotImplementedError

    # def get_mapping(self, eid, key=None, value=None):
    #     raise NotImplementedError

    # def pid2eid(self, pid: str) -> tuple[str, str]:
    #     return pid.split('_')

    # def eid2pid(self, eid):
    #     pnames = self._eid2pnames(eid)
    #     pids = [f'{eid}_{pname}' for pname in pnames]
    #     return (pids, pnames)

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
