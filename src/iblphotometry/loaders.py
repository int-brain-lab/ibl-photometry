import numpy as np
import pynapple as nap
import pandas as pd
from pathlib import Path


class PhotometryLoader:
    def __init__(self, one):
        self.one = one

    def load_photometry_data(self, eid=None, pid=None, signal=None) -> nap.TsdFrame:
        # TODO design choice: should loading data with a pid return a nap.Tsd? I think it should
        # TODO design choice: what should be the column names? Right now they are Region0X, I think they should be brain_regions
        if eid is not None and pid is not None:
            if pid not in self.one.eid2pid(eid)[0]:
                raise ValueError(
                    'both pid and eid are provided, however, the pid does not belong to the eid'
                )
            # TODO discuss: decide what to return in this case. I think Tsd based on pid

        if pid is not None:
            return self._load_data_from_pid(pid, signal=signal)

        if eid is not None:
            return self._load_data_from_eid(eid, signal=signal)

    def load_trials_table(self, eid):
        return self.one.load_dataset(eid, '*trials.table')

    def get_mappable(self, eid):
        return self._load_locations(eid).reset_index().columns

    def get_mapping(self, eid, key=None, value=None):
        locations = self._load_locations(eid)
        return locations.reset_index().set_index(key)[value].to_dict()

    def _load_data_from_eid(self, eid, signal=None) -> nap.TsdFrame:
        # TODO design choice: what if only one channel is measured per eid?
        # Still return a nap.TsdFrame or nap.Tsd?
        raw_photometry_df = self.one.load_dataset(eid, 'photometry.signal.pqt')
        if signal is not None:
            raw_photometry_df = raw_photometry_df.groupby('name').get_group(signal)
        locations = self._load_locations(eid)
        raw_photometry_df = raw_photometry_df.set_index('times')[locations.index]

        # if TsdFrame.columns should be renamed to brain_regions
        # if rename:
        #     rename_map = self.get_mapping(eid, key='ROI', value='brain_region')
        #     raw_photometry_df = raw_photometry_df.rename(rename_map)

        return nap.TsdFrame(raw_photometry_df)

    def _load_locations(self, eid) -> pd.DataFrame:
        return self.one.load_dataset(eid, 'photometryROI.locations.pqt')

    def _load_data_from_pid(self, pid=None, signal=None) -> nap.Tsd:
        eid, pname = self.one.pid2eid(pid)
        locations = self._load_locations(eid)
        roi_name = dict(zip(locations['fiber'], locations.index))[pname]
        return self._load_data_from_eid(eid, signal=signal)[roi_name]

    def pid2eid(self, pid: str) -> tuple[str, str]:
        return self.one.pid2eid(pid)


class AlexLoader(PhotometryLoader):
    def load_photometry_data(self, eid=None, pid=None):
        return super().load_photometry_data(eid=eid, pid=pid, signal='GCaMP')


# class KceniaLoader(PhotometryLoader):
#     def load_photometry_data()

# class BaseLoader(ABC):
#     i = 0

#     @property
#     @abstractmethod
#     def size(self) -> int:
#         ...
#         # has to be set to the number of datasets

#     @abstractmethod
#     def get_data(self) -> tuple[nap.TsdFrame, pd.DataFrame, dict]:
#         ...
#         # should return the photometry data, the behavioral data (a trials table)
#         # and a dict with the metadata

#     def __iter__(self):
#         return self

#     def __next__(self):
#         if self.i < self.size:
#             self.i += 1
#             return self.get_data()
#         else:
#             raise StopIteration


# class OneLoader(BaseLoader):
#     """serves as a base class for ONE compatible data loaders, Kcenias data is still a special case of this"""

#     size = None

#     def __init__(self, one, eids: list[str] = None, pids: list[str] = None, *args):
#         self.one = one
#         if eids is None and pids is None:
#             raise ValueError('either eids or pids must be provided')

#         # time for pydantic
#         # for name, _list in zip(['eids', 'pids'], [eids, pids]):
#         #     if not isinstance(_list, list):
#         #         raise TypeError(f'{name} has to be a list of str')

#         if eids is not None:
#             self.eids = eids
#             if pids is None:
#                 self.pids = list(
#                     chain.from_iterable([one.eid2pid(eid)[0] for eid in self.eids])
#                 )

#         if pids is not None:
#             self.pids = pids
#             if eids is None:
#                 self.eids = list(np.unique([one.pid2eid(pid)[0] for pid in pids]))

#         self.size = len(self.pids)

#     def get_trials_data(self, pid: str) -> pd.DataFrame:
#         # set up like this so it can be overridden in subclass
#         eid, _ = self.pid2eid(pid)
#         return self.one.load_dataset(eid, '*trials.table')

#     def get_photometry_data(self, pid: str, signal_name: str = None) -> nap.TsdFrame:
#         # set up like this so it can be overridden in subclass
#         eid, _ = self.one.pid2eid(pid)

#         # photometry.signal.pqt comes with ROIs as column names.
#         raw_photometry_df = self.one.load_dataset(eid, 'photometry.signal.pqt')
#         if signal_name is not None:
#             raw_photometry_df = raw_photometry_df.groupby('name').get_group(signal_name)

#         # conversion to nap.TsdFrame: time as index, and restricting to regions
#         locations = self.one.load_dataset(eid, 'photometryROI.locations.pqt')
#         raw_photometry = nap.TsdFrame(
#             raw_photometry_df.set_index('times')[locations.index]
#         )
#         return raw_photometry

#     def get_meta_data(self, pid: str) -> dict:
#         # set up like this so it can be overridden in subclass
#         eid, pname = self.pid2eid(pid)
#         return dict(eid=eid, pid=pid, pname=pname)

#     def pid2eid(self, pid: str) -> tuple[str, str]:
#         # set up like this so it can be overridden
#         return self.one.pid2eid(pid)

#     def get_data(self) -> tuple[nap.TsdFrame, pd.DataFrame, dict]:
#         """iterates over pids!"""
#         pid = self.pids[self.i]  # uses the index defined in the base class

#         # Get photometry data and convert to pynapple
#         raw_photometry = self.get_photometry_data(pid)

#         # Get trials data
#         trials = self.get_trials_data(pid)

#         # meta
#         meta = self.get_meta_data(pid)

#         return raw_photometry, trials, meta


# class KceniaLoader(OneLoader):
#     def __init__(self, one, eids: list[str]):
#         self.one = one
#         pids = []
#         for eid in eids:
#             pnames = self.eid2pnames(eid)
#             for pname in pnames:
#                 pids.append(f'{eid}_{pname}')
#         super().__init__(one, eids=eids, pids=pids)

#     def pid2eid(self, pid: str):
#         return pid.split('_')

#     def eid2pnames(self, eid: str):
#         session_path = self.one.eid2path(eid)
#         pnames = [reg.name for reg in session_path.joinpath('alf').glob('Region*')]
#         return pnames

#     def get_photometry_data(self, pid: str):
#         eid, pname = self.pid2eid(pid)
#         session_path = self.one.eid2path(eid)
#         pqt_path = session_path / 'alf' / pname / 'raw_photometry.pqt'
#         raw_photometry_df = pd.read_parquet(pqt_path)
#         raw_photometry = nap.TsdFrame(raw_photometry_df.set_index('times'))
#         return raw_photometry


# class AlexLoader(OneLoader):
#     def __init__(self, one, eids: list[str] = None):
#         eids = (
#             list(one.search(dataset='photometry.signal.pqt')) if eids is None else eids
#         )
#         super().__init__(one, eids=eids)

#     def get_photometry_data(self, pid: str):
#         eid, pname = self.one.pid2eid(pid)

#         # mapping pname to roi
#         locations = self.one.load_dataset(eid, 'photometryROI.locations.pqt')
#         roi = dict(zip(locations['fiber'], locations.index))[pname]

#         # signal_name is the part that is specific to alejaandro
#         raw_photometry = super().get_photometry_data(pid, signal_name='GCaMP')
#         return raw_photometry[roi]


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
