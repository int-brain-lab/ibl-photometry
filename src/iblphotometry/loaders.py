import numpy as np
import pynapple as nap
import pandas as pd
from pathlib import Path
from abc import ABC, abstractmethod
from itertools import chain

""" 
only implements the iterator
"""


class BaseLoader(ABC):
    i = 0

    def __init__(self, dataset_ids):
        self.dataset_ids = dataset_ids  # these can be pids

    @abstractmethod
    def get_data(self):
        # uses the current index (self.i) to return the i-th dataset
        ...

    def __iter__(self):
        return self

    def __next__(self):
        if self.i < len(self.dataset_ids):
            self.i += 1
            return self.get_data()
        else:
            raise StopIteration


"""
the problem is, that both iterating over pids and eids is possible
if iteration is over eids, then a pname is required to get the data
but for two different pids, the same dataframe is fetched
logically, we should iterate over pids
"""


class OneLoader(BaseLoader):
    """serves as a base class for ONE compatible data loaders, Kcenias data is still a special case of this"""

    def __init__(self, one, eids=None, pids=None, *args):
        self.one = one
        if eids is None and pids is None:
            raise ValueError('either eids or pids must be provided')

        if eids is not None:
            self.eids = eids
            self.pids = list(chain([one.eid2pid(eid)[0] for eid in self.eids]))

        if pids is not None:
            self.pids = pids
            self.eids = list(np.unique([one.pid2eid(pid)[0] for pid in pids]))

    def get_trials_data(self, pid) -> pd.DataFrame:
        eid = self.one.eid2pid(pid)
        return self.one.load_dataset(eid, '*trials.table')

    def get_photometry_data(self, pid) -> nap.TsdFrame:
        eid, _ = self.one.pid2eid(pid)
        raw_photometry = self.one.load_dataset(eid, 'photometry.signal.pqt')
        raw_photometry = nap.TsdFrame(raw_photometry.set_index('times'))
        return raw_photometry

    def pid2eid(self, pid):
        return self.one.pid2eid(pid)

    def get_data(self):
        """iterates over pids!"""
        pid = self.pids[self.i]  # uses the index defined in the base class

        # Get photometry data and convert to pynapple
        raw_photometry = self.get_photometry_data(pid)

        # Get trials data
        trials = self.get_trials_data(pid)

        # meta
        eid, pname = self.pid2eid(pid)
        meta = dict(eid=eid, pid=pid, pname=pname)

        return raw_photometry, trials, meta


class KceniaLoader(OneLoader):
    def __init__(self, one, eids):
        pids = list(chain([self.eid2pnames(eid) for eid in eids]))
        super().__init__(one, eids=eids, pids=pids)

    def pid2eid(self, pid):
        return pid.split('-')

    def eid2pnames(self, eid):
        session_path = self.one.eid2path(eid)
        pnames = [reg.name for reg in session_path.joinpath('alf').glob('Region*')]
        return pnames

    def get_photometry_data(self, pid):
        eid, pname = self.pid2eid(pid)
        session_path = self.one.eid2path(eid)
        pqt_path = session_path / 'alf' / pname / 'raw_photometry.pqt'
        raw_photometry = pd.read_parquet(pqt_path)
        return raw_photometry


class AlexLoader(OneLoader):
    def __init__(self, one, eids=None):
        eids = (
            self.one.search(dataset='photometry.signal.pqt') if eids is None else eids
        )
        super().__init__(one, eids=eids)

    def get_photometry_data(self, pid):
        _, pname = self.one.pid2eid(pid)
        raw_photometry = super().get_photometry_data(pid)
        raw_photometry = raw_photometry.groupby('name').get_group('GCaMP')
        return raw_photometry[pname]

    def get_data(pid):
        raw_photometry, trials, meta = super().get_data(pid)

        # add TODO here brain region to meta
        # rois = self.one.load_dataset(eid, 'photometryROI.locations.pqt')
        # brain_region = None

        return raw_photometry, trials, meta


# TODO delete this once analysis settled
def user_config(user):
    path_users = dict()

    match user:
        case 'georg':
            path_users = {
                'dir_results': Path('/home/georg/code/ibl-photometry/qc_results/'),
                'file_websheet': Path(
                    '/home/georg/code/ibl-photometry/src/iblphotometry/website.csv'
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
