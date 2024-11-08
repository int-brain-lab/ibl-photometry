import pynapple as nap
import pandas as pd
from pathlib import Path
from abc import ABC, abstractmethod

class BaseLoader(ABC):
    def __init__(self, one, eids=None):
        self.i_eid = 0
        self.i_probe = 0
        self.one = one
        self.eids = self.set_eids(eids)
        pass

    @abstractmethod
    def get_photometry_data() -> nap.TsdFrame:
        ...

    def get_data(self, eid, pname):
        # Get photometry data and convert to pynapple
        raw_photometry = self.get_photometry_data(eid, pname)
        
        # Get trials data
        trials = self.one.load_dataset(eid, '*trials.table')
        return raw_photometry, trials, eid, pname

    # def get_eids_pnames(self, eids):
    #     # Instantiate dict with keys as eids
    #     dict_a = dict((ikey, list()) for ikey in eids)
    #     for eid in eids:
    #         dict_a[eid] = self.eid2pnames(eid)
    #     return dict_a

    def __next__(self):
        # check if eid iteration is valid
        # if not, end iteration
        if self.i_eid == len(self.eids):
            raise StopIteration
        eid = self.eids[self.i_eid]

        # if eid is valid, get brain regions
        pnames = self.eid2pnames(eid)

        # check if probe iteration is valid
        if self.i_probe < len(pnames):
            pname = pnames[self.i_probe]
            self.i_probe += 1
            return self.get_data(eid, pname)
        else:
            self.i_probe = 0
            self.i_eid += 1
            return self.__next__()


class KceniaLoader(BaseLoader):

    def set_eids(self, eids):
        if eids is None:
            raise ValueError('eids cannot be None for Kcenia loader')
        return eids

    def eid2pnames(self, eid):
        session_path = self.one.eid2path(eid)
        pnames = [reg.name for reg in session_path.joinpath('alf').glob('Region*')]
        return pnames

    def get_photometry_data(self, eid, pname):
        session_path = self.one.eid2path(eid)
        pqt_path = session_path / 'alf' / pname / 'raw_photometry.pqt'
        raw_photometry = pd.read_parquet(pqt_path)
        return nap.TsdFrame(raw_photometry.set_index('times'))


class AlexLoader(BaseLoader):

    def set_eids(self, eids=None):
        eids = self.one.search(dataset='photometry.signal.pqt') if eids is None else eids
        return eids

    def eid2pnames(self, eid):
        rois = self.one.load_dataset(eid, 'photometryROI.locations.pqt')
        pnames = list(rois.index)
        return pnames

    def get_photometry_data(self, eid, pname):
        raw_photometry = self.one.load_dataset(eid, 'photometry.signal.pqt')
        raw_photometry = raw_photometry.groupby('name').get_group('GCaMP')
        raw_photometry = nap.TsdFrame(
            t=raw_photometry['times'].values,
            d=raw_photometry[pname].values,
            columns=['raw_calcium'],
        )
        return raw_photometry


# TODO delete this once analysis settled
def user_config(user):
    path_users = dict()

    match user:
        case 'georg':
            path_users = {
                "dir_results": Path('/home/georg/code/ibl-photometry/qc_results/'),
                "file_websheet": Path('/home/georg/code/ibl-photometry/src/iblphotometry/website.csv'),
                "dir_one": Path('/mnt/h0/kb/data/one')
            }
        case 'gaelle':
            path_users = {
                "dir_results": Path('/Users/gaellechapuis/Desktop/FiberPhotometry/Pipeline_GR'),
                "file_websheet": Path('/Users/gaellechapuis/Desktop/FiberPhotometry/QC_Sheets/'
                                      'website_overview - website_overview.csv'),
                "dir_one": Path('/Users/gaellechapuis/Downloads/ONE/alyx.internationalbrainlab.org/')
            }

    return path_users
