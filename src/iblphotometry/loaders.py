import pynapple as nap
import pandas as pd
from pathlib import Path


class IterateSession():
    def __init__(self, one, eids):
        self.i_eid = 0
        self.i_probe = 0
        self.eids = self.set_eids(eids)
        self.one = one
        pass

    def set_eids(self, eids):
        return eids

    def get_all_data(self, eid, pname):
        # Get photometry data and convert to pynapple
        raw_photometry, eid, pname = self.get_data(eid, pname)
        raw_photometry = nap.TsdFrame(raw_photometry.set_index('times'))
        # Get trials data
        trials = self.one.load_dataset(eid, '*trials.table')
        return raw_photometry, trials, eid, pname

    def __next__(self):
        # check if eid iteration is valid
        # if not, end iteration
        if self.i_eid == len(self.eids):
            raise StopIteration
        eid = self.eids[self.i_eid]

        # if i is valid, get brain regions
        pnames = self.eid2pnames(eid)

        # check if probe iteration is valid
        if self.i_probe < len(pnames):
            pname = pnames[self.i_probe]
            self.i_probe += 1
            return self.get_all_data(eid, pname)
        else:
            self.i_probe = 0
            self.i_eid += 1
            self.__next__()
            return


class KceniaLoader(IterateSession):

    def eid2pnames(self, eid):
        session_path = self.one.eid2path(eid)
        pnames = [reg.name for reg in session_path.joinpath('alf').glob('Region*')]
        return pnames

    def get_data(self, eid, pname):
        session_path = self.one.eid2path(eid)
        pqt_path = session_path / 'alf' / pname / 'raw_photometry.pqt'
        raw_photometry = pd.read_parquet(pqt_path)
        return raw_photometry, eid, pname


class AlexLoader(IterateSession):

    def set_eid(self, eids=None):
        eids = self.one.search(dataset='photometry.signal.pqt') if eids is None else eids
        return eids

    def eid2pnames(self, eid):
        rois = self.one.load_dataset(eid, 'photometryROI.locations.pqt')
        pnames = list(rois.index)
        return pnames

    def get_data(self, eid, pname):
        photometry = self.one.load_dataset(eid, 'photometry.signal.pqt')

        # Restrict the dataframe to only the GCamp signal
        photometry = photometry.groupby('name').get_group('GCaMP')
        # This should be equivalent to :
        # photometry = photometry[photometry['wavelength'] == 470]

        # Create a new dataframe restricted to pname as calcium signal
        raw_photometry = pd.DataFrame()
        raw_photometry["raw_calcium"] = photometry[pname]
        raw_photometry["times"] = photometry['times']

        return raw_photometry, eid, pname


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
