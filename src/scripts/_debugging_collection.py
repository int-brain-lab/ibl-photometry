# %%
from one.api import ONE
import logging
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
logging.basicConfig()

one = ONE(cache_rest=None)
eid = "ef6e33ef-3511-447e-8c38-65042ff62f0b"
file = "_neurophotometrics_fpData.raw.pqt"
collection = "raw_photometry_data"
one.load_dataset(eid, file, collection=collection) # does not work
# one.load_dataset(eid, '/'.join([collection,file])) # works

# %%
import pandas as pd
import re

# this is the return value of the alf function call under 3.12
pat = '^(?P<collection>(?s:raw_photometry_data))/(#(?P<revision>[\\w.-]+)#/)?(?s:_neurophotometrics_fpData\\.raw\\.pqt)\\Z'


# this is the pattern under 3.14
pat = '^(?P<collection>(?s:raw_photometry_data)\\z)/(#(?P<revision>[\\w.-]+)#/)?(?s:_neurophotometrics_fpData\\.raw\\.pqt)\\z'

# %% restart systematically
# this is 3.14
regexp_args = {'collection': '(?s:raw_photometry_data)\\z'}
spec_str = '{collection}/(#{revision}#/)?(?s:_neurophotometrics_fpData\\.raw\\.pqt)\\z'

# this is 3.12
regexp_args = {'collection': '(?s:raw_photometry_data)'}
spec_str = '{collection}/(#{revision}#/)?(?s:_neurophotometrics_fpData\\.raw\\.pqt)\\Z'

# this is 3.14
# this is in str_match
string = 'raw_photometry_data/_neurophotometrics_fpData.raw.pqt'
pat = '^(?P<collection>(?s:raw_photometry_data)\\z)/(#(?P<revision>[\\w.-]+)#/)?(?s:_neurophotometrics_fpData\\.raw\\.pqt)\\z'
re.match(re.compile(pat, flags=re.U), string)

# %%
# series = pd.read_parquet('/home/georg/all_datasets.pqt')['rel_path']

# series = pd.read_csv('/home/georg/all_datasets.csv')['rel_path']
# print(series.str.match(pattern))

# series = pd.read_csv('/home/georg/all_datasets_14.csv')['rel_path']
# print(series.str.match(pattern))

# %%


from one.api import ONE
one = ONE()
eid = "ef6e33ef-3511-447e-8c38-65042ff62f0b"
file = "_neurophotometrics_fpData.raw.pqt"
collection = "raw_photometry_data"
one.load_dataset(eid, file, collection=collection) # does not work
one.load_dataset(eid, '/'.join([collection,file])) # works
