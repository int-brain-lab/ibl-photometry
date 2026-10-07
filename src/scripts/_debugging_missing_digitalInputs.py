# %%
# previously, these sessions extracted here
# now they compain about missing digitalInputs

eids = [
    'ba01bf35-8a0d-4ca3-a66e-b3a540b21128',
    '7c67fbd4-18c1-42f2-b989-8cbfde0d2374',
    'b1e38acd-f65f-4395-ae4f-8fee34ca40c9',
    'ef6e33ef-3511-447e-8c38-65042ff62f0b', # present on sdsc
]

# investigate whether those are available on via one

from one.api import ONE

one = ONE()

for eid in eids:
    datasets = one.list_datasets(eid)
    for dataset in datasets:
        if 'digital' in dataset:
            print(dataset)

# %%
from one.alf.exceptions import ALFObjectNotFound
from iblphotometry.tasks import FibrePhotometryBpodSync
session_path = one.eid2path(eids[0])

task = FibrePhotometryBpodSync(session_path, one=one)

task.get_signatures()
for signature in task.signature['input_files']:
    file, collection, required, _ = signature
    try:
        one.load_dataset(eid, file, collection=collection, download_only=True)
    except ALFObjectNotFound:
        if required:
            raise
        else:
            print(f'optional file {file} not found, skipping')

task.setUp()
task._input_files_to_register()
# %%
from one.api import ONE
eid = "ef6e33ef-3511-447e-8c38-65042ff62f0b"
file = "_neurophotometrics_fpData.raw.pqt"
collection = "raw_photometry_data"
one.load_dataset(eid, file, collection=collection) # does not work
one.load_dataset(eid, '/'.join([collection,file])) # works
