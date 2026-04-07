import unittest
from one.api import ONE
from iblphotometry.tasks import FibrePhotometryDAQSync, FibrePhotometryBpodSync
import os
from one.alf.exceptions import ALFObjectNotFound

RUN_EXTRACTOR_TESTS = True if os.environ.get('RUN_EXTRACTOR_TESTS') == '1' else False


class PhotometryExtractorTest(unittest.TestCase):
    def setUp(self):
        self.daq_eids = [
            '34f55b3a-725e-4cc7-aed3-6e6338f573bf',  # Laura
            'b3b87ca9-2075-474b-b925-9fc824de85a5',  # Carolina
        ]

        self.bpod_eids = [
            'ba01bf35-8a0d-4ca3-a66e-b3a540b21128',
            '7c67fbd4-18c1-42f2-b989-8cbfde0d2374',
            'b1e38acd-f65f-4395-ae4f-8fee34ca40c9',
        ]

    def test_daq_extractor(self):
        if not RUN_EXTRACTOR_TESTS:
            self.skipTest(
                'this test will download large files via ONE and is not meant to be run in a CI, set env var RUN_EXTRACTOR_TESTS=1 to enable'
            )

        self.one = ONE()
        for eid in self.daq_eids:
            experiment_description = self.one.load_dataset(eid, '*experiment.description')
            session_folder = self.one.eid2path(eid)

            task = FibrePhotometryDAQSync(
                session_folder,
                one=self.one,
                on_error='raise',
                **experiment_description['devices']['neurophotometrics'],
            )
            task.get_signatures()
            for file, collection, _, _ in task.signature['input_files']:
                self.one.load_dataset(eid, file, collection=collection, download_only=True)
            assert task.assert_expected_inputs()[0]
            task.run()
            assert task.status == 0

    def test_bpod_extractor(self):
        if not RUN_EXTRACTOR_TESTS:
            self.skipTest(
                'this test will download large files via ONE and is not meant to be run in a CI, set env var RUN_EXTRACTOR_TESTS=1 to enable'
            )

        self.one = ONE()
        for eid in self.bpod_eids:
            experiment_description = self.one.load_dataset(eid, '*experiment.description')
            session_folder = self.one.eid2path(eid)

            task = FibrePhotometryBpodSync(
                session_folder,
                one=self.one,
                on_error='raise',
                **experiment_description['devices']['neurophotometrics'],
            )
            task.get_signatures()
            for signature in task.signature['input_files']:
                file, collection, required, _ = signature
                try:
                    self.one.load_dataset(eid, file, collection=collection, download_only=True)
                except ALFObjectNotFound:
                    if required:
                        raise
                    else:
                        print(f'optional file {file} not found, skipping')

            assert task.assert_expected_inputs()[0]
            task.run()
            assert task.status == 0
