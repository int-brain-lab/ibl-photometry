from pathlib import Path
import unittest


class PhotometryDataTestCase(unittest.TestCase):
    def setUp(self):
        self.versions_path = Path(__file__).parent / 'data' / 'neurophotometrics' / 'versions'
