from pathlib import Path
import pandas as pd
from brainbox.io.one import PhotometrySessionLoader
from one.api import ONE


class PhotometryDataValidator:
    def __init__(self, one=None):
        self.one = ONE() if one is None else one

    def validate_eids(self, eids: list) -> list[str]:
        return [self._validate(eid) for eid in eids]

    def validate_file(self, eids_file: str | Path) -> list[str]:
        with open(eids_file, 'r') as fH:
            eids = [eid.strip() for eid in fH.readlines()]
        return self.validate_eids(eids)

    def validate_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        for eid in df.index.values:
            df.loc[eid, 'validation'] = self._validate(eid)
        return df

    def _validate(self, eid: str) -> str:
        try:
            psl = PhotometrySessionLoader(eid=eid, one=self.one)
            psl.load_photometry()
        except Exception as e:
            return f'{type(e).__name__}:{e}'
        return ''
