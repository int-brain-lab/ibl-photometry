from pathlib import Path
from typing import ClassVar
import unittest

import yaml
from ibllib.tests.base import IntegrationTest
from one.alf.path import ALFPath

SESSIONS_FOR_TESTS_FILE = Path(__file__).parent / 'fixtures' / 'sessions_for_tests.yaml'


def load_sessions_for_tests() -> list[dict]:
    """Read the sessions the integration tests run against.

    Returns
    -------
    list[dict]
        One entry per declared session, with its 'eid', 'session_path' and 'comment'.
    """
    with open(SESSIONS_FOR_TESTS_FILE, 'r') as file_handle:
        return yaml.safe_load(file_handle)


class PhotometryDataTestCase(unittest.TestCase):
    def setUp(self):
        self.versions_path = Path(__file__).parent / 'data' / 'neurophotometrics' / 'versions'


class PhotometryIntegrationTestCase(IntegrationTest):
    """Base class for tests against the real sessions staged in `INTEGRATION_DATA_DIR`.

    The sessions come from `fixtures/sessions_for_tests.yaml` and are exposed as `sessions`, so
    a test iterates over them rather than declaring its own. Use `src/scripts/update_fixtures.py`
    to stage them. The tests auto-skip when `INTEGRATION_DATA_DIR` is not set, and each test runs
    against a fresh symlink mirror of the sessions, so whatever a task writes into a session
    leaves the staged data alone.
    """

    sessions: ClassVar[list[dict]] = load_sessions_for_tests()

    # the session paths in the syntax the parent class mirrors them by
    # required_files: ClassVar[list[str]] = [session['session_path'] for session in sessions]

    # a mirror per test method, as the extractors write their outputs into the session
    _writable_scope = 'test'

    def get_session_path(self, session: dict) -> ALFPath:
        """Return a declared session's path in the mirror the current test may write into.

        Parameters
        ----------
        session : dict
            One entry of `sessions`.

        Returns
        -------
        ALFPath
            The session path inside the mirror.
        """
        return ALFPath(self.data_path / session['session_path'])
