"""Integration tests for the photometry sync tasks of `iblphotometry.tasks`.

Every session declared in `fixtures/sessions_for_tests.yaml` is treated the same: the sync mode
is inferred from the session, the matching task is built from the session's own experiment
description and then run.
"""

from ibllib.io import session_params
from ibllib.tests.base import TEST_DB
from one.api import ONE

from iblphotometry.tasks import FibrePhotometryBpodSync, FibrePhotometryDAQSync, infer_sync_mode
from tests.base_tests import PhotometryIntegrationTestCase

SYNC_TASKS = {
    'bpod': FibrePhotometryBpodSync,
    'daqami': FibrePhotometryDAQSync,
}


class TestPhotometryExtractors(PhotometryIntegrationTestCase):
    def test_extractors(self):
        """Run the sync task matching each session's sync mode."""
        for session in self.sessions:
            with self.subTest(session=session['session_path'], comment=session['comment']):
                session_path = self.get_session_path(session)
                sync_mode = infer_sync_mode(session_path)
                task_class = SYNC_TASKS[sync_mode]

                # the task is configured by the session's own photometry device parameters
                neurophotometrics_params = (
                    session_params.read_params(session_path).get('devices', {}).get('neurophotometrics', {})
                )
                # the one instance is passed explicitly on every task creation: ibllib's Task
                # falls back to a production ONE() in get_data_handler when it is not given one
                task = task_class(
                    session_path,
                    one=ONE(**TEST_DB),
                    on_error='raise',
                    **neurophotometrics_params,
                )

                task.get_signatures()
                self.assertTrue(task.assert_expected_inputs()[0], f'missing inputs for {sync_mode} task')
                self.assertEqual(0, task.run(), task.log)
