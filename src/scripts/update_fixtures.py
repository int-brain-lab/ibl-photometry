"""Stage integration test data from real sessions into the folder set by `INTEGRATION_DATA_DIR`.

Run this to (re)populate the integration data root that the integration tests read from. The
sessions to stage are declared in `tests/fixtures/sessions_for_tests.yaml`, the same file the
integration tests iterate over, and each is placed under its session path, so the root ends up
looking like:

    $INTEGRATION_DATA_DIR/
    ├── Subjects_init/                          # marker folder expected by ibllib's IntegrationTest
    ├── ZFM-03059/2021-08-27/001/
    │   ├── _ibl_experiment.description.yaml
    │   ├── raw_photometry_data/
    │   └── raw_task_data_00/
    └── ZFM-08554/2025-02-26/001/
        └── ...

`LOCATION` selects where the data is taken from:

- 'local' or 'server': a regular ONE instance downloads the two collections into the ONE cache,
  from where they are copied into the fixture root. Both behave the same for now.
- 'sdsc': an `OneSdsc` instance resolves the session on the SDSC filesystem, where the data
  already sits, and the two collections are copied straight from there. Dataset UUIDs are
  stripped from the filenames on the way.

Either way only the raw photometry collection and the raw task collection are staged, both
looked up in the session's experiment description, along with the description file itself. Each
file is reported as copied or skipped, and any dataset that could not be downloaded is named.
"""

import logging
import os
import shutil
from pathlib import Path

import yaml
from ibllib.io import session_params
from one.alf.path import ALFPath
from one.api import ONE

_logger = logging.getLogger(__name__)

LOCATION = 'local'  # one of 'local', 'server', 'sdsc'

SESSIONS_FOR_TESTS_FILE = Path(__file__).parents[2] / 'tests' / 'fixtures' / 'sessions_for_tests.yaml'


def load_sessions_for_tests() -> list[dict]:
    """Read the sessions to stage, the same ones the integration tests run against.

    Returns
    -------
    list[dict]
        One entry per declared session, with its 'eid', 'session_path' and 'comment'.
    """
    with open(SESSIONS_FOR_TESTS_FILE, 'r') as file_handle:
        return yaml.safe_load(file_handle)


def get_one() -> ONE:
    """Instantiate the ONE flavour matching `LOCATION`.

    Returns
    -------
    ONE
        A regular ONE instance, or an `OneSdsc` instance when running on SDSC.
    """
    if LOCATION == 'sdsc':
        # only available on the SDSC infrastructure
        from deploy.iblsdsc import OneSdsc

        return OneSdsc()
    return ONE()


def get_destination_root() -> Path:
    """Return the fixture root and make sure it is a valid ibllib integration data root.

    Returns
    -------
    Path
        The folder set by `INTEGRATION_DATA_DIR`, falling back to the fixtures folder that holds
        the sessions file.
    """
    integration_data_dir = os.environ.get('INTEGRATION_DATA_DIR')
    destination_root = Path(integration_data_dir) if integration_data_dir else SESSIONS_FOR_TESTS_FILE.parent
    _logger.info(f'staging into {destination_root}')

    # ibllib's IntegrationTest validates a data root by the presence of this folder
    destination_root.joinpath('Subjects_init').mkdir(parents=True, exist_ok=True)
    return destination_root


def get_collections(experiment_description: dict) -> tuple[str, str]:
    """Look up the raw photometry and raw task collections of a session.

    Parameters
    ----------
    experiment_description : dict
        The contents of the session's experiment description file.

    Returns
    -------
    tuple[str, str]
        The photometry collection and the task collection of the first protocol.
    """
    if 'neurophotometrics' in experiment_description['devices']:
        photometry_collection = experiment_description['devices']['neurophotometrics']['collection']
    else:
        photometry_collection = 'raw_photometry_data'

    # as elsewhere in this package, the first protocol is the one that is used
    task_protocol = next(iter(experiment_description['tasks'][0]))
    task_collection = session_params.get_task_collection(experiment_description, task_protocol)
    return photometry_collection, task_collection


def gather_downloaded_files(eid: str, one: ONE) -> list[ALFPath]:
    """Download a session's photometry and task data, returning the resulting cache paths.

    Parameters
    ----------
    eid : str
        The session to download.
    one : ONE
        The ONE instance to download with.

    Returns
    -------
    list[ALFPath]
        The downloaded files, in the ONE cache.
    """
    experiment_description = one.load_dataset(eid, '*experiment.description')
    photometry_collection, task_collection = get_collections(experiment_description)

    # the description file itself is a fixture too, the tasks read the collections back from it
    files = [one.load_dataset(eid, '*experiment.description', download_only=True)]
    for collection in (photometry_collection, task_collection):
        _logger.info(f'downloading collection {collection}')
        downloaded = list(one.load_collection(eid, collection, download_only=True))

        # load_collection drops every dataset it could not put on disk without raising, so a
        # failed download is indistinguishable from a smaller collection - name them here
        expected = set(one.list_datasets(eid, collection=collection))
        missing = expected - {str(ALFPath(path).relative_to_session()) for path in downloaded}
        if missing:
            _logger.warning(f'{len(missing)} datasets of {collection} were not downloaded:')
            for rel_path in sorted(missing):
                _logger.warning(f'  not downloaded: {rel_path}')

        files.extend(downloaded)
    return files


def gather_sdsc_files(eid: str, one: ONE) -> list[ALFPath]:
    """List a session's photometry and task files as they sit on the SDSC filesystem.

    Parameters
    ----------
    eid : str
        The session to look up.
    one : ONE
        The `OneSdsc` instance to resolve the session path with.

    Returns
    -------
    list[ALFPath]
        The files of the two raw data collections, plus the experiment description file.
    """
    session_path = one.eid2path(eid).session_path()
    # read_params globs for the description file, which on SDSC carries the dataset UUID
    experiment_description = session_params.read_params(session_path)
    photometry_collection, task_collection = get_collections(experiment_description)

    files = list(session_path.glob('_ibl_experiment.description*.yaml'))
    for collection in (photometry_collection, task_collection):
        _logger.info(f'collecting collection {collection}')
        files.extend(path for path in session_path.joinpath(collection).rglob('*') if path.is_file())
    return files


def copy_to_fixture_root(files: list[ALFPath], destination_root: Path, location: str) -> None:
    """Copy files into the fixture root, keeping the session tree intact.

    Every file is reported as either copied or skipped, the latter meaning it is already staged
    and unchanged.

    Parameters
    ----------
    files : list[ALFPath]
        The files to copy.
    destination_root : Path
        The fixture root to copy them into.
    location : str
        Where the files come from. On SDSC the dataset UUID is part of the filename and is
        stripped on the way.
    """
    n_copied, n_skipped = 0, 0
    for source_file in files:
        source_file = ALFPath(source_file)
        target_file = ALFPath(destination_root / source_file.session_path_short() / source_file.relative_to_session())
        if location == 'sdsc':
            target_file = target_file.without_uuid()
        target_file.parent.mkdir(parents=True, exist_ok=True)

        # an existing target is only kept when it still matches its source, so a partially
        # written or outdated fixture is repaired by re-running this script
        if is_up_to_date(source_file, target_file):
            _logger.info(f'  skipped (unchanged): {target_file.relative_to_session()}')
            n_skipped += 1
            continue
        shutil.copy2(source_file, target_file)
        _logger.info(f'  copied: {target_file.relative_to_session()}')
        n_copied += 1

    _logger.info(f'{n_copied} files copied, {n_skipped} unchanged')


def is_up_to_date(source_file: Path, target_file: Path) -> bool:
    """Report whether a staged file still matches the source it was copied from.

    Parameters
    ----------
    source_file : Path
        The file in the ONE cache or on the SDSC filesystem.
    target_file : Path
        The staged file in the fixture root.

    Returns
    -------
    bool
        True when the target exists and has the size and modification time of the source, which
        `shutil.copy2` preserves.
    """
    if not target_file.exists():
        return False
    source_stat, target_stat = source_file.stat(), target_file.stat()
    return source_stat.st_size == target_stat.st_size and source_stat.st_mtime == target_stat.st_mtime


def main() -> None:
    """Stage every session declared in the sessions file into the fixture root."""
    destination_root = get_destination_root()
    one = get_one()

    failed_sessions = []
    for session in load_sessions_for_tests():
        eid, session_path = session['eid'], session['session_path']
        _logger.info(f'staging {session_path} ({session["comment"]}) from {LOCATION}')
        try:
            if LOCATION == 'sdsc':
                files = gather_sdsc_files(eid, one)
            else:
                files = gather_downloaded_files(eid, one)
            copy_to_fixture_root(files, destination_root, LOCATION)
        except Exception as exception:  # noqa: BLE001 - staging must survive any single session
            # the remaining sessions are staged regardless, so one broken session does not
            # leave the fixture root in a half updated state
            _logger.error(f'failed to stage {session_path}: {exception!r}')
            failed_sessions.append(session_path)
            continue

    _logger.info(f'fixtures staged in {destination_root}')
    if failed_sessions:
        _logger.error(f'{len(failed_sessions)} sessions failed to stage: {", ".join(failed_sessions)}')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()
