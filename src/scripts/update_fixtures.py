"""Stage integration test data from real sessions into the folder set by `INTEGRATION_DATA_DIR`.

Run this to (re)populate the integration data root that the integration tests read from. The
sessions to stage are declared in `tests/fixtures/sessions_for_tests.yaml`, the same file the
integration tests iterate over, and each is placed under its session path, so the root ends up
looking like:

    $INTEGRATION_DATA_DIR/                      # tests/fixtures/photometry when unset
    ├── Subjects_init/                          # marker folder expected by ibllib's IntegrationTest
    ├── ZFM-03059/2021-08-27/001/
    │   ├── _ibl_experiment.description.yaml
    │   ├── raw_photometry_data/
    │   └── raw_task_data_00/
    └── ZFM-08554/2025-02-26/001/
        └── ...

`--location` selects where the data is taken from:

- 'local' or 'server': a regular ONE instance downloads the two collections into the ONE cache,
  from where they are copied into the fixture root. Both behave the same for now.
- 'sdsc': an `OneSdsc` instance resolves the session on the SDSC filesystem, where the data
  already sits, and the two collections are copied straight from there. Dataset UUIDs are
  stripped from the filenames on the way.

Either way only the raw photometry collection and the raw task collection are staged, both
looked up in the session's experiment description, along with the description file itself. Each
file is reported as copied or skipped, and any dataset that could not be downloaded is named.

`--dry` reports what would be copied without writing anything to the fixture root. Note that in
'local' and 'server' mode the datasets are still downloaded into the ONE cache, as that is where
the copy reads from - only the writes to the fixture root are suppressed.
"""

import argparse
import logging
import os
import shutil
from pathlib import Path

import yaml
from ibllib.io import session_params
from one.alf.path import ALFPath
from one.api import ONE

_logger = logging.getLogger(__name__)

LOCATIONS = ('local', 'server', 'sdsc')

SESSIONS_FOR_TESTS_FILE = Path(__file__).parents[2] / 'tests' / 'fixtures' / 'sessions_for_tests.yaml'

# where the sessions are staged when INTEGRATION_DATA_DIR is not set. A folder of its own, kept
# out of version control, so the staged raw data stays separate from the fixtures next to it
DEFAULT_DESTINATION_ROOT = SESSIONS_FOR_TESTS_FILE.parent / 'photometry'


def load_sessions_for_tests() -> list[dict]:
    """Read the sessions to stage, the same ones the integration tests run against.

    Returns
    -------
    list[dict]
        One entry per declared session, with its 'eid', 'session_path' and 'comment'.
    """
    with open(SESSIONS_FOR_TESTS_FILE, 'r') as file_handle:
        return yaml.safe_load(file_handle)


def get_one(location: str) -> ONE:
    """Instantiate the ONE flavour matching the location.

    Parameters
    ----------
    location : str
        Where the data is taken from, one of `LOCATIONS`.

    Returns
    -------
    ONE
        A regular ONE instance, or an `OneSdsc` instance when running on SDSC.
    """
    if location == 'sdsc':
        # only available on the SDSC infrastructure
        from deploy.iblsdsc import OneSdsc

        return OneSdsc()
    return ONE()


def get_destination_root(dry: bool = False) -> Path:
    """Return the fixture root and make sure it is a valid ibllib integration data root.

    Parameters
    ----------
    dry : bool, optional
        If True, report what would be done without writing anything, by default False.

    Returns
    -------
    Path
        The folder set by `INTEGRATION_DATA_DIR`, falling back to `DEFAULT_DESTINATION_ROOT`.
    """
    integration_data_dir = os.environ.get('INTEGRATION_DATA_DIR')
    destination_root = Path(integration_data_dir) if integration_data_dir else DEFAULT_DESTINATION_ROOT
    _logger.info(f'staging into {destination_root}')

    # ibllib's IntegrationTest validates a data root by the presence of this folder
    if not dry:
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


def copy_to_fixture_root(files: list[ALFPath], destination_root: Path, location: str, dry: bool = False) -> None:
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
    dry : bool, optional
        If True, report what would be copied without writing anything, by default False.
    """
    n_copied, n_skipped = 0, 0
    for source_file in files:
        source_file = ALFPath(source_file)
        target_file = ALFPath(destination_root / source_file.session_path_short() / source_file.relative_to_session())
        if location == 'sdsc':
            target_file = target_file.without_uuid()
        if not dry:
            target_file.parent.mkdir(parents=True, exist_ok=True)

        # an existing target is only kept when it still matches its source, so a partially
        # written or outdated fixture is repaired by re-running this script
        if is_up_to_date(source_file, target_file):
            _logger.info(f'  skipped (unchanged): {target_file.relative_to_session()}')
            n_skipped += 1
            continue
        if not dry:
            shutil.copy2(source_file, target_file)
            _logger.info(f'  copied: {target_file.relative_to_session()}')
            n_copied += 1
        else:
            _logger.info(f'  would copy: {source_file} to: {target_file}')
            # n_copied += 1

    if not dry:
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


def parse_args() -> argparse.Namespace:
    """Parse the command line arguments.

    Returns
    -------
    argparse.Namespace
        The parsed 'location' and 'dry' arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        '--location',
        choices=LOCATIONS,
        default='local',
        help="where the data is taken from ('local' and 'server' download, 'sdsc' reads the filesystem)",
    )
    parser.add_argument(
        '--dry',
        action='store_true',
        help='report what would be copied without writing anything to the fixture root',
    )
    return parser.parse_args()


def main(location: str = 'local', dry: bool = False) -> None:
    """Stage every session declared in the sessions file into the fixture root.

    Parameters
    ----------
    location : str, optional
        Where the data is taken from, one of `LOCATIONS`, by default 'local'.
    dry : bool, optional
        If True, report what would be copied without writing anything, by default False.
    """
    if dry:
        _logger.info('dry run: nothing will be written to the fixture root')
    destination_root = get_destination_root(dry=dry)
    one = get_one(location)

    failed_sessions = []
    for session in load_sessions_for_tests():
        eid, session_path = session['eid'], session['session_path']
        _logger.info(f'staging {session_path} ({session["comment"]}) from {location}')
        try:
            if location == 'sdsc':
                files = gather_sdsc_files(eid, one)
            else:
                files = gather_downloaded_files(eid, one)
            copy_to_fixture_root(files, destination_root, location, dry=dry)
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
    args = parse_args()
    main(location=args.location, dry=args.dry)
