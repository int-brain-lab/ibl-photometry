#!/usr/bin/env python3
"""Runner side entry point: submit one blocking Lightning job per matrix leg.

The integration tests need the staged sessions, which are too large to put on a GitHub runner.
They are held in a Lightning data connection instead, mounted into the job at the path given by
INTEGRATION_DATA_DIR. Setting that variable is what makes the integration tests run rather than
skip, so everything outside this job stays a unit test run.

The job clones the repository at the commit under test and hands off to `scripts/ci/run_tests.sh`,
so the steps that actually build the environment and run the suite are versioned with the code.
"""

import os
import re
import sys

from lightning_sdk import Job, Machine, Status

# base image for the Lightning job. python:3.12 ships git, bash and pip; uv installs the actual
# test interpreter from PY_VERSION, so this version only bootstraps `pip install uv`
CI_IMAGE = 'python:3.12'


def sanitize(value: str) -> str:
    """Reduce a string to the characters a Lightning job name accepts.

    Parameters
    ----------
    value : str
        The string to sanitize.

    Returns
    -------
    str
        Lowercased, with every character outside [a-zA-Z0-9-] replaced by a dash.
    """
    return re.sub(r'[^a-zA-Z0-9-]', '-', value).strip('-').lower()


PY_VERSION = os.environ['PY_VERSION']
GITHUB_SHA = os.environ['GITHUB_SHA']
GITHUB_RUN_ID = os.environ['GITHUB_RUN_ID']
RUN_ATTEMPT = os.environ.get('GITHUB_RUN_ATTEMPT', '1')
GITHUB_REPOSITORY = os.environ['GITHUB_REPOSITORY']  # "owner/repo"
GITHUB_REF_NAME = os.environ.get('GITHUB_REF_NAME', '')
REPO_URL = os.environ['REPO_URL']
TEAMSPACE = os.environ['LIGHTNING_TEAMSPACE']  # "owner/teamspace"
DATA_CONNECTION = os.environ['LIGHTNING_DATA_CONNECTION']
INTEGRATION_DATA_DIR = os.environ.get('INTEGRATION_DATA_DIR') or '/data'
PR_NUMBER = os.environ.get('PR_NUMBER', '')
JOB_TIMEOUT = int(os.environ.get('JOB_TIMEOUT_SECONDS', '7200'))

# GITHUB_RUN_ID is only unique per repository, and one teamspace serves several of them
JOB_NAME = f'{sanitize(GITHUB_REPOSITORY)}-py{PY_VERSION.replace(".", "")}-{GITHUB_RUN_ID}-{RUN_ATTEMPT}'

owner, _, teamspace = TEAMSPACE.partition('/')

# for pull_request events GITHUB_SHA is a virtual merge commit that a plain clone does not have,
# so the merge ref is fetched explicitly
if PR_NUMBER:
    checkout_command = f'git fetch origin refs/pull/{PR_NUMBER}/merge && git checkout FETCH_HEAD'
else:
    checkout_command = 'git checkout "$GITHUB_SHA"'

command = (
    'set -e && '
    "command -v git >/dev/null || { echo 'ERROR: git not present in image'; exit 127; } && "
    'git clone --no-checkout "$REPO_URL" /workspace/repo && '
    'cd /workspace/repo && '
    f'{checkout_command} && '
    'bash scripts/ci/run_tests.sh'
)

print(f'submitting {JOB_NAME} (image={CI_IMAGE}, python={PY_VERSION})', flush=True)

job = Job.run(
    name=JOB_NAME,
    image=CI_IMAGE,
    machine=Machine.CPU,
    command=command,
    env={
        'PY_VERSION': PY_VERSION,
        'REPO_URL': REPO_URL,
        # setting this is what enables the integration tests
        'INTEGRATION_DATA_DIR': INTEGRATION_DATA_DIR,
        'GITHUB_SHA': GITHUB_SHA,
        'GITHUB_REPOSITORY': GITHUB_REPOSITORY,
        'GITHUB_REF_NAME': GITHUB_REF_NAME,
    },
    path_mappings={INTEGRATION_DATA_DIR: DATA_CONNECTION},
    teamspace=teamspace,
    org=owner,
    interruptible=False,
)

try:
    job.wait(interval=15, timeout=JOB_TIMEOUT, stop_on_timeout=True)
except TimeoutError:
    print(f'job exceeded {JOB_TIMEOUT}s and was stopped', flush=True)

print('================ JOB LOGS ================', flush=True)
try:
    print(job.logs, flush=True)
except Exception as exception:  # noqa: BLE001 - logs are best effort, never fail the run over them
    print(f'could not fetch logs: {exception}', flush=True)
print('==========================================', flush=True)

status = job.status
print(f'final status: {status}', flush=True)
sys.exit(0 if status == Status.Completed else 1)
