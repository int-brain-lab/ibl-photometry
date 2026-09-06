#!/usr/bin/env bash
# Entry point for the Lightning CI job, invoked by .github/scripts/dispatch.py.
# The current directory is the repo root, already cloned and checked out.
set -euo pipefail

source scripts/ci/prep_env.sh
# shellcheck disable=SC1091
source /workspace/venv/bin/activate  # ensure the venv is active in this shell too

# INTEGRATION_DATA_DIR is set by the dispatch script and points at the mounted data connection,
# which is what makes the integration tests run instead of skipping themselves
: "${INTEGRATION_DATA_DIR:?INTEGRATION_DATA_DIR must be set, the integration tests would all skip}"
echo "running the full test suite against ${INTEGRATION_DATA_DIR}"

export ONE_SAVE_ON_DELETE=false  # stop ONE from writing its cache tables to disk
export NO_PROGRESSBARS=1  # keep the job logs readable
export INTEGRATION_DATA_WRITABLE=0  # the data connection is read only, so sessions are mirrored

python -m unittest discover -s tests -v -t . -p "test_*.py"
