#!/usr/bin/env bash
# Runs INSIDE the Lightning job image.
# Assumes git is present, the repo is cloned and checked out at the commit under test, and the
# current directory is the repo root - all handled by the dispatch command.
set -euo pipefail

: "${PY_VERSION:?PY_VERSION must be set}"

echo "installing uv"
pip install --quiet uv

echo "setting up python ${PY_VERSION}"
uv python install "${PY_VERSION}"
uv venv --python "${PY_VERSION}" /workspace/venv
# shellcheck disable=SC1091
source /workspace/venv/bin/activate

# a fresh resolve rather than uv.lock, so the run picks up the current state of the git
# dependencies - notably the ibllib branch the integration test support lives on
echo "installing project"
uv pip install -e ".[analysis]"

echo "env ready ($(python --version))"
