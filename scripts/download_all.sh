#!/usr/bin/env bash
# One command: download every dataset that is behind (background), report afterwards.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=python
if [ -x .venv/bin/python ]; then PY=.venv/bin/python; fi
exec "$PY" scripts/download_all.py "$@"
