#!/usr/bin/env bash
# One command for the night: start the download in the background, report in the morning.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=python
if [ -x .venv/bin/python ]; then PY=.venv/bin/python; fi
exec "$PY" scripts/night.py "$@"
