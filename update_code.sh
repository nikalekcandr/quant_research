#!/usr/bin/env bash
# One command: update only the code of this folder. data/ is never touched (it comes from the exchanges).
set -euo pipefail
cd "$(dirname "$0")"
PY=python
if [ -x .venv/bin/python ]; then PY=.venv/bin/python; fi
exec "$PY" scripts/get_code.py "$@"
