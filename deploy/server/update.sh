#!/usr/bin/env bash
# Pull the latest code from GitHub and restart; the lab also restarts by itself when its code changes.
set -e
cd "${DIR:-$HOME/quant_research}"
git pull -q
.venv/bin/pip install -q -r requirements.txt
for f in jobs/*.py; do n=$(basename "$f" .py); [ -f "jobs/done/$n.json" ] || echo '{"status": "skipped on the server"}' > "jobs/done/$n.json"; done
systemctl --user restart btcvol-shadow
systemctl --user is-enabled btcvol-demo >/dev/null 2>&1 && systemctl --user restart btcvol-demo || true
echo "updated to $(git log --oneline -1)"
