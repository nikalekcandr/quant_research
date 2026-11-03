#!/usr/bin/env bash
# What is running and how it is doing.
cd "${DIR:-$HOME/quant_research}"
systemctl --user --no-pager --plain list-units 'btcvol-*'
echo; echo "lab heartbeat:"; cat data/lab/heartbeat.json 2>/dev/null; echo
for m in shadow demo live; do
  [ -f data/trader/heartbeat_$m.json ] && { echo "executor $m:"; cat data/trader/heartbeat_$m.json; echo; }
done
.venv/bin/python -m trader lab-sprt 2>/dev/null | tail -2
