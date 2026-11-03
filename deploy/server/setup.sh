#!/usr/bin/env bash
# One-time setup of the lab + executor on an Ubuntu 22.04/24.04 server (2 vCPU, 4 GB RAM, 40+ GB disk).
#   curl -fsSL https://raw.githubusercontent.com/nikalekcandr/quant_research/main/deploy/server/setup.sh | bash
# or, after cloning:  bash deploy/server/setup.sh
# Creates user-level systemd services (no root needed after the apt step):
#   btcvol-lab           lab_daemon.py, restarted whenever it exits (code change, crash)
#   btcvol-shadow        python -m trader run --mode shadow (production books, no orders)
#   btcvol-demo          python -m trader run --profile demo_plumbing (needs the demo key in data/trader/keys/demo/)
#   btcvol-live.timer    live_once.py every 10 minutes (the course-project logger)
#   btcvol-daily.timer   daily_update.py at 06:20 UTC
# Live trading is NOT installed here: start it by hand (docs/SERVER.md §5).
set -euo pipefail
REPO=${REPO:-https://github.com/nikalekcandr/quant_research.git}
DIR=${DIR:-$HOME/quant_research}
BRANCH=${BRANCH:-main}

sudo apt-get update -qq
sudo apt-get install -y -qq python3 python3-venv python3-pip git tmux chrony >/dev/null
sudo timedatectl set-ntp true || true            # request signatures carry a timestamp: keep the clock exact

[ -d "$DIR/.git" ] || git clone -q -b "$BRANCH" "$REPO" "$DIR"
cd "$DIR"
python3 -m venv .venv
.venv/bin/pip install -q --upgrade pip
.venv/bin/pip install -q -r requirements.txt cryptography

# jobs/ holds one-off scripts the lab starts on the Windows computer (history downloads, packing, the Windows
# shadow/demo runners). On the server they must not run: mark every existing job as done.
mkdir -p jobs/done logs/jobs data/trader/keys/demo data/trader/keys/prod
for f in jobs/*.py; do
  n=$(basename "$f" .py)
  [ -f "jobs/done/$n.json" ] || echo '{"status": "skipped on the server"}' > "jobs/done/$n.json"
done

.venv/bin/python -c "import lab.engine, lab.report, lab_daemon, trader.executor; print('imports ok')"

UNIT=$HOME/.config/systemd/user
mkdir -p "$UNIT"
svc() {  # name, description, command
cat > "$UNIT/$1.service" <<UNITEOF
[Unit]
Description=$2
After=network-online.target

[Service]
WorkingDirectory=$DIR
ExecStart=$DIR/.venv/bin/python $3
Restart=always
RestartSec=30
Environment=PYTHONUNBUFFERED=1
StandardOutput=append:$DIR/logs/$1.out
StandardError=append:$DIR/logs/$1.out

[Install]
WantedBy=default.target
UNITEOF
}
svc btcvol-lab    "BTC alpha lab (wings)"                 "lab_daemon.py"
svc btcvol-shadow "Kalshi executor, shadow mode"          "-m trader run --mode shadow"
svc btcvol-demo   "Kalshi executor, demo exchange"        "-m trader run --profile demo_plumbing"

oneshot() {  # name, command, OnCalendar
cat > "$UNIT/$1.service" <<UNITEOF
[Unit]
Description=$1
[Service]
Type=oneshot
WorkingDirectory=$DIR
ExecStart=$DIR/.venv/bin/python $2
UNITEOF
cat > "$UNIT/$1.timer" <<UNITEOF
[Unit]
Description=$1 timer
[Timer]
OnCalendar=$3
Persistent=true
[Install]
WantedBy=timers.target
UNITEOF
}
oneshot btcvol-live  "live_once.py"     "*:0/10"
oneshot btcvol-daily "daily_update.py"  "*-*-* 06:20:00 UTC"

loginctl enable-linger "$USER" || sudo loginctl enable-linger "$USER"   # services keep running after logout
systemctl --user daemon-reload
systemctl --user enable --now btcvol-lab btcvol-shadow btcvol-live.timer btcvol-daily.timer
echo
echo "Running: lab, shadow executor, logger timers.  Status: bash deploy/server/status.sh"
echo "Demo: copy the key into $DIR/data/trader/keys/demo/ then: systemctl --user enable --now btcvol-demo"
