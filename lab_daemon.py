"""Alpha lab daemon: runs 24/7 (Task Scheduler task 'btcvol-lab' keeps it alive), see lab/*.py.

    python lab_daemon.py              run forever (what the scheduled task starts)
    python lab_daemon.py --once       one minute cycle + report, then exit (test)
    python lab_daemon.py --report     only rebuild reports/lab.md and reports/lab.html

Writes data/lab/ (ticks, fast ticks, market quotes, ledger.csv, positions.json, volzoo.csv),
reports/lab.md, reports/lab.html, results/lab/*.csv, logs/lab_YYYYMMDD.log.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time

import pandas as pd

import config
from lab.core import HEARTBEAT, code_version, log, log_exc, utcnow


def another_running() -> bool:
    try:
        hb = json.loads(HEARTBEAT.read_text(encoding="utf-8"))
        age = (utcnow() - pd.Timestamp(hb["ts"])).total_seconds()
        return age < 150 and int(hb.get("pid", -1)) != os.getpid()
    except Exception:
        return False


def keep_awake():
    if sys.platform == "win32":
        try:
            import ctypes
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
        except Exception:
            pass


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args(argv)
    if not (a.once or a.report) and another_running():
        print("lab already running (fresh heartbeat); exit")
        return
    from lab.engine import Lab
    lab = Lab()
    if a.report:
        lab.report(utcnow(), force=True)
        return
    keep_awake()
    if a.once:
        now = utcnow()
        lab.minute(now)
        lab.hourly(now.floor("h") + pd.Timedelta(minutes=5))
        lab.report(now, force=True)
        log("lab: one cycle done")
        return
    stop = threading.Event()
    from lab.signals import RETIRED
    loops = [lab.fast_loop, lab.fast_markets_loop]
    if not {"D19", "D20"} <= set(RETIRED):          # the trade-flow websocket serves only D19/D20
        loops.append(lab.flow.run)
    for target in loops:
        threading.Thread(target=target, args=(stop,), daemon=True).start()
    lab.report(utcnow(), force=True)
    while True:
        try:
            now = utcnow()
            lab.minute(now)
            lab.hourly(now)
            lab.report(now)
        except KeyboardInterrupt:
            break
        except Exception:
            log_exc("main loop")
        try:
            from lab import jobs
            jobs.poll()
        except Exception:
            log_exc("jobs")
        keep_awake()
        if utcnow().minute % 5 == 4 and code_version() != lab.book.code:
            # a patched rule file: exit cleanly; the scheduled task starts the new code within 5 minutes
            log("lab: code changed on disk, restarting with the new version")
            lab.book.save()
            stop.set()
            try:
                HEARTBEAT.unlink()
            except Exception:
                pass
            return
        nxt = utcnow().floor("min") + pd.Timedelta(minutes=1, seconds=3)
        time.sleep(max(1.0, (nxt - utcnow()).total_seconds()))
    stop.set()


if __name__ == "__main__":
    main()
