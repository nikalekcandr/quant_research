"""One cycle of the live logger + live forecasts, for the Windows Task Scheduler (every 5 minutes).

    python live_once.py            one cycle: order-book snapshot (at :00, :10, :20, ...) and every forecast due now
    python live_once.py --force-snapshot

Same work as one turn of the loop in live.py, but the scheduler starts it, so nothing is lost when a terminal
is closed, frozen by a click, or the laptop sleeps: the task is registered with "wake the computer to run".
Logs go to logs/live_YYYYMMDD.log. A run that finds another one still working exits at once.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback

import pandas as pd

import config
from src import runner
from src.utils import Http, log, utcnow

PM_EVENTS = lambda: config.LIVE / "pm_events.json"          # noqa: E731  open Polymarket ladders, refreshed hourly
PAUSED = lambda: config.LIVE / "paused.json"                # noqa: E731  sources blocked (401/403) and until when
RUN_LOCK = lambda: config.LIVE / ".live_once.lock"          # noqa: E731


def _read_json(p, default):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return default


def pm_events(http: Http) -> list[dict]:
    from src.download import polymarket
    cache = _read_json(PM_EVENTS(), {})
    now = pd.Timestamp(utcnow())
    if cache.get("ts") and now - pd.Timestamp(cache["ts"]) < pd.Timedelta(hours=1):
        return cache["events"]
    try:
        ev = polymarket.open_events(http)
        PM_EVENTS().write_text(json.dumps({"ts": str(now), "events": ev}), encoding="utf-8")
        log(f"polymarket: {len(ev)} open ladders")
        return ev
    except Exception as e:
        log(f"polymarket event refresh failed: {e!r}")
        return cache.get("events", [])


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--force-snapshot", action="store_true")
    ap.add_argument("--no-forecast", action="store_true")
    a = ap.parse_args(argv)
    config.LIVE.mkdir(parents=True, exist_ok=True)
    runner.log_to_file("live", also_console=sys.__stdout__ is not None)
    lock = RUN_LOCK()
    if lock.exists() and time.time() - lock.stat().st_mtime < 4 * 60:
        log("another live_once run is still working; exit")
        return
    lock.write_text(str(os.getpid()), encoding="utf-8")
    try:
        import live   # the snapshot code of the loop version
        live.PAUSED.update({k: pd.Timestamp(v) for k, v in _read_json(PAUSED(), {}).items()})
        http = Http(min_interval=0.05, retries=3)
        now = pd.Timestamp(utcnow())
        # forecasts first: an hourly forecast must be made within 2 minutes of HH:05
        if not a.no_forecast:
            try:
                from src import live_forecast
                live_forecast.run_due(now)
            except Exception:
                log("live forecast failed:\n" + traceback.format_exc())
        if a.force_snapshot or now.minute % live.INTERVAL_MIN < live.WAKE_MIN:
            live.snapshot(http, pm_events(http))
        PAUSED().write_text(json.dumps({k: str(v) for k, v in live.PAUSED.items()
                                        if v > pd.Timestamp(utcnow())}), encoding="utf-8")
    finally:
        try:
            lock.unlink()
        except FileNotFoundError:
            pass


if __name__ == "__main__":
    main()
