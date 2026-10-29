"""Download ALL available history once, then rebuild every table and model.

    python backfill.py --probe       find the earliest Kalshi / Polymarket history (~2 minutes), write
                                     data/manual/starts.json and print what the backfill will download
    python backfill.py               probe (if not done yet) + download everything missing + rebuild all
    python backfill.py --no-kalshi   everything except Kalshi (Kalshi is the slow source: ~80 s per day)
    python backfill.py --only-build  no downloads, only rebuild (after a download in another window)

Every download skips files that already exist, so an interrupted run is resumed by starting it again.
Order: the new data (from config.NEW_DATA_START) first, then the old history, so that the out-of-sample
period is complete even if the long Kalshi backfill is still running.
"""
from __future__ import annotations

import argparse
import json
import sys

import pandas as pd

import config
from src import runner
from src.utils import Http, log


def probe(write: bool = True) -> dict:
    from src.download import kalshi, polymarket
    http = Http(min_interval=0.06)
    out = {}
    log("probe: earliest Polymarket daily ladder (scan from 2024-01) ...")
    try:
        out["POLYMARKET_START"] = polymarket.probe_start(http, "2024-01-01", config.POLYMARKET_START)
    except Exception as e:
        log(f"probe polymarket failed: {e!r}")
    log("probe: earliest Kalshi KXBTCD hourly event (scan from 2023-01) ...")
    try:
        out["KALSHI_START"] = kalshi.probe_start(http, "2023-01-01", config.KALSHI_START)
    except Exception as e:
        log(f"probe kalshi failed: {e!r}")
    out = {k: v for k, v in out.items() if v}
    starts = [pd.Timestamp(v) for v in out.values()] + [pd.Timestamp(config.POLYMARKET_START), pd.Timestamp(config.KALSHI_START)]
    pm0 = pd.Timestamp(out.get("POLYMARKET_START", config.POLYMARKET_START))
    # Deribit at-the-money IV is a predictor of the Polymarket horizons: start a month before the first ladder
    out["DERIBIT_START"] = min(pd.Timestamp(config.DERIBIT_START), pm0 - pd.Timedelta(days=31)).strftime("%Y-%m-%d")
    # Evaluation starts once the ladder models have their minimum history (60 days of ladders for Polymarket,
    # 14 for Kalshi) plus a month, never later than the original 1 January 2026
    first_ladder = min(starts)
    ev = (first_ladder + pd.Timedelta(days=90)).to_period("M").to_timestamp()
    out["EVAL_START"] = min(ev, pd.Timestamp("2026-01-01")).strftime("%Y-%m-%d")
    log("probe result: " + json.dumps(out))
    if write:
        path = config.MANUAL / "starts.json"
        path.write_text(json.dumps(out, indent=1), encoding="utf-8")
        log(f"written {path}; config.py reads it from now on")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="store_true", help="only probe the earliest history")
    ap.add_argument("--reprobe", action="store_true", help="probe again even if starts.json exists")
    ap.add_argument("--no-kalshi", action="store_true")
    ap.add_argument("--only-build", action="store_true")
    ap.add_argument("--no-build", action="store_true")
    ap.add_argument("--kalshi-from", default=None, help="do not download Kalshi before this date (it costs ~80 s per day)")
    a = ap.parse_args()
    runner.log_to_file("backfill")
    runner.keep_awake()
    if a.probe:
        probe()
        return
    failed = []
    with runner.lock("backfill"):
        if not a.only_build:
            if a.reprobe or not (config.MANUAL / "starts.json").exists():
                probe()
                import importlib
                importlib.reload(config)
            if a.kalshi_from and pd.Timestamp(config.KALSHI_START) < pd.Timestamp(a.kalshi_from):
                config.KALSHI_START = a.kalshi_from
            log(f"history: Polymarket from {config.POLYMARKET_START}, Kalshi from {config.KALSHI_START}, "
                f"Deribit trades from {config.DERIBIT_START}, evaluation from {config.EVAL_START}")
            failed = runner.download_all(kalshi=not a.no_kalshi)
        if not a.no_build:
            # the first full build writes in place (nothing is live yet); later rebuilds use daily_update.py
            runner.run_steps()
            from oos_report import main as oos
            oos([])
    log("backfill finished" + (f" — but these sources failed: {', '.join(failed)}; run it again later" if failed else ""))
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
