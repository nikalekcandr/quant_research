"""Nightly update: download the new data of every source, rebuild all tables and models in side folders,
swap them in, score the live forecasts and write the out-of-sample report.

    python daily_update.py               what the scheduled task runs every night
    python daily_update.py --no-download only rebuild (e.g. after a manual download)

Nothing about the models changes here: same code, same windows (checked by oos_report.py against the hashes
of the version frozen on 15 October). The only thing that grows is the data, so every day adds a few
genuinely out-of-sample forecasts.
"""
from __future__ import annotations

import argparse
import traceback

import pandas as pd

import config
from src import runner
from src.utils import log


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-download", action="store_true")
    ap.add_argument("--no-kalshi", action="store_true")
    a = ap.parse_args(argv)
    runner.log_to_file("daily_update")
    runner.keep_awake()
    if runner.busy("backfill"):
        log("backfill is running: the nightly update is skipped today")
        return
    with runner.lock("daily_update"):
        log(f"===== daily update {pd.Timestamp.now():%Y-%m-%d %H:%M} =====")
        if not a.no_download:
            # Kalshi: only the new period (days already on disk are skipped in a second)
            runner.download_all(kalshi=not a.no_kalshi, kalshi_first=config.NEW_DATA_START)
            # the old history is the backfill's job, not the nightly one
        orig = runner.stage_paths()
        try:
            runner.run_steps()
        except Exception:
            log("!!! rebuild failed; the previous tables stay in place")
            traceback.print_exc()
            for k, v in orig.items():
                setattr(config, k, v)
            raise
        runner.swap_in(orig)
        rep = config.REPORTS / "draft_report.md"
        if rep.exists():   # figure links were written relative to the *_next folders
            rep.write_text(rep.read_text(encoding="utf-8").replace("results_next", "results"), encoding="utf-8")
        from oos_report import main as oos
        oos([])
        for name in ("alpha_live", "alpha_extra", "research"):   # live alpha; historical follow-ups; research layer
            try:
                __import__(name).main([])
            except Exception:
                log(f"!!! {name} failed")
                traceback.print_exc()
        from src.live_forecast import ledger_head
        head = ledger_head()
        if head:
            log(f"ledger head: {head['n']} live forecasts, chain hash {head['chain'][:16]}… "
                f"(send this line to yourself or the teacher: it proves which forecasts existed today)")
    log("daily update finished")


if __name__ == "__main__":
    main()
