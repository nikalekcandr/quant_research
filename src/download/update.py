"""One entry point for the downloads, with the audit before and after.

    python -m src.download.update --audit                 what is stale, how stale, and the command for each source
    python -m src.download.update --all                   everything that is behind, newest data first
    python -m src.download.update --only kalshi,binance   just these sources
    python -m src.download.update --only kalshi --since 2026-10-27 --until 2026-11-27
    python -m src.download.update --all --workers 4        Kalshi: 4 days at a time (~10 s per day instead of 60)
    python -m src.download.update --only kalshi --build   then rebuild the interim tables of those sources

Run with no arguments at all (an IDE "Run" button) it only prints the audit and the hint, it never starts a
download: pulling every source by accident is exactly the kind of surprise a "Run" click must not produce.

Both invocation styles work: `python -m src.download.update` from the repository root and
`python src/download/update.py` (or the green triangle in an IDE), which is why the file bootstraps sys.path
below. Without that, Python puts src/download/ on the path instead of the root and `import config` fails with
`ModuleNotFoundError: No module named 'config'`.

Every downloader skips what is already on disk, so an interrupted run is resumed by running the same command
again. Kalshi is the slow one (~80 s per day of hourly ladders, ~50 requests per day) — always run it first for
the recent weeks, then let the long backfill run in the background.
"""
from __future__ import annotations

import argparse
import sys
import time
import traceback

import pandas as pd

# --- runnable both ways -------------------------------------------------------
# `python -m src.download.NAME` and `python src/download/NAME.py` (an IDE "Run" button) must both work:
# when this file is executed as a script, Python puts src/download/ on sys.path instead of the repository
# root, and `import config` fails. Put the root back before importing anything from the project.
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
# -----------------------------------------------------------------------------

import config
from src import runner
from src.utils import log

# name -> (what it is, function(since, until, http))
def _binance(since, until, http):
    from src.download import binance
    return binance.download(since, http)


def _funding(since, until, http):
    from src.download import funding
    return funding.download(since, http)


def _deribit(since, until, http):
    from src.download import deribit
    out = {"dvol": len(deribit.download_dvol(config.DVOL_START, http) or [])}
    out["trades"] = len(deribit.download_trades(since, until, http) or [])
    return out


def _polymarket(since, until, http):
    from src.download import polymarket
    return polymarket.download_history(since, until, http)


def _kalshi(since, until, http):
    from src.download import kalshi
    return kalshi.download_history(since, until, http=http, workers=WORKERS, min_interval=MIN_INTERVAL)


def _macro(since, until, http):
    from src.download import macro
    return macro.download(http, dry_run=False)


def _perp(since, until, http):
    from src.download import perp
    return perp.download(since, http)


SOURCES = {
    "binance": ("Binance 1-minute spot candles", lambda s, u, h: _binance(s or config.BINANCE_START, u, h)),
    "funding": ("Binance / Bybit funding rates", lambda s, u, h: _funding(s or config.BINANCE_START, u, h)),
    "deribit": ("Deribit DVOL and option trades", lambda s, u, h: _deribit(s or config.DERIBIT_START, u, h)),
    "polymarket": ("Polymarket daily ladders", lambda s, u, h: _polymarket(s or config.POLYMARKET_START, u, h)),
    "kalshi": ("Kalshi hourly ladders (the slow one: see --workers)", lambda s, u, h: _kalshi(s or config.KALSHI_START, u, h)),
    "macro": ("US macro release calendar", lambda s, u, h: _macro(s, u, h)),
    "perp": ("Binance perp candles + basis", lambda s, u, h: _perp(s, u, h)),
}

# newest data first: the out-of-sample weeks matter more than the deep history
ORDER = ["binance", "funding", "deribit", "kalshi", "polymarket", "perp", "macro"]

WORKERS = 4            # Kalshi: days fetched at the same time (--workers)
MIN_INTERVAL = 0.15    # Kalshi: seconds between requests, aggregate (--min-interval)

BUILDERS = {
    "binance": lambda: __import__("src.download.binance", fromlist=["x"]).build_interim(),
    "kalshi": lambda: __import__("src.download.kalshi", fromlist=["x"]).build_interim(),
    "polymarket": lambda: __import__("src.download.polymarket", fromlist=["x"]).build_interim(),
    "deribit": lambda: __import__("src.download.deribit", fromlist=["x"]).build_interim_trades(),
    "perp": lambda: __import__("src.download.perp", fromlist=["x"]).build_interim(),
}


def plan(only: str | None) -> list[str]:
    """Source names in download order (newest data first)."""
    if not only:
        return ORDER
    wanted = [s.strip() for s in only.split(",") if s.strip()]
    unknown = [s for s in wanted if s not in SOURCES]
    if unknown:
        raise SystemExit(f"unknown source(s): {', '.join(unknown)}; choose from {', '.join(SOURCES)}")
    return [s for s in ORDER if s in wanted]


def audit_table() -> pd.DataFrame:
    from src.download import gaps
    return gaps.audit()


def main(argv=None) -> int:
    global WORKERS, MIN_INTERVAL
    ap = argparse.ArgumentParser(description="download every source that is behind, one source at a time")
    ap.add_argument("--all", action="store_true", help="download every stale source (required to start a download)")
    ap.add_argument("--only", default=None, help="comma-separated subset of " + ", ".join(SOURCES))
    ap.add_argument("--since", default=None, help="first date to download (default: config.*_START)")
    ap.add_argument("--until", default=None, help="last date to download (default: now)")
    ap.add_argument("--audit", action="store_true", help="only print what is stale, do not download")
    ap.add_argument("--build", action="store_true", help="rebuild the interim tables of the chosen sources")
    ap.add_argument("--no-keep-awake", action="store_true", help="do not stop the machine from sleeping")
    ap.add_argument("--workers", type=int, default=WORKERS,
                    help=f"Kalshi: days fetched at the same time (default {WORKERS}); 1 = old sequential behaviour")
    ap.add_argument("--min-interval", type=float, default=MIN_INTERVAL,
                    help=f"Kalshi: seconds between requests, aggregate (default {MIN_INTERVAL}; raise on 429s)")
    a = ap.parse_args(argv)
    WORKERS, MIN_INTERVAL = a.workers, a.min_interval

    table = audit_table()
    if a.audit or not (a.all or a.only):
        print(table.to_string(index=False))
        behind = table[(table["gap days"] > 0.5) & table["command"].str.startswith("python")]
        print(f"\n{len(behind)} source(s) behind.")
        if not (a.all or a.only):
            print("Ничего не скачано: это только диагностика.")
            print("Чтобы качать — добавьте источник:")
            print("  python -m src.download.update --all                     # всё, что отстало")
            print("  python -m src.download.update --only binance,funding    # только эти")
        return 0

    names = ORDER if a.all and not a.only else plan(a.only)
    if not a.no_keep_awake:
        kept = runner.keep_awake()
        log(f"sleep inhibitor: {'installed' if kept else 'not available on this machine'}")

    from src.utils import Http
    http = Http(min_interval=a.min_interval)   # polite: the venues rate-limit, and this is not a race
    failed = []
    for name in names:
        what, fn = SOURCES[name]
        log(f"--- {what} ({a.since or 'default since'} .. {a.until or 'now'})")
        t0 = time.time()
        try:
            out = fn(a.since, a.until, http)
            log(f"--- {name}: ok ({(time.time() - t0) / 60:.1f} min) {out if out is not None else ''}")
        except Exception as e:
            failed.append(name)
            log(f"!!! {name} failed: {e!r}")
            traceback.print_exc()

    if a.build:
        for name in names:
            if name not in BUILDERS:
                continue
            try:
                log(f"--- rebuilding {name}")
                BUILDERS[name]()
            except Exception as e:
                failed.append(f"{name}:build")
                log(f"!!! rebuilding {name} failed: {e!r}")

    print()
    print(audit_table().to_string(index=False))
    if failed:
        log(f"!!! failed: {', '.join(failed)} — the rest is on disk; run the same command again to resume")
        return 1
    log("all sources downloaded")
    return 0


if __name__ == "__main__":
    sys.exit(main())
