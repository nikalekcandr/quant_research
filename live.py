"""Live logger + live forecasts. Keep this running in a terminal for the whole live period.

    python live.py              # snapshots every 10 minutes, forecasts when they are due
    python live.py --once       # one snapshot and exit (to test the connection)
    python live.py --no-forecast

What it writes (append-only, one gzip CSV per day):
    data/live/pm_books/YYYY-MM-DD.csv.gz     Polymarket ladders: best bid/ask, sizes, depth for every strike
    data/live/deribit_chain/YYYY-MM-DD.csv.gz Deribit BTC option chain: bid/ask/mark/mark_iv for every option
    data/live/kalshi/YYYY-MM-DD.csv.gz        Kalshi hourly ladder: top of book for quoted strikes
    data/live/kalshi_range/YYYY-MM-DD.csv.gz  Kalshi hourly range buckets (series KXBTC), same settlement
    data/live/binance/YYYY-MM-DD.csv.gz       BTC/USDT last price at the snapshot

Why: order books cannot be downloaded after the fact. Without them the trading test would have to assume
spreads instead of measuring them, and the Deribit implied volatility would come only from trades.

The computer must stay awake (disable sleep while plugged in) and online.
"""
from __future__ import annotations

import argparse
import sys
import time
import traceback

import pandas as pd

import config
from src.download import binance, deribit, kalshi, polymarket, venues
from src.utils import Blocked, Http, append_csv_gz, log, utcnow

INTERVAL_MIN = 10   # order-book snapshots every 10 minutes
WAKE_MIN = 5        # the loop wakes every 5 minutes so that hourly forecasts are made at HH:05


PAUSED: dict = {}   # source -> time until which it is skipped after a 401/403
LATE_WAKE = pd.Timedelta(minutes=2)   # a wake-up later than this means the computer slept or the process froze


def keep_awake():
    """Ask Windows not to put the computer to idle sleep while this process runs (no admin rights needed).
    It does not stop sleep from closing a laptop lid or from the Start menu."""
    if sys.platform != "win32":
        return
    try:
        import ctypes
        ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
        if ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED):
            log("windows: idle sleep blocked while live.py runs")
    except Exception as e:
        log(f"windows: could not block idle sleep ({e!r}); turn sleep off in the power settings")


def kalshi_series(http: Http) -> list[str]:
    """Bitcoin series on Kalshi, discovered once a day and cached in data/live/kalshi_series.json."""
    import json
    f = config.LIVE / "kalshi_series.json"
    now = pd.Timestamp(utcnow())
    try:
        c = json.loads(f.read_text(encoding="utf-8"))
        if now - pd.Timestamp(c["ts"]) < pd.Timedelta(hours=24) and c["series"]:
            return c["series"]
    except Exception:
        pass
    series = venues.kalshi_btc_series(http)
    f.write_text(json.dumps({"ts": str(now), "series": series}), encoding="utf-8")
    log(f"kalshi: {len(series)} Bitcoin series: {', '.join(series)}")
    return series


def snapshot(http: Http, pm_events: list[dict]):
    snap = pd.Timestamp(utcnow()).floor("s")
    day = f"{snap:%Y-%m-%d}"
    config.LIVE.mkdir(parents=True, exist_ok=True)
    jobs = {
        "pm_books": lambda: polymarket.snapshot_books(http, pm_events, snap),
        "deribit_chain": lambda: deribit.snapshot_chain(http, snap),
        "kalshi": lambda: kalshi.snapshot(http, snap),
        "kalshi_range": lambda: kalshi.snapshot(http, snap, series="KXBTC"),
        "binance": lambda: pd.DataFrame([{"snap_ts": snap, "price": float(binance.api_get(
            http, "/api/v3/ticker/price", {"symbol": "BTCUSDT"})["price"])}]),
    }
    # v6.1: other Bitcoin markets and a second options venue (see src/download/venues.py)
    jobs.update({
        "kalshi_other": lambda: venues.kalshi_other(http, snap, kalshi_series(http)),
        "pm_other": lambda: venues.pm_other(http, snap),
        "binance_options": lambda: venues.binance_options(http, snap),
        "okx_options": lambda: venues.okx_options(http, snap),
        "bybit_options": lambda: venues.bybit_options(http, snap),
        "limitless": lambda: venues.limitless_btc(http, snap),
    })
    counts = {}
    for name, job in jobs.items():
        if name in PAUSED and snap < PAUSED[name]:
            counts[name] = "paused (blocked)"
            continue
        try:
            df = job()
            append_csv_gz(df, config.LIVE / name / f"{day}.csv.gz")
            counts[name] = 0 if df is None else len(df)
        except Blocked as e:
            PAUSED[name] = snap + pd.Timedelta(hours=1)
            counts[name] = "BLOCKED, retry in 1h"
            with open(config.LIVE / "errors.log", "a", encoding="utf-8") as f:
                f.write(f"{snap} {name}: {e}\n")
        except Exception as e:  # one failing source must not stop the others
            counts[name] = f"ERROR {e!r}"[:120]
            with open(config.LIVE / "errors.log", "a", encoding="utf-8") as f:
                f.write(f"{snap} {name}\n{traceback.format_exc()}\n")
    log(f"snapshot {snap:%Y-%m-%d %H:%M} " + ", ".join(f"{k}={v}" for k, v in counts.items()))
    return snap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--no-forecast", action="store_true")
    args = ap.parse_args()
    config.LIVE.mkdir(parents=True, exist_ok=True)
    keep_awake()
    http = Http(min_interval=0.05)
    pm_events, pm_refreshed = [], None
    expected = None   # when the previous sleep should have ended
    forecaster = None
    if not args.no_forecast:
        try:
            from src import live_forecast
            forecaster = live_forecast
        except Exception as e:
            log(f"live forecasts disabled ({e!r})")
    while True:
        now = pd.Timestamp(utcnow())
        if expected is not None and now - expected > LATE_WAKE:
            gap = (now - expected).total_seconds() / 60
            log(f"WARNING: woke up {gap:.0f} min late — the computer slept or the window was frozen "
                f"(a click in the terminal pauses it); snapshots in between are lost")
            with open(config.LIVE / "errors.log", "a", encoding="utf-8") as f:
                f.write(f"{now} gap: woke {gap:.1f} min late (expected {expected})\n")
        if pm_refreshed is None or now - pm_refreshed > pd.Timedelta(hours=1):
            try:
                pm_events = polymarket.open_events(http)
                pm_refreshed = now
                log(f"polymarket: {len(pm_events)} open ladders: " + ", ".join(e['slug'] for e in pm_events))
            except Exception as e:
                log(f"polymarket event refresh failed: {e!r}")
        now = pd.Timestamp(utcnow())
        snap = now.floor("s")
        if args.once or now.minute % INTERVAL_MIN < WAKE_MIN:
            snap = snapshot(http, pm_events)
        if forecaster is not None:
            try:
                forecaster.run_due(snap)
            except Exception:
                with open(config.LIVE / "errors.log", "a", encoding="utf-8") as f:
                    f.write(f"{snap} forecast\n{traceback.format_exc()}\n")
                log("live forecast failed, see data/live/errors.log")
        if args.once:
            break
        nxt = (pd.Timestamp(utcnow()).floor(f"{WAKE_MIN}min") + pd.Timedelta(minutes=WAKE_MIN, seconds=5))
        expected = nxt
        time.sleep(max(1.0, (nxt - pd.Timestamp(utcnow())).total_seconds()))


if __name__ == "__main__":
    main()
