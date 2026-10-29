"""One-off check of every venue the logger reads: what it returns, how markets are named, whether the
network is allowed. Saves raw samples to data/live/discovery/ so the parsers can be checked against reality.

    python discover_venues.py
"""
from __future__ import annotations

import json
import traceback

import pandas as pd

import config
from src import runner
from src.download import venues
from src.utils import Http, ensure, log, utcnow


def dump(name, obj):
    d = ensure(config.LIVE / "discovery")
    (d / f"{name}.json").write_text(json.dumps(obj, indent=1, default=str)[:2_000_000], encoding="utf-8")


def main():
    runner.log_to_file("discover_venues")
    http = Http(min_interval=0.06, retries=2)
    snap = pd.Timestamp(utcnow()).floor("s")
    summary = {}
    # Kalshi
    try:
        series = venues.kalshi_btc_series(http)
        summary["kalshi_series"] = series
        df = venues.kalshi_other(http, snap, series)
        summary["kalshi_other_markets"] = len(df)
        if len(df):
            summary["kalshi_other_by_series"] = df.groupby("series").size().to_dict()
            dump("kalshi_other_sample", df.groupby("series").head(5).to_dict("records"))
        raw = http.get(f"{venues.KALSHI}/series", {"category": "Crypto"})
        dump("kalshi_series_raw", raw)
    except Exception as e:
        summary["kalshi_error"] = repr(e)
        traceback.print_exc()
    # Polymarket
    try:
        evs = venues.pm_btc_events(http)
        dump("pm_btc_events_raw", evs[:40])
        df = venues.pm_other(http, snap, include_ladders=True)
        summary["pm_markets"] = len(df)
        if len(df):
            summary["pm_by_kind"] = df.groupby("kind").size().to_dict()
            summary["pm_titles"] = df.groupby("kind")["title"].apply(lambda s: sorted(set(s))[:6]).to_dict()
            summary["pm_best_bid_present"] = float(df["best_bid"].notna().mean())
    except Exception as e:
        summary["pm_error"] = repr(e)
        traceback.print_exc()
    # Binance options
    try:
        df = venues.binance_options(http, snap)
        summary["binance_options"] = len(df)
        if len(df):
            summary["binance_options_expiries"] = sorted(df["expiry"].astype(str).unique().tolist())[:10]
            dump("binance_options_sample", df.head(20).to_dict("records"))
    except Exception as e:
        summary["binance_options_error"] = repr(e)
    # OKX and Bybit options
    for name, fn in (("okx_options", venues.okx_options), ("bybit_options", venues.bybit_options)):
        try:
            df = fn(http, snap)
            summary[name] = len(df)
            if len(df):
                summary[name + "_expiries"] = sorted(df["expiry"].astype(str).unique().tolist())[:8]
                summary[name + "_iv_median"] = float(df["iv"].median())
                dump(name + "_sample", df.head(20).to_dict("records"))
        except Exception as e:
            summary[name + "_error"] = repr(e)
    # Limitless
    try:
        lim = venues.limitless_probe(http)
        dump("limitless_raw", lim)
        summary["limitless"] = {k: (type(v).__name__ if not isinstance(v, str) else v[:120]) for k, v in lim.items()}
    except Exception as e:
        summary["limitless_error"] = repr(e)
    dump("summary", summary)
    print(json.dumps(summary, indent=1, default=str, ensure_ascii=False))
    log(f"saved to {config.LIVE / 'discovery'}")


if __name__ == "__main__":
    main()
