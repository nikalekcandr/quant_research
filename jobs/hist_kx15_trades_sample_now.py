"""Kalshi KXBTC15M prints, sampled and aggregated (a full market has ~30 000 prints, a day ~1 million).
Six markets per day (closes at hh:15 UTC for hh = 1, 5, 9, 13, 17, 21) over the last 220 days, every page of prints,
aggregated per market × taker side × 1¢ price × minute before the close: contracts, YES notional, prints.
-> data/raw/kx15_trades_agg/<day>.csv   (resumable by day; newest first)"""
from __future__ import annotations

import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from src.utils import Http  # noqa: E402

KX = "https://api.elections.kalshi.com/trade-api/v2"
OUT = config.RAW / "kx15_trades_agg"
MK = config.RAW / "kx15" / "markets.csv"
HOURS = (1, 5, 9, 13, 17, 21)
DAYS = 220
MAX_PAGES = 120


def say(m):
    print(f"[{datetime.now():%H:%M:%S}] {m}", flush=True)


def f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return np.nan


def prints(http, tk, hist):
    for path in (("/historical/trades", "/markets/trades") if hist else ("/markets/trades", "/historical/trades")):
        rows, cur = [], None
        for _ in range(MAX_PAGES):
            p = {"ticker": tk, "limit": 1000}
            if cur:
                p["cursor"] = cur
            try:
                res = http.get(f"{KX}{path}", p) or {}
            except Exception:
                break
            tr = res.get("trades") or []
            rows += [(t.get("created_time"), f(t.get("yes_price_dollars", t.get("yes_price"))),
                      f(t.get("count_fp", t.get("count"))), t.get("taker_side")) for t in tr]
            cur = res.get("cursor")
            if not cur or not tr:
                break
        if rows:
            return rows
    return []


def aggregate(tk, close, rows):
    d = pd.DataFrame(rows, columns=["ts", "p", "n", "side"]).dropna()
    if d.empty:
        return d
    d["p"] = np.where(d["p"] > 1, d["p"] / 100, d["p"])
    d["ts"] = pd.to_datetime(d["ts"], utc=True, format="mixed")
    d["left"] = np.floor((close - d["ts"]).dt.total_seconds() / 60).clip(0, 15).astype(int)
    d["pc"] = (d["p"] * 100).round().clip(0, 100).astype(int)
    d["notional"] = d["n"] * d["p"]
    g = d.groupby(["side", "pc", "left"]).agg(contracts=("n", "sum"), notional=("notional", "sum"),
                                              prints=("n", "size")).reset_index()
    g.insert(0, "ticker", tk)
    return g


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    mk = pd.read_csv(MK)
    mk["close_time"] = pd.to_datetime(mk["close_time"], utc=True)
    mk = mk[(mk["close_time"].dt.minute == 15) & mk["close_time"].dt.hour.isin(HOURS)]
    mk["day"] = mk["close_time"].dt.date
    today = datetime.now(timezone.utc).date()
    days = [d for d in sorted(mk["day"].unique(), reverse=True) if d >= today - timedelta(days=DAYS) and d < today]
    http = Http(min_interval=0.15, retries=4)
    say(f"{len(days)} days x {len(HOURS)} markets")
    for i, d in enumerate(days):
        fo = OUT / f"{d}.csv"
        if fo.exists() and fo.stat().st_size > 1000:
            continue
        g = mk[mk["day"] == d]
        with ThreadPoolExecutor(max_workers=3) as ex:
            parts = list(ex.map(lambda r: aggregate(r.ticker, r.close_time, prints(http, r.ticker, bool(r.historical))),
                                g.itertuples()))
        out = pd.concat([p for p in parts if len(p)], ignore_index=True) if any(len(p) for p in parts) else pd.DataFrame()
        out.to_csv(fo, index=False)
        say(f"{d}: {len(g)} markets, {int(out['prints'].sum()) if len(out) else 0} prints ({i + 1}/{len(days)})")
    say("done")


if __name__ == "__main__":
    main()
