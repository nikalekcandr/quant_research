"""Who earns on Kalshi 15-minute Bitcoin markets: the resting (maker) side or the aggressor (taker)?

Prints of settled KXBTC15M markets (data/raw/kx15_trades_agg, aggregated per market × side × 1¢ × minute) is matched with the market's result. A print at
YES price p with taker_side = yes means the taker bought YES at p and the maker sold YES at p (and vice versa).
Per contract:  maker gross = (p − y) if the taker bought YES, (y − p) if the taker bought NO   (y = 1 if YES won)
               taker gross = − maker gross
Fees: taker 0.07·p·(1−p), maker 0.0175·p·(1−p) per contract (see fee()). t-statistics are clustered by market (prints of one market share one outcome).

    python studies/kx15_makers.py <raw dir> [out dir]
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/raw")
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("results/studies")


def fee(p, rate):
    """Kalshi charges ceil(rate·C·P·(1−P)) per order of C contracts; prints here average dozens of contracts, so the
    per-contract fee is rate·P·(1−P) (the rounding is spread over the order)."""
    return rate * p * (1 - p)


def clustered(x, g, w=None):
    """Mean of x (weighted by w) and its t-stat with clusters g."""
    d = pd.DataFrame({"x": x, "g": g, "w": 1.0 if w is None else w})
    tot = d.groupby("g").apply(lambda h: pd.Series({"sx": (h.x * h.w).sum(), "sw": h.w.sum()}), include_groups=False)
    m = tot.sx.sum() / tot.sw.sum()
    u = tot.sx - m * tot.sw
    n = len(tot)
    se = math.sqrt(n / (n - 1) * (u ** 2).sum()) / tot.sw.sum() if n > 1 else np.nan
    return m, (m / se if se else np.nan), n


def load():
    """Aggregated prints (jobs/hist_kx15_trades_sample_now.py): ticker, side, pc, left, contracts, notional, prints."""
    mk = pd.read_csv(RAW / "kx15" / "markets.csv", usecols=["ticker", "close_time", "result"])
    files = [f for f in sorted((RAW / "kx15_trades_agg").glob("*.csv")) if f.stat().st_size > 10]
    tr = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    tr = tr.merge(mk, on="ticker")
    tr = tr[tr["result"].isin(["yes", "no"]) & tr["side"].isin(["yes", "no"]) & (tr["contracts"] > 0)].copy()
    tr["yes_price"] = tr["notional"] / tr["contracts"]
    tr = tr[tr["yes_price"].between(0.005, 0.995)]
    tr["taker_side"] = tr["side"]
    tr["count"] = tr["contracts"]
    tr["left_min"] = tr["left"] + 0.5
    tr["y"] = (tr["result"] == "yes").astype(float)
    p = tr["yes_price"]
    sgn = np.where(tr["taker_side"] == "yes", 1.0, -1.0)        # +1: taker bought YES
    tr["taker_gross"] = sgn * (tr["y"] - p)
    tr["taker_px"] = np.where(sgn > 0, p, 1 - p)                # price the taker paid for his side
    tr["taker_net"] = tr["taker_gross"] - fee(tr["taker_px"], 0.07)
    tr["maker_net"] = -tr["taker_gross"] - fee(tr["taker_px"], 0.0175)
    tr["ts"] = pd.to_datetime(tr["close_time"], utc=True)
    return tr


def table(tr, by, name):
    rows = []
    for k, g in tr.groupby(by, observed=True):
        if g["ticker"].nunique() < 30:
            continue
        mt, tt, n = clustered(g["taker_gross"], g["ticker"], g["count"])
        mm, tm, _ = clustered(g["maker_net"], g["ticker"], g["count"])
        mtn, ttn, _ = clustered(g["taker_net"], g["ticker"], g["count"])
        rows.append({name: str(k), "markets": n, "prints": int(g["prints"].sum()), "contracts": g["count"].sum(),
                     "taker gross ¢": 100 * mt, "t": tt, "taker net ¢": 100 * mtn, "t net": ttn,
                     "maker net ¢": 100 * mm, "t maker": tm})
    return pd.DataFrame(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    tr = load()
    print(f"{int(tr['prints'].sum())} prints, {tr['ticker'].nunique()} markets, {tr['count'].sum():.0f} contracts, "
          f"{tr['ts'].min()} .. {tr['ts'].max()}")
    tabs = {
        "all": table(tr.assign(all="all"), "all", "sample"),
        "by_taker_price": table(tr, pd.cut(tr["taker_px"], [0, .05, .1, .2, .35, .5, .65, .8, .9, .95, 1]), "taker paid"),
        "by_time_left": table(tr, pd.cut(tr["left_min"], [0, 1, 2, 5, 10, 16]), "minutes left"),
        "by_side": table(tr, "taker_side", "taker bought"),
        "by_month": table(tr, tr["ts"].dt.strftime("%Y-%m"), "month"),
    }
    for k, v in tabs.items():
        v.to_csv(OUT / f"kx15_makers_{k}.csv", index=False)
        print(f"\n== {k}\n{v.round(3).to_string(index=False)}")


if __name__ == "__main__":
    main()
