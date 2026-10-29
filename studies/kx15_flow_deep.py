"""Deeper look at Kalshi 15-minute prints: where the taker losses come from.
Inputs: data/raw/kx15_trades_agg (per market × taker side × 1¢ × minute left), data/raw/kx15/markets.csv.
1. price bucket × minutes left: taker P&L per contract (contract-weighted, clustered by market)
2. three weightings: per contract, per print, per market (equal weight) — is it a few whales?
3. leave-one-day-out / drop the top-5 markets — is it a few outcomes?
4. markout: price move in the same market over the next 1 and 3 minutes after a taker's print, in the taker's
   direction (VWAP of all prints of that later minute). Informed takers: positive markout. Lottery: none.
5. who buys longshots: share of YES vs NO longshot buys, size of prints
    python studies/kx15_flow_deep.py <raw dir>
"""
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/raw")
BINS = [0, .05, .1, .2, .35, .5, .65, .8, .9, .95, 1]


def cl(x, g, w):
    d = pd.DataFrame({"x": x, "g": g, "w": w})
    s = d.assign(xw=d.x * d.w).groupby("g")[["xw", "w"]].sum()
    m = s.xw.sum() / s.w.sum()
    u = s.xw - m * s.w
    n = len(s)
    se = math.sqrt(n / (n - 1) * (u ** 2).sum()) / s.w.sum() if n > 1 else np.nan
    return m, m / se if se else np.nan, n


def main():
    mk = pd.read_csv(RAW / "kx15" / "markets.csv", usecols=["ticker", "close_time", "result"])
    tr = pd.concat([pd.read_csv(f) for f in sorted((RAW / "kx15_trades_agg").glob("*.csv")) if f.stat().st_size > 10])
    tr = tr.merge(mk, on="ticker")
    tr = tr[tr.result.isin(["yes", "no"]) & tr.side.isin(["yes", "no"]) & (tr.contracts > 0)].copy()
    tr["p"] = tr.notional / tr.contracts
    tr = tr[tr.p.between(0.005, 0.995)]
    tr["y"] = (tr.result == "yes").astype(float)
    sgn = np.where(tr.side == "yes", 1.0, -1.0)
    tr["sgn"] = sgn
    tr["paid"] = np.where(sgn > 0, tr.p, 1 - tr.p)
    tr["won"] = np.where(sgn > 0, tr.y, 1 - tr.y)
    tr["gross"] = tr.won - tr.paid
    tr["net"] = tr.gross - 0.07 * tr.paid * (1 - tr.paid)
    tr["day"] = pd.to_datetime(tr.close_time, utc=True).dt.date
    tr["bucket"] = pd.cut(tr.paid, BINS)
    print(f"{tr.ticker.nunique()} markets, {int(tr.prints.sum())} prints, {tr.contracts.sum():.0f} contracts, "
          f"{tr.day.min()} .. {tr.day.max()}\n")

    # 1. bucket x minutes left
    tr["left_b"] = pd.cut(tr.left, [-1, 0, 2, 5, 9, 15], labels=["last min", "1-2", "3-5", "6-9", "10-15"])
    rows = []
    for (b, lb), g in tr.groupby(["bucket", "left_b"], observed=True):
        if g.ticker.nunique() >= 20:
            m, t, n = cl(g.net, g.ticker, g.contracts)
            rows.append({"taker paid": str(b), "minutes left": lb, "markets": n, "net ¢": 100 * m, "t": t})
    t1 = pd.DataFrame(rows).pivot(index="taker paid", columns="minutes left", values=["net ¢", "t"])
    print("== 1. taker net P&L, ¢ per contract (and t), by price paid × minutes left\n", t1.round(1).to_string(), "\n")

    # 2. weightings + win rate vs price
    rows = []
    for b, g in tr.groupby("bucket", observed=True):
        mc, tc, n = cl(g.net, g.ticker, g.contracts)
        mp, tp, _ = cl(g.net, g.ticker, g.prints)
        per_m = g.groupby("ticker").apply(lambda h: np.average(h.net, weights=h.contracts), include_groups=False)
        rows.append({"taker paid": str(b), "markets": n, "avg paid": np.average(g.paid, weights=g.contracts),
                     "win rate": np.average(g.won, weights=g.contracts), "net ¢ per contract": 100 * mc, "t": tc,
                     "net ¢ per print": 100 * mp, "t print": tp, "net ¢ per market (equal)": 100 * per_m.mean(),
                     "median market ¢": 100 * per_m.median(), "share of markets < 0": (per_m < 0).mean(),
                     "contracts per print": g.contracts.sum() / g.prints.sum()})
    print("== 2. weightings\n", pd.DataFrame(rows).round(3).to_string(index=False), "\n")

    # 3. robustness for longshot buyers (paid <= 20c)
    ls = tr[tr.paid <= 0.20]
    m0, t0, n0 = cl(ls.net, ls.ticker, ls.contracts)
    contrib = ls.assign(c=ls.net * ls.contracts).groupby("ticker").c.sum().sort_values()
    drop = ls[~ls.ticker.isin(contrib.index[:5])]
    m5, t5, _ = cl(drop.net, drop.ticker, drop.contracts)
    days = []
    for d in sorted(ls.day.unique()):
        h = ls[ls.day != d]
        days.append(cl(h.net, h.ticker, h.contracts)[0])
    print(f"== 3. longshot buyers (paid ≤ 20¢): {100 * m0:.2f}¢ (t {t0:.1f}, {n0} markets); without the 5 worst "
          f"markets {100 * m5:.2f}¢ (t {t5:.1f}); leave-one-day-out range {100 * min(days):.2f} … {100 * max(days):.2f}¢\n")

    # 4. markout: VWAP of the same market k minutes later (left smaller), in the taker's direction
    vw = tr.groupby(["ticker", "left"]).apply(lambda h: np.average(h.p, weights=h.contracts), include_groups=False).rename("vwap")
    vw = vw.reset_index()
    out = []
    for k in (1, 3):
        nxt = vw.assign(left=vw.left + k).rename(columns={"vwap": f"vwap_{k}"})
        tr = tr.merge(nxt, on=["ticker", "left"], how="left")
        tr[f"mo_{k}"] = tr.sgn * (tr[f"vwap_{k}"] - tr.p)
    rows = []
    for b, g in tr.groupby("bucket", observed=True):
        rec = {"taker paid": str(b)}
        for k in (1, 3):
            h = g.dropna(subset=[f"mo_{k}"])
            m, t, n = cl(h[f"mo_{k}"], h.ticker, h.contracts)
            rec[f"markout {k} min ¢"], rec[f"t {k}"] = 100 * m, t
        m, t, n = cl(g.gross, g.ticker, g.contracts)
        rec["to settlement ¢"], rec["t settle"] = 100 * m, t
        rows.append(rec)
    print("== 4. markouts in the taker's direction (positive = price moved his way after the print)\n",
          pd.DataFrame(rows).round(2).to_string(index=False), "\n")

    # 5. who buys longshots
    g = tr[tr.paid <= 0.10]
    print("== 5. longshot buys (paid ≤ 10¢): share bought as YES (betting BTC goes up) "
          f"{g[g.side == 'yes'].contracts.sum() / g.contracts.sum():.2f}; contracts per print "
          f"{g.contracts.sum() / g.prints.sum():.0f} vs all {tr.contracts.sum() / tr.prints.sum():.0f}")
    for s, h in g.groupby("side"):
        m, t, n = cl(h.net, h.ticker, h.contracts)
        print(f"   taker bought {s} at ≤ 10¢: {100 * m:.2f}¢ (t {t:.1f}, {n} markets)")


if __name__ == "__main__":
    main()
