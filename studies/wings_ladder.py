"""Selling cheap wings on the Kalshi hourly ladder (KXBTCD), Jul–Oct 2026: what works, if anything?

Data: 1-minute book candles of the hourly ladder (data/raw/kalshi_kxbtcd_60m: best YES bid/ask per strike and
minute, every second strike within ±1.5–2.5% of spot, the whole hour), the settled result, Binance minutes.
A strike's "cheap side" is YES when its bid is in the band, or NO when 1 − YES ask is in the band. We SELL the cheap
side at its bid as a taker (= buy the other side at its ask), hold to settlement, Kalshi taker fee.

Fair value of the cheap side (for the filter): P(60-s BRTI average at the close beyond the strike) from Binance spot
× the BRTI/Binance basis of the previous 48 settlements, variance = EWMA level of deseasonalised 1-minute returns ×
the hour-of-week profile × (minutes left − 2/3), normal or Student-t(4).
Grid (chosen on Jul–Aug, read on Sep–Oct): cheap price band × minutes left × model filter × strikes per hour.
Clusters = hours (all strikes of one hour share one outcome path).
    python studies/wings_ladder.py <ladder dir> <minutes.parquet> <out dir>
"""
from __future__ import annotations

import math
import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm, t as student_t

LAD = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/raw/kalshi_kxbtcd_60m")
MIN = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("data/interim/minutes.parquet")
OUT = Path(sys.argv[3]) if len(sys.argv) > 3 else Path("results/studies")
SPLIT = pd.Timestamp("2026-09-01", tz="UTC")


def fee(p):
    return np.ceil(np.round(0.07 * p * (1 - p) * 100, 6)) / 100


def variance_level(close):
    r2 = np.log(close).diff() ** 2
    idx = close.index
    how = idx.dayofweek * 24 + idx.hour
    df = pd.DataFrame({"r2": r2.values, "how": how, "day": idx.floor("D")}, index=pd.DatetimeIndex(idx, name="ts"))
    daily = df.groupby(["day", "how"])["r2"].mean().unstack()
    sea = daily.rolling(28, min_periods=7).mean().shift(1)
    sea = sea.div(sea.mean(axis=1), axis=0).stack().rename("sea").reset_index()
    df = df.reset_index().merge(sea, on=["day", "how"], how="left").set_index("ts")
    df["sea"] = df["sea"].fillna(1.0)
    df["lvl"] = (df["r2"] / df["sea"]).ewm(halflife=60, min_periods=120).mean()
    return df[["sea", "lvl"]]


def load():
    d = pd.concat([pd.read_csv(f) for f in sorted(LAD.glob("*.csv*"))], ignore_index=True)
    d = d[(d.strike_type == "greater") & (d.yes_bid > 0) & (d.yes_ask < 1) & (d.yes_ask - d.yes_bid <= 0.03)]
    d["ts"] = pd.to_datetime(d.end_ts, unit="s", utc=True)
    d["T"] = pd.to_datetime(d.close_time, utc=True)
    d["left"] = (d["T"] - d["ts"]).dt.total_seconds() / 60
    d["y"] = (d.result == "yes").astype(float)
    m = pd.read_parquet(MIN, columns=["open_time", "close"])
    m = m.set_index(pd.to_datetime(m.open_time, utc=True) + pd.Timedelta(minutes=1))["close"]   # close time
    m = m[m.index >= d.ts.min() - pd.Timedelta(days=40)]
    m = m[~m.index.duplicated()].sort_index()
    vm = variance_level(m)
    d["S"] = m.reindex(d.ts).values
    d["lvl"] = vm["lvl"].reindex(d.ts).values
    d["sea"] = vm["sea"].reindex(d.ts).values
    ev = d.groupby("T").expiration_value.first().sort_index()
    ratio = (ev / m.reindex(ev.index).values).shift(1).rolling(48, min_periods=5).median()
    d["basis"] = d["T"].map(ratio)
    d = d.dropna(subset=["S", "lvl", "basis"])
    var = d.lvl * d.sea * np.maximum(d.left - 2 / 3, 1 / 3)
    x = np.log(d.S * d.basis / d.strike) / np.sqrt(var)
    d["p_n"] = norm.cdf(x)
    d["p_t"] = student_t.cdf(x * math.sqrt(2), 4)            # t4 with the same variance
    # the cheap side of each row
    cy = d.yes_bid.between(0.005, 0.20)
    cn = (1 - d.yes_ask).between(0.005, 0.20)
    a = d[cy].assign(cheap="yes", c_bid=d.yes_bid, c_fair_n=d.p_n, c_fair_t=d.p_t, c_won=d.y)
    b = d[cn].assign(cheap="no", c_bid=1 - d.yes_ask, c_fair_n=1 - d.p_n, c_fair_t=1 - d.p_t, c_won=1 - d.y)
    c = pd.concat([a, b], ignore_index=True)
    c["px"] = 1 - c.c_bid                                    # what we pay for the other side
    c["pnl"] = c.c_bid - c.c_won - fee(c.px)
    return c


def run(c, lo, hi, l0, l1, filt, margin, cap):
    g = c[c.c_bid.between(lo, hi) & c.left.between(l0, l1)]
    if filt != "none":
        fair = g.c_fair_n if filt == "normal" else g.c_fair_t
        g = g[g.c_bid - fair - fee(g.px) >= margin]
    g = g.sort_values(["T", "ts", "c_bid"], ascending=[True, True, False])
    g = g.drop_duplicates(["T", "ticker"])                   # one trade per strike: the first signal
    if cap:
        g = g.groupby("T").head(cap)
    return g


def stats(g):
    if g.empty:
        return {"trades": 0}
    h = g.groupby("T").pnl.sum()
    n = len(h)
    return {"trades": len(g), "hours": n, "¢/trade": 100 * g.pnl.mean(), "$/hour": h.mean(),
            "t (hours)": h.mean() / (h.std(ddof=1) / math.sqrt(n)) if n > 2 and h.std() > 0 else np.nan,
            "hit": (g.pnl > 0).mean(), "worst hour $": h.min()}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    c = load()
    print(f"{c['T'].nunique()} hours, {c.ticker.nunique()} strikes, {len(c)} cheap-side minutes, "
          f"{c['T'].min():%Y-%m-%d} .. {c['T'].max():%Y-%m-%d}")
    base = stats(run(c, 0.02, 0.06, 15, 60, "none", 0, 0))
    print("W01-like (2–6¢, ≥ 15 min, no filter, every strike):", {k: round(v, 3) if isinstance(v, float) else v for k, v in base.items()})
    rows = []
    for (lo, hi), (l0, l1), (filt, mg), cap in product(
            [(0.01, 0.03), (0.03, 0.06), (0.06, 0.10), (0.10, 0.20), (0.02, 0.06)],
            [(5, 15), (15, 30), (30, 45), (45, 60), (15, 60)],
            [("none", 0), ("normal", 0.0), ("normal", 0.01), ("normal", 0.02), ("t4", 0.0), ("t4", 0.01)],
            [1, 3, 0]):
        g = run(c, lo, hi, l0, l1, filt, mg, cap)
        tr, te = g[g["T"] < SPLIT], g[g["T"] >= SPLIT]
        rec = {"band ¢": f"{100 * lo:.0f}–{100 * hi:.0f}", "minutes left": f"{l0}–{l1}", "filter": filt,
               "margin ¢": 100 * mg, "strikes/hour": cap or "all"}
        for half, h in (("train", tr), ("test", te)):
            s = stats(h)
            for k in ("trades", "hours", "¢/trade", "t (hours)", "hit", "worst hour $"):
                rec[f"{k} {half}"] = s.get(k, np.nan)
        rows.append(rec)
    G = pd.DataFrame(rows)
    G.to_csv(OUT / "wings_ladder_grid.csv", index=False)
    ok = G[(G["hours train"] >= 40)]
    top = ok.sort_values("t (hours) train", ascending=False).head(10)
    print("\n== top 10 on Jul–Aug (train) and the same rules on Sep–Oct (test)\n", top.round(2).to_string(index=False))
    # multiple-testing reality check: how do the test results of the 10 best train rules compare with all rules
    print("\nshare of all", len(ok), "rules with t > 2 on the test half:", round((ok["t (hours) test"] > 2).mean(), 3),
          "; among the top 10 by train:", round((top["t (hours) test"] > 2).mean(), 2))
    print("mean test ¢/trade of all rules", round(ok["¢/trade test"].mean(), 2), "| top 10:", round(top["¢/trade test"].mean(), 2))


if __name__ == "__main__":
    main()
