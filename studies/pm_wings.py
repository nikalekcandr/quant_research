"""The Kalshi wing rules on the Polymarket hourly ladder, history Jul–Nov 2026 (04.11.2026).

Polymarket "Bitcoin above K on <date>, <h> ET" settles on the Binance BTC/USDT 1-hour candle close: the spot IS the
settlement source (no basis, no averaging window). History has no book: prices-history gives one price per minute
(≈ mid / last). The cheap side's bid is assumed at price − h, h = half the spread (live books 28.10–02.11: median
spread 2¢ on 6–20¢ wings → h = 1¢; also 0.5 and 1.5¢). Taker fee 0.07·P·(1−P). Fair value: the W03/W04 model
(EWMA level × hour-of-week profile × minutes left, normal / t4) on Binance minutes.
Result is an upper bound: no depth (live median top size 10–18 shares vs 600–970 on Kalshi).
    python studies/pm_wings.py <pm hist dir> <binance minutes csv/parquet ...>
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm, t as student_t

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from studies import wings_ladder as wl  # noqa: E402

HIST = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/raw/pm_ladder_1h/hist")
EXTRA = [Path(x) for x in sys.argv[2:]]
SPLIT = pd.Timestamp("2026-09-01", tz="UTC")


def minutes():
    m = pd.read_parquet("data/interim/minutes.parquet", columns=["open_time", "close"])
    parts = [m]
    for f in EXTRA:
        parts.append(pd.read_csv(f)[["open_time", "close"]])
    m = pd.concat(parts)
    m["open_time"] = pd.to_datetime(m.open_time, utc=True)
    s = m.drop_duplicates("open_time", keep="last").set_index("open_time")["close"].sort_index()
    s.index = s.index + pd.Timedelta(minutes=1)            # close time
    return s[s.index >= pd.Timestamp("2026-05-01", tz="UTC")]


def load(S):
    d = pd.concat([pd.read_csv(f) for f in sorted(HIST.glob("*.csv")) if f.stat().st_size > 200], ignore_index=True)
    d = d.dropna(subset=["t", "p", "outcome"])
    d["ts"] = pd.to_datetime(d.t, unit="s", utc=True).dt.floor("min")
    d["T"] = pd.to_datetime(d["end"], utc=True)
    d = d.sort_values("t").drop_duplicates(["market", "ts"], keep="last")
    d["left"] = (d["T"] - d["ts"]).dt.total_seconds() / 60
    vm = wl.variance_level(S)
    d["S"] = S.reindex(d.ts).values
    d["lvl"], d["sea"] = vm["lvl"].reindex(d.ts).values, vm["sea"].reindex(d.ts).values
    d = d.dropna(subset=["S", "lvl"])
    var = d.lvl * d.sea * np.maximum(d.left, 1 / 3)
    x = np.log(d.S / d.strike) / np.sqrt(var)
    d["p_n"], d["p_t"] = norm.cdf(x), student_t.cdf(x * math.sqrt(2), 4)
    # check: does our Binance settlement reproduce the outcome?
    st = S.reindex(d["T"]).values
    d["y_check"] = (st > d.strike).astype(float)
    return d


def frame(d, h):
    yes = d.assign(cheap="yes", c_bid=d.p - h, c_fair_n=d.p_n, c_fair_t=d.p_t, c_won=d.outcome)
    no = d.assign(cheap="no", c_bid=(1 - d.p) - h, c_fair_n=1 - d.p_n, c_fair_t=1 - d.p_t, c_won=1 - d.outcome)
    c = pd.concat([yes, no], ignore_index=True)
    c["px"] = 1 - c.c_bid
    c["pnl"] = c.c_bid - c.c_won - 0.07 * c.px * (1 - c.px)
    return c


def run(c, lo, hi, l0, l1, filt, mg):
    g = c[c.c_bid.between(lo, hi) & c.left.between(l0, l1)]
    if filt != "none":
        fair = g.c_fair_n if filt == "normal" else g.c_fair_t
        g = g[g.c_bid - fair - 0.07 * g.px * (1 - g.px) >= mg]
    g = g.sort_values(["T", "ts", "c_bid"], ascending=[True, True, False]).drop_duplicates(["T", "market"])
    return g.drop_duplicates("T")


def st(g):
    if len(g) < 3:
        return np.nan, np.nan, len(g)
    return 100 * g.pnl.mean(), g.pnl.mean() / (g.pnl.std(ddof=1) / math.sqrt(len(g))), len(g)


def main():
    S = minutes()
    d = load(S)
    ok = (d.y_check == d.outcome).mean()
    lines = ["# Kalshi wing rules on the Polymarket hourly ladder (studies/pm_wings.py)", "",
             f"{d['T'].nunique()} hours, {d.market.nunique()} strikes, {d['T'].min():%Y-%m-%d} .. {d['T'].max():%Y-%m-%d}; "
             f"Binance close reproduces the outcome in {100 * ok:.1f}% of rows.", ""]
    rules = {"W09 = W03 (10–20¢, 5–15 min, normal, ≥1¢)": (0.10, 0.20, 5, 15, "normal", 0.01),
             "W10 = W04 (6–10¢, 15–30 min, t4, ≥0)": (0.06, 0.10, 15, 30, "t4", 0.0),
             "no model, 10–20¢, 5–15 min": (0.10, 0.20, 5, 15, "none", 0),
             "no model, 6–10¢, 15–30 min": (0.06, 0.10, 15, 30, "none", 0)}
    rows = []
    for h in (0.005, 0.01, 0.015):
        c = frame(d, h)
        for nm, p in rules.items():
            g = run(c, *p)
            rec = {"rule": nm, "half spread ¢": 100 * h}
            for half, gg in (("Jul–Aug", g[g["T"] < SPLIT]), ("Sep–Nov", g[g["T"] >= SPLIT])):
                m, t, n = st(gg)
                rec[f"{half} n"], rec[f"{half} ¢"], rec[f"{half} t"] = n, m, t
            wk = g[g["T"].dt.dayofweek >= 5]
            rec["weekend ¢"], rec["weekend n"] = (100 * wk.pnl.mean(), len(wk)) if len(wk) else (np.nan, 0)
            rows.append(rec)
    R = pd.DataFrame(rows)
    lines += ["## One trade per hour, held to settlement", "", R.round(2).to_markdown(index=False), ""]
    c = frame(d, 0.01)
    u = c[c.c_bid.between(0.03, 0.20) & c.left.between(5, 30)].sort_values("ts").drop_duplicates(["T", "market"])
    u["bk"] = pd.cut(u.c_bid, [0.03, 0.06, 0.10, 0.15, 0.20])
    cal = u.groupby("bk", observed=True).agg(n=("pnl", "size"), bid=("c_bid", "mean"), paid=("c_won", "mean"),
                                            pnl=("pnl", "mean"))
    cal[["bid", "paid", "pnl"]] *= 100
    lines += ["## Favourite–longshot on Polymarket (every cheap side, first minute in 5–30 min, h = 1¢)", "",
              cal.round(2).to_markdown(), ""]
    Path("results/studies/pm_wings.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
