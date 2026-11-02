"""How to execute the wing trades: taker now, maker first, or passive-then-aggressive? Kalshi hourly ladder,
Jul–Oct 2026 book candles (data/raw/kalshi_kxbtcd_60m), the W03 and W04 signals of studies/wings_ladder.py.

At the signal minute t0 the rule wants to SELL the cheap side (bid b0, ask a0). Policies:
  taker            sell at b0 now (taker fee) — the backtest of record
  maker(τ)         offer the cheap side at a0 (post-only) for τ minutes, nothing after
  hybrid(τ)        offer at a0 for τ minutes; if not filled, at t0+τ re-price the fair value on that minute's book
                   and sell at the bid if bid − fair − fee ≥ the rule's margin (Almgren–Chriss / Harris:
                   patience while the deadline is far, urgency when it is near)
  hybrid-in(τ)     as hybrid, but the offer is one tick inside the spread (a0 − 1¢) when the spread is ≥ 2¢,
                   which puts us first in the queue

Fill model. Candles carry the closing book and the last trade of each minute, not the queue, so fills are bounded:
  lower bound  "through": a later minute's book shows the cheap side BID ≥ our offer. A bid above a resting offer is
               impossible on a continuous book, so our offer was certainly lifted — and lifted exactly when the
               price moved against us (this is adverse selection, measured, not assumed).
  upper bound  "touch": additionally any minute whose last trade printed at our price with volume > 0.
The truth lies between; with a long queue it is near the lower bound.
Fees: taker ceil(0.07·P(1−P)) per contract, maker ceil(0.0175·P(1−P)) (Kalshi, 1 contract — conservative).
    python studies/wings_execution.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "studies"))
import config  # noqa: E402

OUT = config.RESULTS / "studies"
RULES = {"W03": dict(lo=0.10, hi=0.20, l0=5, l1=15, filt="normal", margin=0.01),
         "W04": dict(lo=0.06, hi=0.10, l0=15, l1=30, filt="t4", margin=0.0)}
SPLIT = pd.Timestamp("2026-09-01", tz="UTC")


def tfee(p):
    return np.ceil(np.round(0.07 * p * (1 - p) * 100, 6)) / 100


def mfee(p):
    return np.ceil(np.round(0.0175 * p * (1 - p) * 100, 6)) / 100


def load():
    sys.argv = ["x", str(config.RAW / "kalshi_kxbtcd_60m"), str(config.INTERIM / "minutes.parquet"), str(OUT)]
    import wings_ladder as W
    c = W.load()                                  # cheap-side rows with fair values (no look-ahead)
    raw = pd.concat([pd.read_csv(f, usecols=["ticker", "end_ts", "yes_bid", "yes_ask", "last", "volume"])
                     for f in sorted((config.RAW / "kalshi_kxbtcd_60m").glob("*.csv*"))], ignore_index=True)
    raw["ts"] = pd.to_datetime(raw["end_ts"], unit="s", utc=True)
    return W, c, raw.set_index(["ticker", "ts"]).sort_index()


def simulate(W, c, raw, hid, taus=(1, 2, 3, 5, 7, 10)):
    r = RULES[hid]
    sig = W.run(c, r["lo"], r["hi"], r["l0"], r["l1"], r["filt"], r["margin"], 1)
    fair_col = "c_fair_n" if r["filt"] == "normal" else "c_fair_t"
    cidx = c.set_index(["ticker", "cheap", "ts"]).sort_index()
    out = []
    for s in sig.itertuples(index=False):
        try:
            path = raw.loc[s.ticker]
        except KeyError:
            continue
        path = path[(path.index > s.ts) & (path.index <= s.T - pd.Timedelta(minutes=1))]
        b0 = s.c_bid
        a0 = (s.yes_ask if s.cheap == "yes" else 1 - s.yes_bid)          # cheap side's ask
        spread = round(a0 - b0, 2)
        won = s.c_won
        rec = {"rule": hid, "T": s.T, "ts": s.ts, "ticker": s.ticker, "cheap": s.cheap, "b0": b0, "a0": a0,
               "spread": spread, "left": s.left, "won": won, "taker": b0 - won - tfee(1 - b0)}
        # cheap side's bid and last trade on the cheap side, minute by minute
        cb = path["yes_bid"] if s.cheap == "yes" else 1 - path["yes_ask"]
        last = path["last"] if s.cheap == "yes" else 1 - path["last"]
        for inside in (False, True):
            px = round(a0 - 0.01, 2) if (inside and spread >= 0.02) else a0
            through = cb >= px - 1e-9
            touch = through | ((last.round(2) >= px - 1e-9) & (path["volume"] > 0))
            for tau in taus:
                win = path.index <= s.ts + pd.Timedelta(minutes=tau)
                for kind, f in (("lo", through), ("hi", touch)):
                    filled = bool(f[win].any())
                    maker = px - won - mfee(1 - px) if filled else 0.0
                    # fallback: re-price at t0+τ with that minute's fair and bid
                    fb = 0.0
                    if not filled:
                        t1 = s.ts + pd.Timedelta(minutes=tau)
                        try:
                            row = cidx.loc[(s.ticker, s.cheap, t1)]
                            if isinstance(row, pd.DataFrame):
                                row = row.iloc[0]
                            b1, fair1 = row["c_bid"], row[fair_col]
                            if (s.T - t1).total_seconds() / 60 >= 2 and b1 - fair1 - tfee(1 - b1) >= r["margin"]:
                                fb = b1 - won - tfee(1 - b1)
                                rec[f"fb_traded_{'in' if inside else 'at'}_{tau}_{kind}"] = 1
                        except KeyError:
                            pass
                    tag = f"{'in' if inside else 'at'}_{tau}_{kind}"
                    rec[f"fill_{tag}"] = filled
                    rec[f"maker_{tag}"] = maker
                    rec[f"hybrid_{tag}"] = maker if filled else fb
        out.append(rec)
    return pd.DataFrame(out)


def summarise(x: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (rule, half), g in [((r, h), g) for r, gg in x.groupby("rule")
                             for h, g in (("all", gg), ("Sep–Oct", gg[gg["T"] >= SPLIT]))]:
        n = len(g)

        def st(v, name, fill=None, traded=None):
            h = v.groupby(g["T"]).sum()
            rows.append({"rule": rule, "sample": half, "policy": name, "signals": n,
                         "fill %": None if fill is None else 100 * fill.mean(),
                         "traded %": 100 * (traded.mean() if traded is not None else 1.0),
                         "¢ per signal": 100 * v.mean(), "t (hours)": h.mean() / (h.std(ddof=1) / math.sqrt(len(h))),
                         "total $": v.sum()})
        st(g["taker"], "taker now")
        for inside in ("at", "in"):
            for tau in (1, 3, 5, 10):
                for kind in ("lo", "hi"):
                    tag = f"{inside}_{tau}_{kind}"
                    lbl = f"{'1¢ inside' if inside == 'in' else 'at ask'}, {tau} min, fills {'lower' if kind == 'lo' else 'upper'} bound"
                    fill = g[f"fill_{tag}"].astype(float)
                    st(g[f"maker_{tag}"], f"maker only — {lbl}", fill, fill)
                    fb = g.get(f"fb_traded_{tag}", pd.Series(0, index=g.index)).fillna(0)
                    st(g[f"hybrid_{tag}"], f"hybrid — {lbl}", fill, ((fill > 0) | (fb > 0)).astype(float))
    return pd.DataFrame(rows)


def adverse(x: pd.DataFrame) -> pd.DataFrame:
    """Adverse selection: how often the wing we sold paid off, given that a maker offer was (or was not) lifted."""
    rows = []
    for rule, g in x.groupby("rule"):
        for tau in (3, 10):
            f = g[f"fill_at_{tau}_lo"]
            rows.append({"rule": rule, "τ min": tau, "P(wing pays) all": g["won"].mean(),
                         "P(wing pays | lifted, lower bound)": g.loc[f, "won"].mean(),
                         "P(wing pays | not lifted)": g.loc[~f, "won"].mean(),
                         "share lifted": f.mean(), "mean price b0": g["b0"].mean()})
    return pd.DataFrame(rows)


def main():
    W, c, raw = load()
    x = pd.concat([simulate(W, c, raw, h) for h in RULES], ignore_index=True)
    x.to_csv(OUT / "wings_execution_rows.csv", index=False)
    s = summarise(x)
    s.to_csv(OUT / "wings_execution.csv", index=False)
    a = adverse(x)
    a.to_csv(OUT / "wings_execution_adverse.csv", index=False)
    pd.set_option("display.width", 250)
    print(s.round(2).to_string(index=False))
    print(a.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
