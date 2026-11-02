"""Stand in the book and wait for the lottery buyer: resting offers on cheap wings, Kalshi hourly ladder, Jul–Oct 2026.

The idea: instead of hitting a bid minutes before the hour (W03/W04), offer the cheap side early, at a price that
is well above our fair value, and let longshot buyers come to us. For each strike, hour and cheap side, the first
minute with `left` in [l0, l1] minutes and the cheap side's bid in [lo, hi] where the model edge of an offer at
    q = max(cheap ask, fair + m + maker fee)          (never below the ask: post-only)
is ≥ m, we post q and keep it until 2 minutes before the close (or cancel it when the model's edge at q falls below
m/2 — variant "cancel"). Fair value: the lab's EWMA × seasonal variance, normal (studies/wings_ladder.py).

Fill bounds (candles have the closing book and the last trade of each minute, not the queue):
  lower  a later minute's cheap-side BID ≥ q — impossible unless our offer was lifted (and lifted by a move against us)
  upper  plus any minute whose last trade printed at ≥ q with volume
P&L per filled contract: q − payout − maker fee; unfilled orders earn 0. t-statistics are clustered by hour.
    python studies/wings_maker.py [1-5,6-10,10-20]
"""
from __future__ import annotations

import math
import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "studies"))
import config  # noqa: E402
from wings_execution import load, mfee  # noqa: E402

OUT = config.RESULTS / "studies"
SPLIT = pd.Timestamp("2026-09-01", tz="UTC")


def orders(c, raw, lo, hi, l0, l1, m, cancel, at_ask=False):
    """at_ask=False: offer at max(ask, fair + m + fee) — wait for the price to come to us;
    at_ask=True: only where the ask itself is ≥ fair + m + fee (the wing is rich now), offer at the ask."""
    g = c[c["c_bid"].between(lo, hi) & c["left"].between(l0, l1)].copy()
    ask = np.round(g["c_bid"] + (g["yes_ask"] - g["yes_bid"]), 2)
    g["q"] = ask if at_ask else np.maximum(ask, np.ceil((g["c_fair_n"] + m + mfee(1 - g["c_fair_n"])) * 100 - 1e-9) / 100)
    g["edge"] = g["q"] - g["c_fair_n"] - mfee(1 - g["q"])
    g = g[(g["edge"] >= m - 1e-9) & (g["q"] <= 0.99)]
    g = g.sort_values("ts").drop_duplicates(["ticker", "cheap"])          # first chance per strike and side
    rows = []
    cidx = c.set_index(["ticker", "cheap"]).sort_index() if cancel else None
    for s in g.itertuples(index=False):
        try:
            path = raw.loc[s.ticker]
        except KeyError:
            continue
        end = s.T - pd.Timedelta(minutes=2)
        path = path[(path.index > s.ts) & (path.index <= end)]
        if path.empty:
            continue
        cb = path["yes_bid"] if s.cheap == "yes" else 1 - path["yes_ask"]
        last = path["last"] if s.cheap == "yes" else 1 - path["last"]
        live = pd.Series(True, index=path.index)
        if cancel:
            try:
                fr = cidx.loc[(s.ticker, s.cheap)]
                fr = fr[(fr["ts"] > s.ts) & (fr["ts"] <= end)].set_index("ts")["c_fair_n"]
                gone = fr[s.q - fr - mfee(1 - s.q) < m / 2]
                if len(gone):
                    live = path.index < gone.index[0]
                    live = pd.Series(live, index=path.index)
            except KeyError:
                pass
        through = (cb >= s.q - 1e-9) & live
        touch = through | (((last.round(2) >= s.q - 1e-9) & (path["volume"] > 0)) & live)
        rec = {"T": s.T, "ts": s.ts, "ticker": s.ticker, "cheap": s.cheap, "q": s.q, "bid": s.c_bid, "fair": s.c_fair_n,
               "left": s.left, "won": s.c_won}
        for k, f in (("lo", through), ("hi", touch)):
            filled = bool(f.any())
            rec[f"fill_{k}"] = filled
            rec[f"pnl_{k}"] = (s.q - s.c_won - mfee(1 - s.q)) if filled else 0.0
        rows.append(rec)
    return pd.DataFrame(rows)


def stat(x, col):
    h = x.groupby("T")[col].sum()
    n = len(h)
    return h.mean() / (h.std(ddof=1) / math.sqrt(n)) if n > 2 and h.std() > 0 else np.nan


def main():
    arg = sys.argv[1] if len(sys.argv) > 1 else None            # load() rewrites sys.argv
    W, c, raw = load()
    rows = []
    bands = {"1-5": (0.01, 0.05), "6-10": (0.06, 0.10), "10-20": (0.10, 0.20)}
    pick = [bands[b] for b in (arg.split(",") if arg else bands)]
    for (lo, hi), (l0, l1), (m, at_ask), cancel in product(pick, [(10, 30), (30, 58)],
                                                           [(0.02, False), (0.06, False), (0.01, True), (0.03, True)],
                                                           [False, True]):
        x = orders(c, raw, lo, hi, l0, l1, m, cancel, at_ask)
        if x.empty:
            continue
        for half, h in (("all", x), ("Sep–Oct", x[x["T"] >= SPLIT])):
            rec = {"band ¢": f"{100 * lo:.0f}–{100 * hi:.0f}", "minutes left": f"{l0}–{l1}", "edge ≥ ¢": 100 * m,
                   "price": "ask (rich now)" if at_ask else "fair + edge (wait)",
                   "cancel": cancel, "sample": half, "offers": len(h), "hours": h["T"].nunique()}
            for k in ("lo", "hi"):
                f = h[f"fill_{k}"]
                rec[f"fill % {k}"] = 100 * f.mean()
                rec[f"¢/offer {k}"] = 100 * h[f"pnl_{k}"].mean()
                rec[f"¢/fill {k}"] = 100 * h.loc[f, f"pnl_{k}"].mean() if f.any() else np.nan
                rec[f"t {k}"] = stat(h, f"pnl_{k}")
                rec[f"pays|fill {k}"] = 100 * h.loc[f, "won"].mean() if f.any() else np.nan
            rec["pays (all) %"] = 100 * h["won"].mean()
            rec["fair (mean) %"] = 100 * h["fair"].mean()
            rec["offer (mean) ¢"] = 100 * h["q"].mean()
            rows.append(rec)
        print(rows[-2]["band ¢"], rows[-2]["minutes left"], m, at_ask, cancel, "offers", rows[-2]["offers"],
              "fill lo/hi %.0f/%.0f" % (rows[-2]["fill % lo"], rows[-2]["fill % hi"]),
              "¢/offer lo/hi %.2f/%.2f" % (rows[-2]["¢/offer lo"], rows[-2]["¢/offer hi"]), flush=True)
    g = pd.DataFrame(rows)
    tag = arg.replace(",", "_") if arg else "all"
    g.to_csv(OUT / f"wings_maker_grid_{tag}.csv", index=False)
    pd.set_option("display.width", 250)
    print(g.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
