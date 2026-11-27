"""Weekend wings, one per side of spot (W11) vs one per hour (W08) — history, unseen holdout, live (09.11.2026).

Why a second leg: the weekend premium is broad (studies/wings_mechanism.py §5, studies/wings_flow.py §1), and the
dearest 6–10¢ wing above spot (cheap YES) and the dearest one below spot (cheap NO) cannot both pay in the same hour,
so the second leg adds premium without raising the worst hour (one loss of ~0.93 per hour either way).

Samples
  history  Kalshi hourly-ladder book candles Jul–Aug (where W08 was found) and Sep–Oct, first minute with 15–30 min
           left per strike (studies/wings_ladder logic), taker at the cheap side's bid, Kalshi taker fee
  holdout  28.10–09.11: the lab's own book snapshots of the ladder (jobs/claude_compact_wings_now.py →
           _export/compact/lab_kx) with outcomes from data/lab/resolutions.json — never used to choose anything
  live     the lab's paper ledger (W08 since 07.11)
    python studies/wings_weekend2.py
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lab.report import md  # noqa: E402
from studies import wings_ladder as wl  # noqa: E402

LAD = ROOT / "data/raw/kalshi_kxbtcd_60m"
OUT = ROOT / "results/studies"
LAB_KX = ROOT / "_export/compact/lab_kx"
RES = ROOT / "data/lab/resolutions.json"


def stats(g):
    if len(g) == 0:
        return {"trades": 0}
    h = g.groupby("T").pnl.sum()
    n = len(h)
    d = g.groupby(g["T"].dt.floor("D")).pnl.sum()
    return {"trades": len(g), "hours": n, "¢/trade": 100 * g.pnl.mean(), "$/hour": h.mean(),
            "t (hours)": h.mean() / (h.std(ddof=1) / math.sqrt(n)) if n > 2 and h.std() > 0 else np.nan,
            "$/weekend day": d.mean(), "hit %": 100 * (g.pnl > 0).mean(), "worst hour $": h.min(),
            "losing hours %": 100 * (h < 0).mean()}


def pick(u):
    """Enter on the first eligible minute, ranking only quotes visible at that minute."""
    u = u.sort_values(["T", "ts", "c_bid", "ticker"], ascending=[True, True, False, True])
    return u.drop_duplicates("T"), u.drop_duplicates(["T", "cheap"])


def history():
    parts = []
    for file in sorted(p for p in LAD.glob("*.csv*") if p.name.endswith((".csv", ".csv.gz"))):
        d = pd.read_csv(file)
        d["T"] = pd.to_datetime(d.close_time, utc=True)
        d["ts"] = pd.to_datetime(d.end_ts, unit="s", utc=True)
        left = (d["T"] - d.ts).dt.total_seconds() / 60
        spread = (d.yes_ask - d.yes_bid).round(6)
        d = d[(d.strike_type == "greater") & d.result.isin(["yes", "no"]) & (d.yes_bid > 0)
              & (d.yes_ask < 1) & spread.between(0, .03) & left.between(15, 30) & (d["T"].dt.day_of_week >= 5)]
        for side, price in (("yes", d.yes_bid), ("no", (1 - d.yes_ask).round(6))):
            g = d.loc[price.between(.06, .10), ["ts", "T", "ticker", "result"]].copy()
            g["cheap"], g["c_bid"] = side, price
            g["pnl"] = g.c_bid - (g.result == side).astype(float) - wl.fee(1 - g.c_bid)
            parts.append(g)
    return pd.concat(parts, ignore_index=True)


def holdout():
    files = sorted(LAB_KX.glob("*.parquet"))
    if not files or not RES.exists():
        return None
    l = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    l["market"] = l.market.astype(str)
    r = json.loads(RES.read_text(encoding="utf-8"))
    l["y"] = l.market.map(lambda m: r.get("kalshi|" + m))
    l = l.dropna(subset=["y"]).reset_index(drop=True)
    l = l[(l.bid > 0) & (l.ask < 1) & (l.ask - l.bid).round(6).between(0, 0.03)]
    l["left"] = (l.end - l.ts).dt.total_seconds() / 60
    a = l[l.bid.between(0.005, 0.20)].assign(cheap="yes", c_bid=lambda x: x.bid, c_won=lambda x: x.y)
    b = l[(1 - l.ask).between(0.005, 0.20)].assign(cheap="no", c_bid=lambda x: 1 - x.ask, c_won=lambda x: 1 - x.y)
    c = pd.concat([a, b], ignore_index=True)
    c["T"], c["ticker"] = c.end, c.market
    c["pnl"] = c.c_bid - c.c_won - wl.fee(1 - c.c_bid)
    c = c[c["T"].dt.dayofweek >= 5]
    return c[c.c_bid.round(6).between(0.06, 0.10) & c.left.between(15, 30)]


def main():
    rows = []
    u = history()
    u["sample"] = np.where(u["T"] < wl.SPLIT, "Jul–Aug (found)", "Sep–Oct")
    u["month"] = u["T"].dt.strftime("%Y-%m")
    for key in ("sample", "month"):
        for k, g in u.groupby(key):
            w08, w11 = pick(g)
            second = w11[~w11.index.isin(w08.index)]
            for nm, x in (("W08 one per hour", w08), ("W11 one per side", w11), ("second leg alone", second)):
                rows.append({"sample": k, "rule": nm, **stats(x)})
    h = holdout()
    if h is not None and len(h):
        w08, w11 = pick(h)
        for nm, x in (("W08 one per hour", w08), ("W11 one per side", w11)):
            rows.append({"sample": f"holdout {h['T'].min():%d.%m}–{h['T'].max():%d.%m} (lab snapshots)", "rule": nm, **stats(x)})
    R = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    R.to_csv(OUT / "wings_weekend2.csv", index=False)
    w08, w11 = pick(u)
    pv = w11.pivot_table(index="T", columns="cheap", values="pnl").dropna()
    lines = ["# Weekend wings: one per side (W11) vs one per hour (W08) — studies/wings_weekend2.py", "",
             __doc__.split("Samples")[0].strip(), "",
             md(R, 3), "",
             f"Hours with both legs: {len(pv)} of {w11['T'].nunique()}; correlation of the two legs' P&L {pv.corr().iloc[0, 1]:+.2f}; "
             f"hours where both legs lost: {int(((pv < 0).sum(axis=1) == 2).sum())}.", ""]
    (OUT / "wings_weekend2.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
