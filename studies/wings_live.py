"""Live paper results of the wing rules from the lab ledger (data/lab/ledger.csv), voided trades excluded.
Per rule: trades, ¢ per contract, t over settlement hours, P&L, hit rate; split by period and weekend; the
sequential test (trader/monitor.sprt) on the 'early' slot (W04/W06) and on W08.
    python studies/wings_live.py            -> results/studies/wings_live.md"""
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lab.report import void_mask  # noqa: E402
from trader.monitor import sprt  # noqa: E402

d = pd.read_csv(ROOT / "data/lab/ledger.csv")
d = d[~void_mask(d)]
c = d[(d.event == "CLOSE") & d.hyp.astype(str).str.match(r"W\d")].copy()
c["ts"] = pd.to_datetime(c.ts, utc=True, format="mixed")
c["hour"] = pd.to_datetime(c.exit_ts, utc=True, format="mixed").dt.floor("h")
c["c"] = 100 * c.pnl / c.qty
c["days"] = np.where(c.hour.dt.dayofweek >= 5, "weekend", "weekday")


def st(g):
    h = g.groupby("hour").c.sum()
    n = len(h)
    return pd.Series({"trades": len(g), "¢/trade": g.c.mean(),
                      "t (hours)": h.mean() / (h.std(ddof=1) / math.sqrt(n)) if n > 2 and h.std() > 0 else np.nan,
                      "$ (1 contract)": g.pnl.sum(), "hit %": 100 * (g.pnl > 0).mean(), "worst ¢": g.c.min(),
                      "first": g.ts.min().strftime("%d %b"), "last": g.ts.max().strftime("%d %b")})


A = c.groupby("hyp").apply(st, include_groups=False)
B = c[c.hyp.isin(["W03", "W04", "W06", "W08"])].groupby(["hyp", "days"]).apply(st, include_groups=False)
early = c[c.hyp.isin(["W04", "W06"])].sort_values("ts").drop_duplicates("hour").groupby("hour").c.sum() / 100
w08 = c[c.hyp == "W08"].groupby("hour").c.sum() / 100
lines = [f"# Live paper wings, lab ledger to {c.ts.max():%d.%m.%Y %H:%M} UTC (studies/wings_live.py)", "",
         A.round(2).to_markdown(), "", "## By weekend / weekday", "", B.round(2).to_markdown(), "",
         "## Sequential test (1 contract per slot; trader/monitor.sprt)", "",
         f"* early slot W04/W06: {sprt(early)}", f"* W08: {sprt(w08)}", ""]
(ROOT / "results/studies/wings_live.md").write_text("\n".join(lines), encoding="utf-8")
print("\n".join(lines))
