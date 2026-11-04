"""Weekend wings check (04.11.2026): every 6–10¢ cheap side, 15–30 min left, no model, first minute per strike —
by weekday/weekend, half, month; per-hour book (all strikes of an hour = one bet) and one strike per hour."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.argv = sys.argv[:1] + ["data/raw/usd_minutes"]
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from studies import brti_proxy as bp  # noqa: E402
from studies import wings_ladder as wl  # noqa: E402
wl.LAD, wl.MIN = Path("data/raw/kalshi_kxbtcd_60m"), Path("data/interim/minutes.parquet")

c = bp.wings_frame(bp.binance(), bp.settlements())
out = []
for (lo, hi), (l0, l1) in (((0.06, 0.10), (15, 30)), ((0.03, 0.06), (15, 30)), ((0.10, 0.20), (5, 15)),
                           ((0.06, 0.10), (30, 45))):
    u = c[c.c_bid.between(lo, hi) & c.left.between(l0, l1)].sort_values("ts").drop_duplicates(["T", "ticker"])
    u["wk"] = np.where(u["T"].dt.dayofweek >= 5, "weekend", "weekday")
    # Kalshi hour T is the close; Friday 20:00 ET .. Sunday: dayofweek of the UTC close
    u["half"] = np.where(u["T"] < wl.SPLIT, "Jul–Aug", "Sep–Oct")
    u["month"] = u["T"].dt.strftime("%m")
    for keys in (["wk", "half"], ["wk", "month"]):
        for k, g in u.groupby(keys):
            h = g.groupby("T").pnl.sum()
            one = g.sort_values(["T", "c_bid"], ascending=[True, False]).drop_duplicates("T")
            out.append({"band": f"{100*lo:.0f}–{100*hi:.0f}¢ {l0}–{l1} min", "split": "/".join(keys), "value": " ".join(map(str, k)),
                        "strikes": len(g), "hours": len(h), "¢/strike": 100 * g.pnl.mean(),
                        "$/hour": h.mean(), "t (hours)": h.mean() / (h.std(ddof=1) / np.sqrt(len(h))),
                        "1/hour ¢": 100 * one.pnl.mean(),
                        "1/hour t": one.pnl.mean() / (one.pnl.std(ddof=1) / np.sqrt(len(one))),
                        "paid %": 100 * g.c_won.mean(), "bid ¢": 100 * g.c_bid.mean()})
R = pd.DataFrame(out)
R.to_csv("results/studies/wings_weekend.csv", index=False)
print(R.round(2).to_markdown(index=False))
