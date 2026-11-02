"""Tables for the O04 / intraday-gamma write-up (reads results/studies/o04_panel.csv and gamma_intraday_panel.csv).
    python studies/o04_report.py      → results/studies/o04_tables.md
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
import gamma_intraday as G  # noqa: E402
import o04_straddle as O  # noqa: E402

OUT = config.RESULTS / "studies"
SPLIT = pd.Timestamp("2025-01-01", tz="UTC")


def md(df: pd.DataFrame, digits=2) -> str:
    d = df.copy()
    for c in d.columns:
        if pd.api.types.is_float_dtype(d[c]):
            d[c] = d[c].map(lambda v: "" if pd.isna(v) else f"{v:,.{digits}f}")
    return d.to_markdown(index=False)


def qlike(rv, f):
    x = rv / f
    return x - np.log(x) - 1


def main():
    b = pd.read_csv(OUT / "o04_panel.csv", parse_dates=["entry", "expiry"])
    b["hour"] = b["entry"].dt.hour
    b["year"] = b["expiry"].dt.year
    parts = []
    parts.append(f"Panel: {len(b)} entries, {b['expiry'].nunique()} expiries, {b['expiry'].min():%Y-%m-%d} – "
                 f"{b['expiry'].max():%Y-%m-%d}; entries per hour {b['hour'].value_counts().sort_index().to_dict()}; "
                 f"settlement {b['settle'].value_counts().to_dict()}")

    # 1. who forecasts the variance to expiry better: Deribit IV or the lab's seasonal EWMA?
    rows = []
    for h, x in [("all", b)] + list(b.groupby("hour")):
        x = x[(x.rv > 0)]
        qi, qf = qlike(x.rv, x.var_iv), qlike(x.rv, x.var_fc)
        rows.append({"entry UTC": h, "n": len(x), "hours to expiry": x.tau_h.mean(), "IV, % ann": 100 * x.iv.mean(),
                     "lab forecast, %": 100 * x.fc_vol.mean(),
                     "realised, %": 100 * np.sqrt((x.rv / (x.tau_h / 8760)).mean()),
                     "RV/IV var (geo)": np.exp(np.log(x.rv / x.var_iv).mean()),
                     "QLIKE IV": qi.mean(), "QLIKE lab": qf.mean(), "DM t (lab − IV)": O.hac_t((qf - qi).values)})
    parts.append("## 1. Forecast of the variance to expiry: Deribit IV vs the lab's model\n\n" + md(pd.DataFrame(rows), 3))

    # 2. the variance risk premium in straddle terms
    rows = []
    for (h, y), x in b.groupby(["hour", "year"]):
        rows.append({"entry UTC": h, "year": y, "n": len(x), "mean payoff $": x.payoff.mean(), "mean mid $": x.mid.mean(),
                     "payoff / mid": x.payoff.sum() / x.mid.sum()})
    parts.append("## 2. Straddle payoff vs its mid price (payoff/mid < 1: options are dear)\n\n" + md(pd.DataFrame(rows), 3))

    # 3. O04 as frozen: θ = 10%, both sides, first entry per expiry that fires
    rows = []
    for label, x in (("all", b), ("2023–2024", b[b.expiry < SPLIT]), ("2025–2026", b[b.expiry >= SPLIT])):
        for side in ("both", "long", "short"):
            t = O.trade(x, 0.10, side)
            s = O.stats(t, f"O04 {side}")
            lo, hi = O.boot_ci(t.pnl.values) if len(t) >= 20 else (np.nan, np.nan)
            rows.append({"sample": label, **s, "95% CI $/trade": f"[{lo:,.0f}; {hi:,.0f}]"})
    parts.append("## 3. O04 as frozen (θ = 10%, fair value from the lab model)\n\n" + md(pd.DataFrame(rows)))

    # 4. robustness grid
    rows = []
    for th in (0.0, 0.05, 0.10, 0.20):
        for side in ("long", "short"):
            for sp in (0.3, 1.0):
                for hrs, hl in ((None, "all"), ([0], "00"), ([9], "09"), ([16, 17], "16")):
                    t = O.trade(b, th, side, sp, hrs)
                    s = O.stats(t, "")
                    rows.append({"θ": th, "side": side, "spread ×": sp, "entry": hl, "trades": s["trades"],
                                 "$/trade": s.get("$/trade"), "ret %": s.get("ret/premium %"), "t HAC": s.get("t HAC")})
    for side in ("long", "short"):
        for sp in (0.3, 1.0):
            for hrs, hl in ((None, "all"), ([0], "00"), ([9], "09"), ([16, 17], "16")):
                t = O.trade(b, None, side, sp, hrs)
                s = O.stats(t, "")
                rows.append({"θ": "always", "side": side, "spread ×": sp, "entry": hl, "trades": s["trades"],
                             "$/trade": s.get("$/trade"), "ret %": s.get("ret/premium %"), "t HAC": s.get("t HAC")})
    g = pd.DataFrame(rows)
    g.to_csv(OUT / "o04_grid.csv", index=False)
    parts.append("## 4. Grid: threshold × side × spread × entry hour (one trade per expiry)\n\n" + md(g))

    # 5. intraday gamma: delta-hedged straddles by entry hour
    gp = OUT / "gamma_intraday_panel.csv"
    if gp.exists():
        gg = pd.read_csv(gp, parse_dates=["entry", "expiry"])
        s = G.summarise(gg)
        s.to_csv(OUT / "gamma_intraday_summary.csv", index=False)
        keep = s[s.hedge.isin(["nohedge", "every 15 min", "every 60 min", "band 0.25", "every 1 min (theory)"])]
        parts.append("## 5. Delta-hedged straddles by entry hour\n\n" + md(keep.drop(columns=["worst $"])))
        rows = []
        for h, x in gg.groupby("hour"):
            rows.append({"entry UTC": h, "n": len(x), "IV %": 100 * x.iv.mean(), "realised %": 100 * np.sqrt((x.rv_ann ** 2).mean()),
                         "share of days RV > IV %": 100 * (x.rv_ann > x.iv).mean(),
                         "long gamma at mid, no costs, 1-min hedge, $": (x.payoff - x.mid + x["long|every 1 min (theory)|gross"]).mean(),
                         "t": O.hac_t((x.payoff - x.mid + x["long|every 1 min (theory)|gross"]).values)})
        parts.append("### Realised vs implied by entry hour\n\n" + md(pd.DataFrame(rows)))
    (OUT / "o04_tables.md").write_text("\n\n".join(parts) + "\n", encoding="utf-8")
    print("\n\n".join(parts))


if __name__ == "__main__":
    main()
