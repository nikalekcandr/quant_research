"""Sub-minute lead–lag between spot and the 15-minute binaries (04.11.2026), lab books every ~15 s, 29.10–02.11.
For each market and snapshot: mid(t), the model fair from the spot at t (fair = what the price should be now).
1. Does the book catch up? Δmid(t → t+k) on gap(t) = fair(t) − mid(t), k = 1, 2, 4 snapshots (15, 30, 60 s), with
   and without the past move of the mid. β > 0 = the mid moves toward the spot-implied value later = a lag.
2. Tradeable? Buy YES at the ask when fair − ask − fee ≥ x (sell when bid − fair − fee ≥ x); value at the mid
   k snapshots later (markout) and at settlement is not available here, so markout is the measure.
Clusters: market.  python studies/fast_leadlag.py
"""
import math
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm

d = pd.concat([pd.read_csv(f) for f in sorted(Path("data/lab/markets_fast_pc").glob("*.csv"))], ignore_index=True)
d["ts"] = pd.to_datetime(d.ts, utc=True, format="mixed")
d = d[(d.bid > 0) & (d.ask > d.bid) & (d.ask - d.bid <= 0.05) & d.fair.notna()]
d["mid"] = (d.bid + d.ask) / 2
d["end"] = pd.to_datetime(d["end"], utc=True)
d["left_s"] = (d["end"] - d.ts).dt.total_seconds()
d = d[d.left_s > 60].sort_values(["market", "ts"])
lines = ["# Sub-minute lead–lag: spot vs 15-minute binaries (studies/fast_leadlag.py)", "",
         f"{len(d)} snapshots, {d.market.nunique()} markets, {d.ts.min():%Y-%m-%d %H:%M} .. {d.ts.max():%Y-%m-%d %H:%M}; "
         f"median gap between snapshots {d.groupby('market').ts.diff().dt.total_seconds().median():.0f} s.", ""]
rows = []
for venue, g in d.groupby("kind"):
    g = g.copy()
    grp = g.groupby("market")
    g["gap"] = g.fair - g.mid
    g["dmid_prev"] = grp.mid.diff()
    for k in (1, 2, 4):
        g[f"dmid{k}"] = grp.mid.shift(-k) - g.mid
        g[f"dt{k}"] = (grp.ts.shift(-k) - g.ts).dt.total_seconds()
        x = g.dropna(subset=["gap", "dmid_prev", f"dmid{k}"])
        x = x[x[f"dt{k}"] < 25 * k]
        for spec, cols in (("gap", ["gap"]), ("gap + past move", ["gap", "dmid_prev"])):
            m = sm.OLS(x[f"dmid{k}"], sm.add_constant(x[cols])).fit(cov_type="cluster",
                                                                      cov_kwds={"groups": pd.factorize(x.market)[0]})
            rows.append({"market": venue, "horizon s": 15 * k, "spec": spec, "n": len(x),
                         "β gap": m.params["gap"], "t": m.tvalues["gap"],
                         "β past move": m.params.get("dmid_prev", np.nan), "R²": m.rsquared})
    # tradeable: taker at the touch when the spot-implied fair is beyond it
    fr = g.fee_rate.fillna(0.0)
    fee_b = np.where(g.kind == "kx_15m", np.ceil(np.round(0.07 * g.ask * (1 - g.ask) * 100, 6)) / 100, fr * g.ask * (1 - g.ask))
    fee_s = np.where(g.kind == "kx_15m", np.ceil(np.round(0.07 * g.bid * (1 - g.bid) * 100, 6)) / 100, fr * g.bid * (1 - g.bid))
    for xth in (0.0, 0.02, 0.04):
        buy = (g.fair - g.ask - fee_b) >= xth
        sell = (g.bid - g.fair - fee_s) >= xth
        for k in (1, 2, 4):
            fb = pd.DataFrame({"x": (g.mid + g[f"dmid{k}"] - g.ask - fee_b)[buy], "c": g.market[buy]})
            fs = pd.DataFrame({"x": (g.bid - fee_s - (g.mid + g[f"dmid{k}"]))[sell], "c": g.market[sell]})
            ff = pd.concat([fb, fs], ignore_index=True).dropna()
            mk, cl = ff.x, ff.c
            if len(mk) > 5:
                gg = pd.DataFrame({"x": mk.values, "c": cl.values}).groupby("c").x.agg(["sum", "size"])
                mu = mk.mean()
                se = math.sqrt(((gg["sum"] - mu * gg["size"]) ** 2).sum()) / len(mk)
                rows.append({"market": venue, "horizon s": 15 * k, "spec": f"taker, edge ≥ {100*xth:.0f}¢: markout ¢",
                             "n": len(mk), "β gap": 100 * mu, "t": mu / se if se else np.nan, "β past move": np.nan, "R²": np.nan})
R = pd.DataFrame(rows)
lines += ["β gap > 0: the mid later moves toward the value implied by the current spot (the book lags). "
          "Taker rows: markout = mid k snapshots later − entry price − fee, in ¢ (the 'β gap' column).", "",
          R.round(3).to_markdown(index=False), ""]
Path("results/studies/fast_leadlag.md").write_text("\n".join(lines), encoding="utf-8")
print("\n".join(lines))
