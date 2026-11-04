"""Storm filter for the wing rules (04.11.2026): skip a wing when the last minutes moved much more than the model
expects. Signal R_w = realised variance of the last w minutes / the model's variance for those w minutes (EWMA level
× hour-of-week profile, the same numbers W03/W04 use). Strictly past data: minutes closed at or before the decision.

Part 1 (7 243 settlements, Dec 2025 – Nov 2026): tails of z = log(E/(S·basis))/sd_model by R bucket — does a storm
predict the fat tails?
Part 2 (wings backtest, Jul–Oct 2026): W03/W04 with "skip if R_w > c", c chosen on Jul–Aug, read on Sep–Oct; and the
same with the variance scaled up by R instead of skipping.
    python studies/storm_filter.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.argv = sys.argv[:1] + ["data/raw/usd_minutes"]
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from studies import brti_proxy as bp  # noqa: E402
from studies import wings_ladder as wl  # noqa: E402

wl.LAD, wl.MIN = Path("data/raw/kalshi_kxbtcd_60m"), Path("data/interim/minutes.parquet")
OUT = Path("results/studies")
WINDOWS = (5, 15, 60)


def storm(S: pd.Series) -> pd.DataFrame:
    vm = wl.variance_level(S)
    r2 = np.log(S).diff() ** 2
    exp = vm["lvl"] * vm["sea"]                       # model variance of one minute, known at the minute
    out = {}
    for w in WINDOWS:
        out[f"R{w}"] = r2.rolling(w, min_periods=w).sum() / (exp.shift(1).rolling(w, min_periods=w).sum())
    return pd.DataFrame(out)


def part1(E, S, name):
    R = storm(S)
    b = bp.lagged_basis(E, S)
    vm = wl.variance_level(S)
    rows = []
    for tau in (5, 15, 30):
        t0 = E.index - pd.Timedelta(minutes=tau)
        var = vm["lvl"].reindex(t0).values * vm["sea"].reindex(t0).values * max(tau - 2 / 3, 1 / 3)
        z = pd.Series(np.log(E.values / (S.reindex(t0).values * b.values)) / np.sqrt(var), index=E.index)
        for w in WINDOWS:
            r = pd.Series(R[f"R{w}"].reindex(t0).values, index=E.index)
            d = pd.DataFrame({"z": z, "r": r}).dropna()
            d = d[np.isfinite(d.z)]
            d["bk"] = pd.cut(d.r, [0, 0.5, 1, 2, 3, 5, np.inf])
            for bk, g in d.groupby("bk", observed=True):
                rows.append({"spot": name, "τ": tau, "window": w, "R bucket": str(bk), "n": len(g),
                             "share %": 100 * len(g) / len(d), "var(z)": g.z.var(),
                             "P|z|>2 %": 100 * (g.z.abs() > 2).mean(), "P|z|>3 %": 100 * (g.z.abs() > 3).mean()})
    return pd.DataFrame(rows)


RULES = {"W03": (0.10, 0.20, 5, 15, "normal", 0.01, 1), "W04": (0.06, 0.10, 15, 30, "t4", 0.0, 1)}


def part2(E, S, name):
    c = bp.wings_frame(S, E)
    R = storm(S)
    for w in WINDOWS:
        c[f"R{w}"] = R[f"R{w}"].reindex(c.ts).values
    rows = []
    for rule, p in RULES.items():
        g0 = wl.run(c, *p)
        for w in WINDOWS:
            for cut in (np.inf, 6, 4, 3, 2.5, 2, 1.5):
                g = g0[~(g0[f"R{w}"] > cut)]
                rec = {"spot": name, "rule": rule, "window": w, "skip if R >": cut}
                for half, gg in (("Jul–Aug", g[g["T"] < wl.SPLIT]), ("Sep–Oct", g[g["T"] >= wl.SPLIT])):
                    s = wl.stats(gg)
                    sk = g0[(g0["T"] < wl.SPLIT) == (half == "Jul–Aug")]
                    skipped = sk[sk[f"R{w}"] > cut]
                    rec[f"{half} n"] = s.get("trades", 0)
                    rec[f"{half} ¢/tr"] = s.get("¢/trade", np.nan)
                    rec[f"{half} $"] = gg.pnl.sum()
                    rec[f"{half} t"] = s.get("t (hours)", np.nan)
                    rec[f"{half} skipped ¢/tr"] = 100 * skipped.pnl.mean() if len(skipped) else np.nan
                    rec[f"{half} skipped n"] = len(skipped)
                rows.append(rec)
    return pd.DataFrame(rows)


def main():
    E = bp.settlements()
    cb, bs = bp.candles("cb"), bp.candles("bs")
    spots = {"binance×basis": bp.binance(), "cb+bs×basis": ((cb.c + bs.c) / 2).dropna()}
    lines = ["# Storm filter for the wings (studies/storm_filter.py)", "",
             "R_w = realised variance of the last w minutes / model variance for those minutes.", ""]
    P1 = pd.concat([part1(E, S, n) for n, S in spots.items()])
    P1.to_csv(OUT / "storm_part1.csv", index=False)
    lines += ["## 1. Tails of z by storm level (7 243 settlements, Dec 2025 – Nov 2026)",
              "Normal: P|z|>2 4.55%, P|z|>3 0.27%.", "",
              P1[P1.window == 15].round(2).to_markdown(index=False), ""]
    P2 = pd.concat([part2(E, S, n) for n, S in spots.items()])
    P2.to_csv(OUT / "storm_part2.csv", index=False)
    lines += ["## 2. W03/W04 with 'skip if R_w > c' (Jul–Aug train, Sep–Oct test)", "",
              P2.round(2).to_markdown(index=False), ""]
    (OUT / "storm_filter.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
