"""Recommendation to test: add the variance of the last hour (and shorter windows) to the HAR models.

The hourly model (kx1h) already uses the last hour (`l_rv_d` = RV of [t − 1 h, t)), the last 24 h and 7 days.
The daily and weekly models (pm24h, pm6d) use 1 day, 7 and 30 days only. Here we add:
  kx1h   + log RV of the last 15 minutes and of the last 5 minutes
  pm24h  + log RV of the last hour and of the last 4 hours
  pm6d   + log RV of the last hour and of the last 4 hours
to `har` and to the project's best model `harx_all`, with the project's own rolling OLS (src/models.rolling_ols:
same windows, refits and minimum samples), and score every variant on the same rows from 2026-01-01 with CRPS of
log RV (normal, closed form), QLIKE and Diebold–Mariano (Newey–West) against the variant without the new lags.
    python studies/short_lags.py   (from the project root)
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from src import models as M  # noqa: E402

NEW = {"kx1h": {"l_rv_15m": 15, "l_rv_5m": 5}, "pm24h": {"l_rv_1h": 60, "l_rv_4h": 240},
       "pm6d": {"l_rv_1h": 60, "l_rv_4h": 240}}


def crps_normal(y, m, s):
    z = (y - m) / s
    return s * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / math.sqrt(math.pi))


def dm(d, h):
    d = d[np.isfinite(d)]
    u = d - d.mean()
    L = max(h, int(4 * (len(u) / 100) ** (2 / 9)))
    s = (u @ u + 2 * sum((1 - j / (L + 1)) * (u[j:] @ u[:-j]) for j in range(1, L + 1))) / len(u)
    return d.mean() / math.sqrt(s / len(u))


def main():
    mn = pd.read_parquet(config.INTERIM / "minutes.parquet", columns=["open_time", "close"])
    mn = mn.set_index(pd.to_datetime(mn["open_time"], utc=True))["close"]
    cs = (np.log(mn).diff() ** 2).fillna(0).cumsum()
    out = []
    for h in ("kx1h", "pm24h", "pm6d"):
        p = pd.read_parquet(config.INTERIM / f"panel_{h}.parquet").sort_values("t").reset_index(drop=True)
        t = pd.DatetimeIndex(p["t"])
        for name, mins in NEW[h].items():
            a = cs.reindex(t - pd.Timedelta(minutes=mins + 1), method="ffill").values
            b = cs.reindex(t - pd.Timedelta(minutes=1), method="ffill").values
            p[name] = np.log(np.clip(b - a, 1e-12, None))
        sp = M.specs(h)
        res = {}
        for base in ("har", "harx_all"):
            f0 = sp[base]["features"]
            mk = sp[base].get("market", False)
            for tag, feats in ((base, f0), (f"{base} + short lags", f0 + list(NEW[h]))):
                m, s, _ = M.rolling_ols(p, feats, h, market=mk)
                res[tag] = (m, s)
        y = p["y"].to_numpy(float)
        ok = (p["t"] >= pd.Timestamp(config.EVAL_START, tz="UTC")).to_numpy() & np.isfinite(y)
        for tag, (m, s) in res.items():
            ok &= np.isfinite(m) & np.isfinite(s)
        rv = np.exp(y)
        for tag, (m, s) in res.items():
            c = crps_normal(y, m, s)
            lvl = np.exp(m + s ** 2 / 2)
            q = rv / lvl - np.log(rv / lvl) - 1
            res[tag] = (m, s, c, q)
        overlap = 1 if h == "kx1h" else (1 if h == "pm24h" else 6)
        for base in ("har", "harx_all"):
            c0, q0 = res[base][2], res[base][3]
            c1, q1 = res[f"{base} + short lags"][2], res[f"{base} + short lags"][3]
            out.append({"horizon": h, "model": base, "added": ", ".join(NEW[h]), "n": int(ok.sum()),
                        "CRPS": c0[ok].mean(), "CRPS + lags": c1[ok].mean(), "CRPS gain %": 100 * (1 - c1[ok].mean() / c0[ok].mean()),
                        "DM t (CRPS)": dm(c1[ok] - c0[ok], overlap), "QLIKE": q0[ok].mean(), "QLIKE + lags": q1[ok].mean(),
                        "DM t (QLIKE)": dm(q1[ok] - q0[ok], overlap)})
        print(pd.DataFrame(out[-2:]).round(4).to_string(index=False), flush=True)
    d = pd.DataFrame(out)
    (config.RESULTS / "studies").mkdir(parents=True, exist_ok=True)
    d.to_csv(config.RESULTS / "studies" / "short_lags.csv", index=False)


if __name__ == "__main__":
    main()
