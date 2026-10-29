"""Volatility model horse race on Binance BTC/USDT minutes, 2023-01 → 2026-10, three horizons:
15 minutes (Kalshi / Polymarket 15-minute markets), 1 hour (Kalshi hourly ladder), 1 day.

Models (all forecast log realised variance of the next bucket; smearing back to variance):
  EWMA            RiskMetrics on bucket RV (λ = 0.94 per bucket)
  GARCH(1,1)      on bucket returns, Student-t (1 h and 1 day only; refit monthly)
  HAR             log RV on lags (1 bucket, 1 day, 1 week) + hour-of-week seasonal mean
  HAR deseas.     the same with the seasonal mean also removed from the 1-bucket lag
  HAR + activity  + log trades, log $ volume, log Kyle λ, |flow imbalance|   (Kyle–Obizhaeva / microstructure)
  GBM             gradient-boosted trees (HistGradientBoosting) on the HAR + activity features + calendar
  GBM (HAR only)  the same trees without activity
  Combo           equal-weight average of log forecasts of HAR + activity and GBM
Walk-forward: expanding window from 2024-01-01, refit every 30 days, no look-ahead (all features end at the
forecast time; seasonal means from the training window). Score: QLIKE (the robust loss for variance, Patton 2011),
MSE of log; Diebold–Mariano vs HAR with Newey–West; gain by year.
    python studies/vol_horserace.py <bn_minutes.parquet> <out dir>
"""
from __future__ import annotations

import math
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
SRC = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("bn_minutes.parquet")
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("results/studies")


def buckets(m, freq):
    r = np.log(m["close"]).diff()
    flow = 2 * m["tbuy"] - m["vol"]
    g = pd.DataFrame({"r2": r ** 2, "r": r, "n": m["n"], "qv": m["qvol"], "flow": flow})
    b = g.resample(freq).agg({"r2": "sum", "n": "sum", "qv": "sum", "r": "sum"})
    cov = (g["r"] * g["flow"]).resample(freq).mean() - g["r"].resample(freq).mean() * g["flow"].resample(freq).mean()
    b["lam"] = (cov / g["flow"].resample(freq).var(ddof=0)).clip(lower=1e-9)
    b["absflow"] = flow.abs().resample(freq).sum() / m["vol"].resample(freq).sum()
    return b[(b["r2"] > 0) & (b["n"] > 0)]


def design(b, per_day, freq):
    d = pd.DataFrame(index=b.index)
    lr = np.log(b["r2"])
    d["y"] = lr.shift(-1)
    d["l1"] = lr
    d["l_day"] = lr.rolling(per_day).mean()
    d["l_week"] = lr.rolling(7 * per_day).mean()
    nxt = b.index + pd.tseries.frequencies.to_offset(freq)
    if per_day == 1:
        d["how_next"] = nxt.dayofweek
    elif per_day == 24:
        d["how_next"] = nxt.dayofweek * 24 + nxt.hour
    else:
        d["how_next"] = nxt.dayofweek * 96 + nxt.hour * 4 + nxt.minute // 15
    d["hour_next"], d["dow_next"] = nxt.hour, nxt.dayofweek
    d["how_cur"] = d["how_next"].shift(1).fillna(d["how_next"])
    d["ln"] = np.log(b["n"])
    d["ln_day"] = d["ln"].rolling(per_day).mean()
    d["lqv"] = np.log(b["qv"])
    d["llam"] = np.log(b["lam"])
    d["absflow"] = b["absflow"]
    d["ret"] = b["r"]
    d["ret_next"] = b["r"].shift(-1)
    d["rv"] = b["r2"]
    d["rv_next"] = b["r2"].shift(-1)
    return d.dropna()


HAR = ["l1", "l_day", "l_week"]
ACT = ["ln", "ln_day", "lqv", "llam", "absflow"]


def ols(tr, te, cols):
    sea = tr.groupby("how_next")["y"].mean()
    ytr = tr["y"] - tr["how_next"].map(sea)
    X = np.c_[np.ones(len(tr)), tr[cols].values]
    b, *_ = np.linalg.lstsq(X, ytr.values, rcond=None)
    res = ytr.values - X @ b
    pred = np.c_[np.ones(len(te)), te[cols].values] @ b + te["how_next"].map(sea).fillna(sea.mean()).values
    return pred + math.log(np.mean(np.exp(res)))


def ols_sa(tr, te, cols):
    """HAR on deseasonalised logs: the seasonal mean of the bucket's own hour-of-week is removed from the target AND
    from the lags (a lag measured at the US open is not the level of the quiet hour that follows)."""
    sea = tr.groupby("how_next")["y"].mean()
    s_now = lambda d: d["how_next"].map(sea.shift(1, fill_value=sea.iloc[-1]).reindex(sea.index)).fillna(sea.mean())
    cur = {"l1": "how_cur"}
    def adj(d):
        x = d[cols].copy()
        sc = d["how_cur"].map(sea).fillna(sea.mean())
        x["l1"] = x["l1"] - sc
        x["l_day"] = x["l_day"] - sea.mean()
        x["l_week"] = x["l_week"] - sea.mean()
        return x
    ytr = tr["y"] - tr["how_next"].map(sea)
    X = np.c_[np.ones(len(tr)), adj(tr).values]
    b, *_ = np.linalg.lstsq(X, ytr.values, rcond=None)
    res = ytr.values - X @ b
    pred = np.c_[np.ones(len(te)), adj(te).values] @ b + te["how_next"].map(sea).fillna(sea.mean()).values
    return pred + math.log(np.mean(np.exp(res)))


def gbm(tr, te, cols):
    from sklearn.ensemble import HistGradientBoostingRegressor
    f = cols + ["hour_next", "dow_next"]
    trn = tr.iloc[-200_000:]
    g = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=200,
                                      l2_regularization=1.0, random_state=0)
    g.fit(trn[f].values, trn["y"].values)
    res = trn["y"].values - g.predict(trn[f].values)
    return g.predict(te[f].values) + math.log(np.mean(np.exp(res)))


def ewma(d, idx):
    v = d["rv"].ewm(alpha=0.06, adjust=False).mean()
    return np.log(v.reindex(idx).values)


def garch(tr, te, scale):
    from arch import arch_model
    x = tr["ret"].iloc[-5000:] * scale
    am = arch_model(x, mean="Zero", vol="GARCH", p=1, q=1, dist="t").fit(disp="off")
    w, a, b = am.params["omega"], am.params["alpha[1]"], am.params["beta[1]"]
    s2 = float(am.conditional_volatility.iloc[-1] ** 2)
    r_prev = float(x.iloc[-1])
    out = []
    for r in te["ret"].values * scale:          # one-step recursion through the test month (uses only past returns)
        s2 = w + a * r_prev ** 2 + b * s2
        out.append(s2)
        r_prev = r
    # out[i] is the variance of the bucket after te row i's predecessor; shift so row i forecasts row i + 1
    f = np.r_[out[1:], w + a * r_prev ** 2 + b * s2]
    return np.log(np.array(f) / scale ** 2)


def run(d, label, use_garch):
    preds = {k: [] for k in ["EWMA", "GARCH(1,1)", "HAR", "HAR deseasonalised", "HAR + activity", "GBM (HAR only)",
                             "GBM", "Combo"]}
    if not use_garch:
        preds.pop("GARCH(1,1)")
    idx, t = [], pd.Timestamp("2024-01-01", tz="UTC")
    while t < d.index[-1]:
        t1 = t + pd.Timedelta(days=30)
        tr, te = d[d.index < t], d[(d.index >= t) & (d.index < t1)]
        if len(te):
            idx.append(te.index)
            preds["EWMA"].append(ewma(d, te.index))
            if use_garch:
                try:
                    preds["GARCH(1,1)"].append(garch(tr, te, 100.0))
                except Exception:
                    preds["GARCH(1,1)"].append(np.full(len(te), np.nan))
            preds["HAR"].append(ols(tr, te, HAR))
            preds["HAR deseasonalised"].append(ols_sa(tr, te, HAR))
            ha = ols(tr, te, HAR + ACT)
            preds["HAR + activity"].append(ha)
            preds["GBM (HAR only)"].append(gbm(tr, te, HAR))
            gb = gbm(tr, te, HAR + ACT)
            preds["GBM"].append(gb)
            preds["Combo"].append((ha + gb) / 2)
        t = t1
    ix = idx[0].append(idx[1:])
    rv = d.loc[ix, "rv_next"].values
    loss, rows = {}, []
    for k, v in preds.items():
        f = np.exp(np.concatenate(v))
        ok = np.isfinite(f) & (f > 0)
        q = np.where(ok, rv / f - np.log(rv / f) - 1, np.nan)
        loss[k] = q
        rows.append({"horizon": label, "model": k, "n": int(ok.sum()), "QLIKE": np.nanmean(q),
                     "MSE log": np.nanmean(np.where(ok, (np.log(rv) - np.log(f)) ** 2, np.nan))})
    tab = pd.DataFrame(rows)
    base = loss["HAR"]
    yrs = pd.DatetimeIndex(ix).year
    for k, q in loss.items():
        dd = base - q
        m = np.isfinite(dd)
        dd = dd[m]
        if k == "HAR" or not len(dd):
            continue
        u = dd - dd.mean()
        L = int(4 * (len(u) / 100) ** (2 / 9))
        s = (u @ u + 2 * sum((1 - j / (L + 1)) * (u[j:] @ u[:-j]) for j in range(1, L + 1))) / len(u)
        tab.loc[tab.model == k, "gain vs HAR"] = dd.mean()
        tab.loc[tab.model == k, "DM t"] = dd.mean() / math.sqrt(s / len(u))
        for y in sorted(set(yrs)):
            mm = (yrs == y) & np.isfinite(base - q)
            tab.loc[tab.model == k, f"gain {y}"] = np.nanmean((base - q)[mm])
    return tab.sort_values("QLIKE")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    m = pd.read_parquet(SRC)
    res = []
    for freq, per_day, label, g in (("15min", 96, "15 minutes", False), ("1h", 24, "1 hour", True), ("1D", 1, "1 day", True)):
        d = design(buckets(m, freq), per_day, freq)
        t = run(d, label, g)
        print(t.round(4).to_string(index=False), "\n", flush=True)
        res.append(t)
    pd.concat(res).to_csv(OUT / "vol_horserace.csv", index=False)


if __name__ == "__main__":
    main()
