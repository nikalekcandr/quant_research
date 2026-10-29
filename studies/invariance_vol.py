"""Microstructure-based volatility forecasts on Binance BTC/USDT minutes (2023-01 .. 2026-10).

Question: does trading activity (number of trades N, dollar volume, Kyle's lambda from signed taker flow) forecast
next-period realised variance beyond a seasonal HAR? Two horizons: 1 hour (Kalshi hourly ladder) and 15 minutes
(Kalshi / Polymarket 15-minute markets).

Kyle–Obizhaeva invariance: the variance per bet is invariant, so returns variance scales with the number of bets;
Clark (1973) / Ané–Geman (2000): returns are Gaussian in trade time.  M3 forecasts log(RV/N) and log N separately.
Out of sample: expanding window from 2024-01, refit every 30 days; QLIKE; Diebold–Mariano with Newey–West.
    python invariance_vol.py bn_minutes.parquet out_dir
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SRC = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("bn_minutes.parquet")
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("out")


def buckets(m, freq):
    r = np.log(m["close"]).diff()
    flow = 2 * m["tbuy"] - m["vol"]                       # taker buy − taker sell, BTC
    g = pd.DataFrame({"r2": r ** 2, "r": r, "n": m["n"], "qv": m["qvol"], "flow": flow})
    b = g.resample(freq).agg({"r2": "sum", "n": "sum", "qv": "sum"})
    # Kyle's lambda per bucket: slope of 1-minute returns on signed flow (bp per 100 BTC)
    cov = (g["r"] * g["flow"]).resample(freq).mean() - g["r"].resample(freq).mean() * g["flow"].resample(freq).mean()
    var = g["flow"].resample(freq).var(ddof=0)
    b["lam"] = (cov / var).clip(lower=1e-9)
    b["absflow"] = flow.abs().resample(freq).sum() / m["vol"].resample(freq).sum()
    b = b[(b["r2"] > 0) & (b["n"] > 0)]
    return b


def design(b, per_day):
    d = pd.DataFrame(index=b.index)
    lr = np.log(b["r2"])
    d["y"] = lr.shift(-1)
    d["l1"] = lr
    d["l_day"] = lr.rolling(per_day).mean()
    d["l_week"] = lr.rolling(7 * per_day).mean()
    d["how"] = (b.index.dayofweek * 24 + b.index.hour) if per_day == 24 else (b.index.dayofweek * 96 + b.index.hour * 4 + b.index.minute // 15)
    d["how_next"] = d["how"].shift(-1)
    d["ln"] = np.log(b["n"])
    d["ln_day"] = d["ln"].rolling(per_day).mean()
    d["lqv"] = np.log(b["qv"])
    d["llam"] = np.log(b["lam"])
    d["absflow"] = b["absflow"]
    d["y_rn"] = (lr - d["ln"]).shift(-1)                    # log variance per trade, next bucket
    d["rn1"] = lr - d["ln"]
    d["rn_day"] = d["rn1"].rolling(per_day).mean()
    d["rn_week"] = d["rn1"].rolling(7 * per_day).mean()
    d["yn"] = d["ln"].shift(-1)
    d["ln_week"] = d["ln"].rolling(7 * per_day).mean()
    d["rv_next"] = b["r2"].shift(-1)
    return d.dropna()


def fit_predict(tr, te, target, cols):
    # hour-of-week seasonal mean of the target (train only), then OLS of the deseasonalised target
    sea = tr.groupby("how_next")[target].mean()
    ytr = tr[target] - tr["how_next"].map(sea)
    X = np.c_[np.ones(len(tr)), tr[cols].values]
    b, *_ = np.linalg.lstsq(X, ytr.values, rcond=None)
    res = ytr.values - X @ b
    pred = np.c_[np.ones(len(te)), te[cols].values] @ b + te["how_next"].map(sea).fillna(sea.mean()).values
    return pred, res


MODELS = {
    "HAR": ("y", ["l1", "l_day", "l_week"]),
    "HAR + trades": ("y", ["l1", "l_day", "l_week", "ln", "ln_day"]),
    "HAR + $ volume": ("y", ["l1", "l_day", "l_week", "lqv"]),
    "HAR + Kyle lambda": ("y", ["l1", "l_day", "l_week", "llam"]),
    "HAR + |flow imbalance|": ("y", ["l1", "l_day", "l_week", "absflow"]),
    "HAR + all": ("y", ["l1", "l_day", "l_week", "ln", "ln_day", "lqv", "llam", "absflow"]),
}


def run(d, label):
    start = pd.Timestamp("2024-01-01", tz="UTC")
    preds = {k: [] for k in list(MODELS) + ["Invariance (var per trade × trades)"]}
    idx = []
    t = start
    while t < d.index[-1]:
        t1 = t + pd.Timedelta(days=30)
        tr, te = d[d.index < t], d[(d.index >= t) & (d.index < t1)]
        if len(te):
            idx.append(te.index)
            for k, (target, cols) in MODELS.items():
                p, res = fit_predict(tr, te, target, cols)
                preds[k].append(p + math.log(np.mean(np.exp(res))))          # smearing: E[exp(e)]
            # invariance: forecast log(RV/N) and log N separately (errors combined by smearing jointly)
            p1, r1 = fit_predict(tr, te, "y_rn", ["rn1", "rn_day", "rn_week"])
            p2, r2 = fit_predict(tr, te, "yn", ["ln", "ln_day", "ln_week"])
            preds["Invariance (var per trade × trades)"].append(p1 + p2 + math.log(np.mean(np.exp(r1 + r2))))
        t = t1
    ix = idx[0].append(idx[1:]) if len(idx) > 1 else idx[0]
    rv = d.loc[ix, "rv_next"].values
    rows, loss = [], {}
    for k, v in preds.items():
        f = np.exp(np.concatenate(v))
        q = rv / f - np.log(rv / f) - 1
        loss[k] = q
        rows.append({"horizon": label, "model": k, "n": len(q), "QLIKE": q.mean(),
                     "MSE log": np.mean((np.log(rv) - np.log(f)) ** 2)})
    tab = pd.DataFrame(rows)
    base = loss["HAR"]
    L = int(4 * (len(base) / 100) ** (2 / 9))
    for k in loss:
        dd = base - loss[k]
        if k == "HAR":
            continue
        u = dd - dd.mean()
        g0 = (u @ u) / len(u)
        s = g0 + 2 * sum((1 - j / (L + 1)) * (u[j:] @ u[:-j]) / len(u) for j in range(1, L + 1))
        tab.loc[tab["model"] == k, "gain vs HAR"] = dd.mean()
        tab.loc[tab["model"] == k, "DM t (NW)"] = dd.mean() / math.sqrt(s / len(dd))
    # stability: gain by year
    yrs = pd.DatetimeIndex(ix).year
    for k in loss:
        if k == "HAR":
            continue
        for y in sorted(set(yrs)):
            m = yrs == y
            tab.loc[tab["model"] == k, f"gain {y}"] = (base[m] - loss[k][m]).mean()
    return tab


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    m = pd.read_parquet(SRC)
    out = []
    for freq, per_day, label in (("1h", 24, "1 hour"), ("15min", 96, "15 minutes")):
        d = design(buckets(m, freq), per_day)
        t = run(d, label)
        print(t.round(4).to_string(index=False), flush=True)
        out.append(t)
    pd.concat(out).to_csv(OUT / "invariance_vol.csv", index=False)


if __name__ == "__main__":
    main()
