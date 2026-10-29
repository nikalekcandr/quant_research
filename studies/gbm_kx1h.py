"""Gradient boosting trained on three years of Binance minutes, scored on the course project's own target: the
variance of BTC log returns from HH:05 to the next full hour (the Kalshi hourly ladder window, `kx1h`).

For each panel row (t = HH:05, end e) features are computed from minutes before t only: log RV of the last 5, 60
minutes, of the last day and week (per hour), trades, $ volume, Kyle λ and |flow imbalance| of the last hour, hour
and weekday of the window. Walk-forward: train on every HH:05 since 2023-01 before the month, refit monthly.
The forecast density of log RV is normal with the training residual sd. Every model of the project is rescored on
the same rows with the same CRPS (closed form for normal mixtures, Grimit et al. 2006) and QLIKE.
    python studies/gbm_kx1h.py <bn_minutes.parquet> <results/forecasts_kx1h.parquet> <out dir>
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

BN = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("bn_minutes.parquet")
FC = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("results/forecasts_kx1h.parquet")
OUT = Path(sys.argv[3]) if len(sys.argv) > 3 else Path("results/studies")


def A(mu, var):
    s = np.sqrt(var)
    z = mu / s
    return 2 * s * norm.pdf(z) + mu * (2 * norm.cdf(z) - 1)


def crps_mix(y, ms, ss, ws):
    """ms, ss, ws: arrays (n, k) with NaN for unused components."""
    ws = np.nan_to_num(ws)
    ms = np.nan_to_num(ms)
    ss = np.where(np.isfinite(ss), ss, 1.0)
    t1 = sum(ws[:, i] * A(y - ms[:, i], ss[:, i] ** 2) for i in range(ms.shape[1]))
    t2 = sum(ws[:, i] * ws[:, j] * A(ms[:, i] - ms[:, j], ss[:, i] ** 2 + ss[:, j] ** 2)
             for i in range(ms.shape[1]) for j in range(ms.shape[1]))
    return t1 - 0.5 * t2


def features(m, times, ends):
    r2 = np.log(m["close"]).diff() ** 2
    flow = 2 * m["tbuy"] - m["vol"]
    cs = lambda s: s.fillna(0).cumsum()
    C = {"r2": cs(r2), "n": cs(m["n"]), "qv": cs(m["qvol"]), "af": cs(flow.abs()), "v": cs(m["vol"]),
         "rf": cs(np.log(m["close"]).diff() * flow), "r": cs(np.log(m["close"]).diff()), "f": cs(flow), "ff": cs(flow ** 2)}

    def win(key, a, b):
        x = C[key]
        return x.reindex(b, method="ffill").values - x.reindex(a, method="ffill").values

    t = pd.DatetimeIndex(times)
    e = pd.DatetimeIndex(ends)
    m1 = pd.Timedelta(minutes=1)
    X = pd.DataFrame(index=range(len(t)))
    X["lrv5"] = np.log(win("r2", t - pd.Timedelta(minutes=5) - m1, t - m1) + 1e-12)
    X["lrv60"] = np.log(win("r2", t - pd.Timedelta(minutes=60) - m1, t - m1) + 1e-12)
    X["lrv_day"] = np.log(win("r2", t - pd.Timedelta(days=1) - m1, t - m1) / 24 + 1e-12)
    X["lrv_week"] = np.log(win("r2", t - pd.Timedelta(days=7) - m1, t - m1) / 168 + 1e-12)
    n60 = win("n", t - pd.Timedelta(minutes=60) - m1, t - m1)
    X["ln60"] = np.log(n60 + 1)
    X["lqv60"] = np.log(win("qv", t - pd.Timedelta(minutes=60) - m1, t - m1) + 1)
    X["absflow60"] = win("af", t - pd.Timedelta(minutes=60) - m1, t - m1) / (win("v", t - pd.Timedelta(minutes=60) - m1, t - m1) + 1e-9)
    # Kyle λ over the last hour: cov(r, flow) / var(flow) with 60 one-minute observations
    n = 60
    sr, sf = win("r", t - pd.Timedelta(minutes=60) - m1, t - m1), win("f", t - pd.Timedelta(minutes=60) - m1, t - m1)
    srf, sff = win("rf", t - pd.Timedelta(minutes=60) - m1, t - m1), win("ff", t - pd.Timedelta(minutes=60) - m1, t - m1)
    lam = (srf / n - sr * sf / n ** 2) / (sff / n - (sf / n) ** 2 + 1e-12)
    X["llam"] = np.log(np.clip(lam, 1e-9, None))
    X["hour"], X["dow"] = e.hour, e.dayofweek
    X["minutes"] = (e - t).total_seconds() / 60
    y = np.log(win("r2", t - m1, e - m1) + 1e-12)
    return X, y


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    from sklearn.ensemble import HistGradientBoostingRegressor
    m = pd.read_parquet(BN)
    last = m.index[-1]
    # training grid: every HH:05 → HH+1:00 since 2023-01-08
    grid_t = pd.date_range(pd.Timestamp("2023-01-08 00:05", tz="UTC"), last - pd.Timedelta(hours=1), freq="h")
    grid_e = grid_t + pd.Timedelta(minutes=55)
    Xg, yg = features(m, grid_t, grid_e)
    ok = np.isfinite(yg) & (yg > -30)
    Xg, yg, grid_t = Xg[ok].reset_index(drop=True), yg[ok], grid_t[ok]
    fc = pd.read_parquet(FC)
    fc["t"] = pd.to_datetime(fc["t"], utc=True)
    fc["e"] = pd.to_datetime(fc["e"], utc=True)
    rows = fc.drop_duplicates("t")[["t", "e", "y", "rv"]]
    rows = rows[(rows.t >= "2026-01-15") & (rows.e <= last)].reset_index(drop=True)
    Xr, yr = features(m, rows.t, rows.e)
    pred, sd = np.full(len(rows), np.nan), np.full(len(rows), np.nan)
    months = pd.date_range(rows.t.min().floor("D").replace(day=1), rows.t.max() + pd.offsets.MonthBegin(1), freq="MS")
    for a, b in zip(months[:-1], months[1:]):
        te = (rows.t >= a) & (rows.t < b)
        if not te.any():
            continue
        tr = grid_t + pd.Timedelta(minutes=55) <= a
        g = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=100,
                                          l2_regularization=1.0, random_state=0).fit(Xg[tr].values, yg[tr])
        # residual sd from the last 20% of the training window, refit-free (honest out-of-bag proxy)
        k = int(tr.sum() * 0.8)
        g2 = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=100,
                                           l2_regularization=1.0, random_state=0).fit(Xg[tr].values[:k], yg[tr][:k])
        res = yg[tr][k:] - g2.predict(Xg[tr].values[k:])
        pred[te.values] = g.predict(Xr[te.values].values)
        sd[te.values] = res.std()
        print(f"{a:%Y-%m}: {te.sum()} rows, trained on {tr.sum()} hours, resid sd {res.std():.3f}", flush=True)
    rows["m_gbm"], rows["s_gbm"] = pred, sd
    rows["y_bn"] = yr
    # score every project model + GBM-long + combos on the same rows
    f = fc.merge(rows[["t"]], on="t")
    out = []
    samples = {"history (15.01–12.10)": rows.t < "2026-10-13", "new (13.10–28.10)": rows.t >= "2026-10-13"}
    def score(name, ms, ss, ws, level, key):
        y = key["y"].values
        c = crps_mix(y, ms, ss, ws)
        q = key["rv"].values / level - np.log(key["rv"].values / level) - 1
        return c, q
    res_rows = {}
    for model, g in f.groupby("model"):
        g = g.set_index("t").reindex(rows.t)
        ms = g[["m1", "m2", "m3"]].values
        ss = g[["s1", "s2", "s3"]].values
        ws = g[["w1", "w2", "w3"]].values
        res_rows[model] = score(model, ms, ss, ws, g["level"].values, rows)
    lvl = np.exp(rows.m_gbm + rows.s_gbm ** 2 / 2)
    res_rows["GBM (3 years of minutes)"] = score("gbm", rows[["m_gbm"]].values, rows[["s_gbm"]].values,
                                                 np.ones((len(rows), 1)), lvl.values, rows)
    for other in ("harx_all", "mkt_ln_raw"):
        g = f[f.model == other].set_index("t").reindex(rows.t)
        mm = (g["mean_log"].values + rows.m_gbm.values) / 2
        ssd = np.sqrt((rows.s_gbm.values ** 2 + np.nan_to_num(g["s1"].values, nan=1.0) ** 2) / 2)
        res_rows[f"GBM + {other} (average)"] = score("c", mm[:, None], ssd[:, None], np.ones((len(rows), 1)),
                                                     np.exp(mm + ssd ** 2 / 2), rows)
    # one common sample: hours where the ladder, the project's best model and the GBM all have a forecast
    common = np.ones(len(rows), dtype=bool)
    for k in ("harx_all", "mkt_ln_raw", "GBM (3 years of minutes)"):
        common &= np.isfinite(res_rows[k][0]) & np.isfinite(res_rows[k][1])
    samples = {k: v & common for k, v in samples.items()}
    for name, (c, q) in res_rows.items():
        rec = {"model": name}
        for sname, msk in samples.items():
            ok = msk.values & np.isfinite(c) & np.isfinite(q)
            rec[f"CRPS {sname}"] = c[ok].mean()
            rec[f"QLIKE {sname}"] = q[ok].mean()
            rec[f"n {sname}"] = int(ok.sum())
        out.append(rec)
    tab = pd.DataFrame(out).sort_values("CRPS new (13.10–28.10)")
    # DM on CRPS vs the project's best model (harx_all) and the ladder centre
    for ref in ("harx_all", "mkt_ln_raw"):
        for sname, msk in samples.items():
            c0 = res_rows[ref][0]
            for name, (c, q) in res_rows.items():
                d = (c - c0)[msk.values]
                d = d[np.isfinite(d)]
                if len(d) > 10 and name != ref:
                    u = d - d.mean()
                    L = int(4 * (len(u) / 100) ** (2 / 9))
                    s = (u @ u + 2 * sum((1 - j / (L + 1)) * (u[j:] @ u[:-j]) for j in range(1, L + 1))) / len(u)
                    tab.loc[tab.model == name, f"DM vs {ref}, {sname}"] = d.mean() / math.sqrt(s / len(u))
    print(tab.round(4).to_string(index=False))
    tab.to_csv(OUT / "gbm_kx1h.csv", index=False)
    rows.to_parquet(OUT / "gbm_kx1h_forecasts.parquet")


if __name__ == "__main__":
    main()
