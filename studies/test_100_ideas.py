"""Test 100 ideas/models for BTC volatility forecasting (and related return predictability).

Every idea is defined as a feature set + estimator, evaluated strictly out-of-sample:
  - OLS / Ridge: rolling estimation (window 2160 h = 90 d, refit weekly, train only on
    occasions with e_j <= t, i.e. no look-ahead), same discipline as src/models.py.
  - Lasso / GBM / RandomForest / ExtraTrees: single time split (train < 2025-06-01,
    test >= 2025-06-01, 1-day purge), because refitting trees 190 times is wasteful
    and a single split is the honest walk-forward analogue for ML.
  - Raw forecasts (DVOL / ladder alone): used as-is, density spread from past errors.
  - Directional ideas (91-99): return regression, OOS R2, Newey-West t, sign-strategy
    Sharpe net of 5 bp/side.

Metrics for volatility ideas: QLIKE (Patton 2011, main point loss), MSE on log RV,
Gaussian CRPS for log RV. Diebold-Mariano (HAC Newey-West + HLN correction) vs the
HAR baseline (idea 01) on the common sample; Holm correction across all 100 tests.

Run:  python studies/test_100_ideas.py
Out:  results/ideas100/results.csv, results/ideas100/summary.md,
      results/ideas100/forecasts_*.parquet (per-idea OOS forecasts for audit)
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

OUT = config.RESULTS / "ideas100"
OUT.mkdir(parents=True, exist_ok=True)
TINY = 1e-12

# ---------------------------------------------------------------- features from minutes
def build_hourly_features():
    mn = pd.read_parquet(config.INTERIM / "minutes.parquet")
    mn = mn.sort_values("open_time").reset_index(drop=True)
    lc = np.log(mn["close"].to_numpy(float))
    r = np.diff(lc, prepend=lc[0])
    r[0] = 0.0
    absr = np.abs(r)
    # 5-minute returns
    r5 = lc - np.concatenate([np.full(5, lc[0]), lc[:-5]])
    # MedRV per minute: median of 3 consecutive |r| (vectorised via padded array)
    pad = np.concatenate([[absr[0]], absr, [absr[-1]]])
    med3 = np.median(np.stack([pad[:-2], pad[1:-1], pad[2:]]), axis=0)
    medrv_i = (np.pi / 6.0) * med3 ** 2
    bv_i = (np.pi / 2.0) * absr * np.concatenate([[0.0], absr[:-1]])
    trunc_i = np.minimum(r ** 2, 0.005 ** 2)  # winsorise |r| at 50 bp
    filled = mn["filled"].to_numpy(bool) if "filled" in mn.columns else np.zeros(len(r), bool)
    df = pd.DataFrame({
        "open_time": pd.to_datetime(mn["open_time"], utc=True),
        "r": r, "rv_i": r ** 2, "bv_i": bv_i, "medrv_i": medrv_i,
        "rv5_i": r5 ** 2, "trunc_i": trunc_i,
        "vol": mn["volume"].to_numpy(float),
        "absr": absr, "rq_i": r ** 4, "filled": filled.astype(float),
    })
    df["hour"] = df["open_time"].dt.floor("h")
    g = df.groupby("hour").agg(
        rv_h=("rv_i", "sum"), bv_h=("bv_i", "sum"), medrv_h=("medrv_i", "sum"),
        rv5_h=("rv5_i", "sum"), trunc_h=("trunc_i", "sum"),
        vol_h=("vol", "sum"), ret_h=("r", "sum"), absr_h=("absr", "sum"),
        rq_h=("rq_i", "sum"), n_min=("r", "size"), filled_share=("filled", "mean"),
    ).reset_index()
    # exchange-outage hours (forward-filled closes, like src/measure/realized.py):
    # mark variance/volume as missing instead of zero (BV/MedRV would be exactly 0)
    bad_h = g["filled_share"] > 0.2
    for c in ("rv_h", "bv_h", "medrv_h", "rv5_h", "trunc_h", "vol_h", "ret_h", "absr_h", "rq_h"):
        g.loc[bad_h, c] = np.nan
    # full hourly grid (fill gaps with NaN -> rolling will propagate honestly)
    full = pd.date_range(g["hour"].min(), g["hour"].max(), freq="h", tz="UTC")
    g = g.set_index("hour").reindex(full).rename_axis("hour").reset_index()
    g["l_rv_h"] = np.log(np.clip(g["rv_h"], TINY, None))
    g["l_bv_h"] = np.log(np.clip(g["bv_h"], TINY, None))
    g["l_medrv_h"] = np.log(np.clip(g["medrv_h"], TINY, None))
    g["l_rv5_h"] = np.log(np.clip(g["rv5_h"] / 5.0, TINY, None))
    g["l_trunc_h"] = np.log(np.clip(g["trunc_h"], TINY, None))
    g["l_vol_h"] = np.log1p(g["vol_h"].clip(lower=0))
    # lagged rolling sums ending at hour-1 (shift 1 so hour H uses [H-1, H) etc.)
    def roll_sum(col, win):
        return g[col].shift(1).rolling(win, min_periods=win).sum()
    def roll_mean(col, win):
        return g[col].shift(1).rolling(win, min_periods=win).mean()
    g["rv_1h"] = roll_sum("rv_h", 1)
    g["rv_4h"] = roll_sum("rv_h", 4)
    g["rv_24h"] = roll_sum("rv_h", 24)
    g["rv_7d"] = roll_sum("rv_h", 168)
    g["bv_1h"] = roll_sum("bv_h", 1)
    g["bv_24h"] = roll_sum("bv_h", 24)
    g["bv_7d"] = roll_sum("bv_h", 168)
    g["medrv_1h"] = roll_sum("medrv_h", 1)
    g["medrv_24h"] = roll_sum("medrv_h", 24)
    g["medrv_7d"] = roll_sum("medrv_h", 168)
    g["rv5_1h"] = roll_sum("rv5_h", 1) / 5.0
    g["rv5_24h"] = roll_sum("rv5_h", 24) / 5.0
    g["trunc_1h"] = roll_sum("trunc_h", 1)
    g["trunc_24h"] = roll_sum("trunc_h", 24)
    g["vol_1h"] = roll_sum("vol_h", 1)
    g["vol_24h"] = roll_sum("vol_h", 24)
    g["vol_7d"] = roll_sum("vol_h", 168)
    g["ret_1h"] = roll_sum("ret_h", 1)
    g["ret_24h"] = roll_sum("ret_h", 24)
    g["ret_7d"] = roll_sum("ret_h", 168)
    g["absr_24h"] = roll_sum("absr_h", 24)
    g["rq_24h"] = roll_sum("rq_h", 24)
    g["l_rv_5m_proxy"] = np.nan  # filled below from minute cumsum for precision
    g["l_rv_15m_proxy"] = np.nan
    # EWMA RV (halflife 6h) lagged — short-component alternative to HAR daily
    g["ewma_rv"] = g["rv_h"].shift(1).ewm(halflife=6, adjust=False).mean()
    # vol-of-vol: std of daily log RV over past 30d (daily RV from hourly sums)
    daily = g.set_index("hour")["rv_h"].resample("D").sum()
    l_daily = np.log(np.clip(daily, TINY, None))
    vov = l_daily.shift(1).rolling(30, min_periods=30).std()
    g["vov_30d"] = g["hour"].dt.floor("D").map(vov)
    # volume volatility: std of hourly log volume over 24h
    g["vol_std_24h"] = g["l_vol_h"].shift(1).rolling(24, min_periods=24).std()
    # short lags from minute cumsum (exact 5m/15m before t, not hour-aligned)
    cs = pd.Series(np.cumsum(df["rv_i"].to_numpy(float)), index=df["open_time"])
    cs5 = pd.Series(np.cumsum(df["rv5_i"].to_numpy(float)), index=df["open_time"])
    g["cs_rv"] = cs.reindex(g["hour"], method="ffill").values
    g["cs_rv5"] = cs5.reindex(g["hour"], method="ffill").values
    # 5m/15m RV ending at hour H (i.e. last 5/15 min of previous hour) via cumsum diff
    # approximate with hourly rv scaled: use minute-level lookup at H-5m, H-15m
    idx = cs.index
    def cs_at(ts):
        return cs.reindex(pd.DatetimeIndex(ts), method="ffill").values
    g["l_rv_5m"] = np.log(np.clip(cs_at(g["hour"] - pd.Timedelta(minutes=1)) - cs_at(g["hour"] - pd.Timedelta(minutes=6)), TINY, None))
    g["l_rv_15m"] = np.log(np.clip(cs_at(g["hour"] - pd.Timedelta(minutes=1)) - cs_at(g["hour"] - pd.Timedelta(minutes=16)), TINY, None))
    g["l_rv_1h_h"] = np.log(np.clip(g["rv_1h"], TINY, None))
    g["l_rv_4h_h"] = np.log(np.clip(g["rv_4h"], TINY, None))
    g["l_rv_24h_h"] = np.log(np.clip(g["rv_24h"] / 24.0, TINY, None))
    g["l_rv_7d_h"] = np.log(np.clip(g["rv_7d"] / 168.0, TINY, None))
    g["l_bv_1h"] = np.log(np.clip(g["bv_1h"], TINY, None))
    g["l_bv_24h"] = np.log(np.clip(g["bv_24h"] / 24.0, TINY, None))
    g["l_bv_7d"] = np.log(np.clip(g["bv_7d"] / 168.0, TINY, None))
    g["l_medrv_1h"] = np.log(np.clip(g["medrv_1h"], TINY, None))
    g["l_medrv_24h"] = np.log(np.clip(g["medrv_24h"] / 24.0, TINY, None))
    g["l_medrv_7d"] = np.log(np.clip(g["medrv_7d"] / 168.0, TINY, None))
    g["l_rv5_1h"] = np.log(np.clip(g["rv5_1h"], TINY, None))
    g["l_rv5_24h"] = np.log(np.clip(g["rv5_24h"] / 24.0, TINY, None))
    g["l_trunc_1h"] = np.log(np.clip(g["trunc_1h"], TINY, None))
    g["l_trunc_24h"] = np.log(np.clip(g["trunc_24h"] / 24.0, TINY, None))
    g["l_vol_1h"] = np.log1p(g["vol_1h"].clip(lower=0))
    g["l_vol_24h"] = np.log1p(g["vol_24h"].clip(lower=0))
    g["l_vol_7d"] = np.log1p(g["vol_7d"].clip(lower=0))
    g["jump_24h"] = np.clip((g["rv_24h"] - g["bv_24h"]) / g["rv_24h"], 0, 1)
    g["jump_1h"] = np.clip((g["rv_1h"] - g["bv_h"].shift(1)) / g["rv_1h"], 0, 1)
    g["l_ewma_rv"] = np.log(np.clip(g["ewma_rv"], TINY, None))
    g["rq_level_24h"] = np.log(np.clip(g["rq_24h"] * 24 / 3.0, TINY, None))
    return g


def build_panel():
    p = pd.read_parquet(config.INTERIM / "panel_kx1h.parquet").sort_values("t").reset_index(drop=True)
    p["t"] = pd.to_datetime(p["t"], utc=True)
    p["e"] = pd.to_datetime(p["e"], utc=True)
    g = build_hourly_features()
    p["hour"] = p["t"].dt.floor("h")
    keep = ["hour", "l_rv_5m", "l_rv_15m", "l_rv_1h_h", "l_rv_4h_h", "l_rv_24h_h", "l_rv_7d_h",
            "l_bv_1h", "l_bv_24h", "l_bv_7d", "l_medrv_1h", "l_medrv_24h", "l_medrv_7d",
            "l_rv5_1h", "l_rv5_24h", "l_trunc_1h", "l_trunc_24h",
            "l_vol_1h", "l_vol_24h", "l_vol_7d", "jump_24h", "jump_1h", "ret_1h", "ret_24h",
            "ret_7d", "absr_24h", "l_ewma_rv", "vov_30d", "vol_std_24h", "rq_level_24h"]
    p = p.merge(g[keep], on="hour", how="left")
    # funding: last rate before t (merge_asof needs identical datetime dtype)
    try:
        fd = pd.read_csv(config.RAW / "funding" / "btc_funding.csv.gz")
        fd["ts"] = pd.to_datetime(fd["ts"], unit="ms", utc=True).astype("datetime64[ns, UTC]")
        fd = fd[fd["venue"] == "binance"].sort_values("ts")[["ts", "rate"]].rename(columns={"rate": "funding"})
        p["t"] = p["t"].astype("datetime64[ns, UTC]")
        p = pd.merge_asof(p.sort_values("t"), fd, left_on="t", right_on="ts", direction="backward")
        p = p.drop(columns=["ts"], errors="ignore")
    except Exception as e:
        print("funding load failed:", e)
        p["funding"] = np.nan
    # DVOL changes (panel dvol is hourly)
    p["dvol_chg_1h"] = p["dvol"].diff(1)
    p["dvol_chg_24h"] = p["dvol"].diff(24)
    p["dvol_z"] = (p["dvol"] - p["dvol"].rolling(720, min_periods=720).mean()) / p["dvol"].rolling(720, min_periods=720).std()
    # calendar / seasonality
    p["hour_utc"] = p["t"].dt.hour
    p["dow"] = p["t"].dt.dayofweek
    for k in (1, 2, 3, 6):
        p[f"sin_h{k}"] = np.sin(2 * np.pi * k * p["hour_utc"] / 24.0)
        p[f"cos_h{k}"] = np.cos(2 * np.pi * k * p["hour_utc"] / 24.0)
    for k in (1, 2):
        p[f"sin_hw{k}"] = np.sin(2 * np.pi * k * (p["dow"] * 24 + p["hour_utc"]) / 168.0)
        p[f"cos_hw{k}"] = np.cos(2 * np.pi * k * (p["dow"] * 24 + p["hour_utc"]) / 168.0)
    p["us_session"] = ((p["hour_utc"] >= 13) & (p["hour_utc"] <= 20)).astype(float)
    p["asia_session"] = ((p["hour_utc"] >= 0) & (p["hour_utc"] <= 7)).astype(float)
    p["eu_session"] = ((p["hour_utc"] >= 7) & (p["hour_utc"] <= 13)).astype(float)
    p["funding_hour"] = p["hour_utc"].isin([0, 8, 16]).astype(float)
    p["hour0"] = (p["hour_utc"] == 0).astype(float)
    for d in range(7):
        p[f"dow{d}"] = (p["dow"] == d).astype(float)
    # interactions / transforms
    p["l_rv_d_sq"] = p["l_rv_d"] ** 2
    p["l_rv_d_x_jump"] = p["l_rv_d"] * p["jump_share"]
    p["l_rv_d_x_weekend"] = p["l_rv_d"] * p["weekend"]
    p["l_rv_d_x_vol"] = p["l_rv_d"] * p["l_vol_24h"]
    p["l_rv_d_x_dvol"] = p["l_rv_d"] * p["l_dvol_var"]
    p["l_rv_d_x_vov"] = p["l_rv_d"] * p["vov_30d"]
    p["lev_24h"] = p["ret_24h"] * p["l_rv_d"]  # rescaled: return x log-RV (O(0.1), numerically stable)
    p["vol_chg"] = p["l_vol_1h"] - p["l_vol_24h"]
    p["vol_x_rv"] = p["l_vol_24h"] * p["l_rv_d"]
    p["amihud"] = np.abs(p["ret_1h"]) / (p["l_vol_1h"] + 1.0)
    p["vol_imb"] = p["ret_1h"] * p["l_vol_1h"]
    p["rv_level_d"] = np.exp(p["l_rv_d"])
    p["r_next"] = np.log(p["S_e"] / p["S_t"])
    p["abs_r_next"] = np.abs(p["r_next"])
    p["ladder_drift"] = p["mkt_m"] - np.log(p["S_t"])
    p["vrp_proxy"] = p["l_dvol_var"] - p["l_rv_d"]
    p["funding_abs"] = np.abs(p["funding"])
    p["funding_x_vol"] = p["funding"] * p["l_vol_24h"]
    p["mkt_s_sq"] = p["mkt_s"] ** 2
    p["mkt_nu_inv"] = 1.0 / p["mkt_nu"]
    return p


# ---------------------------------------------------------------- estimators
def rolling_ols(p, feats, window=2160, refit=168, min_train=720, ridge=0.0):
    n = len(p)
    X = p[feats].to_numpy(float)
    y = p["y"].to_numpy(float)
    okx = np.isfinite(X).all(axis=1)
    oky = np.isfinite(y)
    e = p["e"].to_numpy(dtype="datetime64[ns]")
    m_out = np.full(n, np.nan)
    s_out = np.full(n, np.nan)
    rows = np.flatnonzero(okx & oky)
    # chunk by refit on rows index
    for c0 in range(0, len(rows), refit):
        chunk = rows[c0:c0 + refit]
        t0 = p["t"].iloc[chunk[0]]
        t0ns = np.datetime64(t0.tz_localize(None) if t0.tzinfo else t0)
        k = int(np.searchsorted(e, t0ns, side="right"))
        tr = np.flatnonzero(okx[:k] & oky[:k])[-window:]
        if len(tr) < min_train:
            continue
        A = np.column_stack([np.ones(len(tr)), X[tr]])
        if ridge > 0:
            # ridge via augmented rows (no penalty on intercept)
            P = np.eye(A.shape[1]) * math.sqrt(ridge)
            P[0, 0] = 0.0
            A_aug = np.vstack([A, P])
            y_aug = np.concatenate([y[tr], np.zeros(A.shape[1])])
            beta, *_ = np.linalg.lstsq(A_aug, y_aug, rcond=None)
        else:
            beta, *_ = np.linalg.lstsq(A, y[tr], rcond=None)
        res = y[tr] - A @ beta
        sig = float(np.sqrt(res @ res / max(len(tr) - A.shape[1], 1)))
        Ac = np.column_stack([np.ones(len(chunk)), X[chunk]])
        m_out[chunk] = Ac @ beta
        s_out[chunk] = sig
    return m_out, s_out


def rolling_raw(p, x, window=2160, refit=168, min_train=336):
    y = p["y"].to_numpy(float)
    ok = np.isfinite(x) & np.isfinite(y)
    e = p["e"].to_numpy(dtype="datetime64[ns]")
    s_out = np.full(len(p), np.nan)
    err2 = np.where(ok, (y - x) ** 2, np.nan)
    rows = np.flatnonzero(ok)
    for c0 in range(0, len(rows), refit):
        chunk = rows[c0:c0 + refit]
        t0 = p["t"].iloc[chunk[0]]
        t0ns = np.datetime64(t0.tz_localize(None) if t0.tzinfo else t0)
        k = int(np.searchsorted(e, t0ns, side="right"))
        past = err2[:k]
        past = past[np.isfinite(past)][-window:]
        if len(past) < min_train:
            continue
        s_out[chunk] = float(np.sqrt(np.mean(past)))
    return x.copy(), s_out


def split_ml(p, feats, kind, target="y"):
    """Single time split: train < 2025-06-01, test >= 2025-06-01 (+1d purge)."""
    from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor, ExtraTreesRegressor
    from sklearn.linear_model import Lasso, ElasticNet, Ridge, LinearRegression
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    cut = pd.Timestamp("2025-06-01", tz="UTC")
    purge = pd.Timestamp("2025-06-02", tz="UTC")
    X = p[feats].to_numpy(float)
    y = p[target].to_numpy(float)
    finite = np.isfinite(X).all(axis=1) & np.isfinite(y)
    tr = (p["t"] < cut).to_numpy() & finite
    te = (p["t"] >= purge).to_numpy() & finite
    if tr.sum() < 500 or te.sum() < 500:
        # ladder features exist only from Dec 2025: fall back to chronological
        # 60/40 split on the rows where features exist (still strictly OOS in time)
        idx = np.flatnonzero(finite)
        if len(idx) < 1000:
            return np.full(len(p), np.nan), np.full(len(p), np.nan)
        cut_i = idx[int(0.6 * len(idx))]
        tr = finite & (np.arange(len(p)) <= cut_i)
        te = finite & (np.arange(len(p)) > cut_i + 24)  # 1-day purge in rows
        if tr.sum() < 200 or te.sum() < 200:
            return np.full(len(p), np.nan), np.full(len(p), np.nan)
    if kind == "lasso":
        est = make_pipeline(SimpleImputer(strategy="median"), Lasso(alpha=0.01, max_iter=5000))
    elif kind == "elastic":
        est = make_pipeline(SimpleImputer(strategy="median"), ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=5000))
    elif kind == "ridge_ml":
        # standardised OLS for return regressions: features live on wildly different
        # scales (funding ~1e-4, dummies 0/1); unstandardised Ridge would shrink
        # small-scale features to zero and give identical constant predictions.
        est = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LinearRegression())
    elif kind == "gbm":
        est = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_depth=4,
                                            min_samples_leaf=50, l2_regularization=1.0, random_state=0)
    elif kind == "rf":
        est = RandomForestRegressor(n_estimators=300, max_depth=8, min_samples_leaf=50,
                                    n_jobs=-1, random_state=0)
    elif kind == "et":
        est = ExtraTreesRegressor(n_estimators=300, max_depth=None, min_samples_leaf=50,
                                  n_jobs=-1, random_state=0)
    else:
        raise ValueError(kind)
    est.fit(X[tr], y[tr])
    pred_tr = est.predict(X[tr])
    sig = float(np.sqrt(np.mean((y[tr] - pred_tr) ** 2)))
    m_out = np.full(len(p), np.nan)
    s_out = np.full(len(p), np.nan)
    m_out[te] = est.predict(X[te])
    s_out[te] = sig
    return m_out, s_out


def crps_normal(y, m, s):
    z = (y - m) / s
    return s * (z * (2 * stats.norm.cdf(z) - 1) + 2 * stats.norm.pdf(z) - 1 / math.sqrt(math.pi))


def nw_lag(n):
    return int(max(1, math.floor(4 * (n / 100) ** (2 / 9))))


def dm_test(loss_a, loss_b):
    d = np.asarray(loss_a) - np.asarray(loss_b)
    d = d[np.isfinite(d)]
    n = len(d)
    if n < 30:
        return np.nan, np.nan, n
    u = d - d.mean()
    L = nw_lag(n)
    v = (u @ u) / n
    for j in range(1, L + 1):
        v += 2 * (1 - j / (L + 1)) * (u[j:] @ u[:-j]) / n
    se = math.sqrt(max(v, 1e-300) / n)
    if se == 0:
        return np.nan, np.nan, n
    dm = d.mean() / se
    corr = math.sqrt(max((n + 1 - 2 + 0) / n, 1e-9))
    dm_hln = dm * corr
    p = 2 * stats.t.sf(abs(dm_hln), df=n - 1)
    return dm_hln, p, n


def holm(pvals):
    p = np.asarray(pvals, float)
    m = len(p)
    order = np.argsort(p)
    adj = np.full(m, np.nan)
    running = 0.0
    for rank, idx in enumerate(order):
        if not np.isfinite(p[idx]):
            continue
        val = (m - rank) * p[idx]
        running = max(running, val)
        adj[idx] = min(running, 1.0)
    return adj


# ---------------------------------------------------------------- 100 ideas
HAR = ["l_rv_d", "l_rv_w", "l_rv_m", "l_F"]
HARX = HAR + ["l_rs_pos", "l_rs_neg", "jump_share", "harq", "macro", "weekend"]
FOURIER6 = ["sin_h1", "cos_h1", "sin_h2", "cos_h2", "sin_h3", "cos_h3"]
FOURIER_HW = ["sin_hw1", "cos_hw1", "sin_hw2", "cos_hw2"]
DOW = [f"dow{d}" for d in range(7)]

IDEAS = [
    # id, name, category, features, estimator, note
    (1, "HAR baseline", "HAR core", HAR, "ols", "Corsi 2009 + seasonal factor"),
    (2, "HAR + semivariances", "HAR core", HAR + ["l_rs_pos", "l_rs_neg"], "ols", "Patton-Sheppard good/bad vol"),
    (3, "HAR + jump share", "HAR core", HAR + ["jump_share"], "ols", "BNS jump share (4h)"),
    (4, "HAR + HARQ", "HAR core", HAR + ["harq"], "ols", "Bollerslev-Patton-Quaedvlieg 2016"),
    (5, "HAR-X (semivar+jump+HARQ+dummies)", "HAR core", HARX, "ols", "project harx"),
    (6, "HAR daily only", "HAR core", ["l_rv_d", "l_F"], "ols", "ablation"),
    (7, "HAR weekly only", "HAR core", ["l_rv_w", "l_F"], "ols", "ablation"),
    (8, "Naive log RV only", "HAR core", ["l_rv_naive"], "ols", "persistence benchmark"),
    (9, "HAR + naive", "HAR core", HAR + ["l_rv_naive"], "ols", "short + long persistence"),
    (10, "HAR + squared term", "HAR core", HAR + ["l_rv_d_sq"], "ols", "nonlinearity in level"),
    (11, "HAR + RV x jump interaction", "HAR core", HAR + ["l_rv_d_x_jump"], "ols", "state-dependent persistence"),
    (12, "HAR + RV x weekend interaction", "HAR core", HAR + ["l_rv_d_x_weekend"], "ols", "weekend persistence differs"),
    (13, "HAR + macro dummy", "HAR core", HAR + ["macro"], "ols", "US macro releases"),
    (14, "HAR + weekend dummy", "HAR core", HAR + ["weekend"], "ols", "weekend level shift"),
    (15, "HAR + dummies + semivar", "HAR core", HAR + ["macro", "weekend", "l_rs_pos", "l_rs_neg"], "ols", "calendar + asymmetry"),
    (16, "HAR-BV (bipower instead of RV)", "Jump-robust", ["l_bv_1h", "l_bv_24h", "l_bv_7d", "l_F"], "ols", "Barndorff-Nielsen-Shephard BV"),
    (17, "HAR-MedRV", "Jump-robust", ["l_medrv_1h", "l_medrv_24h", "l_medrv_7d", "l_F"], "ols", "MedRV (Andersen-Bollerslev-Diebold)"),
    (18, "HAR + BV + RV", "Jump-robust", HAR + ["l_bv_24h", "l_bv_7d"], "ols", "continuous + total"),
    (19, "HAR + MedRV + RV", "Jump-robust", HAR + ["l_medrv_24h", "l_medrv_7d"], "ols", "robust + total"),
    (20, "HAR-BV naive", "Jump-robust", ["l_bv_1h"], "ols", "jump-robust persistence"),
    (21, "HAR + jump share 24h (BV-based)", "Jump-robust", HAR + ["jump_24h"], "ols", "daily jump share"),
    (22, "HAR + jump share 1h", "Jump-robust", HAR + ["jump_1h"], "ols", "hourly jump share"),
    (23, "HAR-truncated RV (winsorised 50bp)", "Jump-robust", ["l_trunc_1h", "l_trunc_24h", "l_rv_w", "l_rv_m", "l_F"], "ols", "outlier-robust RV"),
    (24, "HAR-RV5 (5-min returns)", "Jump-robust", ["l_rv5_1h", "l_rv5_24h", "l_rv_w", "l_rv_m", "l_F"], "ols", "microstructure-robust"),
    (25, "HAR + RV5", "Jump-robust", HAR + ["l_rv5_24h"], "ols", "1m + 5m"),
    (26, "HAR + quarticity level", "Jump-robust", HAR + ["rq_level_24h"], "ols", "measurement noise (HARQ level)"),
    (27, "HAR + HARQ level (not scaled)", "Jump-robust", HAR + ["rq_level_24h", "harq"], "ols", "HARQ variants"),
    (28, "HAR + BV x jump interaction", "Jump-robust", HAR + ["l_bv_24h", "jump_24h"], "ols", "jump state"),
    (29, "HAR + semivariance asymmetry", "Jump-robust", HAR + ["l_rs_pos", "l_rs_neg", "jump_24h"], "ols", "good/bad + jumps"),
    (30, "HAR + leverage (r x RV)", "Jump-robust", HAR + ["lev_24h"], "ols", "leverage effect"),
    (31, "HAR + log volume 24h", "Volume", HAR + ["l_vol_24h"], "ols", "activity predicts vol (Andersen et al.)"),
    (32, "HAR + log volume 1h", "Volume", HAR + ["l_vol_1h"], "ols", "short activity"),
    (33, "HAR + volume change", "Volume", HAR + ["vol_chg"], "ols", "volume surprise"),
    (34, "HAR + volume x RV", "Volume", HAR + ["l_rv_d_x_vol"], "ols", "volume-conditioned persistence"),
    (35, "HAR + volume imbalance proxy", "Volume", HAR + ["vol_imb"], "ols", "signed volume"),
    (36, "HAR + volume volatility", "Volume", HAR + ["vol_std_24h"], "ols", "activity uncertainty"),
    (37, "HAR + volume HAR (d/w)", "Volume", HAR + ["l_vol_24h", "l_vol_7d"], "ols", "volume term structure"),
    (38, "HAR + Amihud illiquidity", "Volume", HAR + ["amihud"], "ols", "price impact"),
    (39, "HAR + volume + funding", "Volume", HAR + ["l_vol_24h", "funding"], "ols", "activity + leverage demand"),
    (40, "HAR + volume + jump", "Volume", HAR + ["l_vol_24h", "jump_24h"], "ols", "activity + jumps"),
    (41, "HAR + hour dummies (24)", "Seasonality", HAR + [f"hour_{h}" for h in range(24)], "ols", "full intraday profile"),
    (42, "HAR + Fourier K=3 (hour)", "Seasonality", HAR + FOURIER6, "ols", "smooth intraday"),
    (43, "HAR + Fourier K=6 (hour-of-week)", "Seasonality", HAR + FOURIER6 + FOURIER_HW, "ols", "weekly profile"),
    (44, "HAR + US session dummy", "Seasonality", HAR + ["us_session"], "ols", "US open effect"),
    (45, "HAR + session dummies (Asia/EU/US)", "Seasonality", HAR + ["asia_session", "eu_session", "us_session"], "ols", "global sessions"),
    (46, "HAR + funding-hour dummy", "Seasonality", HAR + ["funding_hour"], "ols", "00/08/16 UTC funding"),
    (47, "HAR + hour-0 dummy (:00 spike)", "Seasonality", HAR + ["hour0"], "ols", "minute-00 spike (R13)"),
    (48, "HAR + day-of-week dummies", "Seasonality", HAR + DOW, "ols", "weekday profile"),
    (49, "HAR + weekend x hour (Fourier)", "Seasonality", HAR + FOURIER6 + ["weekend"], "ols", "weekend intraday differs"),
    (50, "HAR + macro x hour", "Seasonality", HAR + ["macro", "us_session"], "ols", "macro timing"),
    (51, "HAR + DVOL level", "Deribit/IV", HAR + ["l_dvol_var"], "ols", "options info"),
    (52, "HAR + DVOL change 1h", "Deribit/IV", HAR + ["dvol_chg_1h"], "ols", "IV momentum"),
    (53, "HAR + DVOL change 24h", "Deribit/IV", HAR + ["dvol_chg_24h"], "ols", "IV trend"),
    (54, "HAR + DVOL x RV", "Deribit/IV", HAR + ["l_rv_d_x_dvol"], "ols", "IV-conditioned persistence"),
    (55, "DVOL alone (raw)", "Deribit/IV", ["l_dvol_var"], "raw", "Deribit as-is"),
    (56, "HAR + DVOL + semivar", "Deribit/IV", HAR + ["l_dvol_var", "l_rs_pos", "l_rs_neg"], "ols", "options + asymmetry"),
    (57, "HAR + DVOL z-score", "Deribit/IV", HAR + ["dvol_z"], "ols", "IV relative to 30d"),
    (58, "HAR + DVOL + volume", "Deribit/IV", HAR + ["l_dvol_var", "l_vol_24h"], "ols", "options + activity"),
    (59, "HAR + DVOL + Fourier", "Deribit/IV", HAR + ["l_dvol_var"] + FOURIER6, "ols", "options + seasonality"),
    (60, "HAR + DVOL + jump + volume", "Deribit/IV", HAR + ["l_dvol_var", "jump_24h", "l_vol_24h"], "ols", "kitchen sink (ex-ladder)"),
    (61, "Ladder MFIV alone (raw)", "Ladder/market", ["l_mkt_mfiv"], "raw", "BJN model-free, as-is"),
    (62, "HAR + ladder MFIV", "Ladder/market", HAR + ["l_mkt_mfiv"], "ols", "market info test"),
    (63, "HAR-X + ladder MFIV", "Ladder/market", HARX + ["l_mkt_mfiv"], "ols", "harx_m analogue"),
    (64, "HAR-X + DVOL + ladder (harx_all)", "Ladder/market", HARX + ["l_dvol_var", "l_mkt_mfiv"], "ols", "encompassing"),
    (65, "Ladder lognormal var alone (raw)", "Ladder/market", ["l_mkt_var_c"], "raw", "lognormal fit"),
    (66, "Ladder Student-t var alone (raw)", "Ladder/market", ["l_mkt_var_t"], "raw", "fat-tail fit"),
    (67, "HAR + ladder + skew (nu)", "Ladder/market", HAR + ["l_mkt_mfiv", "mkt_nu"], "ols", "tail shape info"),
    (68, "HAR + ladder + fit RMSE", "Ladder/market", HAR + ["l_mkt_mfiv", "mkt_rmse"], "ols", "ladder quality"),
    (69, "HAR + ladder vol (mkt_s)", "Ladder/market", HAR + ["l_mkt_mfiv", "mkt_s"], "ols", "ladder level"),
    (70, "Ladder + DVOL average (raw combo)", "Ladder/market", ["l_mkt_mfiv", "l_dvol_var"], "combo_raw", "two markets averaged"),
    (71, "Ridge HAR-X (alpha=1)", "ML/nonlinear", HARX, "ridge1", "shrinkage"),
    (72, "Lasso HAR-X", "ML/nonlinear", HARX, "lasso", "sparsity"),
    (73, "ElasticNet HAR-X", "ML/nonlinear", HARX, "elastic", "sparse + grouped"),
    (74, "GBM HAR-X", "ML/nonlinear", HARX, "gbm", "nonlinear HAR"),
    (75, "RandomForest HAR-X", "ML/nonlinear", HARX, "rf", "trees"),
    (76, "GBM HAR-X+DVOL+ladder", "ML/nonlinear", HARX + ["l_dvol_var", "l_mkt_mfiv"], "gbm", "nonlinear encompassing"),
    (77, "GBM all features", "ML/nonlinear", HARX + ["l_dvol_var", "l_mkt_mfiv", "l_vol_24h", "jump_24h", "l_bv_24h", "l_medrv_24h"] + FOURIER6, "gbm", "full feature set"),
    (78, "ExtraTrees HAR-X", "ML/nonlinear", HARX, "et", "ensembles"),
    (79, "GBM HAR + short lags (5m/15m)", "ML/nonlinear", HAR + ["l_rv_5m", "l_rv_15m"], "gbm", "short lags + trees"),
    (80, "HAR log + level (both)", "ML/nonlinear", HAR + ["rv_level_d"], "ols", "log + level"),
    (81, "Combo HAR + DVOL-MZ + ladder-MZ", "Combinations", HAR + ["l_dvol_var", "l_mkt_mfiv"], "ols", "equal-weight analogue via OLS"),
    (82, "HAR + short lags 5m/15m", "Combinations", HAR + ["l_rv_5m", "l_rv_15m"], "ols", "short_lags replication"),
    (83, "HAR-X + short lags 5m/15m", "Combinations", HARX + ["l_rv_5m", "l_rv_15m"], "ols", "short lags on harx"),
    (84, "HAR + EWMA RV", "Combinations", ["l_ewma_rv", "l_rv_w", "l_rv_m", "l_F"], "ols", "EWMA vs HAR daily"),
    (85, "HAR + vol-of-vol", "Combinations", HAR + ["vov_30d"], "ols", "uncertainty (Corsi et al.)"),
    (86, "HAR + RV x vol-of-vol", "Combinations", HAR + ["l_rv_d_x_vov"], "ols", "VoV-conditioned persistence"),
    (87, "HAR + funding level", "Combinations", HAR + ["funding"], "ols", "leverage demand -> vol"),
    (88, "HAR + |funding|", "Combinations", HAR + ["funding_abs"], "ols", "funding stress"),
    (89, "HAR + VRP proxy (DVOL-RV)", "Combinations", HAR + ["vrp_proxy"], "ols", "variance premium"),
    (90, "HAR + ladder drift", "Combinations", HAR + ["ladder_drift"], "ols", "market directional info -> vol"),
    (91, "Return: overnight/US session drift", "Directional", ["us_session"], "return", "R2-style session return"),
    (92, "Return: funding -> next return", "Directional", ["funding"], "return", "funding predicts return?"),
    (93, "Return: DVOL change -> next return", "Directional", ["dvol_chg_1h"], "return", "R12 IV leads returns"),
    (94, "Return: volume -> next return", "Directional", ["l_vol_24h"], "return", "volume predicts return?"),
    (95, "Return: ladder drift -> next return", "Directional", ["ladder_drift"], "return", "market drift info"),
    (96, "Return: VRP -> next return", "Directional", ["vrp_proxy"], "return", "VRP predicts return (BTZ)"),
    (97, "Return: 24h momentum -> next return", "Directional", ["ret_24h"], "return", "momentum"),
    (98, "Return: 1h reversal -> next return", "Directional", ["ret_1h"], "return", "reversal"),
    (99, "Return: funding x volume squeeze", "Directional", ["funding_x_vol"], "return", "squeeze proxy"),
    (100, "Abs return: HAR predicts |r|", "Directional", HAR, "absret", "vol -> absolute return"),
]


def main():
    print("building panel + hourly features ...", flush=True)
    p = build_panel()
    # hour dummies for idea 41
    for h in range(24):
        p[f"hour_{h}"] = (p["hour_utc"] == h).astype(float)
    print(f"panel: {p.shape}, t {p['t'].min()} .. {p['t'].max()}", flush=True)
    p.to_parquet(OUT / "panel_with_features.parquet", index=False)

    y = p["y"].to_numpy(float)
    rv = np.exp(y)
    results = []
    forecasts = {}
    baseline_qlike = None
    baseline_idx = None

    for (iid, name, cat, feats, est, note) in IDEAS:
        print(f"[{iid:3d}/100] {name} ({est}) ...", flush=True)
        # check features exist
        missing = [f for f in feats if f not in p.columns]
        if missing:
            print(f"  SKIP missing {missing}")
            results.append({"id": iid, "name": name, "category": cat, "estimator": est,
                            "note": note, "status": "skipped_missing_features", "missing": ",".join(missing)})
            continue
        try:
            if est == "ols":
                m, s = rolling_ols(p, feats)
            elif est == "ridge1":
                m, s = rolling_ols(p, feats, ridge=1.0)
            elif est == "raw":
                x = p[feats[0]].to_numpy(float)
                m, s = rolling_raw(p, x)
            elif est == "combo_raw":
                x = np.nanmean(np.column_stack([p[feats[0]].to_numpy(float), p[feats[1]].to_numpy(float)]), axis=1)
                m, s = rolling_raw(p, x)
            elif est in ("lasso", "elastic", "gbm", "rf", "et", "ridge_ml"):
                m, s = split_ml(p, feats, est)
            elif est == "return":
                m, s = split_ml(p, feats, "ridge_ml", target="r_next")
                # evaluate as return regression
                ok = np.isfinite(m) & np.isfinite(p["r_next"].to_numpy(float))
                if ok.sum() < 100:
                    raise ValueError("too few return obs")
                yt = p["r_next"].to_numpy(float)[ok]
                yp = m[ok]
                # OOS R2 vs zero and vs mean
                ss_res = np.sum((yt - yp) ** 2)
                ss_tot = np.sum((yt - yt.mean()) ** 2)
                r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
                corr = np.corrcoef(yt, yp)[0, 1] if np.std(yp) > 0 else np.nan
                # sign strategy Sharpe (gross, 5bp cost per flip approx via turnover)
                pos = np.sign(yp)
                gross = pos * yt
                # turnover cost: 10bp per round trip when position flips (approx)
                flips = np.sum(np.abs(np.diff(pos)) > 0)
                net = gross - 0.001 * flips / len(gross)
                sharpe = float(np.mean(net) / np.std(net) * math.sqrt(24 * 365)) if np.std(net) > 0 else np.nan
                # NW t on mean net
                u = net - net.mean()
                L = nw_lag(len(u))
                v = (u @ u) / len(u)
                for j in range(1, L + 1):
                    v += 2 * (1 - j / (L + 1)) * (u[j:] @ u[:-j]) / len(u)
                tstat = float(net.mean() / math.sqrt(max(v, 1e-300) / len(u))) if len(u) > 0 else np.nan
                pval = float(2 * stats.t.sf(abs(tstat), df=len(u) - 1)) if np.isfinite(tstat) else np.nan
                results.append({"id": iid, "name": name, "category": cat, "estimator": est, "note": note,
                                "status": "ok", "n_test": int(ok.sum()), "oos_r2": r2, "corr": corr,
                                "sharpe_net_ann": sharpe, "t_nw": tstat, "p_value": pval,
                                "mean_net_per_trade": float(np.mean(net))})
                forecasts[iid] = pd.DataFrame({"t": p["t"][ok].values, "y": yt, "pred": yp})
                continue
            elif est == "absret":
                m, s = rolling_ols(p, feats)  # placeholder; re-evaluate below on abs target
                # redo with abs target via split ridge for speed + rolling OLS on abs
                # rolling OLS on abs_r_next: temporarily swap y
                p_tmp = p.copy()
                # manual rolling on abs target
                X = p[feats].to_numpy(float)
                yt_all = p["abs_r_next"].to_numpy(float)
                okx = np.isfinite(X).all(axis=1) & np.isfinite(yt_all)
                e = p["e"].to_numpy(dtype="datetime64[ns]")
                m_abs = np.full(len(p), np.nan)
                rows = np.flatnonzero(okx)
                for c0 in range(0, len(rows), 168):
                    chunk = rows[c0:c0 + 168]
                    t0 = p["t"].iloc[chunk[0]]
                    t0ns = np.datetime64(t0.tz_localize(None) if t0.tzinfo else t0)
                    k = int(np.searchsorted(e, t0ns, side="right"))
                    tr = np.flatnonzero(okx[:k])[-2160:]
                    if len(tr) < 720:
                        continue
                    A = np.column_stack([np.ones(len(tr)), X[tr]])
                    beta, *_ = np.linalg.lstsq(A, yt_all[tr], rcond=None)
                    Ac = np.column_stack([np.ones(len(chunk)), X[chunk]])
                    m_abs[chunk] = Ac @ beta
                ok = np.isfinite(m_abs) & np.isfinite(yt_all)
                mse = float(np.mean((yt_all[ok] - m_abs[ok]) ** 2))
                # baseline: mean abs return (expanding)
                base = np.full(len(p), np.nan)
                for c0 in range(0, len(rows), 168):
                    chunk = rows[c0:c0 + 168]
                    t0 = p["t"].iloc[chunk[0]]
                    t0ns = np.datetime64(t0.tz_localize(None) if t0.tzinfo else t0)
                    k = int(np.searchsorted(e, t0ns, side="right"))
                    tr = np.flatnonzero(okx[:k])[-2160:]
                    if len(tr) < 720:
                        continue
                    base[chunk] = float(np.mean(yt_all[tr]))
                okb = np.isfinite(base) & np.isfinite(yt_all)
                mse_base = float(np.mean((yt_all[okb] - base[okb]) ** 2))
                # DM on squared errors vs baseline
                dm, pval, n = dm_test((yt_all - m_abs) ** 2, (yt_all - base) ** 2)
                results.append({"id": iid, "name": name, "category": cat, "estimator": est, "note": note,
                                "status": "ok", "n_test": int(ok.sum()), "mse_abs": mse, "mse_abs_base": mse_base,
                                "dm_vs_base": dm, "p_value": pval})
                forecasts[iid] = pd.DataFrame({"t": p["t"][ok].values, "y": yt_all[ok], "pred": m_abs[ok]})
                continue
            else:
                raise ValueError(est)
        except Exception as e:
            print(f"  ERROR {e!r}")
            results.append({"id": iid, "name": name, "category": cat, "estimator": est, "note": note,
                            "status": f"error: {e!r}"})
            continue

        ok = np.isfinite(m) & np.isfinite(s) & np.isfinite(y) & (s > 0)
        if ok.sum() < 100:
            results.append({"id": iid, "name": name, "category": cat, "estimator": est, "note": note,
                            "status": "too_few_obs", "n_test": int(ok.sum())})
            continue
        lvl = np.exp(m + s ** 2 / 2.0)
        qlike = rv / lvl - np.log(rv / lvl) - 1
        mse_log = (y - m) ** 2
        crps = crps_normal(y, m, s)
        row = {"id": iid, "name": name, "category": cat, "estimator": est, "note": note,
               "status": "ok", "n_test": int(ok.sum()),
               "qlike": float(np.mean(qlike[ok])), "mse_log": float(np.mean(mse_log[ok])),
               "crps": float(np.mean(crps[ok])),
               "test_start": str(p["t"][ok].min()), "test_end": str(p["t"][ok].max())}
        # store per-obs losses for DM vs baseline
        row["_qlike_series"] = qlike
        row["_ok"] = ok
        results.append(row)
        forecasts[iid] = pd.DataFrame({"t": p["t"][ok].values, "y": y[ok], "m": m[ok], "s": s[ok],
                                       "qlike": qlike[ok], "crps": crps[ok]})

    # DM vs baseline (id 1) on common sample
    base = next((r for r in results if r.get("id") == 1 and r.get("status") == "ok"), None)
    if base is not None:
        bq = base["_qlike_series"]
        bok = base["_ok"]
        for r in results:
            if r.get("status") != "ok" or r.get("id") == 1 or "_qlike_series" not in r:
                continue
            common = bok & r["_ok"]
            if common.sum() < 100:
                continue
            dm, pval, n = dm_test(r["_qlike_series"][common], bq[common])
            r["dm_vs_har"] = dm
            r["p_value"] = pval
            r["n_common"] = int(n)
            r["qlike_gain_pct"] = float(100 * (1 - r["qlike"] / base["qlike"])) if base["qlike"] != 0 else np.nan
    # Holm across vol ideas with p-values
    vol_idx = [i for i, r in enumerate(results) if "p_value" in r and np.isfinite(r.get("p_value", np.nan))]
    if vol_idx:
        adj = holm([results[i]["p_value"] for i in vol_idx])
        for j, i in enumerate(vol_idx):
            results[i]["p_holm"] = float(adj[j])
    # strip series
    for r in results:
        r.pop("_qlike_series", None)
        r.pop("_ok", None)
    res = pd.DataFrame(results).sort_values("id")
    res.to_csv(OUT / "results.csv", index=False)
    # save forecasts (one file per idea is too many; save long file for vol ideas)
    long_rows = []
    for iid, df in forecasts.items():
        d = df.copy()
        d["idea_id"] = iid
        long_rows.append(d)
    if long_rows:
        pd.concat(long_rows, ignore_index=True).to_parquet(OUT / "forecasts_long.parquet", index=False)
    # summary md
    lines = ["# 100 ideas — results summary", "",
             f"Panel kx1h: {p.shape[0]} occasions, {p['t'].min()} .. {p['t'].max()}.",
             "Baseline = idea 01 (HAR). DM < 0 means the idea beats HAR on QLIKE.",
             "Holm-adjusted p across all tested ideas. See results.csv for full numbers.", ""]
    ok_res = res[res["status"] == "ok"].copy()
    if "qlike" in ok_res.columns:
        ok_res = ok_res.sort_values("qlike")
        lines.append("## Volatility ideas ranked by QLIKE (lower better)")
        lines.append("")
        lines.append("| id | idea | n | QLIKE | CRPS | DM vs HAR | p (Holm) |")
        lines.append("|---|---|---|---|---|---|---|")
        for _, r in ok_res.iterrows():
            if not np.isfinite(r.get("qlike", np.nan)):
                continue
            lines.append(f"| {int(r['id'])} | {r['name']} | {int(r.get('n_test',0))} | {r['qlike']:.4f} | {r.get('crps',np.nan):.4f} | {r.get('dm_vs_har',np.nan):.2f} | {r.get('p_holm',np.nan):.3g} |")
        lines.append("")
    ret = res[res["category"] == "Directional"] if "category" in res.columns else pd.DataFrame()
    if len(ret):
        lines.append("## Directional ideas")
        lines.append("")
        lines.append("| id | idea | n | OOS R2 | corr | Sharpe net | t (NW) | p |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for _, r in ret.iterrows():
            lines.append(f"| {int(r['id'])} | {r['name']} | {r.get('n_test','')} | {r.get('oos_r2',np.nan):.4f} | {r.get('corr',np.nan):.3f} | {r.get('sharpe_net_ann',np.nan):.2f} | {r.get('t_nw',np.nan):.2f} | {r.get('p_value',np.nan):.3g} |")
        lines.append("")
    lines.append("## Skipped / errors")
    lines.append("")
    for _, r in res[res["status"] != "ok"].iterrows():
        lines.append(f"- {int(r['id'])} {r['name']}: {r['status']}")
    (OUT / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(res.to_string())
    print(f"\nwrote {OUT}/results.csv and summary.md")


if __name__ == "__main__":
    main()
