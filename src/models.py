"""Forecasting models for y = log realized variance over the window [t, e).

Every model returns a predictive density for y. Most are Gaussian N(m, s^2); the combination is an
equal-weight mixture of Gaussians. Stored as up to three components (w_k, m_k, s_k).

Estimation is rolling and strictly real-time: the model used at time t is estimated only on occasions
whose target window had already ended (e_j <= t), using the last ROLL_WINDOW of them.

Model list (numbers match the project plan):
  5  naive        log RV of the previous window of the same length (seasonally adjusted for kx1h)
  6  garch        GARCH(1,1) with Student-t errors on 24h returns (pm24h, pm6d only)
  7  har          HAR-RV: daily, weekly, monthly components (+ intraday seasonal factor for kx1h)
  8  harx         HAR + semivariances, jump share, HARQ term, macro-release and weekend dummies
  9  harx_d       HAR-X + Deribit: DVOL and ATM implied variance for the window
  7m har_m        HAR + ladder-implied variance (Britten-Jones-Neuberger, model-free): the direct test
  10 harx_m       HAR-X + prediction-market implied variance
  11 harx_all     HAR-X + Deribit + prediction market (encompassing test)
  12 iv_raw       Deribit implied variance as is (DVOL for kx1h)
  13 iv_mz        Deribit implied variance, rolling Mincer-Zarnowitz recalibration
  14 mkt_raw      ladder-implied BJN variance as is; mkt_ln_raw: lognormal fit; mkt_t_raw: Student-t fit
  15 mkt_mz       ladder-implied variance, rolling Mincer-Zarnowitz recalibration
  16 combo        equal-weight mixture of harx, iv_mz and mkt_mz
  17 gbm_all      gradient-boosted trees on the harx_all features (non-linear HAR-X + Deribit + ladder)
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

import config

HAR = ["l_rv_d", "l_rv_w", "l_rv_m"]
EXTRA = ["l_rs_pos", "l_rs_neg", "jump_share", "harq", "macro", "weekend"]


def specs(horizon: str) -> dict:
    kx = horizon == "kx1h"
    har = HAR + (["l_F", "l_F_naive"] if kx else [])
    harx = har + EXTRA
    deribit = ["l_dvol_var"] + ([] if kx else ["l_iv_var"])
    iv_col = "l_dvol_var" if kx else "l_iv_var"
    s = {
        "naive": {"kind": "raw", "x": "naive"},
        "har": {"kind": "ols", "features": har},
        "harx": {"kind": "ols", "features": harx},
        "harx_d": {"kind": "ols", "features": harx + deribit},
        "har_m": {"kind": "ols", "features": har + ["l_mkt_mfiv"], "market": True},
        "harx_m": {"kind": "ols", "features": harx + ["l_mkt_mfiv"], "market": True},
        "harx_all": {"kind": "ols", "features": harx + deribit + ["l_mkt_mfiv"], "market": True},
        "iv_raw": {"kind": "raw", "x": iv_col},
        "iv_mz": {"kind": "ols", "features": [iv_col]},
        "mkt_raw": {"kind": "raw", "x": "l_mkt_mfiv", "market": True},
        "mkt_ln_raw": {"kind": "raw", "x": "l_mkt_var_c", "market": True},
        "mkt_t_raw": {"kind": "raw", "x": "l_mkt_var_t", "market": True},
        "mkt_mz": {"kind": "ols", "features": ["l_mkt_mfiv"], "market": True},
        "combo": {"kind": "combo", "parts": ["harx", "iv_mz", "mkt_mz"]},
        "gbm_all": {"kind": "gbm", "features": harx + deribit + ["l_mkt_mfiv"]},
    }
    if not kx:
        s["garch"] = {"kind": "garch"}
    return s


# ------------------------------------------------------------------ rolling estimation
def _train_index(e_sorted: np.ndarray, t0) -> int:
    """Number of occasions whose window ended by t0 (panel sorted by t, hence by e)."""
    return int(np.searchsorted(e_sorted, np.datetime64(t0.tz_localize(None) if t0.tzinfo else t0), side="right"))


def rolling_ols(panel: pd.DataFrame, feats: list[str], horizon: str, market: bool = False):
    n = len(panel)
    X = panel[feats].to_numpy(float)
    y = panel["y"].to_numpy(float)
    okx = np.isfinite(X).all(axis=1)
    oky = np.isfinite(y)
    m_out, s_out = np.full(n, np.nan), np.full(n, np.nan)
    coefs = []
    e_sorted = panel["e"].dt.tz_localize(None).to_numpy(dtype="datetime64[ns]")
    win = config.ROLL_WINDOW[horizon]
    min_n = config.MIN_TRAIN_MARKET[horizon] if market else config.MIN_TRAIN[horizon]
    step = config.REFIT_EVERY[horizon]
    rows = np.flatnonzero(okx)
    for c0 in range(0, len(rows), step):
        chunk = rows[c0:c0 + step]
        t0 = panel["t"].iloc[chunk[0]]
        k = _train_index(e_sorted, t0)
        tr = np.flatnonzero(okx[:k] & oky[:k])[-win:]
        if len(tr) < min_n:
            continue
        A = np.column_stack([np.ones(len(tr)), X[tr]])
        beta, *_ = np.linalg.lstsq(A, y[tr], rcond=None)
        res = y[tr] - A @ beta
        dof = max(len(tr) - A.shape[1], 1)
        sig = np.sqrt(res @ res / dof)
        Ac = np.column_stack([np.ones(len(chunk)), X[chunk]])
        m_out[chunk] = Ac @ beta
        s_out[chunk] = sig
        coefs.append({"t": t0, "n_train": len(tr), "sigma": sig, **{f"b_{f}": b for f, b in zip(["const"] + feats, beta)}})
    return m_out, s_out, pd.DataFrame(coefs)


def rolling_raw(panel: pd.DataFrame, x: np.ndarray, horizon: str, market: bool = False):
    """Point forecast x (in logs) used as is; density spread from past forecast errors."""
    n = len(panel)
    y = panel["y"].to_numpy(float)
    ok = np.isfinite(x)
    err2 = np.where(ok & np.isfinite(y), (y - x) ** 2, np.nan)
    e_sorted = panel["e"].dt.tz_localize(None).to_numpy(dtype="datetime64[ns]")
    s_out = np.full(n, np.nan)
    win = config.ROLL_WINDOW[horizon]
    min_n = config.MIN_TRAIN_MARKET[horizon] if market else config.MIN_TRAIN[horizon]
    step = config.REFIT_EVERY[horizon]
    rows = np.flatnonzero(ok)
    for c0 in range(0, len(rows), step):
        chunk = rows[c0:c0 + step]
        k = _train_index(e_sorted, panel["t"].iloc[chunk[0]])
        past = err2[:k]
        past = past[np.isfinite(past)][-win:]
        if len(past) < min_n:
            continue
        s_out[chunk] = np.sqrt(past.mean())
    m_out = np.where(np.isfinite(s_out), x, np.nan)
    return m_out, s_out


GBM_REFIT = {"kx1h": 24 * 7, "pm24h": 10, "pm6d": 10}   # trees are refitted less often than OLS


def rolling_gbm(panel: pd.DataFrame, feats: list[str], horizon: str) -> np.ndarray:
    """Point forecast of y from gradient-boosted regression trees, rolling and real-time like rolling_ols.

    Trees find non-linearities and interactions (e.g. the ladder matters more after calm days) that a linear HAR
    cannot. They handle missing values natively, so days without a ladder still get a forecast. Shallow trees,
    a small learning rate and early stopping on the last 20% of the training window keep the model from
    memorising a year of noisy data."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    n = len(panel)
    X = panel[feats].to_numpy(float)
    y = panel["y"].to_numpy(float)
    base_ok = np.isfinite(panel[HAR].to_numpy(float)).all(axis=1)
    oky = np.isfinite(y)
    out = np.full(n, np.nan)
    e_sorted = panel["e"].dt.tz_localize(None).to_numpy(dtype="datetime64[ns]")
    win, min_n, step = config.ROLL_WINDOW[horizon], config.MIN_TRAIN[horizon], GBM_REFIT[horizon]
    rows = np.flatnonzero(base_ok)
    for c0 in range(0, len(rows), step):
        chunk = rows[c0:c0 + step]
        k = _train_index(e_sorted, panel["t"].iloc[chunk[0]])
        tr = np.flatnonzero(base_ok[:k] & oky[:k])[-win:]
        if len(tr) < min_n:
            continue
        gb = HistGradientBoostingRegressor(learning_rate=0.03, max_iter=400, max_leaf_nodes=8, min_samples_leaf=20,
                                           l2_regularization=1.0, early_stopping=True, validation_fraction=0.2,
                                           n_iter_no_change=30, random_state=0)
        # a feature that is (almost) never observed in the window (the ladder before it existed) is left out
        cols = np.isfinite(X[tr]).sum(axis=0) >= 30
        gb.fit(X[tr][:, cols], y[tr])
        out[chunk] = gb.predict(X[chunk][:, cols])
    return out


def naive_x(panel: pd.DataFrame) -> np.ndarray:
    x = panel["l_rv_naive"].to_numpy(float)
    if panel["horizon"].iloc[0] == "kx1h":
        x = x + panel["l_F"].to_numpy(float) - panel["l_F_naive"].to_numpy(float)
    return x


def garch_x(panel: pd.DataFrame, m) -> np.ndarray:
    """GARCH(1,1)-t on 24h returns between consecutive forecast times; refit every 5 days on <= 1500 days."""
    from arch import arch_model
    horizon = panel["horizon"].iloc[0]
    steps = 1 if horizon == "pm24h" else 6
    t = panel["t"]
    # 24h returns ending at each t (same clock time, so the series is regular)
    p_now = m.price_at(t)
    p_prev = m.price_at(t - pd.Timedelta(hours=24))
    r = 100 * np.log(p_now / p_prev)
    out = np.full(len(panel), np.nan)
    res = None
    for i in range(len(panel)):
        hist = r[: i + 1]
        hist = hist[np.isfinite(hist)][-1500:]
        if len(hist) < 250:
            continue
        if res is None or i % 5 == 0:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                am = arch_model(hist, mean="Zero", vol="GARCH", p=1, q=1, dist="t")
                try:
                    res = am.fit(disp="off", show_warning=False)
                except Exception:
                    res = None
                    continue
            params = res.params
        am = arch_model(hist, mean="Zero", vol="GARCH", p=1, q=1, dist="t")
        fc = am.fix(params).forecast(horizon=steps, reindex=False)
        out[i] = np.log(fc.variance.to_numpy()[-1].sum() / 1e4)
    return out


def run_all_models(panel: pd.DataFrame, m=None, models: list[str] | None = None) -> tuple[pd.DataFrame, dict]:
    """Returns the long table of forecasts (one row per occasion and model) and estimated coefficients."""
    horizon = panel["horizon"].iloc[0]
    panel = panel.sort_values("t").reset_index(drop=True)
    sp = specs(horizon)
    names = models or list(sp)
    comps, coefs = {}, {}
    for name in names:
        s = sp[name]
        if s["kind"] == "ols":
            mm, ss, cf = rolling_ols(panel, s["features"], horizon, s.get("market", False))
            coefs[name] = cf
        elif s["kind"] == "raw":
            x = naive_x(panel) if s["x"] == "naive" else panel[s["x"]].to_numpy(float)
            mm, ss = rolling_raw(panel, x, horizon, s.get("market", False))
        elif s["kind"] == "garch":
            if m is None:
                continue
            try:
                x = garch_x(panel, m)
            except Exception as e:  # GARCH is only a reference model: never let it stop the run
                print(f"  garch skipped for {horizon}: {e!r}", flush=True)
                continue
            mm, ss = rolling_raw(panel, x, horizon)
        elif s["kind"] == "gbm":
            try:
                x = rolling_gbm(panel, s["features"], horizon)
            except ImportError:
                print("  gbm_all skipped: pip install scikit-learn", flush=True)
                continue
            mm, ss = rolling_raw(panel, x, horizon)   # spread of the density from past out-of-sample errors
        else:
            continue
        comps[name] = (mm, ss)
    for name in names:
        s = sp[name]
        if s["kind"] == "combo" and all(p in comps for p in s["parts"]):
            mlist = [comps[p][0] for p in s["parts"]]
            slist = [comps[p][1] for p in s["parts"]]
            ok = np.all([np.isfinite(a) for a in mlist], axis=0)
            comps[name] = ([np.where(ok, a, np.nan) for a in mlist], [np.where(ok, a, np.nan) for a in slist])
    rows = []
    base = panel[["horizon", "t", "T", "e", "y", "rv"]]
    for name, (mm, ss) in comps.items():
        df = base.copy()
        df["model"] = name
        if isinstance(mm, list):
            k = len(mm)
            for j in range(3):
                df[f"m{j+1}"] = mm[j] if j < k else np.nan
                df[f"s{j+1}"] = ss[j] if j < k else np.nan
                df[f"w{j+1}"] = 1 / k if j < k else np.nan
        else:
            df["m1"], df["s1"], df["w1"] = mm, ss, 1.0
            for j in (2, 3):
                df[f"m{j}"] = df[f"s{j}"] = df[f"w{j}"] = np.nan
        rows.append(df)
    fc = pd.concat(rows, ignore_index=True)
    fc = fc[np.isfinite(fc["m1"])].reset_index(drop=True)
    fc["mean_log"] = sum(fc[f"w{j}"].fillna(0) * fc[f"m{j}"].fillna(0) for j in (1, 2, 3))
    raw = fc["model"].map(lambda n: sp[n]["kind"] in ("raw", "garch"))
    level_ols = sum(fc[f"w{j}"].fillna(0) * np.exp(fc[f"m{j}"].fillna(0) + fc[f"s{j}"].fillna(0) ** 2 / 2) for j in (1, 2, 3))
    fc["level"] = np.where(raw, np.exp(fc["m1"]), level_ols)
    return fc, coefs
