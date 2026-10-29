"""Forecast evaluation: losses, density scores, Mincer-Zarnowitz, Diebold-Mariano, ladder probabilities, trading.

Point losses use the level forecast h (expected realized variance):
  QLIKE(RV, h) = RV/h - log(RV/h) - 1      robust to noise in RV (Patton 2011), our main point loss
  MSE_log      = (log RV - E[log RV forecast])^2
Density scores for y = log RV:
  log score (higher is better), CRPS (lower is better), PIT, coverage of the central 50% and 90% intervals.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

import config

COMP = (1, 2, 3)


def _comps(fc):
    m = np.column_stack([fc[f"m{j}"].to_numpy(float) for j in COMP])
    s = np.column_stack([fc[f"s{j}"].to_numpy(float) for j in COMP])
    w = np.column_stack([fc[f"w{j}"].to_numpy(float) for j in COMP])
    w = np.nan_to_num(w)
    m = np.where(w > 0, m, 0.0)
    s = np.where(w > 0, s, 1.0)
    return m, s, w


def _A(mu, var):
    sd = np.sqrt(var)
    z = mu / sd
    return 2 * sd * stats.norm.pdf(z) + mu * (2 * stats.norm.cdf(z) - 1)


def scores(fc: pd.DataFrame) -> pd.DataFrame:
    y = fc["y"].to_numpy(float)
    rv = fc["rv"].to_numpy(float)
    m, s, w = _comps(fc)
    dens = np.sum(w * stats.norm.pdf((y[:, None] - m) / s) / s, axis=1)
    pit = np.sum(w * stats.norm.cdf((y[:, None] - m) / s), axis=1)
    crps = np.sum(w * _A(y[:, None] - m, s ** 2), axis=1)
    for j in range(3):
        for k in range(3):
            crps -= 0.5 * w[:, j] * w[:, k] * _A(m[:, j] - m[:, k], s[:, j] ** 2 + s[:, k] ** 2)
    h = fc["level"].to_numpy(float)
    out = fc[["horizon", "t", "T", "model"]].copy()
    out["qlike"] = rv / h - np.log(rv / h) - 1
    out["mse_log"] = (y - fc["mean_log"].to_numpy(float)) ** 2
    out["logscore"] = np.log(np.maximum(dens, 1e-300))
    out["crps"] = crps
    out["pit"] = pit
    out["in50"] = ((pit >= 0.25) & (pit <= 0.75)).astype(float)
    out["in90"] = ((pit >= 0.05) & (pit <= 0.95)).astype(float)
    out["y"] = y
    out["mean_log"] = fc["mean_log"].to_numpy(float)
    return out


def nw_lag(n: int, h: int = 1) -> int:
    return int(max(h - 1, np.floor(4 * (n / 100) ** (2 / 9))))


def overlap(horizon: str) -> int:
    """Number of consecutive occasions whose target windows overlap (6 for daily 6-day windows)."""
    return 6 if horizon == "pm6d" else 1


def hac_mean(d: np.ndarray, lag: int):
    d = d[np.isfinite(d)]
    n = len(d)
    if n < 10:
        return np.nan, np.nan, n
    u = d - d.mean()
    v = u @ u / n
    for L in range(1, lag + 1):
        v += 2 * (1 - L / (lag + 1)) * (u[L:] @ u[:-L]) / n
    se = np.sqrt(max(v, 1e-300) / n)
    return d.mean(), se, n


def dm_test(loss_a: np.ndarray, loss_b: np.ndarray, h: int = 1) -> dict:
    """Diebold-Mariano on d = loss_a - loss_b with Newey-West variance and the Harvey-Leybourne-Newbold
    small-sample correction. Negative statistic: model A has lower loss."""
    d = np.asarray(loss_a) - np.asarray(loss_b)
    d = d[np.isfinite(d)]
    n = len(d)
    mean, se, n = hac_mean(d, nw_lag(n, h))
    if not np.isfinite(se) or se == 0:
        return {"n": n, "mean_diff": mean, "dm": np.nan, "p": np.nan}
    dm = mean / se
    corr = np.sqrt(max((n + 1 - 2 * h + h * (h - 1) / n) / n, 1e-9))
    dm_hln = dm * corr
    p = 2 * stats.t.sf(abs(dm_hln), df=n - 1)
    return {"n": n, "mean_diff": mean, "dm": dm_hln, "p": p}


def mincer_zarnowitz(y: np.ndarray, f: np.ndarray, h: int = 1) -> dict:
    ok = np.isfinite(y) & np.isfinite(f)
    if ok.sum() < 20:
        return {"mz_a": np.nan, "mz_b": np.nan, "mz_r2": np.nan, "mz_p": np.nan}
    X = sm.add_constant(f[ok])
    res = sm.OLS(y[ok], X).fit(cov_type="HAC", cov_kwds={"maxlags": nw_lag(ok.sum(), h)})
    wald = res.wald_test("const = 0, x1 = 1", scalar=True)
    return {"mz_a": res.params[0], "mz_b": res.params[1], "mz_r2": res.rsquared, "mz_p": float(wald.pvalue)}


def summary_table(sc: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (h, mdl), g in sc.groupby(["horizon", "model"]):
        mz = mincer_zarnowitz(g["y"].to_numpy(), g["mean_log"].to_numpy(), overlap(h))
        rows.append({"horizon": h, "model": mdl, "n": len(g), "qlike": g["qlike"].mean(),
                     "mse_log": g["mse_log"].mean(), "logscore": g["logscore"].mean(), "crps": g["crps"].mean(),
                     "cov50": g["in50"].mean(), "cov90": g["in90"].mean(), **mz})
    return pd.DataFrame(rows)


def common_sample(sc: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    sub = sc[sc["model"].isin(models)]
    cnt = sub.groupby("t")["model"].nunique()
    keep = cnt[cnt == len(models)].index
    return sub[sub["t"].isin(keep)]


COMPARISONS = [
    ("har_m", "har", "HAR + ladder-implied variance vs HAR (main test)"),
    ("har", "naive", "HAR vs naive"),
    ("har", "garch", "HAR vs GARCH"),
    ("harx", "har", "time-series extensions"),
    ("harx_d", "harx", "adding Deribit"),
    ("harx_all", "harx_d", "adding the ladder on top of Deribit (key test)"),
    ("harx_m", "harx", "adding the ladder alone"),
    ("mkt_mz", "iv_mz", "ladder vs Deribit as standalone forecasts"),
    ("combo", "harx_all", "combination vs best regression"),
    ("gbm_all", "harx_all", "gradient-boosted trees vs the same features in OLS"),
]


def dm_table(sc: pd.DataFrame, losses=("qlike", "logscore")) -> pd.DataFrame:
    rows = []
    for h, gh in sc.groupby("horizon"):
        for a, b, label in COMPARISONS:
            if a not in set(gh["model"]) or b not in set(gh["model"]):
                continue
            pa = gh[gh["model"] == a].set_index("t")
            pb = gh[gh["model"] == b].set_index("t")
            idx = pa.index.intersection(pb.index)
            for loss in losses:
                la, lb = pa.loc[idx, loss].to_numpy(), pb.loc[idx, loss].to_numpy()
                if loss == "logscore":  # turn the score into a loss
                    la, lb = -la, -lb
                r = dm_test(la, lb, overlap(h))
                rows.append({"horizon": h, "A": a, "B": b, "comparison": label, "loss": loss, **r})
    return pd.DataFrame(rows)


def bias_table(panel_by_h: dict) -> pd.DataFrame:
    """Average of log RV minus log implied variance: negative = implied variance too high (variance premium)."""
    rows = []
    for h, p in panel_by_h.items():
        for col, label in [("l_iv_var", "Deribit ATM"), ("l_dvol_var", "DVOL"),
                           ("l_mkt_mfiv", "ladder, model-free (BJN)"), ("l_mkt_mfiv_trunc", "ladder, BJN truncated"),
                           ("l_mkt_var_c", "ladder, lognormal"), ("l_mkt_var_t", "ladder, Student-t")]:
            if col not in p:
                continue
            d = (p["y"] - p[col]).to_numpy(float)
            d = d[np.isfinite(d) & (p["t"] >= pd.Timestamp(config.EVAL_START, tz="UTC")).to_numpy()]
            mean, se, n = hac_mean(d, nw_lag(len(d), overlap(h)))
            if n == 0:
                continue
            # implied variance is an expectation of the level of RV, so the premium is measured on levels:
            # E[RV / IV^2] = 1 without a premium. The mean log ratio is lower by about Var(log RV)/2 even
            # without any premium (Jensen), so it overstates the premium.
            r_mean, r_se, _ = hac_mean(np.exp(d), nw_lag(len(d), overlap(h)))
            rows.append({"horizon": h, "implied": label, "n": n, "mean_log_ratio": mean, "se": se,
                         "t_stat": mean / se if se else np.nan,
                         "implied_over_realized": np.exp(-mean) if np.isfinite(mean) else np.nan,
                         "mean_rv_over_implied": r_mean, "se_ratio": r_se,
                         "t_ratio_vs_1": (r_mean - 1) / r_se if r_se else np.nan})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ ladder probabilities
GH_X, GH_W = np.polynomial.hermite.hermgauss(24)


def digital_prob(S: np.ndarray, K: np.ndarray, m: np.ndarray, s: np.ndarray, w: np.ndarray) -> np.ndarray:
    """P(S_e > K) when log S_e = log S - v/2 + sqrt(v) Z and log v ~ sum_k w_k N(m_k, s_k^2).

    A lognormal mixture: uncertainty about variance fattens the tails without extra parameters.
    S, K: (n,), m, s, w: (n, 3).
    """
    out = np.zeros(len(S))
    for j in range(m.shape[1]):
        wj = w[:, j]
        if not np.any(wj > 0):
            continue
        v = np.exp(m[:, j][:, None] + np.sqrt(2) * s[:, j][:, None] * GH_X[None, :])
        d = (np.log(S / K)[:, None] - v / 2) / np.sqrt(v)
        out += wj * (stats.norm.cdf(d) @ GH_W) / np.sqrt(np.pi)
    return out


def ladder_eval(strikes: pd.DataFrame, fc: pd.DataFrame, panel: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    """Strike-level table: market price, model probabilities, outcome."""
    if strikes is None or strikes.empty:
        return pd.DataFrame()
    h = strikes["horizon"].iloc[0]
    st = strikes.merge(panel[["T", "t", "S_t", "S_e"]].rename(columns={"t": "t_panel"}), on="T", how="left")
    # quotes must be observed at the same moment as the price S_t the model uses; anything else leaks
    st = st[st["t"] == st["t_panel"]].drop(columns="t_panel")
    if h == "kx1h":
        # Kalshi settles on CF Benchmarks BRTI; scale Binance by the median BRTI/Binance ratio of the
        # previous 48 settled events (known before t)
        ev = st.groupby("T").agg(ev=("expiration_value", "first"), S_e=("S_e", "first")).sort_index()
        ratio = (pd.to_numeric(ev["ev"], errors="coerce") / ev["S_e"]).shift(1).rolling(48, min_periods=5).median()
        st["basis"] = st["T"].map(ratio).fillna(1.0)
        st["S_ref"] = st["S_t"] * st["basis"]
        st["outcome"] = st["outcome"].astype(float)
    else:
        st["S_ref"] = st["S_t"]
        # Polymarket settles on the Binance close: recompute the outcome from our own data
        st["outcome_binance"] = (st["S_e"] > st["strike"]).astype(float)
        st.loc[st["S_e"].isna(), "outcome_binance"] = np.nan
        st["outcome"] = st["outcome_binance"].fillna(st["outcome"])
    for mdl in models:
        f = fc[fc["model"] == mdl].set_index("T")
        f = f[~f.index.duplicated()]
        sub = f.reindex(pd.DatetimeIndex(st["T"]))
        m = np.column_stack([sub[f"m{j}"].to_numpy(float) for j in COMP])
        s = np.column_stack([sub[f"s{j}"].to_numpy(float) for j in COMP])
        w = np.nan_to_num(np.column_stack([sub[f"w{j}"].to_numpy(float) for j in COMP]))
        m, s = np.where(w > 0, m, 0), np.where(w > 0, s, 1)
        p = digital_prob(st["S_ref"].to_numpy(float), st["strike"].to_numpy(float), m, s, w)
        p[~np.isfinite(sub["m1"].to_numpy(float))] = np.nan
        st[f"p_{mdl}"] = p
    return st


def ladder_scores(le: pd.DataFrame, models: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Brier and log scores of market vs models (per snapshot sums), calibration by bins, DM per snapshot."""
    if le.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    d = le.dropna(subset=["outcome", "p"] + [f"p_{m}" for m in models]).copy()
    d = d[d["t"] >= pd.Timestamp(config.EVAL_START, tz="UTC")]
    srcs = {"market": "p", **{m: f"p_{m}" for m in models}}
    eps = 1e-4
    for k, c in srcs.items():
        pc = np.clip(d[c].to_numpy(float), eps, 1 - eps)
        d[f"brier_{k}"] = (pc - d["outcome"]) ** 2
        d[f"ll_{k}"] = -(d["outcome"] * np.log(pc) + (1 - d["outcome"]) * np.log(1 - pc))
    snap = d.groupby("T")[[c for c in d.columns if c.startswith(("brier_", "ll_"))]].sum()
    h = d["horizon"].iloc[0] if len(d) else ""
    rows = []
    for k in srcs:
        rows.append({"horizon": h, "source": k, "n_strikes": len(d), "n_snapshots": len(snap),
                     "brier_per_strike": d[f"brier_{k}"].mean(), "logloss_per_strike": d[f"ll_{k}"].mean()})
    summ = pd.DataFrame(rows)
    dm_rows = []
    for mdl in models:
        for loss in ("brier", "ll"):
            r = dm_test(snap[f"{loss}_{mdl}"].to_numpy(), snap[f"{loss}_market"].to_numpy(), overlap(h))
            dm_rows.append({"horizon": h, "model": mdl, "vs": "market", "loss": loss, **r})
    bins = [0, 0.02, 0.05, 0.2, 0.5, 0.8, 0.95, 0.98, 1.0001]
    d["bin"] = pd.cut(d["p"], bins, right=False)
    agg = {"n": ("outcome", "size"), "market_p": ("p", "mean"), "realized": ("outcome", "mean")}
    for mdl in models:
        agg[f"p_{mdl}"] = (f"p_{mdl}", "mean")
    cal = d.groupby("bin", observed=True).agg(**agg).reset_index()
    cal["bin"] = cal["bin"].astype(str)
    cal.insert(0, "horizon", h)
    return summ, pd.DataFrame(dm_rows), cal


def trading(le: pd.DataFrame, model: str, thresholds=config.EDGE_THRESHOLDS) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Buy YES at the ask when model prob - ask > threshold; sell YES at the bid when bid - model prob > threshold.
    Profit per contract at settlement, net of Kalshi's taker fee. Polymarket history has no quotes, so
    bid/ask = price -/+ config.PM_HALF_SPREAD there (live logger books replace this assumption)."""
    d = le.dropna(subset=["outcome", f"p_{model}", "p"]).copy()
    d = d[d["t"] >= pd.Timestamp(config.EVAL_START, tz="UTC")]
    if d.empty:
        return pd.DataFrame(), pd.DataFrame()
    h = d["horizon"].iloc[0]
    if "bid" not in d or d["bid"].isna().all():
        d["bid"] = d["p"] - config.PM_HALF_SPREAD
        d["ask"] = d["p"] + config.PM_HALF_SPREAD
    fee = config.KALSHI_FEE_RATE if h == "kx1h" else 0.0
    rows, trades = [], []
    for th in thresholds:
        q = d[f"p_{model}"]
        buy = q - d["ask"] > th
        sell = d["bid"] - q > th
        pnl = np.where(buy, d["outcome"] - d["ask"] - fee * d["ask"] * (1 - d["ask"]), 0.0)
        pnl = pnl + np.where(sell, d["bid"] - d["outcome"] - fee * d["bid"] * (1 - d["bid"]), 0.0)
        tr = d[buy | sell].copy()
        tr["pnl"] = pnl[(buy | sell).to_numpy()]
        tr["side"] = np.where(buy[(buy | sell)], "buy", "sell")
        tr["threshold"] = th
        tr["model"] = model
        trades.append(tr)
        per_snap = tr.groupby("T")["pnl"].sum()
        mean, se, n = hac_mean(per_snap.to_numpy(), nw_lag(len(per_snap), overlap(h)))
        rows.append({"horizon": h, "model": model, "threshold": th, "n_trades": len(tr), "n_snapshots": n,
                     "pnl_per_contract": tr["pnl"].mean() if len(tr) else np.nan, "pnl_total": tr["pnl"].sum(),
                     "hit_rate": (tr["pnl"] > 0).mean() if len(tr) else np.nan,
                     "t_stat_per_snapshot": mean / se if se else np.nan})
    return pd.DataFrame(rows), pd.concat(trades, ignore_index=True)
