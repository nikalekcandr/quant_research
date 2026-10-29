"""Risk metrics for the alpha lab and for volatility forecasts.

Covers three layers:

  1. Position risk — VaR/ES, exposure, leverage, concentration per family/venue.
  2. Forecast risk — PIT calibration, Christoffersen independence, QLIKE decomposition.
  3. Strategy risk — drawdown, turnover, capacity, slippage, deflated Sharpe.

All functions are pure (DataFrame in → DataFrame/dict out) so they can be called
from notebooks, lab/report.py, or daily_update.py without side effects.

Example:
    from src.risk import position_risk, drawdown_stats, forecast_calibration
    pr = position_risk(read_ledger())
    dd = drawdown_stats(daily_pnl_series)
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats


# ------------------------------------------------------------------ position risk
def position_risk(ledger: pd.DataFrame) -> pd.DataFrame:
    """Exposure and concentration of open positions.

    Returns one row per (family, venue) with gross/net exposure in notional
    units (perps) and in $1 contracts (binaries/options).
    """
    if ledger.empty or "event" not in ledger:
        return pd.DataFrame()
    open_pos = ledger[ledger["event"] == "OPEN"].copy()
    # Remove positions that have a matching CLOSE/INSTANT with same id
    closed_ids = set(ledger[ledger["event"].isin(["CLOSE", "INSTANT"])]["id"].astype(str))
    open_pos = open_pos[~open_pos["id"].astype(str).isin(closed_ids)]
    if open_pos.empty:
        return pd.DataFrame()
    open_pos["notional"] = 1.0  # each order is 1 unit; Portfolio scales this
    # Try to infer family from hyp prefix (D/P/A/V/W/M/O/F)
    open_pos["family"] = open_pos["hyp"].str[0].fillna("?")
    rows = []
    for (fam, venue), g in open_pos.groupby(["family", "venue"]):
        longs = (g["side"] == "long").sum() + (g["side"] == "yes").sum()
        shorts = (g["side"] == "short").sum() + (g["side"] == "no").sum()
        rows.append({
            "family": fam,
            "venue": venue,
            "open_positions": len(g),
            "longs": int(longs),
            "shorts": int(shorts),
            "net_exposure": int(longs - shorts),
            "gross_exposure": int(len(g)),
        })
    return pd.DataFrame(rows)


def var_es(returns: pd.Series, alpha: float = 0.05) -> dict:
    """Historical VaR and Expected Shortfall (one-sided, loss = negative return)."""
    r = pd.Series(returns).dropna().to_numpy(float)
    if len(r) < 20:
        return {"VaR_95": np.nan, "ES_95": np.nan, "n": len(r)}
    var = float(np.quantile(r, alpha))
    es = float(r[r <= var].mean()) if (r <= var).any() else var
    return {"VaR_95": var, "ES_95": es, "n": len(r), "VaR_99": float(np.quantile(r, 0.01))}


def drawdown_stats(cum_pnl: pd.Series) -> dict:
    """Drawdown statistics from a cumulative P&L series (in return or log-return units)."""
    s = pd.Series(cum_pnl).dropna().sort_index()
    if len(s) < 5:
        return {}
    peak = s.cummax()
    dd = peak - s  # in same units as cum_pnl
    # For return units: max DD as fraction of peak notional
    max_dd = float(dd.max())
    # Duration: longest time under water
    underwater = (s < peak).astype(int)
    # Find consecutive underwater stretches
    max_dur = 0
    cur = 0
    for v in underwater:
        if v:
            cur += 1
            max_dur = max(max_dur, cur)
        else:
            cur = 0
    # Calmar: annualised return / max DD (if DD > 0)
    total_ret = float(s.iloc[-1] - s.iloc[0]) if len(s) > 1 else 0.0
    years = (s.index[-1] - s.index[0]).days / 365.25 if hasattr(s.index, "days") or isinstance(s.index[0], pd.Timestamp) else len(s) / 365.25
    ann_ret = total_ret / years if years > 0 else 0.0
    calmar = ann_ret / max_dd if max_dd > 1e-9 else float("inf") if ann_ret > 0 else 0.0
    return {
        "total_return": total_ret,
        "ann_return": ann_ret,
        "max_drawdown": max_dd,
        "max_dd_days": int(max_dur),
        "calmar": calmar,
        "current_dd": float(dd.iloc[-1]),
        "peak": float(peak.iloc[-1]),
    }


def turnover(ledger: pd.DataFrame) -> pd.DataFrame:
    """Daily turnover: number of opened + closed trades per day, by family."""
    if ledger.empty or "ts" not in ledger:
        return pd.DataFrame()
    led = ledger.copy()
    led["ts"] = pd.to_datetime(led["ts"], utc=True, errors="coerce")
    led["day"] = led["ts"].dt.floor("D")
    rows = []
    for (day, fam), g in led.groupby(["day", led["hyp"].str[0].fillna("?")]):
        rows.append({
            "day": day,
            "family": fam,
            "opened": int((g["event"] == "OPEN").sum()),
            "closed": int(g["event"].isin(["CLOSE", "INSTANT"]).sum()),
            "turnover": int(len(g)),
        })
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ forecast calibration
def pit_histogram(pits: pd.Series, bins: int = 10) -> pd.DataFrame:
    """PIT histogram: counts per bin, expected = n/bins. χ² test for uniformity."""
    x = pd.Series(pits).dropna().to_numpy(float)
    x = x[(x >= 0) & (x <= 1)]
    if len(x) < 20:
        return pd.DataFrame()
    counts, edges = np.histogram(x, bins=bins, range=(0, 1))
    expected = len(x) / bins
    chi2 = float(((counts - expected) ** 2 / expected).sum())
    p = float(stats.chi2.sf(chi2, bins - 1))
    rows = []
    for i in range(bins):
        rows.append({
            "bin": f"[{edges[i]:.1f}, {edges[i+1]:.1f})",
            "count": int(counts[i]),
            "expected": expected,
            "excess": int(counts[i] - expected),
        })
    df = pd.DataFrame(rows)
    df.attrs["chi2"] = chi2
    df.attrs["p_uniform"] = p
    df.attrs["n"] = len(x)
    return df


def christoffersen(pits: pd.Series, alpha: float = 0.05) -> dict:
    """Christoffersen (1998) conditional coverage test for VaR violations.

    H0: violations are iid Bernoulli(alpha). Returns LR statistics for
    unconditional coverage, independence, and conditional coverage.
    """
    x = pd.Series(pits).dropna().to_numpy(float)
    if len(x) < 50:
        return {"n": len(x), "note": "too few observations"}
    # VaR violation = PIT < alpha (left tail) — for two-sided we test both tails
    viol = (x < alpha).astype(int)
    n, n1 = len(viol), int(viol.sum())
    n0 = n - n1
    if n1 == 0 or n0 == 0:
        return {"n": n, "n_viol": n1, "expected": n * alpha, "note": "no variation"}
    # Unconditional coverage
    pi = n1 / n
    lr_uc = -2 * (n0 * math.log(1 - alpha) + n1 * math.log(alpha) - n0 * math.log(1 - pi) - n1 * math.log(pi))
    # Independence (first-order Markov)
    n00 = int(((viol[:-1] == 0) & (viol[1:] == 0)).sum())
    n01 = int(((viol[:-1] == 0) & (viol[1:] == 1)).sum())
    n10 = int(((viol[:-1] == 1) & (viol[1:] == 0)).sum())
    n11 = int(((viol[:-1] == 1) & (viol[1:] == 1)).sum())
    pi0 = n01 / (n00 + n01) if (n00 + n01) > 0 else 0
    pi1 = n11 / (n10 + n11) if (n10 + n11) > 0 else 0
    pi2 = n1 / n
    if min(n00, n01, n10, n11) == 0:
        lr_ind = 0.0
    else:
        lr_ind = -2 * (
            (n00 + n01) * math.log(1 - pi2) + (n10 + n11) * math.log(pi2)
            - n00 * math.log(1 - pi0) - n01 * math.log(pi0)
            - n10 * math.log(1 - pi1) - n11 * math.log(pi1) if pi0 not in (0, 1) and pi1 not in (0, 1) else 0
        )
    lr_cc = lr_uc + lr_ind
    return {
        "n": n,
        "n_viol": n1,
        "expected_viol": n * alpha,
        "violation_rate": pi,
        "LR_uc": lr_uc,
        "p_uc": float(stats.chi2.sf(lr_uc, 1)),
        "LR_ind": lr_ind,
        "p_ind": float(stats.chi2.sf(lr_ind, 1)),
        "LR_cc": lr_cc,
        "p_cc": float(stats.chi2.sf(lr_cc, 2)),
    }


def qlike_decomposition(y: pd.Series, f: pd.Series) -> dict:
    """QLIKE loss decomposition: bias, variance, and correlation components.

    QLIKE(y,f) = y/f − log(y/f) − 1.  Reports mean QLIKE, bias of log(f),
    and a Mincer-Zarnowitz regression log(y) on log(f).
    """
    y = pd.Series(y).dropna()
    f = pd.Series(f).reindex(y.index).dropna()
    idx = y.index.intersection(f.index)
    y, f = y.loc[idx].to_numpy(float), f.loc[idx].to_numpy(float)
    y, f = y[np.isfinite(y) & np.isfinite(f) & (y > 0) & (f > 0)], f[np.isfinite(y) & np.isfinite(f) & (y > 0) & (f > 0)]
    if len(y) < 20:
        return {"n": len(y)}
    qlike = y / f - np.log(y / f) - 1
    ly, lf = np.log(y), np.log(f)
    bias = float(lf.mean() - ly.mean())
    # MZ: log(y) = a + b*log(f) + e
    X = np.column_stack([np.ones(len(lf)), lf])
    beta, *_ = np.linalg.lstsq(X, ly, rcond=None)
    resid = ly - X @ beta
    r2 = 1 - (resid @ resid) / ((ly - ly.mean()) @ (ly - ly.mean()))
    return {
        "n": len(y),
        "mean_QLIKE": float(qlike.mean()),
        "bias_log": bias,
        "mz_alpha": float(beta[0]),
        "mz_beta": float(beta[1]),
        "mz_R2": float(r2),
        "corr_log": float(np.corrcoef(ly, lf)[0, 1]),
    }


# ------------------------------------------------------------------ capacity & slippage
def capacity_at_book(
    ladder: pd.DataFrame, side: str = "ask", depth_col: str = "ask_size"
) -> pd.DataFrame:
    """How much notional is available at the top of the book per snapshot.

    ladder: DataFrame with snap_ts, strike, bid, ask, bid_size, ask_size.
    Returns median/mean available size and edge at top of book.
    """
    if ladder.empty:
        return pd.DataFrame()
    col = "ask_size" if side == "ask" else "bid_size"
    if col not in ladder:
        return pd.DataFrame()
    snap_sizes = ladder.groupby("snap_ts")[col].sum()
    return pd.DataFrame([{
        "side": side,
        "snapshots": int(ladder["snap_ts"].nunique()),
        "median_size_at_top": float(snap_sizes.median()),
        "mean_size_at_top": float(snap_sizes.mean()),
        "p90_size_at_top": float(snap_sizes.quantile(0.9)),
        "median_spread_cents": float((ladder["ask"] - ladder["bid"]).median() * 100),
    }])


# ------------------------------------------------------------------ deflated Sharpe (re-export for convenience)
def deflated_sharpe(
    sr: float, n: int, skew: float, kurt: float, n_trials: int
) -> float:
    """Bailey & López de Prado (2014) deflated Sharpe ratio (single number)."""
    if not np.isfinite(sr) or n < 10:
        return float("nan")
    den = math.sqrt(max(1e-12, 1 - skew * sr + (kurt - 1) / 4 * sr**2))
    g = 0.5772156649
    v = 1 / (n - 1)
    sr0 = math.sqrt(v) * ((1 - g) * stats.norm.ppf(1 - 1 / max(2, n_trials)) + g * stats.norm.ppf(1 - 1 / (max(2, n_trials) * math.e)))
    return float(stats.norm.cdf((sr - sr0) * math.sqrt(n - 1) / den))


# ------------------------------------------------------------------ summary table for reports
def risk_summary(ledger: pd.DataFrame, daily_pnl: pd.Series | None = None) -> pd.DataFrame:
    """One-row-per-family risk summary for inclusion in lab reports."""
    if ledger.empty:
        return pd.DataFrame()
    done = ledger[ledger["event"].isin(["CLOSE", "INSTANT"])].copy()
    done["pnl"] = pd.to_numeric(done["pnl"], errors="coerce")
    done = done.dropna(subset=["pnl"])
    if done.empty:
        return pd.DataFrame()
    done["family"] = done["hyp"].str[0].fillna("?")
    rows = []
    for fam, g in done.groupby("family"):
        x = g["pnl"].to_numpy(float)
        ve = var_es(x)
        rows.append({
            "family": fam,
            "trades": len(x),
            "mean_net": float(x.mean()),
            "vol_per_trade": float(x.std(ddof=1)) if len(x) > 1 else 0.0,
            "VaR_95_per_trade": ve["VaR_95"],
            "ES_95_per_trade": ve["ES_95"],
            "hit_rate": float((x > 0).mean()),
            "total_pnl": float(x.sum()),
            "worst_trade": float(x.min()),
            "best_trade": float(x.max()),
        })
    if daily_pnl is not None and len(daily_pnl) > 5:
        dd = drawdown_stats(daily_pnl.cumsum())
        for r in rows:
            r["portfolio_maxDD"] = dd.get("max_drawdown", np.nan)
            r["portfolio_calmar"] = dd.get("calmar", np.nan)
    return pd.DataFrame(rows)
