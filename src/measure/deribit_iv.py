"""At-the-money implied variance from Deribit, interpolated to an arbitrary forecast window.

Step 1, per expiry: take option trades in the hour before the snapshot (or, live, the mark IVs of the whole
chain), compute log-moneyness k = ln(K / index) and fit IV(k) with a weighted quadratic (linear if few
trades). ATM IV = fitted value at k = 0.
Step 2, across expiries: total implied variance w(tau) = IV^2 * tau is interpolated linearly in tau to the
forecast horizon. Before the first expiry the first expiry's IV is used; beyond the last, the last one's.

Settlement alignment: Deribit options settle on the 30-minute average of the Deribit index before 08:00 UTC.
The variance of an average over the last 30 minutes is that of a point price 20 minutes earlier, so the
quoted IV satisfies IV^2 * tau = sigma^2 * (tau - 1/3 h). We convert to the point-price variance rate sigma^2
before interpolating to the windows of Polymarket and Kalshi, which settle on (almost) point prices.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

YEAR_H = 365 * 24


def atm_iv(k: np.ndarray, iv: np.ndarray, w: np.ndarray) -> float:
    k, iv, w = map(lambda a: np.asarray(a, float), (k, iv, w))
    ok = np.isfinite(k) & np.isfinite(iv) & (iv > 0) & (np.abs(k) < 0.3)
    k, iv, w = k[ok], iv[ok], w[ok]
    if len(k) == 0:
        return np.nan
    both = (k.min() < 0) and (k.max() > 0)
    if len(k) >= 6 and both:
        coef = np.polyfit(k, iv, 2, w=np.sqrt(w))
    elif len(k) >= 3 and both:
        coef = np.polyfit(k, iv, 1, w=np.sqrt(w))
    else:
        near = np.abs(k) < 0.01
        return float(np.median(iv[near])) if near.any() else np.nan
    val = float(np.polyval(coef, 0.0))
    return float(np.clip(val, iv.min(), iv.max()))


def term_structure(df: pd.DataFrame, snap: pd.Timestamp, iv_col: str = "iv", weight_col: str = "amount") -> pd.DataFrame:
    """df: rows with expiry, strike, index_price, iv (in %), weight. Returns tau_h and ATM IV per expiry."""
    rows = []
    for e, g in df.groupby("expiry"):
        tau_h = (e - snap).total_seconds() / 3600
        if tau_h <= 1:
            continue
        k = np.log(g["strike"].to_numpy(float) / g["index_price"].to_numpy(float))
        w = g[weight_col].to_numpy(float) if weight_col in g else np.ones(len(g))
        a = atm_iv(k, g[iv_col].to_numpy(float), w)
        if np.isfinite(a):
            rows.append({"expiry": e, "tau_h": tau_h, "atm_iv": a, "n": len(g)})
    return pd.DataFrame(rows).sort_values("tau_h") if rows else pd.DataFrame(columns=["expiry", "tau_h", "atm_iv", "n"])


def implied_variance(ts: pd.DataFrame, horizon_h: float) -> float:
    """Total implied variance of log price over the next horizon_h hours."""
    if ts is None or ts.empty:
        return np.nan
    tau = ts["tau_h"].to_numpy(float)
    rate = (ts["atm_iv"].to_numpy(float) / 100) ** 2 * tau / np.maximum(tau - 1 / 3, 0.5) / YEAR_H  # TWAP correction
    wv = rate * tau
    h = float(horizon_h)
    if h <= tau[0]:
        return float(wv[0] / tau[0] * h)
    if h >= tau[-1]:
        return float(wv[-1] / tau[-1] * h)
    return float(np.interp(h, tau, wv))


def from_trades(trades: pd.DataFrame, horizons_h: dict) -> pd.DataFrame:
    """trades: output of download.deribit.build_interim_trades (one block per snapshot)."""
    out = []
    for snap, g in trades.groupby("snap_ts"):
        g = g[g["ts"] < snap]
        ts = term_structure(g, snap)
        row = {"snap_ts": snap, "n_trades": len(g), "n_expiries": len(ts)}
        for name, h in horizons_h.items():
            row[f"iv_var_{name}"] = implied_variance(ts, h)
        out.append(row)
    return pd.DataFrame(out)


def from_chain(chain: pd.DataFrame, snap: pd.Timestamp, horizons_h: dict) -> dict:
    """Live: mark IVs of the whole chain (logger snapshot). Uses out-of-the-money options only."""
    from src.download.deribit import parse_instrument
    c = chain.copy()
    parsed = c["instrument"].map(parse_instrument)
    c["expiry"] = [p[0] for p in parsed]
    c["strike"] = [p[1] for p in parsed]
    c["cp"] = [p[2] for p in parsed]
    c["index_price"] = c["underlying"]
    otm = ((c["cp"] == "C") & (c["strike"] >= c["index_price"])) | ((c["cp"] == "P") & (c["strike"] < c["index_price"]))
    c = c[otm & c["mark_iv"].notna()]
    c["w"] = 1.0
    ts = term_structure(c, snap, iv_col="mark_iv", weight_col="w")
    return {f"iv_var_{name}": implied_variance(ts, h) for name, h in horizons_h.items()}
