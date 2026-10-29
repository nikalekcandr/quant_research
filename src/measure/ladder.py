"""Implied distributions from a ladder of binary "price above K" contracts.

A contract that pays 1 if S_T > K is a digital call; its fair price is P(S_T > K).
Given prices p_i at strikes K_i we fit a distribution for log S_T:

  lognormal  : P(S_T > K) = 1 - Phi((ln K - m) / s)            parameters m, s
  Student-t  : P(S_T > K) = 1 - F_nu((ln K - m) / s)           parameters m, s, nu

s^2 is the market's forecast of the variance of log S_T over the remaining window, i.e. of realized variance.
For the Student-t the variance is s^2 * nu / (nu - 2); nu measures how heavy the market thinks the tails are.

The fit minimises weighted squared errors in prices (not in probit space, which blows up noise near 0 and 1).
Weights are 1 / spread when the spread is known.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import optimize, stats

import config


def _start(lk, p):
    z = stats.norm.ppf(np.clip(1 - p, 1e-4, 1 - 1e-4))
    if len(lk) >= 2 and np.ptp(lk) > 0:
        b, a = np.polyfit(lk, z, 1)
        if b > 0:
            return -a / b, 1 / b
    return float(np.mean(lk)), 0.02


def fit_lognormal(k, p, w=None, band=config.CENTRAL_BAND, band_wide=config.CENTRAL_BAND_WIDE,
                  min_n: int = config.MIN_STRIKES) -> dict:
    k, p = np.asarray(k, float), np.asarray(p, float)
    w = np.ones_like(p) if w is None else np.asarray(w, float)
    used_band = band
    sel = (p >= band[0]) & (p <= band[1]) & np.isfinite(p)
    if sel.sum() < min_n:
        used_band = band_wide
        sel = (p >= band_wide[0]) & (p <= band_wide[1]) & np.isfinite(p)
    if sel.sum() < min_n:
        return {"m": np.nan, "s": np.nan, "n": int(sel.sum()), "rmse": np.nan, "band": np.nan}
    lk, pp, ww = np.log(k[sel]), p[sel], w[sel]
    m0, s0 = _start(lk, pp)
    s0 = float(np.clip(s0, 1e-4, 0.5))

    def resid(th):
        return np.sqrt(ww) * (1 - stats.norm.cdf((lk - th[0]) / th[1]) - pp)

    res = optimize.least_squares(resid, [m0, s0], bounds=([lk.min() - 1, 1e-4], [lk.max() + 1, 1.0]))
    m, s = res.x
    rmse = float(np.sqrt(np.mean((1 - stats.norm.cdf((lk - m) / s) - pp) ** 2)))
    return {"m": float(m), "s": float(s), "n": int(sel.sum()), "rmse": rmse, "band": used_band[0]}


def fit_student(k, p, w=None, band=config.TAIL_BAND, min_n: int = config.MIN_STRIKES + 1) -> dict:
    k, p = np.asarray(k, float), np.asarray(p, float)
    w = np.ones_like(p) if w is None else np.asarray(w, float)
    sel = (p >= band[0]) & (p <= band[1]) & np.isfinite(p)
    nan = {"m": np.nan, "s": np.nan, "nu": np.nan, "var": np.nan, "n": int(sel.sum()), "rmse": np.nan}
    if sel.sum() < min_n:
        return nan
    lk, pp, ww = np.log(k[sel]), p[sel], w[sel]
    m0, s0 = _start(lk, pp)
    s0 = float(np.clip(s0, 1e-4, 0.5))

    def resid(th):
        return np.sqrt(ww) * (1 - stats.t.cdf((lk - th[0]) / th[1], df=th[2]) - pp)

    try:
        res = optimize.least_squares(resid, [m0, s0, 6.0],
                                     bounds=([lk.min() - 1, 1e-4, 2.1], [lk.max() + 1, 1.0, 60.0]))
    except ValueError:
        return nan
    m, s, nu = res.x
    rmse = float(np.sqrt(np.mean((1 - stats.t.cdf((lk - m) / s, df=nu) - pp) ** 2)))
    return {"m": float(m), "s": float(s), "nu": float(nu), "var": float(s ** 2 * nu / (nu - 2)),
            "n": int(sel.sum()), "rmse": rmse}


def monotone(p: np.ndarray) -> np.ndarray:
    """Pool-adjacent-violators: make ladder prices non-increasing in strike.
    Stale prices of far strikes sometimes violate monotonicity (e.g. 126k priced above 124k)."""
    y = list(map(float, p))
    blocks = [[v, 1] for v in y]
    i = 0
    while i < len(blocks) - 1:
        if blocks[i][0] < blocks[i + 1][0]:
            v = (blocks[i][0] * blocks[i][1] + blocks[i + 1][0] * blocks[i + 1][1]) / (blocks[i][1] + blocks[i + 1][1])
            blocks[i] = [v, blocks[i][1] + blocks[i + 1][1]]
            del blocks[i + 1]
            i = max(i - 1, 0)
        else:
            i += 1
    return np.concatenate([[v] * n for v, n in blocks])


def mfiv(k, p, band=config.TAIL_BAND, truncate: bool = False, n_grid: int = 4001) -> dict:
    """Model-free implied variance of a digital ladder (Britten-Jones & Neuberger 2000).

    For a continuous martingale price S with forward F, the expected integrated variance between t and T is
        E[ int sigma^2 dt ] = 2 E[ S_T/F - 1 - ln(S_T/F) ]
                            = 2 [ int_F^inf d(K) (1/F - 1/K) dK + int_0^F (1 - d(K)) (1/K - 1/F) dK ],
    where d(K) = P(S_T > K) is exactly the price of the contract "above K". The ladder gives d(K) at a few
    strikes; between strikes we interpolate the probit of the CDF linearly in log K (exact for a lognormal,
    so the $2000 strike spacing adds no variance of its own) and beyond the last informative strikes we
    extrapolate with the slope of the outermost segment. truncate=True integrates only between the outermost
    informative strikes (the CBOE/Jiang-Tian truncation, a lower bound). F is the ladder's own median: the
    formula is second-order insensitive to the choice of F.
    """
    k, p = np.asarray(k, float), np.asarray(p, float)
    o = np.argsort(k)
    k, p = k[o], monotone(p[o])
    sel = (p > band[0]) & (p < band[1]) & np.isfinite(p)
    nan = {"mfiv": np.nan, "mfiv_trunc": np.nan, "n": int(sel.sum())}
    if sel.sum() < 2:
        return nan
    # the ladder must bracket the median: when the price has run past every strike (all of them at 94-99.7c,
    # say) the whole distribution would be extrapolated from one tail and the variance is meaningless
    if not (p[sel].min() <= 0.75 and p[sel].max() >= 0.25):
        return nan
    x = np.log(k[sel])
    z = stats.norm.ppf(1 - p[sel])            # probit of the CDF, increasing in log K
    z = np.maximum.accumulate(z + np.arange(len(z)) * 1e-9)
    if np.ptp(z) <= 1e-6:
        return nan
    lo_slope = (z[1] - z[0]) / (x[1] - x[0]) if x[1] > x[0] else np.nan
    hi_slope = (z[-1] - z[-2]) / (x[-1] - x[-2]) if x[-1] > x[-2] else np.nan
    glob = np.polyfit(x, z, 1)[0]
    if not (glob > 0):
        return nan
    # a flat outer segment (e.g. after pooling stale prices) would imply an infinitely fat tail:
    # tails may be at most 4 times wider than the ladder as a whole
    lo_slope = max(lo_slope, glob / 4) if np.isfinite(lo_slope) else glob
    hi_slope = max(hi_slope, glob / 4) if np.isfinite(hi_slope) else glob
    xl = max(x[0] - (z[0] + 8) / lo_slope, x[0] - 2.0)    # where the extrapolated probit reaches -8
    xh = min(x[-1] + (8 - z[-1]) / hi_slope, x[-1] + 2.0)  # ... and +8 (capped at e^2 of the outer strikes)
    grid = np.linspace(xl, xh, n_grid)
    zg = np.interp(grid, x, z)
    zg = np.where(grid < x[0], z[0] + lo_slope * (grid - x[0]), zg)
    zg = np.where(grid > x[-1], z[-1] + hi_slope * (grid - x[-1]), zg)
    G = stats.norm.cdf(zg)                       # P(S_T <= K)
    lnF = float(np.interp(0.0, zg, grid))        # median of the implied distribution
    K = np.exp(grid)
    F = np.exp(lnF)
    integrand = np.where(grid >= lnF, (1 - G) * (1 / F - 1 / K), G * (1 / K - 1 / F)) * K   # dK = K dx
    full = 2 * np.trapezoid(integrand, grid)
    inside = (grid >= x[0]) & (grid <= x[-1])
    trunc = 2 * np.trapezoid(integrand[inside], grid[inside]) if inside.sum() > 2 else np.nan
    return {"mfiv": float(full), "mfiv_trunc": float(trunc), "n": int(sel.sum()), "lnF": lnF}


def fit_snapshot(strikes: pd.DataFrame) -> dict:
    """strikes: columns strike, p (mid or last price), optional spread.
    Returns the model-free BJN variance (main ladder predictor), the lognormal and Student-t fits, and
    diagnostics (number of strikes, monotonicity violations)."""
    g = strikes.sort_values("strike")
    w = None
    if "spread" in g and g["spread"].notna().all():
        w = 1 / np.maximum(g["spread"].to_numpy(float), 0.001)
    pr = g["p"].to_numpy(float)
    c = fit_lognormal(g["strike"], pr, w)
    t = fit_student(g["strike"], pr, w)
    mf = mfiv(g["strike"], pr)
    return {"mkt_mfiv": mf["mfiv"], "mkt_mfiv_trunc": mf["mfiv_trunc"], "mkt_mfiv_n": mf["n"],
            "mkt_m": c["m"], "mkt_s": c["s"], "mkt_n": c["n"], "mkt_rmse": c["rmse"], "mkt_band": c["band"],
            "mkt_var_c": c["s"] ** 2 if np.isfinite(c["s"]) else np.nan,
            "mkt_t_m": t["m"], "mkt_t_s": t["s"], "mkt_nu": t["nu"], "mkt_var_t": t["var"], "mkt_t_n": t["n"],
            "mkt_nonmono": int(np.sum(np.diff(pr) > 1e-9))}
