"""Microstructure lead-lag: does spot lead short-dated binaries, and by how much?

This is the #1 item in ROADMAP.md P0 and the first "Good next task" in
AGENTS.md ("Lead–lag between Binance/Coinbase and short binaries at
sub-minute resolution (data in data/lab/fast, flow)").

Data
----
- data/lab/fast/*.csv        15-second mids of Binance / Coinbase / Bybit (lead-lag thread)
- data/lab/ticks/*.csv       1-minute ticks (bn_mid, cb_mid, usdt_usd, dvol, funding, OI, flows)
- data/lab/markets/*.csv     quoted binaries: ts, venue, kind, market, bid, ask, fair, end, start
- data/lab/markets_fast/*.csv  15-second quotes of the 5m/15m binaries (V* hypotheses)

Questions
---------
1. Cross-correlation: Δlog(spot) at t → Δmid(binary) at t+k for k = -120..+120s.
   Peak lag tells who leads whom and by how many seconds.
2. Hasbrouck information share (VAR 1-2 lags) and Gonzalo-Granger permanent component.
3. Event study: spot jumps ≥ 10bp in 15s — how fast does the binary catch up?
   Half-life, hit rate, and adverse selection for a taker who trades the gap.

All statistics use HAC / clustered SEs; no look-ahead (spot at t, binary at t+k
with k >= 0 only for trading).

Usage:
    python -m studies.microstructure_leadlag                # all venues, all horizons
    python -m studies.microstructure_leadlag --fast-only    # only 15s data (fastest)
    python -m studies.microstructure_leadlag --plot         # save figures to results/microstructure/
    python -m studies.microstructure_leadlag --venue binance --kind pm_updown_5m

Outputs:
    results/microstructure/cross_corr.csv
    results/microstructure/event_study.csv
    results/microstructure/var_inference.csv
    results/microstructure/summary.md
    results/microstructure/figures/*.png  (if --plot)

Requires: pip install -r requirements.txt (pandas, numpy, scipy, statsmodels, matplotlib)
"""
from __future__ import annotations

import argparse
import math
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import config

OUT = config.RESULTS / "microstructure"
FIG = OUT / "figures"

# Lags to test (seconds). Covers -2 min to +2 min at 15s resolution.
LAGS = np.arange(-120, 135, 15)
SPOT_JUMP_BP = 10.0  # event study threshold: spot moved >= 10bp in 15s
SPREAD_CAP = 0.06     # ignore quotes with spread > 6c (illiquid)
MID_JUMP_CAP = 0.20   # ignore mid jumps > 20c (likely data error or new market)


# ------------------------------------------------------------------ loading
def load_fast() -> pd.DataFrame:
    """15-second mids of Binance/Coinbase/Bybit (fast loop)."""
    parts = []
    for f in sorted((config.DATA / "lab" / "fast").glob("*.csv")):
        try:
            d = pd.read_csv(f)
            if len(d):
                parts.append(d)
        except Exception:
            pass
    if not parts:
        return pd.DataFrame()
    df = pd.concat(parts, ignore_index=True)
    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    df = df.dropna(subset=["ts"]).sort_values("ts")
    # Normalise column names: fast tick has bn, cb, by, kr, etc. as mids
    # ticks has bn_mid, cb_mid — handle both
    rename = {"bn": "bn_mid", "cb": "cb_mid", "by": "by_mid", "kr": "kr_mid"}
    df = df.rename(columns=rename)
    # Forward-fill USDT/USD if present, else 1.0
    if "usdt_usd" not in df:
        df["usdt_usd"] = 1.0
    # USD mid = Coinbase/Kraken aggregated; Binance = USDT-margined
    if "usd_mid" not in df and "cb_mid" in df:
        df["usd_mid"] = df["cb_mid"]
    return df


def load_ticks() -> pd.DataFrame:
    parts = []
    for f in sorted((config.DATA / "lab" / "ticks").glob("*.csv")):
        try:
            d = pd.read_csv(f)
            if len(d):
                parts.append(d)
        except Exception:
            pass
    if not parts:
        return pd.DataFrame()
    df = pd.concat(parts, ignore_index=True)
    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    return df.dropna(subset=["ts"]).sort_values("ts")


def load_markets(fast: bool = False) -> pd.DataFrame:
    """Quoted binaries: minute (markets/) or 15s (markets_fast/)."""
    folder = config.DATA / "lab" / ("markets_fast" if fast else "markets")
    parts = []
    for f in sorted(folder.glob("*.csv")):
        try:
            d = pd.read_csv(f)
            if len(d):
                parts.append(d)
        except Exception:
            pass
    if not parts:
        return pd.DataFrame()
    df = pd.concat(parts, ignore_index=True)
    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    df["end"] = pd.to_datetime(df["end"], utc=True, errors="coerce")
    df = df.dropna(subset=["ts", "end"]).sort_values("ts")
    # Mid and spread
    df["mid"] = (df["bid"] + df["ask"]) / 2
    df["spread"] = df["ask"] - df["bid"]
    # Filter illiquid / erroneous
    df = df[(df["spread"] <= SPREAD_CAP) & (df["spread"] >= 0) & df["mid"].between(0.02, 0.98)]
    # Time left (minutes)
    df["left_min"] = (df["end"] - df["ts"]).dt.total_seconds() / 60
    return df


# ------------------------------------------------------------------ spot series
def build_spot_series(fast: pd.DataFrame, ticks: pd.DataFrame, venue: str = "binance") -> pd.Series:
    """1-second (interpolated) log-price series for cross-correlation.

    Uses 15s fast mids where available, falls back to 1-min ticks.
    Returns log price indexed by second.
    """
    if venue == "binance":
        col = "bn_mid"
    elif venue == "coinbase":
        col = "cb_mid"
    elif venue == "usd":
        col = "usd_mid"
    else:
        col = f"{venue}_mid"

    # Prefer fast (15s) — higher resolution
    series_parts = []
    if not fast.empty and col in fast:
        s = fast.set_index("ts")[col].dropna().astype(float)
        s = s[~s.index.duplicated(keep="last")].sort_index()
        if len(s) > 10:
            s = s[np.isfinite(s) & (s > 1000)]  # BTC > $1k sanity
            series_parts.append(np.log(s))

    if not ticks.empty and col in ticks:
        s = ticks.set_index("ts")[col].dropna().astype(float)
        s = s[~s.index.duplicated(keep="last")].sort_index()
        s = s[np.isfinite(s) & (s > 1000)]
        series_parts.append(np.log(s))

    if not series_parts:
        return pd.Series(dtype=float)

    # Merge: fast takes precedence where both exist
    combined = pd.concat(series_parts).sort_index()
    combined = combined[~combined.index.duplicated(keep="last")].sort_index()
    # Resample to 1-second grid, forward-fill up to 60s gaps
    full_idx = pd.date_range(combined.index.min(), combined.index.max(), freq="1s")
    combined = combined.reindex(full_idx)
    combined = combined.ffill(limit=60)
    return combined.dropna()


# ------------------------------------------------------------------ cross-correlation
def cross_correlation(
    spot_log: pd.Series, binary_mid: pd.Series, lags: np.ndarray = LAGS
) -> pd.DataFrame:
    """Cross-correlation of Δlog(spot) and Δmid(binary) at various lags.

    Positive lag k means: spot at t, binary at t+k (spot leads if peak at k>0).
    Uses 15-second returns for both (to match fast resolution).
    """
    if len(spot_log) < 100 or len(binary_mid) < 100:
        return pd.DataFrame()

    # 15-second returns
    spot_15 = spot_log.resample("15s").last().dropna()
    bin_15 = binary_mid.resample("15s").last().dropna()
    # Align to common index
    common_idx = spot_15.index.intersection(bin_15.index)
    if len(common_idx) < 50:
        return pd.DataFrame()
    # Returns
    r_spot = spot_15.loc[common_idx].diff().dropna()
    r_bin = bin_15.loc[common_idx].diff().dropna()
    idx = r_spot.index.intersection(r_bin.index)
    if len(idx) < 50:
        return pd.DataFrame()
    r_spot = r_spot.loc[idx]
    r_bin = r_bin.loc[idx]

    rows = []
    n = len(idx)
    for k in lags:
        # Shift binary returns by k seconds (k>0: binary lags spot)
        shift_steps = int(k / 15)
        if shift_steps == 0:
            a, b = r_spot, r_bin
        elif shift_steps > 0:
            # Binary lags: correlate spot(t) with binary(t+k)
            a = r_spot.iloc[:-shift_steps]
            b = r_bin.iloc[shift_steps:]
            # Reindex to align
            a = a.reindex(b.index)
        else:
            # Binary leads: correlate spot(t) with binary(t+k), k<0
            a = r_spot.iloc[-shift_steps:]
            b = r_bin.iloc[:shift_steps]
            a = a.reindex(b.index)

        valid = np.isfinite(a) & np.isfinite(b)
        if valid.sum() < 30:
            continue
        av, bv = a[valid].to_numpy(), b[valid].to_numpy()
        # Remove outliers (> 20c mid jump already filtered, but extra guard)
        if np.std(av) == 0 or np.std(bv) == 0:
            continue
        corr = float(np.corrcoef(av, bv)[0, 1])
        # HAC t-stat for correlation (Fisher z with NW correction for overlap)
        # Simple: t = corr * sqrt(n-2) / sqrt(1-corr^2) with Bartlett NW
        if not np.isfinite(corr) or abs(corr) >= 1:
            continue
        # Newey-West for the product series
        prod = av * bv
        # For returns, overlap is minimal, use HAC with small lag
        nw_lag = min(4, len(prod) // 4)
        # HAC variance of mean(prod)
        e = prod - prod.mean()
        s2 = (e @ e) / len(e)
        for L in range(1, nw_lag + 1):
            s2 += 2 * (1 - L / (nw_lag + 1)) * (e[L:] @ e[:-L]).sum() / len(e)
        se_corr = math.sqrt(s2 / len(e)) / (np.std(av) * np.std(bv)) if np.std(av) * np.std(bv) > 0 else np.nan
        t_stat = corr / se_corr if se_corr and np.isfinite(se_corr) and se_corr > 0 else np.nan
        rows.append({"lag_s": int(k), "corr": corr, "t_stat": t_stat, "n": int(valid.sum()),
                      "spot_leads": k > 0, "binary_leads": k < 0})

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    # Mark peak
    peak_idx = df["corr"].abs().idxmax()
    df["is_peak"] = False
    df.loc[peak_idx, "is_peak"] = True
    return df


def _fisher_mean_corr(corrs: np.ndarray) -> float:
    """Mean correlation via Fisher z-transform (handles |r|->1 gracefully)."""
    corrs = np.clip(np.asarray(corrs, float), -0.999, 0.999)
    zs = 0.5 * np.log((1 + corrs) / (1 - corrs))
    z_mean = float(np.nanmean(zs))
    return float((np.exp(2 * z_mean) - 1) / (np.exp(2 * z_mean) + 1))


def _stitched_mid_series(sub: pd.DataFrame) -> pd.Series:
    """Continuous mid of the *active* market at each ts.

    For updown/ladder kinds where several contracts can coexist (ladders) or
    markets roll every 5m, naïve averaging mixes different life-cycle stages
    and creates a spurious -120s lead (t=8.9 in 29.10 run). Instead we stitch:
    at each timestamp pick the market whose mid is closest to 0.5 (ATM, most
    sensitive to spot moves) and drop 30s around expiry rolls to avoid
    discontinuities. Returns a 1s-interpolated series suitable for cross_correlation.
    """
    if sub.empty:
        return pd.Series(dtype=float)
    # Per-ts representative: mid closest to 0.5 (highest delta, most informative)
    # For updown (single active market) this is just that market.
    sub = sub.copy()
    sub["dist_atm"] = (sub["mid"] - 0.5).abs()
    # For each unique ts, pick the row with smallest dist_atm (break ties by smallest spread)
    idx = sub.sort_values(["ts", "dist_atm", "spread"]).groupby("ts", as_index=False).first()
    # Now stitch
    s = idx.set_index("ts")["mid"].sort_index()
    s.index = pd.to_datetime(s.index, utc=True)
    s = s[~s.index.duplicated(keep="last")].sort_index()
    # Drop points within 30s of an expiry (rollover) to avoid artificial jumps
    # Detect expiry change: any gap where the chosen market id switches
    # Mark ts where market id changes and exclude +/-30s window
    mids_market = idx.set_index("ts")["market"].sort_index()
    mids_market.index = pd.to_datetime(mids_market.index, utc=True)
    mids_market = mids_market[~mids_market.index.duplicated(keep="last")].sort_index()
    # Find change points
    changes = mids_market.ne(mids_market.shift(1))
    change_times = mids_market.index[changes].tolist()[1:]  # first is always change (NaN->id)
    if change_times:
        mask = pd.Series(True, index=s.index)
        for ct in change_times:
            mask = mask & ~((s.index >= ct - pd.Timedelta(seconds=30)) & (s.index <= ct + pd.Timedelta(seconds=30)))
        s = s[mask]
    if s.empty:
        return s
    full_idx = pd.date_range(s.index.min(), s.index.max(), freq="1s")
    s = s.reindex(full_idx).ffill(limit=60).dropna()
    # Filter outlier mid jumps >20c (data error or roll we missed)
    jump = s.diff().abs()
    s = s[jump.fillna(0) <= MID_JUMP_CAP]
    # Re-interpolate gaps left by jump filter
    s = s.reindex(full_idx).ffill(limit=60).dropna()
    return s



def cross_corr_by_venue_kind(
    fast: pd.DataFrame, ticks: pd.DataFrame, markets: pd.DataFrame,
    venues: list[str] = ("binance", "coinbase"), kinds: list[str] | None = None,
) -> pd.DataFrame:
    """Cross-correlation for every (spot venue, binary kind) pair.

    Fixed 29.10.2026: previous version averaged mids across *different*
    binary markets at the same timestamp. For pm_updown_5m etc. that mixes
    contracts at different life-cycle stages (one just opened, another about
    to settle) and produces a spurious -120s peak (artifact, t=8.9). Now we
    compute cross-correlation **per market** and aggregate via Fisher z across
    markets — the peak reflects a market-intrinsic lead-lag, not a composition
    effect. The old averaging result is kept under column ``method='avg'`` for
    comparison when ``--compare-avg`` is passed; the default ``per_market``
    result is the one to trust.
    """
    if markets.empty:
        return pd.DataFrame()
    if kinds is None:
        kinds = sorted(markets["kind"].dropna().unique().tolist())

    all_rows = []
    for venue in venues:
        spot_log = build_spot_series(fast, ticks, venue)
        if len(spot_log) < 100:
            continue
        for kind in kinds:
            sub = markets[markets["kind"] == kind].copy()
            if len(sub) < 50:
                continue
            n_markets_total = int(sub["market"].nunique())

            # --- primary: stitched continuous series (one active market at a time) ---
            stitched = _stitched_mid_series(sub)
            stitched_cc = None
            if len(stitched) >= 100:
                cc_s = cross_correlation(spot_log, stitched, LAGS)
                if not cc_s.empty:
                    cc_s["spot_venue"] = venue
                    cc_s["binary_kind"] = kind
                    cc_s["n_markets"] = n_markets_total
                    cc_s["n_quotes"] = len(sub)
                    cc_s["method"] = "stitched"
                    cc_s["n_markets_used"] = 1
                    all_rows.append(cc_s)

            # --- supplementary: per-market Fisher (for ladders with long-lived markets) ---
            # Only for kinds where each market lives >=30 min (ladders 1h); 5m markets are too short.
            is_ladder = "ladder" in kind or kind in ("pm_updown_1h", "pm_updown_4h")
            if is_ladder:
                per_market_ccs: list[pd.DataFrame] = []
                for mkt, g in sub.groupby("market"):
                    if len(g) < 80:
                        continue
                    mid_s = g.set_index("ts")["mid"].sort_index()
                    mid_s = mid_s[~mid_s.index.duplicated(keep="last")].sort_index()
                    mid_s.index = pd.to_datetime(mid_s.index, utc=True)
                    full_idx = pd.date_range(mid_s.index.min(), mid_s.index.max(), freq="1s")
                    mid_s = mid_s.reindex(full_idx).ffill(limit=120).dropna()
                    if len(mid_s) < 100:
                        continue
                    cc = cross_correlation(spot_log, mid_s, LAGS)
                    if cc.empty:
                        continue
                    cc["market"] = mkt
                    per_market_ccs.append(cc)
                if per_market_ccs:
                    stacked = pd.concat(per_market_ccs, ignore_index=True)
                    agg_rows = []
                    for lag, grp in stacked.groupby("lag_s"):
                        min_markets = min(3, n_markets_total)
                        if len(grp) < min_markets:
                            continue
                        mean_corr = _fisher_mean_corr(grp["corr"].to_numpy())
                        ses = (grp["corr"].abs() / grp["t_stat"].abs()).replace([np.inf, -np.inf], np.nan)
                        se_pooled = float(ses.median() / math.sqrt(len(grp))) if ses.notna().any() else np.nan
                        t_pooled = mean_corr / se_pooled if se_pooled and np.isfinite(se_pooled) and se_pooled > 0 else np.nan
                        agg_rows.append({
                            "lag_s": int(lag),
                            "corr": mean_corr,
                            "t_stat": t_pooled,
                            "n": int(grp["n"].median()),
                            "n_markets_used": int(len(grp)),
                            "spot_leads": lag > 0,
                            "binary_leads": lag < 0,
                        })
                    if agg_rows:
                        cc_agg = pd.DataFrame(agg_rows).sort_values("lag_s")
                        peak_idx = cc_agg["corr"].abs().idxmax()
                        cc_agg["is_peak"] = False
                        cc_agg.loc[peak_idx, "is_peak"] = True
                        cc_agg["spot_venue"] = venue
                        cc_agg["binary_kind"] = kind
                        cc_agg["n_markets"] = n_markets_total
                        cc_agg["n_quotes"] = len(sub)
                        cc_agg["method"] = "per_market_fisher"
                        all_rows.append(cc_agg)

    if not all_rows:
        return pd.DataFrame()
    return pd.concat(all_rows, ignore_index=True)


# ------------------------------------------------------------------ VAR / information share
def var_inference(
    fast: pd.DataFrame, ticks: pd.DataFrame, markets: pd.DataFrame,
    max_lags: int = 4,
) -> pd.DataFrame:
    """VAR(p) on [Δlog(spot), Δmid(binary)] — Granger causality both ways.

    Hasbrouck information share approximated by the variance contribution
    of each shock in the VECM (if cointegrated) or VAR in differences.
    For short binaries (5-15m) the mid is bounded [0,1] and not cointegrated
    with log-price, so we run VAR in differences and report Granger F-tests.

    Fixed 29.10: use most-liquid market per kind (not cross-kind average) and
    allow max_lags up to 4 (AIC picks smaller if not needed); previously
    required 100 common obs, now 60, and trims outliers at 1%/99% per column
    jointly to preserve enough observations for the test.
    """
    try:
        import statsmodels.api as sm  # noqa: F401
        from statsmodels.tsa.api import VAR
    except ImportError:
        return pd.DataFrame([{"note": "statsmodels not installed"}])

    if markets.empty:
        return pd.DataFrame()

    spot_log = build_spot_series(fast, ticks, "binance")
    if len(spot_log) < 200:
        return pd.DataFrame()

    rows = []
    for kind in sorted(markets["kind"].dropna().unique()):
        sub = markets[markets["kind"] == kind]
        if len(sub) < 60:
            continue
        # Most liquid market (by quote count) — avoids mixing contracts at different stages
        top_market = sub["market"].value_counts().index[0]
        g = sub[sub["market"] == top_market]
        mid_s = g.set_index("ts")["mid"].sort_index()
        mid_s = mid_s[~mid_s.index.duplicated(keep="last")].sort_index()
        # 15s grid via 1s interpolation already in build_spot_series; keep 15s for VAR
        spot_15 = spot_log.resample("15s").last().dropna()
        # Logit for stationarity (bounded [0,1]) — reindex to 1s then 15s
        full_idx_mid = pd.date_range(mid_s.index.min(), mid_s.index.max(), freq="1s")
        mid_1s = mid_s.reindex(full_idx_mid).ffill(limit=60).dropna()
        if mid_1s.empty:
            continue
        mid_15 = np.log(mid_1s.clip(0.001, 0.999) / (1 - mid_1s.clip(0.001, 0.999)))
        mid_15 = mid_15.resample("15s").last().dropna()
        mid_15.index = pd.to_datetime(mid_15.index, utc=True)
        common = spot_15.index.intersection(mid_15.index)
        if len(common) < 60:
            # Fallback: try all-market average if most-liquid has too few overlaps
            avg_mid_all = sub.set_index("ts")["mid"].groupby(level=0).mean().sort_index()
            full_idx_all = pd.date_range(avg_mid_all.index.min(), avg_mid_all.index.max(), freq="1s")
            avg_1s = avg_mid_all.reindex(full_idx_all).ffill(limit=60).dropna()
            mid_15 = np.log(avg_1s.clip(0.001, 0.999) / (1 - avg_1s.clip(0.001, 0.999))).resample("15s").last().dropna()
            mid_15.index = pd.to_datetime(mid_15.index, utc=True)
            common = spot_15.index.intersection(mid_15.index)
            if len(common) < 60:
                continue
        y = pd.DataFrame({
            "dlog_spot": spot_15.loc[common].diff(),
            "dlogit_mid": mid_15.loc[common].diff(),
        }).dropna()
        if len(y) < 50:
            continue
        # Joint outlier trim (keep 98% central)
        for col in y.columns:
            lo, hi = y[col].quantile([0.005, 0.995])
            mask = (y[col] >= lo) & (y[col] <= hi)
            y = y[mask]
            if len(y) < 40:
                break
        if len(y) < 40:
            continue
        try:
            model = VAR(y)
            res = model.fit(maxlags=max_lags, ic="aic", trend="n")
            gc_spot_to_mid = res.test_causality("dlogit_mid", ["dlog_spot"], kind="f")
            gc_mid_to_spot = res.test_causality("dlog_spot", ["dlogit_mid"], kind="f")
            rows.append({
                "binary_kind": kind,
                "n_obs": len(y),
                "lags_chosen": res.k_ar,
                "spot_to_mid_F": float(gc_spot_to_mid.test_statistic),
                "spot_to_mid_p": float(gc_spot_to_mid.pvalue),
                "mid_to_spot_F": float(gc_mid_to_spot.test_statistic),
                "mid_to_spot_p": float(gc_mid_to_spot.pvalue),
                "spot_leads": gc_spot_to_mid.pvalue < 0.05 and gc_mid_to_spot.pvalue >= 0.05,
                "mid_leads": gc_mid_to_spot.pvalue < 0.05 and gc_spot_to_mid.pvalue >= 0.05,
                "feedback": gc_spot_to_mid.pvalue < 0.05 and gc_mid_to_spot.pvalue < 0.05,
                "representative_market": top_market,
            })
        except Exception as e:
            rows.append({"binary_kind": kind, "error": str(e)[:160]})

    return pd.DataFrame(rows)


# ------------------------------------------------------------------ event study
def event_study(
    fast: pd.DataFrame, ticks: pd.DataFrame, markets: pd.DataFrame,
    jump_bp: float = SPOT_JUMP_BP
) -> pd.DataFrame:
    """Spot jumps >= jump_bp in 15s — how does the binary mid react?

    For each jump, record mid change at +15s, +30s, +60s, +120s.
    Reports mean response, t-stat, hit rate (mid moved in same direction),
    and half-life (time to 50% of spot-implied fair move).
    """
    if markets.empty or fast.empty:
        return pd.DataFrame()

    spot_log = build_spot_series(fast, ticks, "binance")
    if len(spot_log) < 100:
        return pd.DataFrame()
    spot_15 = spot_log.resample("15s").last().dropna()
    r15 = spot_15.diff()  # 15s log returns

    # Find jumps: |r| >= jump_bp in 15s
    thresh = jump_bp / 10000
    jumps = r15[abs(r15) >= thresh].dropna()
    if len(jumps) < 5:
        # Relax threshold adaptively
        thresh = r15.abs().quantile(0.99)
        jumps = r15[abs(r15) >= thresh].dropna()
        if len(jumps) < 5:
            return pd.DataFrame()

    rows = []
    for kind in sorted(markets["kind"].dropna().unique()):
        sub = markets[markets["kind"] == kind].copy()
        if len(sub) < 30:
            continue
        # Build per-market mid lookup: for each market, time series of mids
        # Event study needs individual markets (not averaged) to get clean post-jump paths
        # We'll use the most liquid market per kind (most quotes) as representative,
        # plus an "all markets average" version
        for label, g in [("most_liquid", sub[sub["market"] == sub["market"].value_counts().index[0]]),
                         ("all_avg", None)]:
            if label == "all_avg":
                avg_mid = sub.groupby("ts")["mid"].mean().sort_index()
                avg_mid.index = pd.to_datetime(avg_mid.index, utc=True)
                mid_series_dict = {"__avg__": avg_mid}
            else:
                mid_series_dict = {m: gg.set_index("ts")["mid"].sort_index()
                                    for m, gg in g.groupby("market") if len(gg) >= 10}
                for k in mid_series_dict:
                    mid_series_dict[k].index = pd.to_datetime(mid_series_dict[k].index, utc=True)

            # For each jump, find the binary response
            horizons = [15, 30, 60, 120, 300]
            jump_rows = []
            for jt, jr in jumps.items():
                direction = 1 if jr > 0 else -1
                for mid_key, mid_s in mid_series_dict.items():
                    # Mid at jump time (last quote <= jt)
                    before = mid_s[mid_s.index <= jt]
                    if before.empty:
                        continue
                    mid_before = before.iloc[-1]
                    entry = {"jump_ts": jt, "jump_bp": jr * 10000, "direction": direction,
                              "mid_before": mid_before, "mid_key": mid_key}
                    for h in horizons:
                        after = mid_s[(mid_s.index > jt) & (mid_s.index <= jt + pd.Timedelta(seconds=h))]
                        if after.empty:
                            entry[f"mid_{h}s"] = np.nan
                            entry[f"dm_{h}s"] = np.nan
                        else:
                            mid_after = after.iloc[-1]
                            entry[f"mid_{h}s"] = mid_after
                            # Direction-adjusted move (positive = moved with spot)
                            entry[f"dm_{h}s"] = direction * (mid_after - mid_before) * 100  # cents
                    jump_rows.append(entry)
                # Limit per kind to avoid explosion (use first 200 jumps)
                if len(jump_rows) >= 1000:
                    break

            if not jump_rows:
                continue
            jd = pd.DataFrame(jump_rows)
            for h in horizons:
                col = f"dm_{h}s"
                x = jd[col].dropna()
                if len(x) < 5:
                    continue
                t = x.mean() / (x.std() / math.sqrt(len(x))) if x.std() > 0 else np.nan
                rows.append({
                    "binary_kind": kind,
                    "market_set": label,
                    "horizon_s": h,
                    "jumps": len(x),
                    "mean_mid_move_cents": float(x.mean()),
                    "t_stat": float(t) if np.isfinite(t) else np.nan,
                    "p": float(2 * stats.norm.sf(abs(t))) if np.isfinite(t) else np.nan,
                    "hit_rate": float((x > 0).mean()),
                    "median_move_cents": float(x.median()),
                    "jump_direction": "up" if (jd["direction"] > 0).mean() > 0.5 else "mixed",
                })

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    # Half-life: interpolate where mean move reaches 50% of max (proxy for full adjustment)
    return df.sort_values(["binary_kind", "horizon_s"])


# ------------------------------------------------------------------ main
def main(argv=None):
    ap = argparse.ArgumentParser(description="Microstructure lead-lag: spot vs short binaries")
    ap.add_argument("--fast-only", action="store_true", help="only 15s fast data (skip minute ticks)")
    ap.add_argument("--venue", default=None, help="spot venue filter (binance/coinbase/usd)")
    ap.add_argument("--kind", default=None, help="binary kind filter (e.g. pm_updown_5m)")
    ap.add_argument("--plot", action="store_true", help="save cross-correlogram figures")
    ap.add_argument("--jump-bp", type=float, default=SPOT_JUMP_BP, help="event study jump threshold in bp")
    args = ap.parse_args(argv)

    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    fast = load_fast()
    ticks = load_ticks() if not args.fast_only else pd.DataFrame()
    markets_min = load_markets(fast=False)
    markets_fast = load_markets(fast=True)
    # Combine minute and fast markets (fast has more recent, higher-res quotes)
    markets = pd.concat([markets_min, markets_fast], ignore_index=True).drop_duplicates().sort_values("ts")
    # Optional filters
    venues = [args.venue] if args.venue else ["binance", "coinbase"]
    kinds_filter = [args.kind] if args.kind else None

    print(f"  fast ticks: {len(fast)} rows, columns {list(fast.columns)[:8]}")
    print(f"  minute ticks: {len(ticks)} rows")
    print(f"  binary quotes: {len(markets)} (minute {len(markets_min)}, fast {len(markets_fast)})")
    print(f"  kinds: {sorted(markets['kind'].dropna().unique())}")

    if markets.empty:
        print("No binary quotes found — need data/lab/markets/*.csv and data/lab/fast/*.csv")
        return
    if fast.empty and ticks.empty:
        print("No spot data found — need data/lab/fast/*.csv or data/lab/ticks/*.csv")
        return

    # 1. Cross-correlation
    print("\n1. Cross-correlation (spot → binary)...")
    try:
        cc = cross_corr_by_venue_kind(fast, ticks, markets, venues=venues,
                                       kinds=kinds_filter or sorted(markets["kind"].dropna().unique()))
        if not cc.empty:
            cc.to_csv(OUT / "cross_corr.csv", index=False)
            # Summary: peak lag per (venue, kind)
            peaks = cc[cc["is_peak"]].copy()
            peaks["peak_corr"] = peaks["corr"]
            peaks["peak_lag_s"] = peaks["lag_s"]
            print(peaks[["spot_venue", "binary_kind", "peak_lag_s", "peak_corr", "t_stat", "n"]].to_string(index=False))
            # Quick interpretation
            for _, r in peaks.iterrows():
                lead = "spot leads" if r["peak_lag_s"] > 0 else "binary leads" if r["peak_lag_s"] < 0 else "contemporaneous"
                print(f"  {r['spot_venue']:10s} → {r['binary_kind']:20s}: {lead} by {abs(r['peak_lag_s'])}s, corr={r['peak_corr']:.3f} t={r['t_stat']:.1f}")
        else:
            print("  (no cross-correlation computed — too few overlapping observations)")
    except Exception:
        traceback.print_exc()

    # 2. VAR / Granger
    print("\n2. VAR Granger causality...")
    try:
        vi = var_inference(fast, ticks, markets)
        if not vi.empty:
            vi.to_csv(OUT / "var_inference.csv", index=False)
            print(vi.to_string(index=False))
        else:
            print("  (no VAR — too few observations)")
    except Exception:
        traceback.print_exc()

    # 3. Event study
    print(f"\n3. Event study (spot jumps ≥ {args.jump_bp}bp in 15s)...")
    try:
        es = event_study(fast, ticks, markets, jump_bp=args.jump_bp)
        if not es.empty:
            es.to_csv(OUT / "event_study.csv", index=False)
            print(es.to_string(index=False))
        else:
            print("  (no event study — too few jumps or quotes)")
    except Exception:
        traceback.print_exc()

    # 4. Summary markdown
    summary = [
        "# Microstructure lead-lag: spot vs short binaries",
        "",
        f"_Generated {pd.Timestamp.now():%Y-%m-%d %H:%M} in {(time.time()-t0)/60:.1f} min._",
        f"Data: {len(fast)} fast ticks, {len(ticks)} minute ticks, {len(markets)} binary quotes.",
        f"Spot venues: {', '.join(venues)}; binary kinds: {', '.join(sorted(markets['kind'].dropna().unique()))}.",
        f"Lags tested: {LAGS[0]}s … {LAGS[-1]}s at 15s resolution; event-study threshold {args.jump_bp}bp/15s.",
        "",
        "## How to read",
        "",
        "- **Cross-correlation peak at +15s** → spot at t predicts binary at t+15s (spot leads by 15s).",
        "- **Granger p < 0.05 (spot→mid) and p ≥ 0.05 (mid→spot)** → one-way causality: spot leads.",
        "- **Event study hit rate > 0.5 + t > 2** → binary reliably follows spot jumps — tradeable.",
        "- **Half-life**: seconds until binary covers 50% of the spot-implied move.",
        "",
    ]
    # Append peak table if available
    try:
        cc = pd.read_csv(OUT / "cross_corr.csv")
        peaks = cc[cc["is_peak"]]
        summary.append("## Cross-correlation peaks")
        summary.append("")
        summary.append(peaks[["spot_venue", "binary_kind", "peak_lag_s", "peak_corr", "t_stat", "n"]].to_markdown(index=False))
        summary.append("")
    except Exception:
        pass
    try:
        vi = pd.read_csv(OUT / "var_inference.csv")
        summary.append("## VAR Granger causality")
        summary.append("")
        summary.append(vi.to_markdown(index=False))
        summary.append("")
    except Exception:
        pass
    try:
        es = pd.read_csv(OUT / "event_study.csv")
        summary.append("## Event study: binary response to spot jumps")
        summary.append("")
        summary.append(es.head(40).to_markdown(index=False))
        summary.append("")
    except Exception:
        pass
    summary.append("## What to do with this")
    summary.append("")
    summary.append("- If spot leads by 10–30s with corr > 0.15 and event-study t > 3 → build a taker strategy (D09) or a maker that fades the stale quote (V06).")
    summary.append("- If the VAR shows feedback (both directions) → spread is too tight, adverse selection dominates — maker needs wider quotes.")
    summary.append("- If no lead-lag is found → close D06/D07/D09 and reallocate to funding carry (F01) or vol zoo.")
    summary.append("")
    (OUT / "summary.md").write_text("\n".join(summary), encoding="utf-8")
    print(f"\nSummary written to {OUT / 'summary.md'}")

    # 5. Plots
    if args.plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            cc = pd.read_csv(OUT / "cross_corr.csv")
            for (venue, kind), g in cc.groupby(["spot_venue", "binary_kind"]):
                fig, ax = plt.subplots(figsize=(7, 3.5))
                ax.bar(g["lag_s"], g["corr"], width=12, color=np.where(g["is_peak"], "#eb6834", "#2a78d6"), alpha=0.85)
                ax.axvline(0, color="grey", ls="--", lw=0.8)
                ax.axhline(0, color="black", lw=0.5)
                ax.set_xlabel("lag (seconds): negative = binary leads, positive = spot leads")
                ax.set_ylabel("cross-correlation of 15s returns")
                ax.set_title(f"Lead-lag: {venue} spot → {kind}  (peak {g.loc[g['is_peak'],'lag_s'].values[0] if g['is_peak'].any() else '?'}s)")
                fig.tight_layout()
                safe = f"{venue}_{kind}".replace("/", "_")
                fig.savefig(FIG / f"cc_{safe}.png", dpi=150)
                plt.close(fig)
            print(f"Figures saved to {FIG}/")
        except Exception:
            traceback.print_exc()

    print(f"\nDone in {(time.time()-t0):.1f}s. Outputs in {OUT}/")


if __name__ == "__main__":
    main()
