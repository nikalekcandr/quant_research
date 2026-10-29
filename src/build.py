"""Assemble the forecasting panels: one row per forecast occasion, all features known at the forecast time t,
the target measured over the window [t, e) that ends at the contract's settlement.

Horizons (see config.HORIZONS):
  pm24h  t = 12:00 ET on day d-1, settlement T = 12:00 ET on day d, e = T + 1 min (Polymarket candle)
  pm6d   t = T - 144 h,                                           e = T + 1 min
  kx1h   t = T - 55 min (5 min after the hourly Kalshi market opens), e = T (top of every hour)
Occasions exist for every day/hour with Binance data, so time-series models have a long history.
Market features (ladder-implied variance) exist only where a ladder was trading.
"""
from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd

import config
from src.measure import ladder as L
from src.measure.realized import Minutes, seasonal_factor
from src.utils import noon_et_utc, norm_dt

TINY = 1e-10
H = pd.Timedelta(hours=1)


def occasions(horizon: str, start, end) -> pd.DataFrame:
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    if horizon in ("pm24h", "pm6d"):
        hrs = config.HORIZONS[horizon]["hours"]
        days = pd.date_range(start.tz_localize(None).normalize() if start.tzinfo else start.normalize(),
                             end.tz_localize(None).normalize() if end.tzinfo else end.normalize(), freq="D")
        T = pd.DatetimeIndex([noon_et_utc(d) for d in days])
        df = pd.DataFrame({"T": T})
        df["t"] = df["T"] - pd.Timedelta(hours=hrs)
        df["e"] = df["T"] + pd.Timedelta(minutes=1)
    elif horizon == "kx1h":
        T = pd.date_range(start.ceil("h"), end.floor("h"), freq="h", tz="UTC" if start.tzinfo is None else None)
        df = pd.DataFrame({"T": T})
        df["t"] = df["T"] - pd.Timedelta(minutes=55)
        df["e"] = df["T"]
    else:
        raise ValueError(horizon)
    df["horizon"] = horizon
    df["tau_h"] = (df["e"] - df["t"]).dt.total_seconds() / 3600
    return norm_dt(df)


def _log(x):
    return np.log(np.maximum(np.asarray(x, float), TINY))


def realized_features(occ: pd.DataFrame, m: Minutes, profiles: dict) -> pd.DataFrame:
    horizon = occ["horizon"].iloc[0]
    t, e = occ["t"], occ["e"]
    if horizon == "kx1h":
        d, w, mo, aux = H, 24 * H, 168 * H, 4 * H   # lag windows: 1h, 24h, 7d; jumps/semivariance over 4h
        scale_w, scale_m = 24, 168
    else:
        d, w, mo, aux = 24 * H, 7 * 24 * H, 30 * 24 * H, 24 * H
        scale_w, scale_m = 7, 30
    if horizon == "kx1h":
        # Kalshi settles on the average of 60 one-second BRTI prints before T. The variance of
        # ln(average / S_t) is IV[t, T-60s) + IV(last minute)/3, so the last minute counts one third.
        main, last = m.window(t, e - pd.Timedelta(minutes=1)), m.window(e - pd.Timedelta(minutes=1), e)
        tgt = main.copy()
        for c in ("rv", "rs_pos", "rs_neg", "bv", "rv5"):
            tgt[c] = main[c] + last[c] / 3
    else:
        # Polymarket settles on the close of the Binance candle that opens at T: e = T + 1 min
        tgt = m.window(t, e)
    wd = m.window(t - d, t)
    ww = m.window(t - w, t)
    wm = m.window(t - mo, t)
    wa = m.window(t - aux, t)
    wn = m.window(t - (e - t), t)  # the window of the same length just before t (naive forecast)
    f = pd.DataFrame(index=occ.index)
    f["rv"] = tgt["rv"].to_numpy()
    f["rv5"] = tgt["rv5"].to_numpy()
    f["y"] = _log(f["rv"])
    # no target when the window is missing, flat, or mostly forward-filled (exchange outage)
    bad = f["rv"].isna() | (f["rv"] <= 0) | (tgt["filled_share"].to_numpy() > 0.2)
    f.loc[bad.to_numpy(), ["rv", "y"]] = np.nan
    f["l_rv_d"] = _log(wd["rv"])
    f["l_rv_w"] = _log(ww["rv"] / scale_w)
    f["l_rv_m"] = _log(wm["rv"] / scale_m)
    f["l_rv_naive"] = _log(wn["rv"])
    f["l_rs_pos"] = _log(wa["rs_pos"])
    f["l_rs_neg"] = _log(wa["rs_neg"])
    f["jump_share"] = np.clip((wa["rv"] - wa["bv"]) / wa["rv"], 0, 1).to_numpy()
    f["harq"] = (np.sqrt(wd["rq"]) / wd["rv"]).to_numpy() * f["l_rv_d"].to_numpy()
    for c in ["l_rv_d", "l_rv_w", "l_rv_m", "l_rv_naive", "l_rs_pos", "l_rs_neg", "jump_share", "harq"]:
        f.loc[wm["rv"].isna().to_numpy() | wd["rv"].isna().to_numpy(), c] = np.nan
    f["l_F"] = np.log(seasonal_factor(profiles, t, e))
    f["l_F_naive"] = np.log(seasonal_factor(profiles, t - (e - t), t))
    f["S_t"] = m.price_at(t)
    f["S_e"] = m.price_at(e)
    return f


def calendar_features(occ: pd.DataFrame) -> pd.DataFrame:
    cal = pd.read_csv(config.MANUAL / "us_macro_calendar.csv")
    rel = pd.to_datetime(cal["date"] + " " + cal["time_et"]).dt.tz_localize("America/New_York").dt.tz_convert("UTC")
    rel = np.sort(rel.to_numpy(dtype="datetime64[ns]"))
    t = occ["t"].dt.tz_localize(None).to_numpy(dtype="datetime64[ns]")
    e = occ["e"].dt.tz_localize(None).to_numpy(dtype="datetime64[ns]")
    n_rel = np.searchsorted(rel, e) - np.searchsorted(rel, t)
    f = pd.DataFrame(index=occ.index)
    f["macro"] = (n_rel > 0).astype(float)
    # share of the window that falls on Saturday/Sunday (UTC)
    shares = []
    for a, b in zip(occ["t"], occ["e"]):
        mins = pd.date_range(a, b, freq="15min", inclusive="left")
        shares.append(float(np.mean(mins.dayofweek >= 5)) if len(mins) else np.nan)
    f["weekend"] = shares
    return f


def implied_features(occ: pd.DataFrame, dvol: pd.DataFrame | None, deribit_iv: pd.DataFrame | None) -> pd.DataFrame:
    horizon = occ["horizon"].iloc[0]
    f = pd.DataFrame(index=occ.index)
    f["dvol"] = np.nan
    f["l_dvol_var"] = np.nan
    f["l_iv_var"] = np.nan
    if dvol is not None and len(dvol):
        dv = dvol[["ts", "close"]].copy()
        dv["known"] = dv["ts"] + H   # an hourly candle is complete one hour after its timestamp
        dv = dv.sort_values("known")
        q = occ[["t"]].reset_index().sort_values("t")
        mrg = pd.merge_asof(q, dv[["known", "close"]], left_on="t", right_on="known", direction="backward",
                            tolerance=pd.Timedelta(hours=3))
        mrg = mrg.set_index("index").reindex(occ.index)
        f["dvol"] = mrg["close"].to_numpy()
        f["l_dvol_var"] = _log((f["dvol"] / 100) ** 2 * occ["tau_h"] / (365 * 24))
        f.loc[f["dvol"].isna(), "l_dvol_var"] = np.nan
    if deribit_iv is not None and len(deribit_iv) and f"iv_var_{horizon}" in deribit_iv:
        iv = deribit_iv.set_index("snap_ts")[f"iv_var_{horizon}"]
        vals = iv.reindex(pd.DatetimeIndex(occ["t"])).to_numpy()
        f["l_iv_var"] = np.where(np.isfinite(vals), _log(vals), np.nan)
    return f


# ------------------------------------------------------------------ ladders
def polymarket_strikes(lad: pd.DataFrame, px: pd.DataFrame, horizon: str) -> pd.DataFrame:
    """Price of every strike at the forecast time t = settlement - horizon (last price within 30 min)."""
    if lad.empty:
        return pd.DataFrame()
    hrs = config.HORIZONS[horizon]["hours"]
    s = lad.copy()
    s["t"] = s["settle_ts"] - pd.Timedelta(hours=hrs)
    s = s[s["t"] >= s["open_ts"] + pd.Timedelta(minutes=15)]
    if s.empty:
        return pd.DataFrame()
    pxs = px.sort_values("ts")
    s = s.sort_values("t")
    m = pd.merge_asof(s, pxs, left_on="t", right_on="ts", by="yes_token", direction="backward",
                      tolerance=pd.Timedelta(minutes=config.PM_MAX_STALENESS_MIN))
    m = m.dropna(subset=["p"])
    m["horizon"] = horizon
    m["spread"] = np.nan
    m["T"] = m["settle_ts"]
    return m[["horizon", "event_slug", "T", "t", "strike", "p", "spread", "outcome", "yes_token"]]


def economical(g: pd.DataFrame) -> pd.DataFrame:
    """The strike design of the archived Kalshi download: every second strike ($200 apart) within +-1.5% of
    the at-the-money strike. Applied to every snapshot, recent ones included, so that the ladder variance is
    computed the same way over the whole sample (a change of design would shift its level)."""
    k = (g["strike"] + 0.01).round()
    ref = float(k.iloc[(g["p"] - 0.5).abs().argmin()])
    # v6: before 18 March 2026 Kalshi listed strikes $250-$1000 apart; "every second strike" then means every
    # strike. Ladders with $100 spacing (all of July-October) are selected exactly as before.
    ku = np.sort(k.unique())
    native = float(np.median(np.diff(ku))) if len(ku) > 1 else 100.0
    step_ok = (k % 200 == 0) if native < 200 else pd.Series(True, index=g.index)
    return g[((k / ref - 1).abs() <= 0.015) & step_ok]


QUOTE_MAX_AGE = pd.Timedelta(minutes=5)


def kalshi_strikes(kx: pd.DataFrame, uniform: bool = True) -> pd.DataFrame:
    """Quotes of every strike at t = T - 55 min: the last candle that ends at or before t and is at most
    5 minutes old. For ordinary hourly events t = open + 5 min. The 17:00 ET event of every day is a daily
    market (opens 25 h before settlement, 169 h on Fridays for the weekly one); a quote taken at its open
    would be a day or a week older than the panel's t and would leak the future into every comparison that
    uses the price at t. Such events only enter when candles near T - 55 min were downloaded.
    uniform=True keeps only the economical design (see `economical`)."""
    if kx is None or kx.empty:
        return pd.DataFrame()
    k = kx.copy()
    k["t"] = k["settle_ts"] - pd.Timedelta(minutes=55)
    k = k[(k["candle_end"] <= k["t"]) & (k["candle_end"] > k["t"] - QUOTE_MAX_AGE)]
    k = k.sort_values("candle_end").groupby("ticker").tail(1)
    ok = (k["yes_bid"] > 0) & (k["yes_ask"] < 1) & (k["yes_ask"] > k["yes_bid"])
    k = k[ok].copy()
    k["p"] = (k["yes_bid"] + k["yes_ask"]) / 2
    k["spread"] = k["yes_ask"] - k["yes_bid"]
    k["horizon"] = "kx1h"
    k["T"] = k["settle_ts"]
    k = k.rename(columns={"yes_bid": "bid", "yes_ask": "ask"})
    k = k[["horizon", "event", "T", "t", "strike", "p", "bid", "ask", "spread", "outcome", "expiration_value"]]
    if uniform and len(k):
        k = pd.concat([economical(g) for _, g in k.groupby("T")], ignore_index=True)
    return k


def ladder_snapshots(strikes: pd.DataFrame) -> pd.DataFrame:
    if strikes.empty:
        return pd.DataFrame()
    rows = []
    for (h, T), g in strikes.groupby(["horizon", "T"]):
        g = g.sort_values("strike")
        r = {"horizon": h, "T": T, "t": g["t"].iloc[0], "n_strikes": len(g)}
        r.update(L.fit_snapshot(g))
        rows.append(r)
    return pd.DataFrame(rows)


def market_features(occ: pd.DataFrame, snaps: pd.DataFrame) -> pd.DataFrame:
    cols = ["mkt_mfiv", "mkt_mfiv_trunc", "mkt_mfiv_n", "mkt_nonmono",
            "mkt_var_c", "mkt_var_t", "mkt_nu", "mkt_n", "mkt_rmse", "mkt_m", "mkt_s"]
    f = pd.DataFrame(index=occ.index, columns=cols, dtype=float)
    if snaps is not None and len(snaps):
        sn = snaps[snaps["horizon"] == occ["horizon"].iloc[0]].set_index("T")
        sn = sn[~sn.index.duplicated()]
        # a ladder counts only if it was observed exactly at the occasion's forecast time t
        same_t = sn["t"].reindex(pd.DatetimeIndex(occ["T"])).to_numpy() == occ["t"].to_numpy()
        for c in cols:
            if c in sn:
                f[c] = np.where(same_t, sn[c].reindex(pd.DatetimeIndex(occ["T"])).to_numpy(), np.nan)
    f["l_mkt_mfiv"] = np.where(f["mkt_mfiv"] > 0, _log(f["mkt_mfiv"]), np.nan)        # main ladder predictor (BJN)
    f["l_mkt_mfiv_trunc"] = np.where(f["mkt_mfiv_trunc"] > 0, _log(f["mkt_mfiv_trunc"]), np.nan)
    f["l_mkt_var_c"] = np.where(f["mkt_var_c"] > 0, _log(f["mkt_var_c"]), np.nan)
    f["l_mkt_var_t"] = np.where(f["mkt_var_t"] > 0, _log(f["mkt_var_t"]), np.nan)
    return f


def build_panel(horizon: str, m: Minutes, profiles: dict, dvol, deribit_iv, snaps, start=None, end=None) -> pd.DataFrame:
    start = start or (m.first + pd.Timedelta(days=32))
    end = end or m.last
    occ = occasions(horizon, start, end)
    occ = occ[occ["t"] >= m.first + pd.Timedelta(days=31)].reset_index(drop=True)
    parts = [occ, realized_features(occ, m, profiles), calendar_features(occ),
             implied_features(occ, dvol, deribit_iv), market_features(occ, snaps)]
    return pd.concat(parts, axis=1)
