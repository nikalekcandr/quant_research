"""Deribit: BTC option trades (with implied volatility), the DVOL index and live option-chain snapshots.

Verified endpoints (October 2026):
  history.deribit.com/api/v2/public/get_last_trades_by_currency_and_time   every trade since launch, field "iv"
  www.deribit.com/api/v2/public/get_volatility_index_data                  DVOL candles
  www.deribit.com/api/v2/public/get_book_summary_by_currency               live chain (bid/ask/mark/mark_iv)
Options expire at 08:00 UTC. Instrument names look like BTC-16OCT26-80000-C.
"""
from __future__ import annotations

from datetime import timedelta

import pandas as pd

import config
from src.utils import Http, ensure, exists_json, load_json, log, ms, noon_et_utc, norm_dt, save_json, utcnow

HIST = "https://history.deribit.com/api/v2/public"
LIVE = "https://www.deribit.com/api/v2/public"
RAW_DIR = config.RAW / "deribit"


def parse_instrument(name: str):
    """BTC-16OCT26-80000-C -> (expiry 08:00 UTC, strike, 'C')."""
    try:
        _, exp, k, cp = name.split("-")
        expiry = pd.Timestamp(pd.to_datetime(exp, format="%d%b%y")).tz_localize("UTC") + pd.Timedelta(hours=8)
        return expiry, float(k), cp
    except Exception:
        return None, None, None


def trades_window(http: Http, t0, t1) -> list[dict]:
    out, start = [], ms(t0)
    end = ms(t1)
    while True:
        res = http.get(f"{HIST}/get_last_trades_by_currency_and_time",
                       {"currency": "BTC", "kind": "option", "start_timestamp": start,
                        "end_timestamp": end, "count": 1000, "sorting": "asc"})
        tr = ((res or {}).get("result") or {}).get("trades", [])
        out.extend(tr)
        if not ((res or {}).get("result") or {}).get("has_more") or not tr:
            break
        start = tr[-1]["timestamp"] + 1
    return out


def snapshot_times(since: str, until=None) -> list[pd.Timestamp]:
    """One snapshot per day at 12:00 ET (the time of the 24h-ahead Polymarket forecast)."""
    d0 = pd.Timestamp(since).date()
    d1 = pd.Timestamp(until).date() if until else pd.Timestamp(utcnow()).date() + timedelta(days=7)
    out, d = [], d0
    while d <= d1:
        t = noon_et_utc(d)
        if t < pd.Timestamp(utcnow()):
            out.append(t)
        d += timedelta(days=1)
    return out


def download_trades(since: str = config.DERIBIT_START, until=None, http: Http | None = None):
    http = http or Http(min_interval=0.06)
    d_dir = ensure(RAW_DIR / "trades")
    times = snapshot_times(since, until)
    for i, t in enumerate(times):
        f = d_dir / f"{t:%Y%m%d%H%M}.json"
        if exists_json(f):
            continue
        tr = trades_window(http, t - pd.Timedelta(minutes=config.DERIBIT_WINDOW_MIN), t)
        save_json(tr, f)
        if i % 30 == 0:
            log(f"deribit trades: {t:%Y-%m-%d} ({len(tr)} trades)")
    log("deribit trades: finished")


def download_dvol(since: str = config.DVOL_START, http: Http | None = None):
    """Hourly DVOL candles (open, high, low, close in annualised %)."""
    http = http or Http(min_interval=0.06)
    rows, start, end = [], ms(since), ms(utcnow())
    step = 1000 * 3600 * 1000  # 1000 hourly points per request
    t = start
    while t < end:
        res = http.get(f"{LIVE}/get_volatility_index_data",
                       {"currency": "BTC", "start_timestamp": t, "end_timestamp": min(t + step, end), "resolution": 3600})
        data = ((res or {}).get("result") or {}).get("data", [])
        rows.extend(data)
        t += step
    df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close"]).drop_duplicates("ts")
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    ensure(RAW_DIR)
    df.sort_values("ts").to_csv(RAW_DIR / "dvol.csv.gz", index=False)
    log(f"dvol: {len(df)} hourly points")


def build_interim_trades() -> pd.DataFrame:
    rows = []
    for f in sorted((RAW_DIR / "trades").glob("*.json*")):
        snap = pd.Timestamp(pd.to_datetime(f.name[:12], format="%Y%m%d%H%M")).tz_localize("UTC")
        for tr in load_json(f):
            rows.append({"snap_ts": snap, "ts": tr["timestamp"], "instrument": tr["instrument_name"],
                         "iv": tr.get("iv"), "price": tr.get("price"), "index_price": tr.get("index_price"),
                         "amount": tr.get("amount")})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    parsed = df["instrument"].map(parse_instrument)
    df["expiry"] = [p[0] for p in parsed]
    df["strike"] = [p[1] for p in parsed]
    df["cp"] = [p[2] for p in parsed]
    return df.dropna(subset=["expiry", "iv"])


def load_dvol() -> pd.DataFrame:
    df = pd.read_csv(RAW_DIR / "dvol.csv.gz")
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    return norm_dt(df)


# ------------------------------------------------------------------ live snapshot
def snapshot_chain(http: Http, snap_ts) -> pd.DataFrame:
    res = http.get(f"{LIVE}/get_book_summary_by_currency", {"currency": "BTC", "kind": "option"})
    rows = []
    for x in (res or {}).get("result", []):
        rows.append({"snap_ts": snap_ts, "instrument": x.get("instrument_name"), "bid": x.get("bid_price"),
                     "ask": x.get("ask_price"), "mark": x.get("mark_price"), "mark_iv": x.get("mark_iv"),
                     "underlying": x.get("underlying_price"), "underlying_index": x.get("underlying_index"),
                     "open_interest": x.get("open_interest"), "volume": x.get("volume")})
    return pd.DataFrame(rows)
