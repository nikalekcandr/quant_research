"""Binance BTC/USDT 1-minute candles: the price that settles Polymarket ladders and our realized-volatility source.

Archives: https://data.binance.vision/data/spot/monthly/klines/BTCUSDT/1m/BTCUSDT-1m-YYYY-MM.zip
          (daily files under .../daily/... for the current month)
Since 2025 the spot archives store timestamps in MICROseconds; earlier files use milliseconds.
The most recent minutes come from the REST API (api.binance.com/api/v3/klines).
"""
from __future__ import annotations

import io
import zipfile
from datetime import timedelta

import pandas as pd

import config
from src.utils import Http, ensure, log, ms, utcnow

VISION = "https://data.binance.vision/data/spot"
API = "https://api.binance.com/api/v3/klines"
RAW_DIR = config.RAW / "binance"
_GOOD_HOST: list = []   # the host that answered last time is tried first


def api_get(http: Http, path: str, params: dict | None = None):
    """GET a Binance REST endpoint, falling back across config.BINANCE_HOSTS (HTTP 451/403 from one host,
    e.g. api.binance.com from some networks, does not stop the download)."""
    hosts = _GOOD_HOST + [h for h in config.BINANCE_HOSTS if h not in _GOOD_HOST]
    errors = []
    for host in hosts:
        try:
            res = http.get(host + path, params)
            if res is not None:
                _GOOD_HOST[:] = [host]
                return res
        except Exception as e:  # blocked (403/451), unreachable, ...
            errors.append(f"{host.split('//')[1]}: {str(e)[:80]}")
    raise RuntimeError("all Binance hosts failed: " + "; ".join(errors))


COLS = ["open_time", "open", "high", "low", "close", "volume", "close_time", "quote_volume", "trades",
        "taker_base", "taker_quote", "ignore"]


def _to_ts(x: pd.Series) -> pd.Series:
    """Milliseconds or microseconds since epoch -> UTC timestamps."""
    x = pd.to_numeric(x).astype("int64")
    x = x.where(x <= 10 ** 14, x // 1000)
    return pd.to_datetime(x, unit="ms", utc=True)


def read_zip(content: bytes) -> pd.DataFrame:
    with zipfile.ZipFile(io.BytesIO(content)) as z:
        name = z.namelist()[0]
        df = pd.read_csv(z.open(name), header=None)
    if not str(df.iloc[0, 0]).isdigit():  # some files carry a header row
        df = df.iloc[1:]
    df = df.iloc[:, :12]
    df.columns = COLS
    df["open_time"] = _to_ts(df["open_time"].astype("int64"))
    return df[["open_time", "close", "volume"]].astype({"close": float, "volume": float})


def download(since: str = config.BINANCE_START, http: Http | None = None):
    http = http or Http(min_interval=0.05)
    ensure(RAW_DIR)
    start = pd.Timestamp(since)
    now = pd.Timestamp(utcnow())
    month = pd.Timestamp(year=start.year, month=start.month, day=1)
    this_month = pd.Timestamp(year=now.year, month=now.month, day=1)
    while month < this_month:
        f = RAW_DIR / f"BTCUSDT-1m-{month:%Y-%m}.zip"
        if not f.exists():
            content = http.get(f"{VISION}/monthly/klines/BTCUSDT/1m/BTCUSDT-1m-{month:%Y-%m}.zip", raw=True)
            if content is None:  # month archive not published yet: fall back to daily files
                _daily(http, month, month + pd.offsets.MonthBegin(1))
            else:
                f.write_bytes(content)
                log(f"binance: {month:%Y-%m}")
        month += pd.offsets.MonthBegin(1)
    _daily(http, this_month, now.tz_localize(None).normalize())
    _recent(http)


def _daily(http: Http, d0, d1):
    d = pd.Timestamp(d0)
    while d < pd.Timestamp(d1):
        f = RAW_DIR / f"BTCUSDT-1m-{d:%Y-%m-%d}.zip"
        if not f.exists():
            content = http.get(f"{VISION}/daily/klines/BTCUSDT/1m/BTCUSDT-1m-{d:%Y-%m-%d}.zip", raw=True)
            if content:
                f.write_bytes(content)
        d += timedelta(days=1)


def _last_archived_day(now: pd.Timestamp) -> pd.Timestamp | None:
    """Last calendar day (UTC) fully covered by a monthly or daily archive file."""
    last = None
    for f in RAW_DIR.glob("BTCUSDT-1m-*.zip"):
        tag = f.stem.replace("BTCUSDT-1m-", "")
        try:
            d = (pd.Timestamp(tag + "-01") + pd.offsets.MonthEnd(0)) if len(tag) == 7 else pd.Timestamp(tag)
        except ValueError:
            continue
        last = d if last is None or d > last else last
    return last


def _recent(http: Http):
    """Minutes not yet in the archives (today, and yesterday when its daily file is not published yet)
    from the REST API. Starts the day after the last archived day, at most 14 days back."""
    now = pd.Timestamp(utcnow())
    last = _last_archived_day(now)
    start_ts = now.normalize()
    if last is not None:
        start_ts = max(pd.Timestamp(last).tz_localize("UTC") + pd.Timedelta(days=1), now.normalize() - pd.Timedelta(days=14))
    start = ms(start_ts)
    rows = []
    while True:
        res = api_get(http, "/api/v3/klines", {"symbol": "BTCUSDT", "interval": "1m", "startTime": start, "limit": 1000})
        if not res:
            break
        rows.extend(res)
        if len(res) < 1000:
            break
        start = res[-1][0] + 60_000
    if rows:
        df = pd.DataFrame(rows, columns=COLS)
        df.to_csv(RAW_DIR / "recent.csv.gz", index=False)


def build_interim() -> pd.DataFrame:
    """One row per minute: open_time (UTC), close, volume. Duplicates removed, sorted."""
    parts = []
    for f in sorted(RAW_DIR.glob("BTCUSDT-1m-*.zip")):
        parts.append(read_zip(f.read_bytes()))
    rec = RAW_DIR / "recent.csv.gz"
    if rec.exists():
        r = pd.read_csv(rec)
        r["open_time"] = _to_ts(r["open_time"])
        parts.append(r[["open_time", "close", "volume"]])
    df = pd.concat(parts, ignore_index=True).drop_duplicates("open_time").sort_values("open_time")
    df = df.set_index("open_time")
    # reindex to a complete minute grid; gaps (exchange outages) are forward-filled and flagged
    full = pd.date_range(df.index[0], df.index[-1], freq="1min", tz="UTC")
    df = df.reindex(full)
    df["filled"] = df["close"].isna()
    df["close"] = df["close"].ffill()
    df.index.name = "open_time"
    return df.reset_index()
