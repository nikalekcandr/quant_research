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


def download(since: str = config.BINANCE_START, http: Http | None = None) -> dict:
    http = http or Http(min_interval=0.05)
    ensure(RAW_DIR)
    months_new, days_new = 0, 0
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
                months_new += 1
                log(f"binance: {month:%Y-%m}")
        month += pd.offsets.MonthBegin(1)
    days_new = _daily(http, this_month, now.tz_localize(None).normalize())
    rest = _recent(http)
    log(f"binance: {months_new} новых месяцев, {days_new} суточных файлов, {rest['rows']} минут из REST"
        + (f" ({rest['start']:%Y-%m-%d %H:%M} .. {rest['end']:%Y-%m-%d %H:%M})" if rest["rows"] else ""))
    return {"months": months_new, "days": days_new, **rest}


def _daily(http: Http, d0, d1) -> int:
    """Daily archives for a month the monthly file does not cover yet; returns how many were new."""
    written = 0
    d = pd.Timestamp(d0)
    while d < pd.Timestamp(d1):
        f = RAW_DIR / f"BTCUSDT-1m-{d:%Y-%m-%d}.zip"
        if not f.exists():
            content = http.get(f"{VISION}/daily/klines/BTCUSDT/1m/BTCUSDT-1m-{d:%Y-%m-%d}.zip", raw=True)
            if content:
                f.write_bytes(content)
                written += 1
        d += timedelta(days=1)
    return written


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


def _recent(http: Http) -> dict:
    """Minutes the archives do not cover yet, appended to `recent.csv.gz` — never replacing what is there.

    Starts after the last archived day and after the last minute we already hold, so it works both after a long
    pause (a month of missing minutes, fetched in 1000-minute chunks) and on a normal day (a few hundred rows).
    The old version always overwrote the file with the last 14 days and said nothing when the REST call failed,
    which is how `binance: ok за 0.0 мин` could hide a month-long hole.
    """
    now = pd.Timestamp(utcnow())
    rec = RAW_DIR / "recent.csv.gz"
    have = pd.DataFrame()
    if rec.exists():
        try:
            have = pd.read_csv(rec)
            have["open_time"] = _to_ts(have["open_time"].astype("int64"))
        except Exception:
            have = pd.DataFrame()
    start_ts = now.normalize() - pd.Timedelta(days=90)
    last = _last_archived_day(now)
    if last is not None:
        start_ts = max(start_ts, pd.Timestamp(last).tz_localize("UTC") + pd.Timedelta(days=1))
    if len(have):
        start_ts = max(start_ts, have["open_time"].max() + pd.Timedelta(minutes=1))
    start = ms(start_ts)
    rows = []
    while True:
        res = api_get(http, "/api/v3/klines", {"symbol": "BTCUSDT", "interval": "1m", "startTime": start,
                                              "limit": 1000})
        if not res:
            break
        rows.extend(res)
        if len(res) < 1000:
            break
        start = int(res[-1][0]) // 1000 * 1000 + 60_000     # archives may store microseconds
    out = {"rows": 0, "start": None, "end": None}
    if rows:
        new = pd.DataFrame(rows, columns=COLS)
        new["open_time"] = _to_ts(new["open_time"].astype("int64"))
        cols = ["open_time", "close", "volume"]
        merged = (pd.concat([have[cols] if len(have) else new[cols], new[cols]], ignore_index=True)
                  .drop_duplicates("open_time", keep="last").sort_values("open_time"))
        merged.to_csv(rec, index=False)
        out = {"rows": len(new), "start": new["open_time"].min(), "end": new["open_time"].max()}
    elif len(have):
        log(f"binance: REST не дал новых минут; recent.csv.gz не тронут "
            f"(последняя минута {have['open_time'].max():%Y-%m-%d %H:%M})")
    return out


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
    if df["filled"].any():
        hole = int(df["filled"].sum())
        log(f"binance: внимание — {hole} минут без данных, они заполнены предыдущей ценой "
            f"(ищите дыру в архивах: это делает реализованную волатильность ниже настоящей)")
    df["close"] = df["close"].ffill()
    df.index.name = "open_time"
    return df.reset_index()
