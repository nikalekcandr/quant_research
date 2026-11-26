"""BTCUSDT perpetual futures: 1-minute candles and the premium index (basis) — the missing input for carry.

Why this download exists
    The wing book trades Kalshi binaries on BRTI. The second sleeve (funding carry, `studies/carry_portfolio.py`)
    needs the *perp leg*: without a perp price the sleeve can only be scored on the funding cash flow, not on its
    actual P&L, and the basis risk stays unmeasured. The same file also gives a 1-minute BTC price for hours where
    the spot archive has not been refreshed yet (a perp close is within a few bp of spot, the basis column says how
    far), which is what the November diagnostic currently approximates from settlement values.

What is downloaded
    data/raw/perp/BTCUSDT-1m-YYYY-MM.zip        monthly USDT-M klines (close, volume)
    data/raw/perp/BTCUSDT-1m-YYYY-MM-DD.zip     daily klines for the current month
    data/raw/perp/recent_klines.csv.gz          minutes not yet in the archives (REST /fapi/v1/klines)
    data/raw/perp/recent_premium.csv.gz         the same minutes from /fapi/v1/premiumIndexKlines
                                                (close = (mark - index) / index, i.e. the basis in fractions)

    python -m src.download.perp                 download / extend everything above
    python -m src.download.perp --interim       also rebuild data/interim/perp.parquet

Archives: https://data.binance.vision/data/futures/um/monthly/klines/BTCUSDT/1m/
REST:     https://fapi.binance.com/fapi/v1/klines, /fapi/v1/premiumIndexKlines
"""
from __future__ import annotations

import argparse
import io
import os
import zipfile

import pandas as pd

import config
from src.utils import Http, ensure, log, ms, utcnow

OUT_DIR = config.RAW / "perp"
INTERIM = config.INTERIM / "perp.parquet"
VISION = "https://data.binance.vision/data/futures/um"
HOSTS = ["https://fapi.binance.com", "https://fapi1.binance.com", "https://fapi2.binance.com"]
COLS = ["open_time", "open", "high", "low", "close", "volume", "close_time", "quote_volume", "trades",
        "taker_base", "taker_quote", "ignore"]
_GOOD: list[str] = []


def _api(http: Http, path: str, params: dict):
    """GET a futures endpoint, trying every fapi host (one host answering 451 does not stop the download)."""
    errs = []
    for host in _GOOD + [h for h in HOSTS if h not in _GOOD]:
        try:
            res = http.get(host + path, params)
            if res is not None:
                _GOOD[:] = [host]
                return res
        except Exception as e:                                   # blocked / unreachable
            errs.append(f"{host.split('//')[1]}: {str(e)[:60]}")
    raise RuntimeError("all fapi hosts failed: " + "; ".join(errs))


def _ts(x: pd.Series) -> pd.Series:
    """Microseconds (archives since 2025) or milliseconds (REST) since epoch -> UTC."""
    v = pd.to_numeric(x).astype("int64")
    v = v.where(v <= 10 ** 14, v // 1000)
    return pd.to_datetime(v, unit="ms", utc=True)


def parse_zip(content: bytes) -> pd.DataFrame:
    """One archive file -> open_time, close, volume (tolerates the header row some files carry)."""
    with zipfile.ZipFile(io.BytesIO(content)) as z:
        df = pd.read_csv(z.open(z.namelist()[0]), header=None)
    if not str(df.iloc[0, 0]).isdigit():
        df = df.iloc[1:]
    df = df.iloc[:, :12]
    df.columns = COLS
    df["open_time"] = _ts(df["open_time"])
    return df[["open_time", "close", "volume"]].astype({"close": float, "volume": float})


def parse_klines(rows: list) -> pd.DataFrame:
    """REST kline payload -> open_time, close, volume."""
    if not rows:
        return pd.DataFrame(columns=["open_time", "close", "volume"])
    df = pd.DataFrame(rows, columns=COLS[:len(rows[0])])
    df["open_time"] = _ts(df["open_time"])
    return df[["open_time", "close", "volume"]].astype({"close": float, "volume": float})


def parse_premium(rows: list) -> pd.DataFrame:
    """REST premiumIndexKlines payload -> open_time, premium (close of the premium index, fractions)."""
    if not rows:
        return pd.DataFrame(columns=["open_time", "premium"])
    df = pd.DataFrame(rows, columns=COLS[:len(rows[0])])
    df["open_time"] = _ts(df["open_time"])
    df["premium"] = pd.to_numeric(df["close"]).astype(float)
    return df[["open_time", "premium"]]


def _months_through(since: pd.Timestamp, now: pd.Timestamp) -> list[pd.Timestamp]:
    m = pd.Timestamp(year=since.year, month=since.month, day=1)
    this_month = pd.Timestamp(year=now.year, month=now.month, day=1)
    out = []
    while m < this_month:
        out.append(m)
        m += pd.offsets.MonthBegin(1)
    return out


def download(since: str | None = None, http: Http | None = None) -> dict:
    """Fill the perp archives and the recent tail. Returns a small report dict."""
    http = http or Http(min_interval=0.05)
    ensure(OUT_DIR)
    now = pd.Timestamp(utcnow())
    since = pd.Timestamp(since or config.BINANCE_START)
    months, days, missing = 0, 0, 0
    for month in _months_through(since, now):
        f = OUT_DIR / f"BTCUSDT-1m-{month:%Y-%m}.zip"
        if f.exists():
            continue
        try:
            content = http.get(f"{VISION}/monthly/klines/BTCUSDT/1m/BTCUSDT-1m-{month:%Y-%m}.zip", raw=True)
        except Exception as e:
            log(f"perp: month {month:%Y-%m} unavailable ({str(e)[:60]})")
            content = None
        if content is None:
            missing += 1            # not published yet (the month is still running): daily files below cover it
            continue
        f.write_bytes(content)
        months += 1
        log(f"perp: {month:%Y-%m}")
    d = pd.Timestamp(year=now.year, month=now.month, day=1)
    while d < now.tz_localize(None).normalize():
        f = OUT_DIR / f"BTCUSDT-1m-{d:%Y-%m-%d}.zip"
        if not f.exists():
            try:
                content = http.get(f"{VISION}/daily/klines/BTCUSDT/1m/BTCUSDT-1m-{d:%Y-%m-%d}.zip", raw=True)
            except Exception:
                content = None
            if content:
                f.write_bytes(content)
                days += 1
        d += pd.Timedelta(days=1)
    tail = _recent(http, now)
    return {"months": months, "days": days, "unpublished_months": missing, **tail}


def _recent(http: Http, now: pd.Timestamp) -> dict:
    """Minutes not yet in the archives: the last 7 days from REST, for both klines and the premium index."""
    start = ms(now.normalize() - pd.Timedelta(days=7))
    out = {}
    for name, path, extra in (("klines", "/fapi/v1/klines", {"interval": "1m", "symbol": "BTCUSDT"}),
                              ("premium", "/fapi/v1/premiumIndexKlines", {"interval": "1m", "symbol": "BTCUSDT"})):
        rows, t = [], start
        while True:
            res = _api(http, path, {**extra, "startTime": t, "limit": 1500})
            if not res:
                break
            rows.extend(res)
            t = int(res[-1][0]) + 60_000
            if len(res) < 1500:
                break
        if rows:
            pd.DataFrame(rows).to_csv(OUT_DIR / f"recent_{name}.csv.gz", index=False, header=False)
        out[name] = len(rows)
    return out


def build_interim() -> pd.DataFrame:
    """One row per minute: open_time, close, volume, premium (NaN where the premium index has no minute)."""
    parts = [parse_zip(f.read_bytes()) for f in sorted(OUT_DIR.glob("BTCUSDT-1m-*.zip"))]
    rec = OUT_DIR / "recent_klines.csv.gz"
    if rec.exists():
        parts.append(parse_klines(pd.read_csv(rec, header=None).values.tolist()))
    if not parts:
        raise RuntimeError(f"no perp candles in {OUT_DIR}: run `python -m src.download.perp` first")
    k = pd.concat(parts, ignore_index=True).drop_duplicates("open_time").sort_values("open_time")
    prem = OUT_DIR / "recent_premium.csv.gz"
    if prem.exists():
        p = parse_premium(pd.read_csv(prem, header=None).values.tolist())
        k = k.merge(p.drop_duplicates("open_time"), on="open_time", how="left")
    else:
        k["premium"] = float("nan")
    k = k.set_index("open_time").sort_index()
    tmp = INTERIM.with_suffix(".tmp.parquet")
    k.to_parquet(tmp)
    os.replace(tmp, INTERIM)
    log(f"perp: {len(k):,} minutes -> {INTERIM.relative_to(config.ROOT)}")
    return k


def main() -> None:
    ap = argparse.ArgumentParser(description="Binance BTCUSDT perpetual candles and basis")
    ap.add_argument("--since", default=None, help="first month to download (default config.BINANCE_START)")
    ap.add_argument("--interim", action="store_true", help="also rebuild data/interim/perp.parquet")
    a = ap.parse_args()
    report = download(a.since)
    log(f"perp: {report}")
    if a.interim:
        build_interim()


if __name__ == "__main__":
    main()
