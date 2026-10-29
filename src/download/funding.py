"""Perpetual-futures funding rates for BTC (research layer, hypothesis R11).

Binance USDT-M BTCUSDT, every 8 hours since 2019:
    https://fapi.binance.com/fapi/v1/fundingRate?symbol=BTCUSDT&startTime=..&limit=1000
Fallbacks if fapi is blocked from this network: the monthly archives on data.binance.vision, then Bybit
(/v5/market/funding/history, linear BTCUSDT) — same 8-hour grid, nearly identical rates.

    python -m src.download.funding            download / extend data/raw/funding/btc_funding.csv.gz
"""
from __future__ import annotations

import io
import zipfile

import pandas as pd

import config
from src.utils import Http, ensure, log, ms, utcnow

RAW_DIR = config.RAW / "funding"
OUT = RAW_DIR / "btc_funding.csv.gz"
FAPI = ["https://fapi.binance.com", "https://fapi1.binance.com", "https://fapi2.binance.com"]
VISION = "https://data.binance.vision/data/futures/um/monthly/fundingRate/BTCUSDT"
BYBIT = "https://api.bybit.com/v5/market/funding/history"


def _binance_api(http: Http, start: pd.Timestamp) -> pd.DataFrame:
    rows, t = [], ms(start)
    end = ms(pd.Timestamp(utcnow()))
    for host in FAPI:
        try:
            while t < end:
                res = http.get(host + "/fapi/v1/fundingRate", {"symbol": "BTCUSDT", "startTime": t, "limit": 1000})
                if not res:
                    break
                rows += [(int(r["fundingTime"]), float(r["fundingRate"]), "binance") for r in res]
                t = int(res[-1]["fundingTime"]) + 1
                if len(res) < 1000:
                    break
            if rows:
                return pd.DataFrame(rows, columns=["ts", "rate", "venue"])
        except Exception as e:
            log(f"funding: {host} failed ({str(e)[:80]})")
    return pd.DataFrame()


def _binance_vision(http: Http, start: pd.Timestamp) -> pd.DataFrame:
    out, misses = [], 0
    m = start.tz_convert(None).to_period("M").to_timestamp()
    while m < pd.Timestamp(utcnow()).tz_convert(None):
        try:
            content = http.get(f"{VISION}/BTCUSDT-fundingRate-{m:%Y-%m}.zip", raw=True)
        except Exception:
            content = None
            misses += 1
            if misses >= 3:          # blocked from this network: do not walk through every month
                break
        if content:
            with zipfile.ZipFile(io.BytesIO(content)) as z:
                d = pd.read_csv(z.open(z.namelist()[0]))
            d = d.rename(columns={"calc_time": "ts", "last_funding_rate": "rate"})
            out.append(pd.DataFrame({"ts": d["ts"].astype("int64"), "rate": d["rate"].astype(float),
                                     "venue": "binance"}))
        m += pd.offsets.MonthBegin(1)
    return pd.concat(out) if out else pd.DataFrame()


def _bybit(http: Http, start: pd.Timestamp) -> pd.DataFrame:
    rows, end = [], ms(pd.Timestamp(utcnow()))
    s0 = ms(start)
    while end > s0:
        res = http.get(BYBIT, {"category": "linear", "symbol": "BTCUSDT", "endTime": end, "limit": 200})
        lst = (res or {}).get("result", {}).get("list", [])
        if not lst:
            break
        rows += [(int(r["fundingRateTimestamp"]), float(r["fundingRate"]), "bybit") for r in lst]
        end = min(int(r["fundingRateTimestamp"]) for r in lst) - 1
    return pd.DataFrame(rows, columns=["ts", "rate", "venue"])


def download(since: str = config.BINANCE_START, http: Http | None = None) -> pd.DataFrame:
    http = http or Http(min_interval=0.12, retries=2)
    ensure(RAW_DIR)
    old = pd.read_csv(OUT) if OUT.exists() else pd.DataFrame(columns=["ts", "rate", "venue"])
    start = pd.Timestamp(since, tz="UTC")
    if len(old):
        start = max(start, pd.to_datetime(old["ts"].max(), unit="ms", utc=True) + pd.Timedelta(minutes=1))
    new = pd.DataFrame()
    for name, fn in (("binance fapi", _binance_api), ("binance archive", _binance_vision), ("bybit", _bybit)):
        try:
            new = fn(http, start)
        except Exception as e:
            log(f"funding: {name} failed ({str(e)[:100]})")
            new = pd.DataFrame()
        if len(new):
            log(f"funding: {len(new)} new rows from {name}")
            break
    df = pd.concat([old, new], ignore_index=True)
    if len(df):
        df["ts"] = df["ts"].astype("int64")
        # funding times are on the 8-hour grid; round away millisecond jitter
        df["ts"] = (df["ts"] // 60000) * 60000
        df = df.drop_duplicates("ts", keep="first").sort_values("ts")
        df.to_csv(OUT, index=False)
        log(f"funding: {len(df)} rows {pd.to_datetime(df.ts.min(), unit='ms', utc=True):%Y-%m-%d} .. "
            f"{pd.to_datetime(df.ts.max(), unit='ms', utc=True):%Y-%m-%d %H:%M}")
    return df


def load() -> pd.DataFrame | None:
    if not OUT.exists():
        return None
    d = pd.read_csv(OUT)
    d["ts"] = pd.to_datetime(d["ts"], unit="ms", utc=True)
    return d


if __name__ == "__main__":
    download()
