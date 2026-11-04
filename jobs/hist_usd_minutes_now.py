"""BRTI study, 04.11.2026: 1-minute BTC-USD candles of CF BRTI constituent exchanges (the container cannot reach
them) + recent Kalshi KXBTCD settlement values + the lab's own ticks, packed for the analysis.
    Coinbase BTC-USD   /products/BTC-USD/candles  (300 per request)   2025-11-25 -> now
    Bitstamp btcusd    /api/v2/ohlc/btcusd/       (1000 per request)  2025-11-25 -> now
    Kraken XBTUSD      /0/public/OHLC             (last 720 minutes only — the API keeps no more)
    Kalshi KXBTCD      settled markets closed after 2026-10-26: one expiration_value per hour
Output: _export/brti_hist.tar.gz (cb.parquet, bs.parquet, kr.parquet, kx_settle.parquet, lab ticks)."""
import io
import sys
import tarfile
import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "raw" / "usd_minutes"
OUT.mkdir(parents=True, exist_ok=True)
S = requests.Session()
S.headers["User-Agent"] = "btcvol-research"
START = pd.Timestamp("2025-11-25", tz="UTC")
NOW = pd.Timestamp.now(tz="UTC").floor("min")


def get(url, params, tries=6):
    for i in range(tries):
        try:
            r = S.get(url, params=params, timeout=20)
            if r.status_code == 429:
                time.sleep(2 + 2 * i)
                continue
            r.raise_for_status()
            return r.json()
        except Exception as e:  # noqa: BLE001
            print("retry", url, params, str(e)[:120], flush=True)
            time.sleep(2 + 2 * i)
    return None


def coinbase():
    f = OUT / "cb.parquet"
    rows, t = [], START
    while t < NOW:
        e = min(t + pd.Timedelta(minutes=300), NOW)
        j = get("https://api.exchange.coinbase.com/products/BTC-USD/candles",
                {"granularity": 60, "start": t.isoformat(), "end": e.isoformat()})
        if j:
            rows += j            # [time, low, high, open, close, volume]
        t = e
        time.sleep(0.12)
        if len(rows) and len(rows) % 30000 < 300:
            print("coinbase", t, len(rows), flush=True)
    d = pd.DataFrame(rows, columns=["t", "l", "h", "o", "c", "v"]).drop_duplicates("t").sort_values("t")
    d["ts"] = pd.to_datetime(d.t, unit="s", utc=True)
    d.drop(columns="t").to_parquet(f)
    print("coinbase rows", len(d), flush=True)


def bitstamp():
    f = OUT / "bs.parquet"
    rows, t = [], int(START.timestamp())
    while t < NOW.timestamp():
        j = get("https://www.bitstamp.net/api/v2/ohlc/btcusd/", {"step": 60, "limit": 1000, "start": t})
        data = (j or {}).get("data", {}).get("ohlc", [])
        if not data:
            t += 60000
            continue
        rows += data
        t = int(data[-1]["timestamp"]) + 60
        time.sleep(0.15)
    d = pd.DataFrame(rows).drop_duplicates("timestamp")
    for c in ("open", "high", "low", "close", "volume"):
        d[c] = d[c].astype(float)
    d["ts"] = pd.to_datetime(d.timestamp.astype(int), unit="s", utc=True)
    d = d.rename(columns={"open": "o", "high": "h", "low": "l", "close": "c", "volume": "v"})[["ts", "o", "h", "l", "c", "v"]]
    d.sort_values("ts").to_parquet(f)
    print("bitstamp rows", len(d), flush=True)


def kraken():
    j = get("https://api.kraken.com/0/public/OHLC", {"pair": "XBTUSD", "interval": 1})
    res = (j or {}).get("result", {})
    k = [v for key, v in res.items() if key != "last"]
    if not k:
        return
    d = pd.DataFrame(k[0], columns=["t", "o", "h", "l", "c", "vwap", "v", "n"])
    for c in ("o", "h", "l", "c", "vwap", "v"):
        d[c] = d[c].astype(float)
    d["ts"] = pd.to_datetime(d.t, unit="s", utc=True)
    d.drop(columns="t").to_parquet(OUT / "kr.parquet")
    print("kraken rows", len(d), flush=True)


def kalshi():
    out, cursor = [], None
    for _ in range(200):
        j = get("https://api.elections.kalshi.com/trade-api/v2/markets",
                {"series_ticker": "KXBTCD", "status": "settled", "limit": 1000, "cursor": cursor,
                 "min_close_ts": int(pd.Timestamp("2026-10-26", tz="UTC").timestamp())})
        if not j:
            break
        for m in j.get("markets", []):
            out.append({"event": m.get("event_ticker"), "close_time": m.get("close_time"),
                        "expiration_value": m.get("expiration_value")})
        cursor = j.get("cursor")
        if not cursor:
            break
        time.sleep(0.2)
    d = pd.DataFrame(out).drop_duplicates("event")
    d.to_parquet(OUT / "kx_settle.parquet")
    print("kalshi settled hours", len(d), flush=True)


for fn in (kalshi, kraken, bitstamp, coinbase):
    try:
        fn()
    except Exception as e:  # noqa: BLE001
        print(fn.__name__, "failed:", e, flush=True)

exp = ROOT / "_export"
exp.mkdir(exist_ok=True)
with tarfile.open(exp / "brti_hist.tar.gz", "w:gz") as tar:
    for p in OUT.glob("*.parquet"):
        tar.add(p, arcname=f"usd_minutes/{p.name}")
    for p in sorted((ROOT / "data" / "lab" / "ticks").glob("*")):
        tar.add(p, arcname=f"ticks/{p.name}")
print("packed", (exp / "brti_hist.tar.gz").stat().st_size, "bytes", flush=True)
