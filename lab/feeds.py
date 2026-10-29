"""Market data for the lab: one wide 'tick' per minute from ~7 venues, and a 15-second fast tick.

Every source is independent: a failure (or a 403/451 block, paused for an hour) leaves its columns empty.
Slow-moving sources (futures positioning ratios, open interest, DVOL, USDT/USD) are refreshed every 5 minutes.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from lab.core import log, utcnow
from src.download import binance
from src.utils import Blocked

PAUSED: dict = {}
_SLOW_CACHE: dict = {"ts": None, "row": {}}

BN_F = ["https://fapi.binance.com", "https://fapi1.binance.com", "https://fapi2.binance.com"]


def _guard(name, fn, row: dict):
    now = time.time()
    if PAUSED.get(name, 0) > now:
        return
    try:
        out = fn()
        if out:
            row.update(out)
    except Blocked as e:
        PAUSED[name] = now + 3600
        log(f"feed {name} blocked for 1 h: {str(e)[:100]}")
    except Exception as e:
        PAUSED[name] = now + 120          # transient: retry in 2 minutes
        log(f"feed {name} failed: {str(e)[:140]}")


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return np.nan


def _fapi(http, path, params=None):
    err = None
    for h in BN_F:
        try:
            r = http.get(h + path, params)
            if r is not None:
                return r
        except Exception as e:
            err = e
    raise RuntimeError(f"binance futures: {err}")


# ------------------------------------------------------------------ the sources
def bn_spot(http):
    bt = binance.api_get(http, "/api/v3/ticker/bookTicker", {"symbol": "BTCUSDT"})
    dp = binance.api_get(http, "/api/v3/depth", {"symbol": "BTCUSDT", "limit": 20})
    bid, ask = _f(bt["bidPrice"]), _f(bt["askPrice"])
    mid = (bid + ask) / 2
    bq = sum(_f(q) for p, q in dp["bids"] if _f(p) > mid * 0.999)   # within 10 bp of the mid
    aq = sum(_f(q) for p, q in dp["asks"] if _f(p) < mid * 1.001)
    kl = binance.api_get(http, "/api/v3/klines", {"symbol": "BTCUSDT", "interval": "1m", "limit": 6})
    done = kl[:-1]                     # closed minutes only
    vol = sum(_f(k[5]) for k in done)
    tb = sum(_f(k[9]) for k in done)
    return {"bn_bid": bid, "bn_ask": ask, "bn_imb10": (bq - aq) / (bq + aq) if bq + aq > 0 else np.nan,
            "bn_vol5": vol, "bn_taker5": tb / vol if vol > 0 else np.nan}


def bn_perp(http):
    pi = _fapi(http, "/fapi/v1/premiumIndex", {"symbol": "BTCUSDT"})
    bt = _fapi(http, "/fapi/v1/ticker/bookTicker", {"symbol": "BTCUSDT"})
    return {"bf_bid": _f(bt["bidPrice"]), "bf_ask": _f(bt["askPrice"]), "bf_mark": _f(pi["markPrice"]),
            "bf_index": _f(pi["indexPrice"]), "bf_funding": _f(pi["lastFundingRate"]),
            "bf_next_funding": pd.to_datetime(int(pi["nextFundingTime"]), unit="ms", utc=True)}


def bn_perp_slow(http):
    oi = _fapi(http, "/fapi/v1/openInterest", {"symbol": "BTCUSDT"})
    out = {"bf_oi": _f(oi["openInterest"])}
    for key, path in (("bf_taker_ls", "/futures/data/takerlongshortRatio"),
                      ("bf_ls_acct", "/futures/data/globalLongShortAccountRatio"),
                      ("bf_ls_top", "/futures/data/topLongShortPositionRatio")):
        try:
            r = _fapi(http, path, {"symbol": "BTCUSDT", "period": "5m", "limit": 1})
            if r:
                out[key] = _f(r[-1].get("buySellRatio", r[-1].get("longShortRatio")))
        except Exception:
            pass
    return out


def bybit(http):
    r = http.get("https://api.bybit.com/v5/market/tickers", {"category": "linear", "symbol": "BTCUSDT"})
    t = r["result"]["list"][0]
    return {"by_bid": _f(t["bid1Price"]), "by_ask": _f(t["ask1Price"]), "by_mark": _f(t["markPrice"]),
            "by_index": _f(t["indexPrice"]), "by_funding": _f(t["fundingRate"]), "by_oi": _f(t["openInterest"]),
            "by_next_funding": pd.to_datetime(int(t["nextFundingTime"]), unit="ms", utc=True)}


def okx(http):
    t = http.get("https://www.okx.com/api/v5/market/ticker", {"instId": "BTC-USDT-SWAP"})["data"][0]
    return {"ok_bid": _f(t["bidPx"]), "ok_ask": _f(t["askPx"])}


def okx_slow(http):
    f = http.get("https://www.okx.com/api/v5/public/funding-rate", {"instId": "BTC-USDT-SWAP"})["data"][0]
    oi = http.get("https://www.okx.com/api/v5/public/open-interest", {"instType": "SWAP", "instId": "BTC-USDT-SWAP"})["data"][0]
    return {"ok_funding": _f(f["fundingRate"]), "ok_oi": _f(oi.get("oiCcy")),
            "ok_next_funding": pd.to_datetime(int(f["fundingTime"]), unit="ms", utc=True)}


def coinbase(http):
    t = http.get("https://api.exchange.coinbase.com/products/BTC-USD/ticker")
    return {"cb_bid": _f(t["bid"]), "cb_ask": _f(t["ask"])}


def coinbase_slow(http):
    t = http.get("https://api.exchange.coinbase.com/products/USDT-USD/ticker")
    return {"usdt_usd": (_f(t["bid"]) + _f(t["ask"])) / 2}


def kraken(http):
    r = http.get("https://api.kraken.com/0/public/Ticker", {"pair": "XBTUSD"})["result"]
    t = list(r.values())[0]
    return {"kr_bid": _f(t["b"][0]), "kr_ask": _f(t["a"][0])}


def deribit(http):
    r = http.get("https://www.deribit.com/api/v2/public/ticker", {"instrument_name": "BTC-PERPETUAL"})["result"]
    return {"db_bid": _f(r.get("best_bid_price")), "db_ask": _f(r.get("best_ask_price")),
            "db_index": _f(r.get("index_price")), "db_funding8h": _f(r.get("funding_8h"))}


def deribit_slow(http):
    end = int(time.time() * 1000)
    r = http.get("https://www.deribit.com/api/v2/public/get_volatility_index_data",
                 {"currency": "BTC", "resolution": "60", "start_timestamp": end - 3 * 3600 * 1000, "end_timestamp": end})
    data = (r.get("result") or {}).get("data") or []
    return {"dvol": _f(data[-1][4])} if data else {}


FAST_SOURCES = [("bn_spot", bn_spot), ("bn_perp", bn_perp), ("bybit", bybit), ("okx", okx), ("coinbase", coinbase),
                ("kraken", kraken), ("deribit", deribit)]
SLOW_SOURCES = [("bn_perp_slow", bn_perp_slow), ("okx_slow", okx_slow), ("coinbase_slow", coinbase_slow),
                ("deribit_slow", deribit_slow)]


def tick(http, now: pd.Timestamp | None = None) -> dict:
    now = now or utcnow()
    row = {"ts": now.floor("s")}
    for name, fn in FAST_SOURCES:
        _guard(name, lambda fn=fn: fn(http), row)
    if _SLOW_CACHE["ts"] is None or now - _SLOW_CACHE["ts"] >= pd.Timedelta(minutes=5):
        slow = {}
        for name, fn in SLOW_SOURCES:
            _guard(name, lambda fn=fn: fn(http), slow)
        _SLOW_CACHE["row"] = {**_SLOW_CACHE["row"], **slow}
        _SLOW_CACHE["ts"] = now
    row.update(_SLOW_CACHE["row"])
    for v in ("bn", "bf", "by", "ok", "cb", "kr", "db"):
        b, a = row.get(f"{v}_bid"), row.get(f"{v}_ask")
        if b is not None and a is not None and np.isfinite(b) and np.isfinite(a):
            row[f"{v}_mid"] = (b + a) / 2
    if "cb_mid" in row and "kr_mid" in row:
        row["usd_mid"] = (row["cb_mid"] + row["kr_mid"]) / 2       # proxy for the CF BRTI (Kalshi settlement)
    elif "cb_mid" in row:
        row["usd_mid"] = row["cb_mid"]
    return row


def fast_tick(http) -> dict:
    row = {"ts": utcnow()}
    for name, fn in (("bn_spot_fast", lambda: binance.api_get(http, "/api/v3/ticker/bookTicker", {"symbol": "BTCUSDT"})),
                     ("coinbase_fast", lambda: http.get("https://api.exchange.coinbase.com/products/BTC-USD/ticker")),
                     ("bybit_fast", lambda: http.get("https://api.bybit.com/v5/market/tickers",
                                                     {"category": "linear", "symbol": "BTCUSDT"}))):
        if PAUSED.get(name, 0) > time.time():
            continue
        try:
            r = fn()
            if name == "bn_spot_fast":
                row["bn"] = (_f(r["bidPrice"]) + _f(r["askPrice"])) / 2
            elif name == "coinbase_fast":
                row["cb"] = (_f(r["bid"]) + _f(r["ask"])) / 2
            else:
                t = r["result"]["list"][0]
                row["by"] = (_f(t["bid1Price"]) + _f(t["ask1Price"])) / 2
        except Blocked:
            PAUSED[name] = time.time() + 3600
        except Exception:
            PAUSED[name] = time.time() + 60
    return row
