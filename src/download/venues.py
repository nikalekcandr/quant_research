"""Other venues and other Bitcoin markets, logged live next to the main ladders (order books cannot be
downloaded after the fact).

  kalshi_other      every open Kalshi market of a Bitcoin series other than KXBTCD / KXBTC (which live.py logs
                    already): daily/weekly/monthly/yearly "above", max/min, 15-minute up/down, ... — whatever
                    Kalshi lists; series are discovered from the API once a day
  pm_other          every open Polymarket Bitcoin market other than the daily "above" ladder: hourly and 15-minute
                    "Up or Down", "Bitcoin price on <date>" ranges, "What price will Bitcoin hit" barriers, ...
                    Best bid / ask from the gamma API (one paginated call instead of one book per market)
  binance_options   Binance European options on BTC (eapi): bid/ask, mark price and mark IV of every option
                    expiring within 10 days — a second options market next to Deribit
  limitless         Limitless Exchange (Base): only probed by discover_venues.py until its API is confirmed

Every function returns a DataFrame and raises on failure; live.py runs each in its own try block.
The classification of market titles is deliberately loose; the raw title is always kept.
"""
from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd

import config
from src.download.kalshi import _num as kx_num
from src.utils import Http, log, to_utc, utcnow

KALSHI = "https://api.elections.kalshi.com/trade-api/v2"
GAMMA = "https://gamma-api.polymarket.com"
EAPI = "https://eapi.binance.com"
OKX = "https://www.okx.com"
BYBIT = "https://api.bybit.com"
LIMITLESS = "https://api.limitless.exchange"
HORIZON_DAYS = 10          # markets settling later than this are not logged
SKIP_KALSHI = {"KXBTCD", "KXBTC"}
BTC_RE = re.compile(r"\b(bitcoin|btc)\b", re.I)


def _num(x):
    try:
        return None if x is None or x == "" else float(x)
    except (TypeError, ValueError):
        return None


# ------------------------------------------------------------------ Kalshi: discover Bitcoin series
def kalshi_btc_series(http: Http) -> list[str]:
    """Bitcoin series tickers. /series?category=Crypto first; if that endpoint does not answer, page through
    open events and collect the series tickers that contain BTC."""
    out = set()
    try:
        res = http.get(f"{KALSHI}/series", {"category": "Crypto"}) or {}
        for s in res.get("series") or []:
            tk = s.get("ticker") or ""
            if "BTC" in tk.upper() or BTC_RE.search(s.get("title") or ""):
                out.add(tk)
    except Exception as e:
        log(f"kalshi /series failed ({e!r}); falling back to open events")
    if not out:
        cursor = None
        for _ in range(40):
            params = {"status": "open", "limit": 200}
            if cursor:
                params["cursor"] = cursor
            res = http.get(f"{KALSHI}/events", params) or {}
            for ev in res.get("events") or []:
                st = ev.get("series_ticker") or ""
                if "BTC" in st.upper():
                    out.add(st)
            cursor = res.get("cursor")
            if not cursor:
                break
    return sorted(out)


def kalshi_other(http: Http, snap_ts, series: list[str]) -> pd.DataFrame:
    """Open markets of every Bitcoin series. Most of the ~70 series have nothing open in the next days, so the
    series that do are cached for 24 hours (data/live/kalshi_active.json): ~5 requests per snapshot instead of 70,
    which also keeps us under Kalshi's rate limit while a Kalshi download runs in parallel."""
    now = pd.Timestamp(utcnow())
    cache = config.LIVE / "kalshi_active.json"
    active = None
    try:
        c = json.loads(cache.read_text(encoding="utf-8"))
        if now - pd.Timestamp(c["ts"]) < pd.Timedelta(hours=24):
            active = c["series"]
    except Exception:
        pass
    todo = active if active is not None else [x for x in series if x not in SKIP_KALSHI]
    rows, found = [], set()
    for st in todo:
        try:
            rows_st = _kalshi_series_markets(http, st, snap_ts, now)
        except Exception as e:
            log(f"kalshi_other {st}: {str(e)[:120]}")
            found.add(st)            # keep it in the list: a transient error (429) is not "nothing open"
            continue
        if rows_st:
            found.add(st)
            rows += rows_st
    if active is None:
        cache.write_text(json.dumps({"ts": str(now), "series": sorted(found)}), encoding="utf-8")
    df = pd.DataFrame(rows)
    return df


def _kalshi_series_markets(http: Http, st: str, snap_ts, now) -> list[dict]:
    rows = []
    if True:
        cursor = None
        for _ in range(10):
            params = {"series_ticker": st, "status": "open", "limit": 1000}
            if cursor:
                params["cursor"] = cursor
            res = http.get(f"{KALSHI}/markets", params) or {}
            for m in res.get("markets") or []:
                close = m.get("close_time")
                try:
                    if to_utc(close) - now > pd.Timedelta(days=HORIZON_DAYS):
                        continue
                except Exception:
                    pass
                rows.append({"snap_ts": snap_ts, "series": st, "event": m.get("event_ticker"), "ticker": m.get("ticker"),
                             "title": m.get("title"), "subtitle": m.get("yes_sub_title") or m.get("subtitle"),
                             "strike_type": m.get("strike_type"), "floor_strike": _num(m.get("floor_strike")),
                             "cap_strike": _num(m.get("cap_strike")), "close_ts": close,
                             "yes_bid": kx_num(m, "yes_bid"), "yes_ask": kx_num(m, "yes_ask"),
                             "yes_bid_size": kx_num(m, "yes_bid_size"), "yes_ask_size": kx_num(m, "yes_ask_size"),
                             "last": kx_num(m, "last_price"), "volume": kx_num(m, "volume"),
                             "open_interest": kx_num(m, "open_interest")})
            cursor = res.get("cursor")
            if not cursor:
                break
    return rows


# ------------------------------------------------------------------ Polymarket: every Bitcoin market
def classify_pm(title: str, question: str) -> str:
    t = f"{title} || {question}".lower()
    if "up or down" in t:
        return "updown_15m" if re.search(r"\d{1,2}:\d{2}\s*(am|pm)?\s*-\s*\d{1,2}:\d{2}", t) else "updown"
    if "above" in t and " on " in t:
        return "above_ladder"
    if "hit" in t or "reach" in t or "dip to" in t:
        return "barrier"
    if "between" in t or "price on" in t or "range" in t or "less than" in t or "greater than" in t:
        return "range"
    if "above" in t or "below" in t:
        return "above_other"
    return "other"


_MONEY = re.compile(r"\$?\s*([\d]{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)\s*(k\b)?", re.I)


def parse_levels(text: str) -> list[float]:
    out = []
    for num, suf in _MONEY.findall(text or ""):
        v = float(num.replace(",", ""))
        if suf and suf.lower() == "k":
            v *= 1000
        if 1_000 <= v <= 10_000_000:
            out.append(v)
    return out


def pm_btc_events(http: Http, max_pages: int = 8) -> list[dict]:
    """Open Bitcoin events settling within HORIZON_DAYS, nearest first (15-minute markets are numerous, so the
    page limit cuts the far ones, not the near ones)."""
    now = pd.Timestamp(utcnow())
    seen, out = set(), []
    for tag in ("bitcoin", "crypto-prices"):
        for page in range(max_pages):
            params = {"tag_slug": tag, "active": "true", "closed": "false", "limit": 100, "offset": 100 * page,
                      "order": "endDate", "ascending": "true",
                      "end_date_min": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
                      "end_date_max": (now + pd.Timedelta(days=HORIZON_DAYS)).strftime("%Y-%m-%dT%H:%M:%SZ")}
            res = http.get(f"{GAMMA}/events", params) or []
            if not res:
                break
            for ev in res:
                if ev.get("id") in seen:
                    continue
                if not (BTC_RE.search(ev.get("title") or "") or BTC_RE.search(ev.get("slug", "").replace("-", " "))):
                    continue
                seen.add(ev.get("id"))
                out.append(ev)
            if len(res) < 100:
                break
    return out


def pm_other(http: Http, snap_ts, include_ladders: bool = True) -> pd.DataFrame:
    """Every open Bitcoin market. include_ladders=True keeps the "above" ladders too: besides the daily 12:00 ET
    ladder (logged with full books in pm_books) Polymarket lists HOURLY ladders ("... on October 28, 7PM ET")
    that settle at the same moment as Kalshi's hourly events."""
    rows = []
    for ev in pm_btc_events(http):
        for m in ev.get("markets") or []:
            kind = classify_pm(ev.get("title") or "", m.get("question") or "")
            if kind == "above_ladder" and not include_ladders:
                continue
            toks = json.loads(m["clobTokenIds"]) if m.get("clobTokenIds") else [None]
            label = m.get("groupItemTitle") or m.get("question") or ""
            lv = parse_levels(label)
            rows.append({"snap_ts": snap_ts, "event_slug": ev.get("slug"), "title": ev.get("title"),
                         "question": m.get("question"), "label": m.get("groupItemTitle"), "kind": kind,
                         "level1": lv[0] if lv else None, "level2": lv[1] if len(lv) > 1 else None,
                         "start_ts": m.get("eventStartTime") or m.get("startDate") or ev.get("startDate"),
                         "end_ts": m.get("endDate") or ev.get("endDate"), "yes_token": toks[0],
                         "best_bid": _num(m.get("bestBid")), "best_ask": _num(m.get("bestAsk")),
                         "last_trade": _num(m.get("lastTradePrice")), "volume": _num(m.get("volumeNum") or m.get("volume")),
                         "liquidity": _num(m.get("liquidityNum") or m.get("liquidity"))})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ Binance options
def binance_options(http: Http, snap_ts) -> pd.DataFrame:
    now = pd.Timestamp(utcnow())
    tick = http.get(f"{EAPI}/eapi/v1/ticker") or []
    mark = http.get(f"{EAPI}/eapi/v1/mark") or []
    t = pd.DataFrame(tick)
    mk = pd.DataFrame(mark)
    if t.empty and mk.empty:
        return pd.DataFrame()
    df = mk.merge(t, on="symbol", how="outer", suffixes=("", "_t")) if len(t) and len(mk) else (mk if len(mk) else t)
    df = df[df["symbol"].astype(str).str.startswith("BTC-")].copy()
    parts = df["symbol"].str.split("-", expand=True)
    df["expiry"] = pd.to_datetime(parts[1], format="%y%m%d", utc=True) + pd.Timedelta(hours=8)
    df["strike"] = pd.to_numeric(parts[2], errors="coerce")
    df["cp"] = parts[3]
    df = df[df["expiry"] - now <= pd.Timedelta(days=HORIZON_DAYS)]
    keep = ["symbol", "expiry", "strike", "cp", "bidPrice", "askPrice", "markPrice", "bidIV", "askIV", "markIV",
            "delta", "indexPrice", "exercisePrice", "lastPrice", "volume"]
    df = df[[c for c in keep if c in df.columns]].copy()
    df.insert(0, "snap_ts", snap_ts)
    return df


# ------------------------------------------------------------------ OKX and Bybit options
# Both return the whole BTC chain in one or two public requests. Output columns are the same normalised set
# for every options venue: expiry (08:00 UTC), strike, cp, iv (decimal), F (forward / underlying in USD),
# bid, ask (USD per BTC when the venue quotes in BTC, converted with F).
NORM_COLS = ["snap_ts", "venue", "symbol", "expiry", "strike", "cp", "iv", "bid_iv", "ask_iv", "F", "bid", "ask", "delta"]


def okx_options(http: Http, snap_ts) -> pd.DataFrame:
    now = pd.Timestamp(utcnow())
    summ = (http.get(f"{OKX}/api/v5/public/opt-summary", {"uly": "BTC-USD"}) or {}).get("data") or []
    tick = (http.get(f"{OKX}/api/v5/market/tickers", {"instType": "OPTION", "uly": "BTC-USD"}) or {}).get("data") or []
    s = pd.DataFrame(summ)
    if s.empty:
        return pd.DataFrame(columns=NORM_COLS)
    t = pd.DataFrame(tick)[["instId", "bidPx", "askPx"]] if tick else pd.DataFrame(columns=["instId", "bidPx", "askPx"])
    d = s.merge(t, on="instId", how="left")
    parts = d["instId"].str.split("-", expand=True)           # BTC-USD-261030-80000-C
    d["expiry"] = pd.to_datetime(parts[2], format="%y%m%d", utc=True, errors="coerce") + pd.Timedelta(hours=8)
    d["strike"] = pd.to_numeric(parts[3], errors="coerce")
    d["cp"] = parts[4]
    d["F"] = pd.to_numeric(d["fwdPx"], errors="coerce")
    out = pd.DataFrame({"snap_ts": snap_ts, "venue": "okx", "symbol": d["instId"], "expiry": d["expiry"],
                        "strike": d["strike"], "cp": d["cp"], "iv": pd.to_numeric(d["markVol"], errors="coerce"),
                        "bid_iv": pd.to_numeric(d.get("bidVol"), errors="coerce"),
                        "ask_iv": pd.to_numeric(d.get("askVol"), errors="coerce"), "F": d["F"],
                        # coin-margined options are quoted in BTC; the USDT-margined '_UM' series already in USD
                        "bid": pd.to_numeric(d["bidPx"], errors="coerce") * np.where(d["instId"].str.contains("_UM"), 1.0, d["F"]),
                        "ask": pd.to_numeric(d["askPx"], errors="coerce") * np.where(d["instId"].str.contains("_UM"), 1.0, d["F"]),
                        "delta": pd.to_numeric(d.get("delta"), errors="coerce")})
    return out[out["expiry"] - now <= pd.Timedelta(days=HORIZON_DAYS)]


def bybit_options(http: Http, snap_ts) -> pd.DataFrame:
    now = pd.Timestamp(utcnow())
    res = http.get(f"{BYBIT}/v5/market/tickers", {"category": "option", "baseCoin": "BTC"}) or {}
    lst = ((res.get("result") or {}).get("list")) or []
    d = pd.DataFrame(lst)
    if d.empty:
        return pd.DataFrame(columns=NORM_COLS)
    parts = d["symbol"].str.split("-", expand=True)            # BTC-30OCT26-80000-C(-USDT)
    d["expiry"] = pd.to_datetime(parts[1], format="%d%b%y", utc=True, errors="coerce") + pd.Timedelta(hours=8)
    d["strike"] = pd.to_numeric(parts[2], errors="coerce")
    d["cp"] = parts[3]
    out = pd.DataFrame({"snap_ts": snap_ts, "venue": "bybit", "symbol": d["symbol"], "expiry": d["expiry"],
                        "strike": d["strike"], "cp": d["cp"], "iv": pd.to_numeric(d.get("markIv"), errors="coerce"),
                        "bid_iv": pd.to_numeric(d.get("bid1Iv"), errors="coerce"),
                        "ask_iv": pd.to_numeric(d.get("ask1Iv"), errors="coerce"),
                        "F": pd.to_numeric(d.get("underlyingPrice"), errors="coerce"),
                        "bid": pd.to_numeric(d.get("bid1Price"), errors="coerce"),
                        "ask": pd.to_numeric(d.get("ask1Price"), errors="coerce"),
                        "delta": pd.to_numeric(d.get("delta"), errors="coerce")})
    return out[out["expiry"] - now <= pd.Timedelta(days=HORIZON_DAYS)]


# ------------------------------------------------------------------ Limitless (probe only)
def _lim_rows(m: dict, snap_ts, parent: dict | None = None) -> dict:
    tp = m.get("tradePrices") or {}
    buy = (tp.get("buy") or {}).get("market") or [None, None]      # price to BUY yes / no now (= ask)
    sell = (tp.get("sell") or {}).get("market") or [None, None]    # price to SELL yes / no now (= bid)
    meta = (parent or m).get("metadata") or {}
    title = (parent or m).get("title") or ""
    sub = m.get("title") if parent else ""
    lv = parse_levels(sub or title) if "up or down" not in title.lower() else []
    return {"snap_ts": snap_ts, "venue": "limitless", "slug": m.get("slug"), "title": title, "label": sub,
            "kind": classify_pm(title, sub) if "up or down" not in title.lower() else "updown",
            "duration": next((p.get("value", [None])[0] for p in (parent or m).get("properties") or []
                              if p.get("propertyKeySlug") == "duration"), None),
            "level1": lv[0] if lv else None, "open_price": _num(meta.get("openPrice")),
            "window_open": (meta.get("chart") or {}).get("windowOpenAt"), "external_slug": meta.get("externalSlug"),
            "end_ts": pd.to_datetime((parent or m).get("expirationTimestamp") or m.get("expirationTimestamp"), unit="ms", utc=True)
            if ((parent or m).get("expirationTimestamp") or m.get("expirationTimestamp")) else None,
            "mid": (m.get("prices") or [None])[0],
            "yes_ask": None if (parent and _num(buy[0]) in (None, 0.999)) else _num(buy[0]),
            "yes_bid": None if (parent and _num(sell[0]) in (None, 0.001)) else _num(sell[0]),
            "no_ask": _num(buy[1]) if len(buy) > 1 else None, "volume": _num(m.get("volumeFormatted")),
            "fee": bool(meta.get("fee"))}


def limitless_btc(http: Http, snap_ts, max_pages: int = 30) -> pd.DataFrame:
    """Bitcoin markets on Limitless Exchange (Base): Up or Down for 5 / 15 minutes, an hour and a day (these mirror
    Polymarket's markets: same Binance candle, metadata.externalSlug = the Polymarket slug) and daily barriers.
    /markets/active returns 25 markets per page; quotes are the current market buy / sell prices."""
    rows, seen = [], set()
    for page in range(1, max_pages + 1):
        res = http.get(f"{LIMITLESS}/markets/active", {"page": page, "limit": 25}) or {}
        data = res.get("data") or []
        new = [m for m in data if m.get("id") not in seen]
        if not new:
            break
        for m in new:
            seen.add(m.get("id"))
            t = (m.get("title") or "") + " " + json.dumps(m.get("categories") or [])
            if not BTC_RE.search(t):
                continue
            if m.get("marketType") == "group" and m.get("markets"):
                rows += [_lim_rows(x, snap_ts, m) for x in m["markets"]]
            else:
                rows.append(_lim_rows(m, snap_ts))
        if len(data) < 25:
            break
    return pd.DataFrame(rows)


def limitless_probe(http: Http) -> dict:
    out = {}
    for path in ("/markets/active", "/markets/active/crypto", "/markets"):
        try:
            out[path] = http.get(LIMITLESS + path, {"limit": 25})
        except Exception as e:
            out[path] = f"error: {e!r}"
    return out
