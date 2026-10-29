"""Short-dated Bitcoin prediction markets, every minute, with real top of book.

Polymarket (gamma events ending within 5 hours, books from the CLOB):
    updown   5m / 15m / 4h  "Up or Down" on the Chainlink BTC/USD TWAP (start price = price at window start)
    updown   1h             on the Binance BTC/USDT 1-hour candle (close >= open)
    ladder   hourly         "Bitcoin above K on <date>, <h> ET": Binance BTC/USDT 1-hour candle close > K
Kalshi:
    KXBTC15M  "BTC price up in next 15 mins?"   floor_strike = target (CF BRTI at the start)
    KXBTCD    hourly ladder of the current hour  (CF BRTI 60-s average before the hour)

Rows: one per market (yes token / ticker) with bid, ask, sizes, start, end, strike/target, resolution source.
"""
from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd

from lab.core import log, utcnow
from src.download import kalshi, polymarket

GAMMA = "https://gamma-api.polymarket.com"
KALSHI = "https://api.elections.kalshi.com/trade-api/v2"
BTC = re.compile(r"\bbitcoin\b|\bbtc\b", re.I)
_EV_CACHE = {"ts": None, "events": []}


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return np.nan


def _pm_events(http, now):
    """Gamma is cached per minute; events are re-listed every 5 minutes (new windows open every 5 minutes)."""
    if _EV_CACHE["ts"] is not None and now - _EV_CACHE["ts"] < pd.Timedelta(minutes=4, seconds=30):
        return _EV_CACHE["events"]
    out, seen = [], set()
    for tag in ("bitcoin", "crypto-prices"):
        params = {"tag_slug": tag, "active": "true", "closed": "false", "limit": 100, "order": "endDate",
                  "ascending": "true", "end_date_min": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
                  "end_date_max": (now + pd.Timedelta(hours=5)).strftime("%Y-%m-%dT%H:%M:%SZ")}
        for ev in http.get(f"{GAMMA}/events", params) or []:
            if ev.get("id") in seen or not (BTC.search(ev.get("title") or "") or "btc" in (ev.get("slug") or "")):
                continue
            seen.add(ev.get("id"))
            out.append(ev)
    _EV_CACHE.update(ts=now, events=out)
    return out


def _kind(ev, m):
    slug = ev.get("slug") or ""
    title = (ev.get("title") or "") + " " + (m.get("question") or "")
    if "up or down" in title.lower() or "updown" in slug:
        mm = re.search(r"updown-(\d+[mh])-", slug)
        dur = mm.group(1) if mm else ("1h" if "up-or-down" in slug else "?")
        return f"pm_updown_{dur}"
    if "above" in title.lower() and re.search(r"\d{1,2}(am|pm)-et", slug):
        return "pm_ladder_1h"
    return None


def polymarket_short(http, now, kinds=None) -> list[dict]:
    rows, tokens = [], []
    for ev in _pm_events(http, now):
        for m in ev.get("markets") or []:
            kind = _kind(ev, m)
            if kind is None or (kinds is not None and kind not in kinds):
                continue
            end = pd.Timestamp(m.get("endDate") or ev.get("endDate"))
            dur = {"pm_updown_5m": 5, "pm_updown_15m": 15, "pm_updown_1h": 60, "pm_updown_4h": 240}.get(kind)
            if kind.startswith("pm_updown") and dur is None:
                continue
            # the window start: end minus the duration (eventStartTime is not set on every series)
            start = end - pd.Timedelta(minutes=dur) if dur else end - pd.Timedelta(hours=1)
            if end <= now or (kind.startswith("pm_updown") and start > now):
                continue          # finished, or a window that has not started (its price carries no information)
            toks = json.loads(m["clobTokenIds"]) if m.get("clobTokenIds") else []
            if not toks:
                continue
            strike = np.nan
            if kind == "pm_ladder_1h":
                lv = [float(x.replace(",", "")) for x in re.findall(r"\d{1,3}(?:,\d{3})+", m.get("groupItemTitle") or m.get("question") or "")]
                strike = lv[0] if lv else np.nan
            fs = m.get("feeSchedule") or {}
            rows.append({"venue": "polymarket", "kind": kind, "event": ev.get("slug"), "market": m.get("slug"),
                         "token": toks[0], "start": start, "end": end, "strike": strike,
                         "source": "chainlink" if "chain.link" in (m.get("resolutionSource") or ev.get("resolutionSource") or "") else "binance",
                         "fee_rate": _f(fs.get("rate")) if m.get("feesEnabled") else 0.0,
                         "g_bid": _f(m.get("bestBid")), "g_ask": _f(m.get("bestAsk"))})
            tokens.append(toks[0])
    if tokens:
        bk = polymarket.books(http, tokens)
        for r in rows:
            b = bk.get(r["token"])
            if b:
                r["bid"], r["bid_size"], _ = polymarket._top(b.get("bids"), "max")
                r["ask"], r["ask_size"], _ = polymarket._top(b.get("asks"), "min")
    for r in rows:
        r.setdefault("bid", r.pop("g_bid", np.nan))
        r.setdefault("ask", r.pop("g_ask", np.nan))
        r.pop("g_bid", None)
        r.pop("g_ask", None)
    return rows


def kalshi_short(http, now, hourly=True) -> list[dict]:
    rows = []
    res = http.get(f"{KALSHI}/markets", {"series_ticker": "KXBTC15M", "status": "open", "limit": 50}) or {}
    for m in res.get("markets") or []:
        end = pd.Timestamp(m.get("close_time"))
        if end <= now or end - now > pd.Timedelta(minutes=16):
            continue
        row = {"venue": "kalshi", "kind": "kx_15m", "event": m.get("event_ticker"), "market": m.get("ticker"),
               "token": m.get("ticker"), "start": end - pd.Timedelta(minutes=15), "end": end,
               "strike": _f(m.get("floor_strike")), "source": "brti", "fee_rate": 0.07,
               "bid": kalshi._num(m, "yes_bid"), "ask": kalshi._num(m, "yes_ask"),
               "bid_size": kalshi._num(m, "yes_bid_size"), "ask_size": kalshi._num(m, "yes_ask_size")}
        _with_book(http, row)
        rows.append(row)
    if not hourly:
        return rows
    try:
        d = kalshi.snapshot(http, now)
        if len(d):
            # live books only for the strikes that matter (listing mid 1–99¢), at most 25 per minute
            mid = (d["yes_bid"].fillna(0) + d["yes_ask"].fillna(1)) / 2
            d = d.assign(_mid=mid).sort_values("_mid", key=lambda x: (x - 0.5).abs())
        for i, (_, m) in enumerate(d.iterrows()):
            row = {"venue": "kalshi", "kind": "kx_ladder_1h", "event": m["event"], "market": m["ticker"],
                   "token": m["ticker"], "start": pd.Timestamp(m["settle_ts"]) - pd.Timedelta(hours=1),
                   "end": pd.Timestamp(m["settle_ts"]), "strike": _f(m["strike"]), "source": "brti",
                   "fee_rate": 0.07, "bid": m["yes_bid"], "ask": m["yes_ask"],
                   "bid_size": m["yes_bid_size"], "ask_size": m["yes_ask_size"]}
            _with_book(http, row, fetch=i < 25 and 0.01 <= m["_mid"] <= 0.99)
            rows.append(row)
    except Exception as e:
        log(f"kalshi hourly ladder failed: {str(e)[:120]}")
    return rows


_PAUSE: dict = {}


def _with_book(http, row, fetch=True):
    """Kalshi's /markets listing lags the live book by up to a minute (29.10: 6¢ on average on the 15-minute market).
    Trade only on the book: listing quotes are kept as bid_m / ask_m for comparison, bid/ask come from the book or
    are left empty (no book = no trade)."""
    row["bid_m"], row["ask_m"] = row["bid"], row["ask"]
    row["bid"] = row["ask"] = row["bid_size"] = row["ask_size"] = np.nan
    row["book"] = 0
    if not fetch:
        return
    try:
        ob = kalshi_book(http, row["market"])
        if ob is not None:
            row.update(ob)
            row["book"] = 1
    except Exception:
        pass


def _levels(side):
    out = []
    for lv in side or []:
        try:
            p, q = float(lv[0]), float(lv[1])
        except (TypeError, ValueError, IndexError):
            continue
        out.append((p / 100 if p > 1 else p, q))
    return out


def kalshi_book(http, ticker):
    """Best YES bid / ask from GET /markets/{ticker}/orderbook (YES bids and NO bids; YES ask = 1 − best NO bid)."""
    res = http.get(f"{KALSHI}/markets/{ticker}/orderbook", {"depth": 5}) or {}
    ob = res.get("orderbook") or res.get("orderbook_fp") or {}
    yes = _levels(ob.get("yes_dollars") or ob.get("yes"))
    no = _levels(ob.get("no_dollars") or ob.get("no"))
    if not yes and not no:
        return None
    by = max(yes, key=lambda x: x[0]) if yes else (np.nan, np.nan)
    bn = max(no, key=lambda x: x[0]) if no else (np.nan, np.nan)
    return {"bid": by[0], "bid_size": by[1], "ask": 1 - bn[0] if np.isfinite(bn[0]) else np.nan, "ask_size": bn[1]}


def snapshot(http, now, fast=False) -> pd.DataFrame:
    """fast=True: only the 5/15-minute markets (the 15-second loop)."""
    rows = []
    # the 15-s loop serves A12F (Kalshi vs Polymarket 15-min) and V07; the 5-minute markets are no longer used there
    jobs = (("polymarket", lambda h, n: polymarket_short(h, n, ("pm_updown_15m",))),
            ("kalshi", lambda h, n: kalshi_short(h, n, hourly=False))) if fast else \
        (("polymarket", polymarket_short), ("kalshi", kalshi_short))
    for name, fn in jobs:
        pk = name + ("_fast" if fast else "")
        if _PAUSE.get(pk) is not None and now < _PAUSE[pk]:
            continue
        try:
            rows += fn(http, now)
        except Exception as e:
            # 429 = rate limit (e.g. while the Kalshi backfill runs in parallel): skip this venue for 2 minutes
            _PAUSE[pk] = now + pd.Timedelta(minutes=2 if "429" in str(e) else 1)
            if not fast:
                log(f"markets {name} failed: {str(e)[:160]}")
    d = pd.DataFrame(rows)
    if len(d):
        d.insert(0, "ts", now.floor("s"))
        for c in ("bid", "ask"):
            d[c] = pd.to_numeric(d[c], errors="coerce")
        # Kalshi reports an empty side as 0: no offer is not an ask of zero
        d.loc[d["ask"] <= 0, "ask"] = np.nan
        d.loc[d["bid"] <= 0, "bid"] = np.nan
    return d


# ------------------------------------------------------------------ settlement
_RES_CACHE: dict = {}


def resolve(http, venue: str, kind: str, market: str, event: str) -> float | None:
    """1.0 / 0.0 once the venue has settled the YES side (Up for up/down markets), else None."""
    key = (venue, market)
    if key in _RES_CACHE:
        return _RES_CACHE[key]
    out = None
    if venue == "polymarket":
        # gamma lists closed markets only when asked for them explicitly
        cands = []
        for params in ({"slug": market, "closed": "true"}, {"slug": market}):
            cands = http.get(f"{GAMMA}/markets", params) or []
            if cands:
                break
        if not cands and event:
            evs = http.get(f"{GAMMA}/events", {"slug": event, "closed": "true"}) or []
            cands = [m for e in evs for m in (e.get("markets") or []) if m.get("slug") == market] or \
                    [m for e in evs for m in (e.get("markets") or [])][:1]
        for m in cands:
            if m.get("outcomePrices"):
                op = [float(x) for x in json.loads(m["outcomePrices"])]
                if m.get("closed") or max(op) >= 0.999:
                    if max(op) >= 0.99:
                        out = 1.0 if op[0] >= 0.99 else 0.0
                        break
    else:
        res = http.get(f"{KALSHI}/markets/{market}") or {}
        m = res.get("market") or {}
        r = (m.get("result") or "").lower()
        if r in ("yes", "no"):
            out = 1.0 if r == "yes" else 0.0
    if out is not None:
        _RES_CACHE[key] = out
    return out
