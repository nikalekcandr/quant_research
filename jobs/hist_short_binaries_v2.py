"""History of the short Bitcoin binaries with real book quotes, for a clean study (no listing lag, no live-log gaps).

1. Kalshi KXBTC15M ("BTC up in the next 15 minutes?"): every settled market (strike = BRTI at the start, result,
   expiration_value = BRTI average at the end) and its 1-minute candles: best YES bid / ask (OHLC of the book),
   last trade, volume, open interest.                                  -> data/raw/kx15/markets.csv, candles/<day>.csv
2. Coinbase BTC-USD 1-minute candles over the same span (the closest public proxy of the BRTI)
                                                                       -> data/raw/coinbase_1m/<day>.csv
3. Polymarket "Bitcoin Up or Down" 15-minute (45 days) and 5-minute (10 days) windows: outcome, price to beat if
   published, 1-minute price history of the Up token                 -> data/raw/pm_updown/<kind>/<day>.csv
Resumable: finished days are skipped. Started by the lab daemon (lab/jobs.py). v2: live-database candles in batches of 4 tickers.
"""
from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from src.utils import Http  # noqa: E402

KX = "https://api.elections.kalshi.com/trade-api/v2"
GAMMA = "https://gamma-api.polymarket.com"
CLOB = "https://clob.polymarket.com"
CB = "https://api.exchange.coinbase.com/products/BTC-USD/candles"
OUT_KX = config.RAW / "kx15"
OUT_CB = config.RAW / "coinbase_1m"
OUT_PM = config.RAW / "pm_updown"
MAX_DAYS = 400
UTC = timezone.utc


def say(msg):
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def num(d, key):
    for k in (key + "_dollars", key + "_fp", key):
        if isinstance(d, dict) and d.get(k) is not None:
            try:
                return float(d[k])
            except (TypeError, ValueError):
                return None
    return None


def today_utc():
    return datetime.now(UTC).date()


# ------------------------------------------------------------------ 1. Kalshi
def mrow(m, hist):
    return {"ticker": m.get("ticker"), "event": m.get("event_ticker"), "open_time": m.get("open_time"),
            "close_time": m.get("close_time"), "floor_strike": num(m, "floor_strike"), "result": m.get("result"),
            "expiration_value": num(m, "expiration_value"), "volume": num(m, "volume"), "status": m.get("status"),
            "historical": hist}


def kx_list(http, path, params, hist):
    rows, cur, pages = [], None, 0
    while True:
        p = dict(params, limit=1000)
        if cur:
            p["cursor"] = cur
        try:
            res = http.get(f"{KX}{path}", p) or {}
        except Exception as e:
            say(f"kalshi {path}: {str(e)[:150]}")
            break
        mk = res.get("markets") or []
        rows += [mrow(m, hist) for m in mk]
        pages += 1
        cur = res.get("cursor")
        if pages % 10 == 0:
            say(f"kalshi {path}: {len(rows)} markets, page {pages}, last close {mk[-1].get('close_time') if mk else None}")
        if not cur or not mk:
            break
    return rows


def kx_event_fallback(http, known: set, since):
    """If the archive listing ignores series_ticker: one request per 15-minute event, newest first; stop after
    three empty days in a row."""
    from zoneinfo import ZoneInfo
    et = ZoneInfo("America/New_York")
    rows, empty_days = [], 0
    d = today_utc() - timedelta(days=1)
    while d >= since and empty_days < 3:
        got = 0
        for q in range(96):
            close = datetime(d.year, d.month, d.day, tzinfo=UTC) + timedelta(minutes=15 * (q + 1))
            ce = close.astimezone(et)
            ev = f"KXBTC15M-{ce:%y}{ce.strftime('%b').upper()}{ce:%d%H%M}"
            tk = f"{ev}-{ce:%M}"
            if tk in known:
                got += 1
                continue
            try:
                res = http.get(f"{KX}/historical/markets", {"event_ticker": ev, "limit": 10}) or {}
            except Exception:
                continue
            for m in res.get("markets") or []:
                rows.append(mrow(m, True))
                got += 1
        say(f"kalshi fallback {d}: {got} markets")
        empty_days = empty_days + 1 if got == 0 else 0
        d -= timedelta(days=1)
    return rows


def kx_candles_live(http, mk):
    out = []
    mk = mk.sort_values("open_time")
    for i in range(0, len(mk), 4):          # the batch endpoint rejects long spans x many tickers (HTTP 400)
        ch = mk.iloc[i:i + 4]
        t0 = int(pd.Timestamp(ch["open_time"].min()).timestamp()) - 60
        t1 = int(pd.Timestamp(ch["close_time"].max()).timestamp()) + 60
        try:
            res = http.get(f"{KX}/markets/candlesticks", {"market_tickers": ",".join(ch["ticker"]), "start_ts": t0,
                                                           "end_ts": t1, "period_interval": 1}) or {}
        except Exception as e:
            say(f"kalshi candles batch failed: {str(e)[:120]}")
            continue
        got = set()
        for m in res.get("markets") or []:
            got.add(m.get("market_ticker"))
            out += [crow(m.get("market_ticker"), c) for c in m.get("candlesticks") or []]
        missing = [t for t in ch["ticker"] if t not in got]
        if missing:                          # the batch endpoint returns nothing for archived markets
            out += kx_candles_hist(http, mk[mk["ticker"].isin(missing)])
    return out


def kx_candles_hist(http, mk):
    def one(r):
        t0 = int(pd.Timestamp(r.open_time).timestamp()) - 60
        t1 = int(pd.Timestamp(r.close_time).timestamp()) + 60
        for path in (f"/historical/markets/{r.ticker}/candlesticks",
                     f"/series/KXBTC15M/markets/{r.ticker}/candlesticks"):
            try:
                res = http.get(f"{KX}{path}", {"start_ts": t0, "end_ts": t1, "period_interval": 1}) or {}
            except Exception:
                continue
            cs = res.get("candlesticks") or []
            if cs:
                return [crow(r.ticker, c) for c in cs]
        return []

    with ThreadPoolExecutor(max_workers=4) as ex:
        return [x for part in ex.map(one, mk.itertuples()) for x in part]


def crow(tk, c):
    yb, ya, pr = c.get("yes_bid") or {}, c.get("yes_ask") or {}, c.get("price") or {}
    return {"ticker": tk, "end_ts": c.get("end_period_ts"),
            "bid_o": num(yb, "open"), "bid_h": num(yb, "high"), "bid_l": num(yb, "low"), "bid": num(yb, "close"),
            "ask_o": num(ya, "open"), "ask_h": num(ya, "high"), "ask_l": num(ya, "low"), "ask": num(ya, "close"),
            "last": num(pr, "close"), "vwap": num(pr, "mean"), "volume": num(c, "volume"),
            "oi": num(c, "open_interest")}


def kalshi(http):
    OUT_KX.mkdir(parents=True, exist_ok=True)
    (OUT_KX / "candles").mkdir(exist_ok=True)
    try:
        say(f"kalshi cutoff: {json.dumps(http.get(f'{KX}/historical/cutoff'))[:300]}")
    except Exception as e:
        say(f"kalshi cutoff failed: {e}")
    rows = kx_list(http, "/markets", {"series_ticker": "KXBTC15M", "status": "settled"}, False)
    say(f"kalshi live settled: {len(rows)}")
    hist = kx_list(http, "/historical/markets", {"series_ticker": "KXBTC15M"}, True)
    hist = [r for r in hist if (r["ticker"] or "").startswith("KXBTC15M")]
    say(f"kalshi archive via series listing: {len(hist)}")
    since = today_utc() - timedelta(days=MAX_DAYS)
    if len(hist) < 500:
        hist += kx_event_fallback(http, {r["ticker"] for r in rows + hist}, since)
    mk = pd.DataFrame(rows + hist).drop_duplicates("ticker", keep="first")
    mk = mk[mk["result"].isin(["yes", "no"])].copy()
    mk["close_time"] = pd.to_datetime(mk["close_time"], utc=True)
    mk["open_time"] = pd.to_datetime(mk["open_time"], utc=True)
    mk = mk[mk["close_time"].dt.date >= since].sort_values("close_time")
    mk.to_csv(OUT_KX / "markets.csv", index=False)
    say(f"kalshi markets: {len(mk)} settled, {mk['close_time'].min()} .. {mk['close_time'].max()}")
    days = sorted(mk["close_time"].dt.date.unique(), reverse=True)
    for i, d in enumerate(days):
        f = OUT_KX / "candles" / f"{d}.csv"
        if f.exists() and f.stat().st_size > 2000 and d < today_utc() - timedelta(days=1):
            continue
        g = mk[mk["close_time"].dt.date == d]
        live, arch = g[~g["historical"].astype(bool)], g[g["historical"].astype(bool)]
        out = (kx_candles_live(http, live) if len(live) else []) + (kx_candles_hist(http, arch) if len(arch) else [])
        pd.DataFrame(out).to_csv(f, index=False)
        if i % 5 == 0:
            say(f"kalshi candles {d}: {len(g)} markets, {len(out)} candles ({i + 1}/{len(days)} days)")
    return mk["close_time"].min()


# ------------------------------------------------------------------ 2. Coinbase
def coinbase(http, since):
    OUT_CB.mkdir(parents=True, exist_ok=True)
    d = today_utc()
    n = 0
    while d >= since:
        f = OUT_CB / f"{d}.csv"
        if not (f.exists() and d < today_utc() - timedelta(days=1)):
            rows = []
            t = datetime(d.year, d.month, d.day, tzinfo=UTC)
            for k in range(5):
                a, b = t + timedelta(minutes=300 * k), t + timedelta(minutes=300 * (k + 1) - 1)
                try:
                    res = http.get(CB, {"granularity": 60, "start": a.isoformat(), "end": b.isoformat()}) or []
                except Exception as e:
                    say(f"coinbase {a}: {str(e)[:100]}")
                    continue
                rows += res
            if rows:
                c = pd.DataFrame(rows, columns=["t", "low", "high", "open", "close", "volume"]).drop_duplicates("t")
                c.sort_values("t").to_csv(f, index=False)
            n += 1
            if n % 20 == 0:
                say(f"coinbase: {d} ({n} days)")
        d -= timedelta(days=1)


# ------------------------------------------------------------------ 3. Polymarket
def pm_window(http, kind, start):
    slug = f"btc-updown-{kind}-{int(start.timestamp())}"
    try:
        evs = http.get(f"{GAMMA}/events", {"slug": slug}) or []
    except Exception:
        return []
    if not evs:
        return []
    ev = evs[0]
    ms = ev.get("markets") or []
    if not ms:
        return []
    m = ms[0]
    toks = json.loads(m["clobTokenIds"]) if m.get("clobTokenIds") else []
    if not toks:
        return []
    try:
        op = [float(x) for x in json.loads(m.get("outcomePrices") or "[]")]
    except Exception:
        op = []
    outcome = (1.0 if op[0] >= 0.99 else 0.0 if op[0] <= 0.01 else None) if op else None
    meta = ev.get("eventMetadata") or m.get("eventMetadata") or {}
    if isinstance(meta, str):
        try:
            meta = json.loads(meta)
        except Exception:
            meta = {}
    ptb = meta.get("priceToBeat") if isinstance(meta, dict) else None
    dur = 15 if kind == "15m" else 5
    t0, t1 = int(start.timestamp()) - 60, int(start.timestamp()) + dur * 60 + 60
    try:
        h = (http.get(f"{CLOB}/prices-history", {"market": toks[0], "startTs": t0, "endTs": t1, "fidelity": 1}) or {}).get("history") or []
    except Exception:
        h = []
    base = {"slug": slug, "start": start.isoformat(), "outcome": outcome, "price_to_beat": ptb,
            "fee": json.dumps(m.get("feeSchedule")) if m.get("feeSchedule") else None, "volume": m.get("volume")}
    if not h:
        return [dict(base, t=None, p=None)]
    return [dict(base, t=x.get("t"), p=x.get("p")) for x in h]


def polymarket(http):
    for kind, days in (("15m", 45), ("5m", 10)):
        out_dir = OUT_PM / kind
        out_dir.mkdir(parents=True, exist_ok=True)
        step = 15 if kind == "15m" else 5
        for k in range(1, days + 1):
            d = today_utc() - timedelta(days=k)
            f = out_dir / f"{d}.csv"
            if f.exists():
                continue
            t = datetime(d.year, d.month, d.day, tzinfo=UTC)
            starts = [t + timedelta(minutes=step * i) for i in range(1440 // step)]
            with ThreadPoolExecutor(max_workers=4) as ex:
                rows = [x for part in ex.map(lambda s: pm_window(http, kind, s), starts) for x in part]
            pd.DataFrame(rows).to_csv(f, index=False)
            say(f"polymarket {kind} {d}: {len({r['slug'] for r in rows})} windows, {len(rows)} price points")


def main():
    say("start")
    since = None
    try:
        first = kalshi(Http(min_interval=0.1, retries=4))
        since = pd.Timestamp(first).date() if first is not None and pd.notna(first) else None
    except Exception as e:
        say(f"kalshi stage failed: {e!r}")
    try:
        coinbase(Http(min_interval=0.2, retries=4), since or today_utc() - timedelta(days=60))
    except Exception as e:
        say(f"coinbase stage failed: {e!r}")
    try:
        polymarket(Http(min_interval=0.05, retries=3))
    except Exception as e:
        say(f"polymarket stage failed: {e!r}")
    say("done")


if __name__ == "__main__":
    main()
