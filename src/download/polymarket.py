"""Polymarket: daily "Bitcoin above ___ on <date>?" strike ladders.

Verified endpoints (October 2026):
  gamma-api.polymarket.com/events?slug=...       event + its markets (one market per strike)
  clob.polymarket.com/prices-history?market=<token>&startTs&endTs&fidelity=<minutes>
                                                 price history, also for closed markets
  clob.polymarket.com/book?token_id=<token>      current order book (used by the live logger)

Each event settles on the close of the Binance BTC/USDT 1-minute candle that opens at 12:00 ET.
Slugs: "bitcoin-above-on-october-13-2026" for recent events, "bitcoin-above-on-march-2" for older ones,
so we try both and check the event's endDate.
"""
from __future__ import annotations

import json
import re
from datetime import date, timedelta

import pandas as pd

import config
from src.utils import Http, ensure, exists_json, load_json, log, noon_et_utc, save_json, sec, to_utc, utcnow

GAMMA = "https://gamma-api.polymarket.com"
CLOB = "https://clob.polymarket.com"
RAW_DIR = config.RAW / "polymarket"
PRICE_FIDELITY_MIN = 10   # minutes between price points in the downloaded history


def slug_candidates(d: date) -> list[str]:
    month = d.strftime("%B").lower()
    return [f"bitcoin-above-on-{month}-{d.day}-{d.year}", f"bitcoin-above-on-{month}-{d.day}"]


def fetch_event(http: Http, d: date) -> dict | None:
    """Return the gamma event whose settlement is 12:00 ET on date d, or None."""
    target = noon_et_utc(d)
    for slug in slug_candidates(d):
        res = http.get(f"{GAMMA}/events", {"slug": slug})
        if not res:
            continue
        ev = res[0]
        end = to_utc(ev.get("endDate"))
        if abs((end - target).total_seconds()) <= 3600 and "above" in ev.get("title", "").lower():
            return ev
    return None


def _strike(m: dict) -> float | None:
    for txt in (m.get("groupItemTitle"), m.get("question")):
        if not txt:
            continue
        hit = re.search(r"\$?([\d][\d,]*(?:\.\d+)?)\s*(k)?", txt.replace(" ", ""))
        if hit:
            v = float(hit.group(1).replace(",", ""))
            if hit.group(2):
                v *= 1000
            if v > 1000:
                return v
    return None


def parse_event(ev: dict) -> pd.DataFrame:
    rows = []
    for m in ev.get("markets", []):
        toks = json.loads(m["clobTokenIds"]) if m.get("clobTokenIds") else [None, None]
        outcome = None
        if m.get("closed") and m.get("outcomePrices"):
            op = [float(x) for x in json.loads(m["outcomePrices"])]
            if max(op) >= 0.99:
                outcome = 1.0 if op[0] >= 0.99 else 0.0
        rows.append({
            "event_slug": ev["slug"],
            "settle_ts": to_utc(ev["endDate"]),
            "open_ts": to_utc(m.get("startDate") or ev.get("startDate")),
            "strike": _strike(m),
            "market_slug": m.get("slug"),
            "condition_id": m.get("conditionId"),
            "yes_token": toks[0],
            "no_token": toks[1] if len(toks) > 1 else None,
            "volume": float(m.get("volumeNum") or m.get("volume") or 0),
            "outcome": outcome,
        })
    df = pd.DataFrame(rows)
    return df.dropna(subset=["strike", "yes_token"]).sort_values("strike").reset_index(drop=True)


def download_history(since: str = config.POLYMARKET_START, until: str | None = None, http: Http | None = None):
    """Download every daily ladder in [since, until] and the price history of every strike.

    Raw files: data/raw/polymarket/events/<date>.json.gz and prices/<token>.json.gz.
    Re-running skips files that already exist, so an interrupted download can be resumed.
    """
    http = http or Http(min_interval=0.05)
    start = pd.Timestamp(since).date()
    end = pd.Timestamp(until).date() if until else (utcnow().date() + timedelta(days=1))
    ev_dir, px_dir = ensure(RAW_DIR / "events"), ensure(RAW_DIR / "prices")
    d = start
    n_ev = n_px = 0
    while d <= end:
        f = ev_dir / f"{d.isoformat()}.json"
        settled = noon_et_utc(d) < pd.Timestamp(utcnow())
        if exists_json(f) and settled:
            ev = load_json(f)
        else:
            ev = fetch_event(http, d)
            if ev is None:
                d += timedelta(days=1)
                continue
            save_json(ev, f)
        n_ev += 1
        markets = parse_event(ev)
        t0 = sec(markets["open_ts"].min()) - 3600
        t1 = sec(to_utc(ev["endDate"])) + 600
        for tok in markets["yes_token"]:
            pf = px_dir / f"{tok}.json"
            if exists_json(pf) and settled:
                continue
            res = http.get(f"{CLOB}/prices-history",
                           {"market": tok, "startTs": t0, "endTs": t1, "fidelity": PRICE_FIDELITY_MIN})
            save_json(res or {"history": []}, pf)
            n_px += 1
        if n_ev % 20 == 0:
            log(f"polymarket: {d} done ({n_ev} events, {n_px} price files fetched)")
        d += timedelta(days=1)
    log(f"polymarket: finished, {n_ev} events")


def build_interim():
    """Combine raw files into two tables: ladder (one row per strike) and prices (token, t, p)."""
    ev_dir, px_dir = RAW_DIR / "events", RAW_DIR / "prices"
    ladders, prices = [], []
    for f in sorted(ev_dir.glob("*.json*")):
        ev = load_json(f)
        lad = parse_event(ev)
        ladders.append(lad)
        for tok in lad["yes_token"]:
            pf = px_dir / f"{tok}.json"
            if not exists_json(pf):
                continue
            h = (load_json(pf) or {}).get("history", [])
            if h:
                p = pd.DataFrame(h)
                p["yes_token"] = tok
                prices.append(p)
    lad = pd.concat(ladders, ignore_index=True) if ladders else pd.DataFrame()
    px = pd.concat(prices, ignore_index=True) if prices else pd.DataFrame(columns=["t", "p", "yes_token"])
    px["ts"] = pd.to_datetime(px["t"], unit="s", utc=True)
    px = px[["yes_token", "ts", "p"]].sort_values(["yes_token", "ts"]).reset_index(drop=True)
    return lad, px


# ------------------------------------------------------------------ live order books
def open_events(http: Http, days_ahead: int = 8) -> list[dict]:
    today = pd.Timestamp(utcnow()).tz_convert("America/New_York").date()
    out = []
    for k in range(0, days_ahead + 1):
        ev = fetch_event(http, today + timedelta(days=k))
        if ev and to_utc(ev["endDate"]) > pd.Timestamp(utcnow()):
            out.append(ev)
    return out


def _top(levels, best: str):
    if not levels:
        return None, 0.0, 0.0
    lv = sorted(((float(x["price"]), float(x["size"])) for x in levels), key=lambda z: z[0], reverse=(best == "max"))
    depth5 = sum(s for _, s in lv[:5])
    return lv[0][0], lv[0][1], depth5


def books(http: Http, tokens: list[str]) -> dict[str, dict]:
    """Order books for several tokens. Tries the batch endpoint, falls back to one request per token."""
    out = {}
    try:
        res = http.post(f"{CLOB}/books", [{"token_id": t} for t in tokens])
        for b in res or []:
            out[b.get("asset_id")] = b
    except Exception:
        out = {}
    for t in tokens:
        if t not in out:
            b = http.get(f"{CLOB}/book", {"token_id": t})
            if b:
                out[t] = b
    return out


def snapshot_books(http: Http, events: list[dict], snap_ts) -> pd.DataFrame:
    rows = []
    for ev in events:
        lad = parse_event(ev)
        bk = books(http, list(lad["yes_token"]))
        for _, m in lad.iterrows():
            b = bk.get(m["yes_token"])
            if not b:
                continue
            bid, bid_sz, bid_d5 = _top(b.get("bids"), "max")
            ask, ask_sz, ask_d5 = _top(b.get("asks"), "min")
            rows.append({"snap_ts": snap_ts, "event_slug": m["event_slug"], "settle_ts": m["settle_ts"],
                         "strike": m["strike"], "yes_token": m["yes_token"], "bid": bid, "ask": ask,
                         "bid_size": bid_sz, "ask_size": ask_sz, "bid_depth5": bid_d5, "ask_depth5": ask_d5,
                         "last_trade": b.get("last_trade_price"), "book_ts": b.get("timestamp")})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ earliest available history
def probe_start(http: Http | None = None, earliest: str = "2024-01-01", latest: str | None = None) -> str | None:
    """Earliest date with a daily "Bitcoin above ___ on <date>" ladder. Monthly scan (three dates per month,
    ladders may be missing on single days), then a day-by-day walk back inside the month before the hit."""
    http = http or Http(min_interval=0.05)
    d = pd.Timestamp(earliest).normalize()
    stop = pd.Timestamp(latest or config.POLYMARKET_START).normalize()
    hit = None
    while d <= stop and hit is None:
        for day in (1, 10, 20):
            x = d + pd.Timedelta(days=day - 1)
            if fetch_event(http, x.date()) is not None:
                hit = x
                break
        d += pd.offsets.MonthBegin(1)
    if hit is None:
        return None
    first, x, misses = hit, hit - pd.Timedelta(days=1), 0
    lower = hit - pd.Timedelta(days=40)
    while x >= lower and misses < 7:     # stop after a week without ladders
        if fetch_event(http, x.date()) is not None:
            first, misses = x, 0
        else:
            misses += 1
        x -= pd.Timedelta(days=1)
    log(f"polymarket: earliest daily ladder found on {first:%Y-%m-%d}")
    return f"{first:%Y-%m-%d}"
