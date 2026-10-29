"""Kalshi: hourly Bitcoin strike ladders, series KXBTCD ("above K"), and range buckets, series KXBTC.

Verified endpoints (October 2026), base https://api.elections.kalshi.com/trade-api/v2 :
  /markets?event_ticker=KXBTCD-26OCT1210                  strikes of one hourly event (recent)
  /historical/markets?event_ticker=...                    the same for events settled before the cutoff
  /historical/cutoff                                      date before which data moved to /historical
  /markets/candlesticks?market_tickers=a,b&start_ts&end_ts&period_interval=1
                                                          1-minute candles with yes bid/ask OHLC for
                                                          markets that are still in the live database
  /historical/markets/<ticker>/candlesticks?...           the same for archived markets, one ticker per
                                                          request (the batch endpoint returns [] for them)
Event tickers: KXBTCD-YYMONDDHH, HH = hour in New York time when the market settles.
Each event opens one hour before settlement and has ~190 "greater than" strikes spaced $100 apart.
Settlement = average of the 60 CF Benchmarks BRTI prints in the last minute (field expiration_value).

Series KXBTC (same events, same settlement): "BTC between floor_strike and cap_strike" buckets. Summing the
buckets above K gives a second, independent price of "above K" on the same venue and the same expiry.

    python -m src.download.kalshi 2026-07-01 2026-10-14                        KXBTCD, first 10 minutes (default)
    python -m src.download.kalshi 2026-07-01 2026-10-14 --minutes 60           KXBTCD, the whole hour
    python -m src.download.kalshi 2026-07-01 2026-10-14 --series KXBTC --minutes 60   range buckets, whole hour
The non-default variants go to their own folders (data/raw/kalshi_<series>_<minutes>m) and do not touch the
files the main pipeline reads. The archive costs one request per strike whatever the window length, so a
whole-hour download takes about as long as the default one.
"""
from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd

import config
from src.utils import ET, Http, ensure, log, sec, to_utc, utcnow

BASE = "https://api.elections.kalshi.com/trade-api/v2"
RAW_DIR = config.RAW / "kalshi"
SNAP_OFFSET_MIN = 5        # snapshot = market open + 5 minutes
CANDLE_MINUTES = 10        # we keep candles for the first 10 minutes after the open
BAND = 0.025               # strikes within +-2.5% of the price at the open (live events, one batch request)
BAND_HIST = 0.015          # archived events cost one request per strike: +-1.5% (~4 hourly st. dev.)
HIST_STRIKE_STEP = 200     # ... and every second strike ($200 apart)
THREADS = 8                # parallel requests for archived events (the shared rate limit still applies)


def event_ticker(close_et: datetime, series: str = "KXBTCD") -> str:
    return f"{series}-{close_et:%y}{close_et.strftime('%b').upper()}{close_et:%d%H}"


def _num(d: dict, key: str):
    """Kalshi renamed fields (close -> close_dollars, volume -> volume_fp); accept both."""
    for k in (key + "_dollars", key + "_fp", key):
        if k in d and d[k] is not None:
            try:
                return float(d[k])
            except (TypeError, ValueError):
                return None
    return None


def markets(http: Http, ev: str) -> tuple[list[dict], bool]:
    """Strikes of one hourly event and whether it lives in the historical archive."""
    res = http.get(f"{BASE}/markets", {"event_ticker": ev, "limit": 1000})
    mk = (res or {}).get("markets") or []
    if mk:
        return mk, False
    res = http.get(f"{BASE}/historical/markets", {"event_ticker": ev, "limit": 1000})
    return (res or {}).get("markets") or [], True


def _candle_row(ticker, c):
    return {"ticker": ticker, "end_ts": c["end_period_ts"],
            "yes_bid": _num(c.get("yes_bid", {}), "close"), "yes_ask": _num(c.get("yes_ask", {}), "close"),
            "last": _num(c.get("price", {}), "close"), "volume": _num(c, "volume"),
            "open_interest": _num(c, "open_interest")}


def candles(http: Http, tickers: list[str], t0: int, t1: int) -> list[dict]:
    """Live markets: up to 50 tickers per request."""
    rows = []
    for i in range(0, len(tickers), 50):
        chunk = tickers[i:i + 50]
        res = http.get(f"{BASE}/markets/candlesticks",
                       {"market_tickers": ",".join(chunk), "start_ts": t0, "end_ts": t1, "period_interval": 1})
        for m in (res or {}).get("markets", []):
            rows.extend(_candle_row(m.get("market_ticker"), c) for c in m.get("candlesticks", []))
    return rows


def candles_hist(http: Http, tickers: list[str], t0: int, t1: int) -> list[dict]:
    """Archived markets: one request per ticker, run in parallel threads."""
    from concurrent.futures import ThreadPoolExecutor

    def one(tk):
        res = http.get(f"{BASE}/historical/markets/{tk}/candlesticks",
                       {"start_ts": t0, "end_ts": t1, "period_interval": 1})
        return [_candle_row(tk, c) for c in (res or {}).get("candlesticks", [])]

    with ThreadPoolExecutor(max_workers=THREADS) as ex:
        return [r for part in ex.map(one, tickers) for r in part]


def _step_ok(strikes: pd.DataFrame, ranges: bool) -> pd.Series:
    """Every second strike when Kalshi lists them $100 apart (the design of July-October 2026); every strike
    when they are $250 or more apart (before 18 March 2026), otherwise only 2-3 strikes would survive."""
    k = (strikes["strike"] + 0.01).round()
    if ranges:
        return pd.Series(True, index=strikes.index)
    ku = k.drop_duplicates().sort_values().to_numpy()
    native = float(pd.Series(ku).diff().median()) if len(ku) > 1 else 100.0
    return (k % HIST_STRIKE_STEP == 0) if native < HIST_STRIKE_STEP else pd.Series(True, index=strikes.index)


def _sparse(f) -> bool:
    """A day downloaded with the old $200 filter while strikes were $250+ apart: 2-3 strikes per event."""
    if f.with_suffix(".dense").exists():
        return False
    try:
        prev = pd.read_csv(f, usecols=["event", "ticker"])
        return len(prev) > 0 and prev.groupby("event")["ticker"].nunique().median() < 6
    except Exception:
        return False


def raw_dir(series: str = "KXBTCD", minutes: int = CANDLE_MINUTES):
    if series == "KXBTCD" and minutes == CANDLE_MINUTES:
        return RAW_DIR
    return config.RAW / f"kalshi_{series.lower()}_{minutes}m"


def download_history(since: str = config.KALSHI_START, until: str | None = None,
                     hour_step: int = config.KALSHI_HOUR_STEP, http: Http | None = None,
                     series: str = "KXBTCD", minutes: int = CANDLE_MINUTES):
    """One CSV per day: data/raw/kalshi/YYYY-MM-DD.csv.gz with strike metadata and the candles of the first
    `minutes` minutes of each hourly event (the last hour for the daily and weekly 17:00 ET events)."""
    http = http or Http(min_interval=0.06)
    day = pd.Timestamp(since).date()
    # only complete days: today's file would be saved half-empty and then skipped on the next run
    last_day = pd.Timestamp(until).date() if until else pd.Timestamp(utcnow()).tz_convert(ET).date() - timedelta(days=1)
    out_dir = ensure(raw_dir(series, minutes))
    center = None
    ranges = series != "KXBTCD"
    log(f"kalshi: downloading {series} hourly ladders {day} .. {last_day}, {minutes} min of candles, into {out_dir}")
    log("kalshi: one line per day; the first one appears in about 20-60 seconds")
    while day <= last_day:
        f = out_dir / f"{day.isoformat()}.csv.gz"
        if f.exists() and series == "KXBTCD" and _sparse(f):
            log(f"kalshi: {day} was downloaded with too few strikes; downloading it again")
            f.unlink()
            refetched = True
        else:
            refetched = False
        if f.exists():
            prev = pd.read_csv(f)
            if len(prev):
                center = float(prev["expiration_value"].dropna().iloc[-1]) if prev["expiration_value"].notna().any() else center
            day += timedelta(days=1)
            continue
        rows = []
        for hour in range(0, 24, hour_step):
            close_et = datetime(day.year, day.month, day.day, hour, tzinfo=ET)
            close_utc = pd.Timestamp(close_et).tz_convert("UTC")
            if close_utc > pd.Timestamp(utcnow()):
                break
            ev = event_ticker(close_et, series)
            mk, archived = markets(http, ev)
            if not mk:
                continue
            strikes = pd.DataFrame([{
                "ticker": m["ticker"], "strike": m.get("floor_strike"), "cap_strike": m.get("cap_strike"),
                "strike_type": m.get("strike_type"), "result": m.get("result"),
                "expiration_value": m.get("expiration_value"), "open_time": m.get("open_time"),
                "close_time": m.get("close_time"), "volume_total": _num(m, "volume")} for m in mk])
            strikes = strikes.dropna(subset=["strike"])
            ref = center if center else strikes["strike"].median()
            # the 17:00 ET event is a daily (weekly on Fridays) market that opens 25 h (169 h) before
            # settlement: take its candles in the last hour, like every other event
            t_open = max(sec(to_utc(strikes["open_time"].iloc[0])), sec(to_utc(strikes["close_time"].iloc[0])) - 3600)
            t_end = t_open + 60 * minutes
            if archived:
                k = (strikes["strike"] + 0.01).round()
                # range buckets: every bucket near the money; "above" strikes: see _step_ok
                sel = strikes[((k / ref - 1).abs() <= BAND_HIST) & _step_ok(strikes, ranges)]
                cs = pd.DataFrame(candles_hist(http, list(sel["ticker"]), t_open, t_end))
            else:
                sel = strikes[(strikes["strike"] / ref - 1).abs() <= BAND]
                cs = pd.DataFrame(candles(http, list(sel["ticker"]), t_open, t_end))
                if cs.empty:  # the event moved to the archive in the meantime
                    k = (strikes["strike"] + 0.01).round()
                    sel = strikes[((k / ref - 1).abs() <= BAND_HIST) & _step_ok(strikes, ranges)]
                    cs = pd.DataFrame(candles_hist(http, list(sel["ticker"]), t_open, t_end))
            if cs.empty:
                continue
            cs = cs.merge(sel, on="ticker", how="left")
            cs["event"] = ev
            rows.append(cs)
            ev_val = pd.to_numeric(strikes["expiration_value"], errors="coerce").dropna()
            if len(ev_val):
                center = float(ev_val.iloc[0])
        if rows:
            pd.concat(rows, ignore_index=True).to_csv(f, index=False)
            if refetched:
                f.with_suffix(".dense").write_text("re-downloaded with the spacing-aware strike filter", encoding="utf-8")
        log(f"kalshi: {day} ({sum(len(r) for r in rows)} candle rows)")
        day += timedelta(days=1)


def build_interim() -> pd.DataFrame:
    files = sorted(RAW_DIR.glob("*.csv.gz"))
    if not files:
        return pd.DataFrame()
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df["open_ts"] = pd.to_datetime(df["open_time"], utc=True)
    df["settle_ts"] = pd.to_datetime(df["close_time"], utc=True)
    df["candle_end"] = pd.to_datetime(df["end_ts"], unit="s", utc=True)
    df["outcome"] = (df["result"] == "yes").astype(float)
    df.loc[~df["result"].isin(["yes", "no"]), "outcome"] = float("nan")
    return df.drop(columns=["open_time", "close_time", "end_ts"])


# ------------------------------------------------------------------ live snapshot
def snapshot(http: Http, snap_ts, series: str = "KXBTCD") -> pd.DataFrame:
    """Top of book for the hourly event that settles at the next full hour."""
    now_et = pd.Timestamp(snap_ts).tz_convert(ET)
    close_et = (now_et.floor("h") + pd.Timedelta(hours=1)).to_pydatetime()
    ev = event_ticker(close_et, series)
    res = http.get(f"{BASE}/markets", {"event_ticker": ev, "limit": 1000}) or {}
    rows = []
    for m in res.get("markets", []):
        extra = {} if series == "KXBTCD" else {"cap_strike": m.get("cap_strike"), "strike_type": m.get("strike_type")}
        # (KXBTCD keeps exactly its old columns: the day's file is appended to without a new header)
        rows.append({"snap_ts": snap_ts, "event": ev, "ticker": m["ticker"], "strike": m.get("floor_strike"), **extra,
                     "settle_ts": m.get("close_time"), "yes_bid": _num(m, "yes_bid"), "yes_ask": _num(m, "yes_ask"),
                     "yes_bid_size": _num(m, "yes_bid_size"), "yes_ask_size": _num(m, "yes_ask_size"),
                     "last": _num(m, "last_price"), "volume": _num(m, "volume")})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    # keep only strikes that are actually quoted
    return df[(df["yes_bid"].fillna(0) > 0) | (df["yes_ask"].fillna(1) < 1)]


# ------------------------------------------------------------------ earliest available history
def has_event(http: Http, day, series: str = "KXBTCD", hours=(12, 15, 10)) -> bool:
    """True when at least one of the hourly events of `day` (New York hours) exists on Kalshi."""
    d = pd.Timestamp(day).date()
    for h in hours:
        mk, _ = markets(http, event_ticker(datetime(d.year, d.month, d.day, h, tzinfo=ET), series))
        if mk:
            return True
    return False


def probe_start(http: Http | None = None, earliest: str = "2023-01-01", latest: str | None = None,
                series: str = "KXBTCD") -> str | None:
    """Earliest day with hourly events: a monthly scan, then a binary search inside the month before the hit.
    Assumes that once the series started it has events every day (true for KXBTCD)."""
    http = http or Http(min_interval=0.06)
    d = pd.Timestamp(earliest).normalize()
    stop = pd.Timestamp(latest or config.KALSHI_START).normalize()
    prev, hit = None, None
    while d <= stop:
        if has_event(http, d, series):
            hit = d
            break
        prev, d = d, d + pd.offsets.MonthBegin(1)
    if hit is None:
        return None
    lo = prev if prev is not None else hit   # lo has no events (or is the first probe), hit has events
    if lo == hit:
        return f"{hit:%Y-%m-%d}"
    while (hit - lo).days > 1:
        mid = lo + pd.Timedelta(days=(hit - lo).days // 2)
        if has_event(http, mid, series):
            hit = mid
        else:
            lo = mid
    log(f"kalshi: earliest {series} hourly event found on {hit:%Y-%m-%d}")
    return f"{hit:%Y-%m-%d}"


if __name__ == "__main__":
    # python -m src.download.kalshi [since] [until] [--series KXBTC] [--minutes 60]
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("since", nargs="?", default=config.KALSHI_START)
    ap.add_argument("until", nargs="?", default=None)
    ap.add_argument("--series", default="KXBTCD", choices=["KXBTCD", "KXBTC"])
    ap.add_argument("--minutes", type=int, default=CANDLE_MINUTES)
    ap.add_argument("--probe", action="store_true", help="only find the earliest day with hourly events")
    a = ap.parse_args()
    from src.runner import keep_awake
    keep_awake()
    if a.probe:
        print(probe_start(series=a.series))
    else:
        download_history(a.since, a.until, series=a.series, minutes=a.minutes)
