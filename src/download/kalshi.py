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

import threading
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

# --- runnable both ways -------------------------------------------------------
# `python -m src.download.NAME` and `python src/download/NAME.py` (an IDE "Run" button) must both work:
# when this file is executed as a script, Python puts src/download/ on sys.path instead of the repository
# root, and `import config` fails. Put the root back before importing anything from the project.
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
# -----------------------------------------------------------------------------

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
WORKERS = 4                # days fetched at the same time (--workers); the rate limit is shared anyway
MIN_INTERVAL = 0.15        # seconds between requests, aggregate across threads (--min-interval)
REPORT_EVERY = 30.0        # seconds between progress lines (a day can take a minute or two)

# live counters, reset by every download_history() call: used for the progress line and the ETA
STATS = {"requests": 0, "days_done": 0, "days_total": 0, "days_failed": 0, "failed": [],
         "ev_event": 0, "ev_batch": 0, "ev_ticker": 0, "req_per_day_prior": 60}
# Which candle endpoints answer on this machine. Probed once per run: on some networks the event-level batch
# works, on others only the per-ticker archive does — a factor of eight in both requests and hours.
STRATEGY = {"order": ["event", "batch", "ticker"], "event_misses": 0}
_STRATEGY_LOCK = threading.Lock()
_STATS_LOCK = threading.Lock()


def _count(n: int = 1) -> None:
    with _STATS_LOCK:
        STATS["requests"] += n


def _snapshot() -> tuple[int, int]:
    with _STATS_LOCK:
        return STATS["days_done"], STATS["requests"]


def _human(seconds: float) -> str:
    if seconds != seconds or seconds == float("inf"):      # NaN or unknown
        return "?"
    if seconds >= 5400:
        return f"{seconds / 3600:.1f} h"
    if seconds >= 90:
        return f"{seconds / 60:.0f} min"
    return f"{seconds:.0f} s"


PRIOR_REQ_PER_DAY = 60      # fallback: 24 events x ~2.5 requests; the probe sets the real prior


def _reporter(t0: float, total: int, stop: threading.Event, every: float, http: Http | None = None) -> None:
    """One line every `every` seconds: how many days are done, the request rate, and the ETA.

    Without it an archive download looks dead: the first day takes a minute or two (a few hundred
    requests per archived day), and until it finishes nothing is printed at all.
    """
    while not stop.wait(every):
        done, req = _snapshot()
        with _STATS_LOCK:
            failed = STATS["days_failed"]
        el = max(time.time() - t0, 1e-9)
        rate = req / el
        with _STATS_LOCK:
            prior = STATS.get("req_per_day_prior", PRIOR_REQ_PER_DAY)
            ev_event, ev_batch, ev_ticker = STATS["ev_event"], STATS["ev_batch"], STATS["ev_ticker"]
        if done:
            eta = (total - done) * el / done
        else:                                   # no finished day yet: ETA from the requests we have spent
            eta = max(total * prior - req, 0) / rate if rate > 0 else float("inf")
        gap = getattr(http, "min_interval", 0.0)
        n429 = (getattr(http, "stats", None) or {}).get("429", 0)
        level = "пачка" if ev_event else ("батч" if ev_batch else "пострайково")
        log(f"kalshi: {done}/{total} days done, {el / 60:.1f} min elapsed, {req} requests ({rate:.1f}/s), "
            f"gap {gap:.2f} s, 429: {n429}, {level} {ev_event + ev_batch + ev_ticker}/~24x{total} событий, "
            f"ETA {_human(eta)}, {failed} failed")


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


ARCHIVE_AFTER_DAYS = 7     # older than this, ask the archive first: the live call would only return nothing


def markets(http: Http, ev: str, archived_hint: bool = False) -> tuple[list[dict], bool]:
    """Strikes of one hourly event and whether it lives in the historical archive.

    `archived_hint` saves one doomed request per event: for a day two months old the live endpoint is empty by
    definition, so the archive is asked first (24 events x 212 days = 5 thousand requests saved).
    """
    def live():
        _count()
        return (http.get(f"{BASE}/markets", {"event_ticker": ev, "limit": 1000}) or {}).get("markets") or []

    def hist():
        _count()
        return (http.get(f"{BASE}/historical/markets", {"event_ticker": ev, "limit": 1000}) or {}).get("markets") or []

    if archived_hint:
        mk = hist()
        return (mk, True) if mk else (live(), False)
    mk = live()
    if mk:
        return mk, False
    return hist(), True


def _candle_row(ticker, c):
    # `end_period_ts` on every endpoint we use; a missing one drops the row instead of killing the day
    return {"ticker": ticker, "end_ts": c.get("end_period_ts", c.get("end_ts")),
            "yes_bid": _num(c.get("yes_bid", {}), "close"), "yes_ask": _num(c.get("yes_ask", {}), "close"),
            "last": _num(c.get("price", {}), "close"), "volume": _num(c, "volume"),
            "open_interest": _num(c, "open_interest")}


def candles(http: Http, tickers: list[str], t0: int, t1: int) -> list[dict]:
    """Live markets: up to 50 tickers per request."""
    rows = []
    for i in range(0, len(tickers), 50):
        chunk = tickers[i:i + 50]
        _count()
        res = http.get(f"{BASE}/markets/candlesticks",
                       {"market_tickers": ",".join(chunk), "start_ts": t0, "end_ts": t1, "period_interval": 1})
        for m in (res or {}).get("markets", []):
            rows.extend(_candle_row(m.get("market_ticker"), c) for c in m.get("candlesticks", []))
    return rows


def candles_event(http: Http, series: str, ev: str, t0: int, t1: int, max_pages: int = 6) -> list[dict]:
    """Every strike of one hourly event in ONE request.

    Kalshi answers with `market_tickers` plus `market_candlesticks` (one array per ticker) for all markets of the
    event, live or archived — "all events are accessible through this endpoint, even if their associated markets
    are older than the historical cutoff". Above ~5000 candlesticks it paginates and returns `adjusted_end_ts`,
    which is the start of the next page. This is the difference between ~400 requests per archived day (one per
    strike) and ~2 per event: 24 events a day instead of a few hundred requests.
    """
    rows, start = [], int(t0)
    for _ in range(max_pages):
        _count()
        res = http.get(f"{BASE}/series/{series}/events/{ev}/candlesticks",
                       {"start_ts": start, "end_ts": int(t1), "period_interval": 1})
        res = res or {}
        tickers = res.get("market_tickers") or []
        for tk, cs in zip(tickers, res.get("market_candlesticks") or []):
            for c in (cs or []):
                row = _candle_row(tk, c)
                if row["end_ts"] is None or not (t0 <= row["end_ts"] <= t1):
                    continue                     # a page can overshoot the requested range
                rows.append(row)
        nxt = res.get("adjusted_end_ts")
        if not nxt or int(nxt) >= t1 or int(nxt) <= start:
            break
        start = int(nxt)
    return rows


def candles_hist(http: Http, tickers: list[str], t0: int, t1: int) -> list[dict]:
    """Archived markets, one request per ticker — the slow fallback when the event endpoint is unavailable."""
    from concurrent.futures import ThreadPoolExecutor

    def one(tk):
        _count()
        res = http.get(f"{BASE}/historical/markets/{tk}/candlesticks",
                       {"start_ts": t0, "end_ts": t1, "period_interval": 1})
        return [_candle_row(tk, c) for c in (res or {}).get("candlesticks", [])]

    with ThreadPoolExecutor(max_workers=THREADS) as ex:
        return [r for part in ex.map(one, tickers) for r in part]


def _disable(level: str, ev: str, why: str) -> None:
    """A dead endpoint is dropped for the rest of the run instead of costing one request per event."""
    with _STRATEGY_LOCK:
        if level == "event":
            STRATEGY["event_misses"] += 1
            if STRATEGY["event_misses"] >= 3 and "event" in STRATEGY["order"]:
                STRATEGY["order"].remove("event")
                log(f"kalshi: пачка по событию не отдаёт свечи ({why}); "
                    f"дальше качаю {'батчем по тикерам' if 'batch' in STRATEGY['order'] else 'по одному страйку'}")
        elif level in STRATEGY["order"]:
            STRATEGY["order"].remove(level)
            log(f"kalshi: {level}: не работает ({why}); остаётся {', '.join(STRATEGY['order']) or 'ничего'}")


def _event_candles(http: Http, series: str, ev: str, t0: int, t1: int, tickers: list[str], live: bool) -> list[dict]:
    """All strikes of the event, trying the cheapest endpoint that still answers.

    event  — one request per event (every strike in the response);
    batch  — one request per 50 tickers (`/markets/candlesticks`), the live market endpoint;
    ticker — one request per strike against the archive: 8x more requests, the old behaviour.

    The order is probed once per run and re-checked only on failure, so a changed or blocked endpoint costs a
    few requests, not a whole day.
    """
    want = {str(t) for t in tickers}
    with _STRATEGY_LOCK:
        order = list(STRATEGY["order"])
    for level in order:
        try:
            if level == "event":
                rows = candles_event(http, series, ev, t0, t1)
                ok = bool(rows)
                if rows:
                    with _STATS_LOCK:
                        STATS["ev_event"] += 1
                        STRATEGY["event_misses"] = 0
            elif level == "batch":
                rows = candles(http, tickers, t0, t1)
                ok = bool(rows)
                if ok:
                    with _STATS_LOCK:
                        STATS["ev_batch"] += 1
            else:
                rows = candles_hist(http, tickers, t0, t1)
                ok = bool(rows)
                if ok:
                    with _STATS_LOCK:
                        STATS["ev_ticker"] += 1
        except Exception as e:
            _disable(level, ev, f"{type(e).__name__}: {str(e)[:90]}")
            continue
        if ok:
            if level == "event":
                picked = [r for r in rows if str(r["ticker"]) in want]
                if picked:
                    return picked
                # the endpoint answered, but with tickers we do not hold: not the same markets, try the next
                _disable(level, ev, "в ответе нет нужных страйков")
                continue
            return rows
        _disable(level, ev, "пустой ответ")
    return []


def _step_ok(strikes: pd.DataFrame, ranges: bool) -> pd.Series:
    """Every second strike when Kalshi lists them $100 apart (the design of July-October 2026); every strike
    when they are $250 or more apart (before 18 March 2026), otherwise only 2-3 strikes would survive."""
    k = (strikes["strike"] + 0.01).round()
    if ranges:
        return pd.Series(True, index=strikes.index)
    ku = k.drop_duplicates().sort_values().to_numpy()
    native = float(pd.Series(ku).diff().median()) if len(ku) > 1 else 100.0
    return (k % HIST_STRIKE_STEP == 0) if native < HIST_STRIKE_STEP else pd.Series(True, index=strikes.index)


def _marker(f):
    """Where we note "this day was re-downloaded with the spacing-aware strike filter".

    It must NOT end in .csv: readers glob `*.csv*`, and a marker named `2026-01-19.csv.dense` was parsed as data —
    "Usecols do not match columns" killed twenty studies. Old files are still recognised, see `_sparse`.
    """
    return f.parent / (f.name.split(".csv")[0] + ".dense")


def _sparse(f) -> bool:
    """A day downloaded with the old $200 filter while strikes were $250+ apart: 2-3 strikes per event."""
    # new name, then the two legacy shapes: `X.csv.dense` (from `X.csv.gz`) and `X.csv.dense` (from `X.csv`)
    if _marker(f).exists() or Path(str(f) + ".dense").exists() or f.with_suffix(".dense").exists():
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


def _one_day(http: Http, day, **kw) -> None:
    """Download one day; a failure is logged and counted, and never aborts the other days.

    A run of a few hundred days will hit a 429 or a timeout at some point. Those days are simply retried
    by the next run of the same command — the file is either complete or absent, never half written.
    """
    try:
        _one_day_impl(http, day, **kw)
    except Exception as e:
        with _STATS_LOCK:
            STATS["days_failed"] += 1
            if len(STATS["failed"]) < 200:
                STATS["failed"].append(str(day))
        log(f"kalshi: {day} failed ({type(e).__name__}: {str(e)[:180]}) — the next run will retry it")
    finally:
        with _STATS_LOCK:
            STATS["days_done"] += 1


def _one_day_impl(http: Http, day, *, minutes: int, series: str, hour_step: int, out_dir, center: dict,
                  lock, workers_note: str = "") -> None:
    old_day = (pd.Timestamp(utcnow()).tz_convert(ET).date() - day).days > ARCHIVE_AFTER_DAYS
    """Download one calendar day into its own CSV (idempotent: an existing file is kept and skipped)."""
    f = out_dir / f"{day.isoformat()}.csv.gz"
    events_n, strikes_n = 0, 0
    refetched = False
    if f.exists() and series == "KXBTCD" and _sparse(f):
        log(f"kalshi: {day} was downloaded with too few strikes; downloading it again")
        f.unlink()
        refetched = True
    if f.exists():
        prev = pd.read_csv(f)
        if len(prev) and prev["expiration_value"].notna().any():
            with lock:
                center["v"] = float(prev["expiration_value"].dropna().iloc[-1])
        return
    rows = []
    for hour in range(0, 24, hour_step):
        close_et = datetime(day.year, day.month, day.day, hour, tzinfo=ET)
        close_utc = pd.Timestamp(close_et).tz_convert("UTC")
        if close_utc > pd.Timestamp(utcnow()):
            break
        ev = event_ticker(close_et, series)
        mk, archived = markets(http, ev, archived_hint=old_day)
        if not mk:
            continue
        strikes = pd.DataFrame([{
            "ticker": m["ticker"], "strike": m.get("floor_strike"), "cap_strike": m.get("cap_strike"),
            "strike_type": m.get("strike_type"), "result": m.get("result"),
            "expiration_value": m.get("expiration_value"), "open_time": m.get("open_time"),
            "close_time": m.get("close_time"), "volume_total": _num(m, "volume")} for m in mk])
        strikes = strikes.dropna(subset=["strike"])
        with lock:
            ref = center["v"] if center["v"] else strikes["strike"].median()
        # the 17:00 ET event is a daily (weekly on Fridays) market that opens 25 h (169 h) before
        # settlement: take its candles in the last hour, like every other event
        t_open = max(sec(to_utc(strikes["open_time"].iloc[0])), sec(to_utc(strikes["close_time"].iloc[0])) - 3600)
        t_end = t_open + 60 * minutes
        ranges = series != "KXBTCD"
        k = (strikes["strike"] + 0.01).round()
        if archived:
            # range buckets: every bucket near the money; "above" strikes: see _step_ok
            sel = strikes[((k / ref - 1).abs() <= BAND_HIST) & _step_ok(strikes, ranges)]
        else:
            sel = strikes[(strikes["strike"] / ref - 1).abs() <= BAND]
        cs = pd.DataFrame(_event_candles(http, series, ev, t_open, t_end, list(sel["ticker"]), live=not archived))
        if cs.empty and not archived:  # the event moved to the archive in the meantime
            sel = strikes[((k / ref - 1).abs() <= BAND_HIST) & _step_ok(strikes, ranges)]
            cs = pd.DataFrame(candles_hist(http, list(sel["ticker"]), t_open, t_end))
        if cs.empty:
            continue
        events_n += 1
        strikes_n += len(sel)
        cs = cs.merge(sel, on="ticker", how="left")
        cs["event"] = ev
        rows.append(cs)
        ev_val = pd.to_numeric(strikes["expiration_value"], errors="coerce").dropna()
        if len(ev_val):
            with lock:
                center["v"] = float(ev_val.iloc[0])
    if rows:
        pd.concat(rows, ignore_index=True).to_csv(f, index=False)
        if refetched:
            _marker(f).write_text("re-downloaded with the spacing-aware strike filter", encoding="utf-8")
    per_strike = events_n + strikes_n          # one markets call + one candle call per strike, worst case
    log(f"kalshi: {day} ({sum(len(r) for r in rows)} candle rows, {events_n} событий, {strikes_n} страйков "
        f"→ ~{per_strike} запросов, если пачка не работает){workers_note}")


def probe_candles(http: Http | None = None, day: str | None = None, series: str = "KXBTCD") -> dict:
    """Which candle endpoint answers here — asked on one real archived event before a multi-hour run.

    Without this the run looks identical whether a day costs 50 requests or 430, and the ETA is a guess.
    """
    http = http or Http(min_interval=0.25)
    d = pd.Timestamp(day).date() if day else (pd.Timestamp(utcnow()).tz_convert(ET).date() - timedelta(days=10))
    ev = event_ticker(datetime(d.year, d.month, d.day, 12, tzinfo=ET), series)
    try:
        mk, archived = markets(http, ev)
    except Exception as e:
        log(f"kalshi: проверка эндпоинтов не удалась ({type(e).__name__}: {str(e)[:100]})")
        return {}
    if not mk:
        log(f"kalshi: проверка эндпоинтов: событие {ev} не найдено, проверяю вслепую")
        return {}
    tickers = [m["ticker"] for m in mk]
    t_end = int(sec(to_utc(mk[0]["close_time"])))
    t0 = t_end - 3600
    found = {}
    for level, fn in (("event", lambda: candles_event(http, series, ev, t0, t_end)),
                      ("batch", lambda: candles(http, tickers[:50], t0, t_end)),
                      ("ticker", lambda: candles_hist(http, tickers[:1], t0, t_end))):
        try:
            found[level] = bool(fn())
        except Exception:
            found[level] = False
    best = next((lv for lv in ("event", "batch", "ticker") if found.get(lv)), None)
    with _STRATEGY_LOCK:
        STRATEGY["order"] = ([best] + [lv for lv in ("event", "batch", "ticker") if lv != best and found.get(lv)]
                             if best else ["ticker"])
    per_day = {"event": 60, "batch": 72, "ticker": 430, None: 430}[best]
    with _STATS_LOCK:
        STATS["req_per_day_prior"] = per_day
    names = {"event": "все страйки события одной пачкой", "batch": "батч по 50 тикеров",
             "ticker": "по одному страйку (архив)"}
    log(f"kalshi: проверка на {ev} (архив): " + ", ".join(
        f"{lv} — {'да' if found.get(lv) else 'нет'}" for lv in ("event", "batch", "ticker")))
    log(f"kalshi: значит каждая пара «событие + страйки» стоит ~{per_day / 24:.0f} запросов; "
        f"сутки ≈ {per_day} запросов, способ: {names.get(best, 'неизвестен')}")
    return {"found": found, "best": best, "req_per_day": per_day}


def download_history(since: str = config.KALSHI_START, until: str | None = None,
                     hour_step: int = config.KALSHI_HOUR_STEP, http: Http | None = None,
                     series: str = "KXBTCD", minutes: int = CANDLE_MINUTES,
                     workers: int = WORKERS, min_interval: float = MIN_INTERVAL,
                     report_every: float = REPORT_EVERY, probe: bool = True):
    """One CSV per day: data/raw/kalshi/YYYY-MM-DD.csv.gz with strike metadata and the candles of the first
    `minutes` minutes of each hourly event (the last hour for the daily and weekly 17:00 ET events).

    `workers` days are fetched at the same time, sharing one polite rate limit. A day costs about 2 requests per
    hourly event (~48 round-trips: the event's markets, then all its strikes in one batch response), so the
    aggregate rate in requests per second — not the latency of one request — sets the pace.
    """
    http = http or Http(min_interval=min_interval)
    t0 = time.time()
    day = pd.Timestamp(since).date()
    # only complete days: today's file would be saved half-empty and then skipped on the next run
    last_day = (pd.Timestamp(until).date() if until
                else pd.Timestamp(utcnow()).tz_convert(ET).date() - timedelta(days=1))
    out_dir = ensure(raw_dir(series, minutes))
    window = [day + timedelta(days=i) for i in range((last_day - day).days + 1)]
    days = []
    for d in window:                       # already on disk (and not one of the old sparse days)? not our work
        f = out_dir / f"{d.isoformat()}.csv.gz"
        if f.exists() and not (series == "KXBTCD" and _sparse(f)):
            continue
        days.append(d)
    # Newest first: the studies live on the last months, and the newest days are exactly the ones the venue still
    # serves in batches (cheap). Oldest first means the useful data arrives last, after days of history.
    days.sort(reverse=True)
    if not days:
        log(f"kalshi: {series} {day} .. {last_day}: всё уже на диске ({len(window)} суток), качать нечего")
        return
    log(f"kalshi: downloading {series} hourly ladders {day} .. {last_day}, {minutes} min of candles, "
        f"{len(days)} day(s) to fetch ({len(window) - len(days)} already on disk), {max(1, workers)} worker(s), "
        f"into {out_dir}")
    if probe:
        probe_candles(http, series=series)      # only when there is work: no days, no probe
    else:
        log(f"kalshi: about {1 / max(min_interval, 1e-3):.0f} requests/s shared by all workers")
    log("kalshi: newest days first (the studies need the last months; those are also the cheapest); one line per "
        f"finished day, progress with the ETA every {report_every:.0f} s")
    center, lock = {"v": None}, threading.Lock()
    kw = dict(minutes=minutes, series=series, hour_step=hour_step, out_dir=out_dir, center=center, lock=lock)
    with _STATS_LOCK:
        STATS.update(requests=0, days_done=0, days_total=len(days), days_failed=0, failed=[])
    stop = threading.Event()
    watchdog = threading.Thread(target=_reporter, args=(t0, len(days), stop, report_every), kwargs={"http": http},
                                daemon=True)
    watchdog.start()
    try:
        if workers <= 1:
            for d in days:
                _one_day(http, d, **kw)
        else:
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=workers) as ex:
                list(ex.map(lambda d: _one_day(http, d, **kw), days))
    finally:
        stop.set()
        watchdog.join(timeout=1)
    done, req = _snapshot()
    with _STATS_LOCK:
        failed = STATS["days_failed"]
        failed_days = list(STATS["failed"])
    log(f"kalshi: finished {done}/{len(days)} days in {_human(time.time() - t0)}, {req} requests, "
        f"{failed} failed")
    if failed:
        log(f"kalshi: failed days ({failed}): {', '.join(failed_days[:10])}"
            + (" ..." if failed > 10 else ""))
        log("kalshi: nothing is lost — run the same command again, it skips the days already on disk")


def ladder_dirs() -> list:
    """Every ladder dataset on disk, coarsest first: 10-minute candles, then the whole hour.

    The archived `kalshi_kxbtcd_60m` files contain the first 10 minutes too, so reading both and keeping the
    last copy per (ticker, candle_end) gives the finest series available for every day, without losing the
    months that were only ever downloaded at 10 minutes.
    """
    out = []
    for minutes in sorted({CANDLE_MINUTES, 60}):
        d = raw_dir("KXBTCD", minutes)
        if d.exists():
            out.append(d)
    return out or [RAW_DIR]


def build_interim() -> pd.DataFrame:
    files = sorted(f for d in ladder_dirs() for f in d.glob("*.csv.gz"))
    if not files:
        return pd.DataFrame()
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    if len(files) > 1:                       # 60-minute rows win over the 10-minute ones
        df = (df.assign(_f=df["ticker"].astype(str))
                .drop_duplicates(subset=["_f", "end_ts", "event"], keep="last")
                .drop(columns="_f"))
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
    _count()
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
    ap.add_argument("--workers", type=int, default=WORKERS, help="days fetched at the same time (default 4)")
    ap.add_argument("--min-interval", type=float, default=MIN_INTERVAL,
                    help="seconds between requests, aggregate (default 0.15; raise it if Kalshi answers 429)")
    a = ap.parse_args()
    from src.runner import keep_awake
    keep_awake()
    if a.probe:
        print(probe_start(series=a.series))
    else:
        download_history(a.since, a.until, series=a.series, minutes=a.minutes,
                         workers=a.workers, min_interval=a.min_interval)
