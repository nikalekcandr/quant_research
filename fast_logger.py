"""High-frequency logger for the microstructure hypothesis: do prediction-market quotes lag the Bitcoin price?

Every few seconds it records the Binance BTC price, the top of book of every strike of the next Kalshi hourly
ladder near the money, and — in the last hour before 12:00 New York — every strike of today's Polymarket
"above" ladder. alpha_live.py then measures, for each quote, how far it was from the fair price implied by the
current Binance price and whether the quote moved towards it within 15-60 seconds (a "mark-out").

    python fast_logger.py                       60 minutes, a snapshot every 10 seconds
    python fast_logger.py --minutes 90 --every 5
    python fast_logger.py --noon-window         only if 12:00 New York is less than 75 minutes away: run until
                                                12:03 New York (what the scheduled task uses)

Output: data/live/fast/YYYY-MM-DD.csv.gz (one row per quote per snapshot).
"""
from __future__ import annotations

import argparse
import time
import traceback

import pandas as pd

import config
from src import runner
from src.download import binance, kalshi, polymarket
from src.utils import ET, Http, append_csv_gz, log, noon_et_utc, utcnow

BAND = 0.012     # Kalshi strikes within +-1.2% of the price (about +-3 hourly standard deviations)


def kalshi_rows(http: Http, ts, price: float) -> list[dict]:
    now_et = pd.Timestamp(ts).tz_convert(ET)
    close_et = (now_et.floor("h") + pd.Timedelta(hours=1)).to_pydatetime()
    ev = kalshi.event_ticker(close_et)
    res = http.get(f"{kalshi.BASE}/markets", {"event_ticker": ev, "limit": 1000}) or {}
    out = []
    for m in res.get("markets", []):
        k = kalshi._num(m, "floor_strike") or m.get("floor_strike")
        if k is None or abs(float(k) / price - 1) > BAND:
            continue
        out.append({"ts": ts, "venue": "kalshi", "event": ev, "settle_ts": m.get("close_time"), "strike": float(k),
                    "bid": kalshi._num(m, "yes_bid"), "ask": kalshi._num(m, "yes_ask"),
                    "bid_size": kalshi._num(m, "yes_bid_size"), "ask_size": kalshi._num(m, "yes_ask_size"),
                    "btc": price})
    return out


def pm_rows(http: Http, ts, price: float, lad: pd.DataFrame) -> list[dict]:
    bk = polymarket.books(http, list(lad["yes_token"]))
    out = []
    for _, m in lad.iterrows():
        b = bk.get(m["yes_token"])
        if not b:
            continue
        bid, bsz, _ = polymarket._top(b.get("bids"), "max")
        ask, asz, _ = polymarket._top(b.get("asks"), "min")
        out.append({"ts": ts, "venue": "polymarket", "event": m["event_slug"], "settle_ts": str(m["settle_ts"]),
                    "strike": float(m["strike"]), "bid": bid, "ask": ask, "bid_size": bsz, "ask_size": asz, "btc": price})
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=60)
    ap.add_argument("--every", type=float, default=10, help="seconds between snapshots")
    ap.add_argument("--noon-window", action="store_true")
    a = ap.parse_args(argv)
    runner.log_to_file("fast_logger")
    now = pd.Timestamp(utcnow())
    noon = noon_et_utc(now.tz_convert(ET).date())
    if a.noon_window:
        if not (pd.Timedelta(0) < noon - now <= pd.Timedelta(minutes=75)):
            log(f"noon New York is {(noon - now).total_seconds() / 60:.0f} min away: nothing to do")
            return
        end = noon + pd.Timedelta(minutes=3)
    else:
        end = now + pd.Timedelta(minutes=a.minutes)
    http = Http(min_interval=0.03, retries=2)
    lad = pd.DataFrame()
    ev = polymarket.fetch_event(http, now.tz_convert(ET).date())
    if ev is not None:
        lad = polymarket.parse_event(ev)
    log(f"fast logger until {end:%H:%M:%S} UTC, every {a.every:.0f} s; Polymarket ladder: {len(lad)} strikes")
    n = 0
    while pd.Timestamp(utcnow()) < end:
        t0 = time.time()
        ts = pd.Timestamp(utcnow()).floor("s")
        rows = []
        try:
            price = float(binance.api_get(http, "/api/v3/ticker/price", {"symbol": "BTCUSDT"})["price"])
            rows.append({"ts": ts, "venue": "binance", "btc": price})
            try:
                rows += kalshi_rows(http, ts, price)
            except Exception as e:
                log(f"kalshi: {e!r}"[:200])
            if len(lad) and pd.Timedelta(0) < noon - ts <= pd.Timedelta(minutes=70):
                try:
                    rows += pm_rows(http, ts, price, lad)
                except Exception as e:
                    log(f"polymarket: {e!r}"[:200])
        except Exception:
            log(traceback.format_exc()[-300:])
        append_csv_gz(pd.DataFrame(rows), config.LIVE / "fast" / f"{ts:%Y-%m-%d}.csv.gz")
        n += 1
        if n % 30 == 0:
            log(f"{n} snapshots, last {len(rows)} rows")
        time.sleep(max(0.5, a.every - (time.time() - t0)))
    log(f"fast logger finished: {n} snapshots")


if __name__ == "__main__":
    main()
