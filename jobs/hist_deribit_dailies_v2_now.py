"""Deribit BTC option trades of the nearest expiries (≤ 32 h away), 2023-01 → today, for the O04 straddle backtest.

For every day and each entry time 00:00, 09:00 and 16:00 UTC: all option trades of the 60 minutes before the entry
(history.deribit.com get_last_trades_by_currency_and_time), keeping only instruments that expire within 32 hours.
Plus the official daily settlement prices (get_delivery_prices, index btc_usd).
Output (resumable, one file per month):
    data/raw/deribit/dailies/YYYY-MM.csv.gz    entry, ts, instrument, price, mark_price, iv, index_price, direction, amount
    data/raw/deribit/delivery_btc_usd.csv      date, delivery_price
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from src.utils import Http  # noqa: E402

HIST = "https://history.deribit.com/api/v2/public"
LIVE = "https://www.deribit.com/api/v2/public"
OUT = config.RAW / "deribit" / "dailies"
OUT.mkdir(parents=True, exist_ok=True)
ENTRIES = (0, 9, 16)          # UTC; the next 08:00 expiry is then 8, 23 and 16 hours away
h = Http(min_interval=0.06, retries=4)


def ms(t):
    return int(pd.Timestamp(t).value // 10 ** 6)


def window(t0, t1, depth=0):
    """All trades in [t0, t1]; on an HTTP error retry, then split the window in halves (skip after 3 splits)."""
    import time
    out, start = [], ms(t0)
    while True:
        r = None
        for attempt in range(3):
            try:
                r = h.get(f"{HIST}/get_last_trades_by_currency_and_time",
                          {"currency": "BTC", "kind": "option", "start_timestamp": start, "end_timestamp": ms(t1),
                           "count": 1000, "sorting": "asc"})
                break
            except Exception as e:  # noqa: BLE001
                print(f"  retry {attempt} {pd.Timestamp(start, unit='ms')}: {str(e)[:120]}", flush=True)
                time.sleep(2 + 3 * attempt)
        if r is None:
            if depth >= 3:
                print(f"  SKIP window {pd.Timestamp(start, unit='ms')} .. {t1}", flush=True)
                return out
            mid = pd.Timestamp(start, unit="ms", tz="UTC") + (pd.Timestamp(t1) - pd.Timestamp(start, unit="ms", tz="UTC")) / 2
            return out + window(pd.Timestamp(start, unit="ms", tz="UTC"), mid, depth + 1) + window(mid, t1, depth + 1)
        res = (r or {}).get("result") or {}
        tr = res.get("trades") or []
        out.extend(tr)
        if not res.get("has_more") or not tr:
            return out
        start = tr[-1]["timestamp"] + 1


def expiry(name):
    try:
        return pd.Timestamp(pd.to_datetime(name.split("-")[1], format="%d%b%y")).tz_localize("UTC") + pd.Timedelta(hours=8)
    except Exception:
        return None


# settlement prices: jobs/hist_deribit_delivery_now.py

# 2. trades of the nearest expiries
now = pd.Timestamp.now(tz="UTC")
for month in pd.period_range("2023-01", now.strftime("%Y-%m"), freq="M"):
    f = OUT / f"{month}.csv.gz"
    if f.exists() and month != pd.Period(now, "M"):
        continue
    recs = []
    for day in pd.date_range(month.start_time, month.end_time.normalize(), freq="D", tz="UTC"):
        for hh in ENTRIES:
            t = day + pd.Timedelta(hours=hh)
            if t > now:
                continue
            for x in window(t - pd.Timedelta(minutes=60), t):
                e = expiry(x["instrument_name"])
                if e is None or not (t < e <= t + pd.Timedelta(hours=32)):
                    continue
                recs.append({"entry": t, "ts": x["timestamp"], "instrument": x["instrument_name"], "price": x.get("price"),
                             "mark_price": x.get("mark_price"), "iv": x.get("iv"), "index_price": x.get("index_price"),
                             "direction": x.get("direction"), "amount": x.get("amount")})
        print(f"  {day:%Y-%m-%d}: {len(recs)}", flush=True) if day.day % 7 == 0 else None
    pd.DataFrame(recs).to_csv(f, index=False)
    print(f"{month}: {len(recs)} trades", flush=True)
print("done", flush=True)
