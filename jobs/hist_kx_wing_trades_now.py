"""Maker test data, 04.11.2026: (1) every trade print of the Kalshi hourly-ladder strikes that had a 3–20¢ cheap side
in the last 45 minutes, 15.08–28.10.2026 (jobs/kx_wing_tickers.csv) -> data/raw/kx_wing_trades/<day>.csv;
(2) Binance BTC/USDT 1-second klines for the same days (data.binance.vision daily zips) -> data/raw/binance_1s/<day>.csv.gz.
Resumable by day. Then packs into _export/maker_data_<n>.tar.gz parts (≤ 400 MB each)."""
import io
import sys
import tarfile
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
KX = "https://api.elections.kalshi.com/trade-api/v2"
OUT_T, OUT_B = ROOT / "data/raw/kx_wing_trades", ROOT / "data/raw/binance_1s"
OUT_T.mkdir(parents=True, exist_ok=True)
OUT_B.mkdir(parents=True, exist_ok=True)
S = requests.Session()
S.headers["User-Agent"] = "btcvol-research"


def say(m):
    print(f"[{datetime.now():%H:%M:%S}] {m}", flush=True)


def get(url, params=None, tries=6):
    for i in range(tries):
        try:
            r = S.get(url, params=params, timeout=30)
            if r.status_code == 429:
                time.sleep(1 + 2 * i)
                continue
            if r.status_code == 404:
                return None
            r.raise_for_status()
            return r
        except Exception:  # noqa: BLE001
            time.sleep(1 + i)
    return None


def prints(tk):
    out = []
    for path in ("/markets/trades", "/historical/trades"):
        cur = None
        for _ in range(200):
            p = {"ticker": tk, "limit": 1000}
            if cur:
                p["cursor"] = cur
            r = get(KX + path, p)
            if r is None:
                break
            j = r.json()
            tr = j.get("trades") or []
            for t in tr:
                out.append({"ticker": tk, "created_time": t.get("created_time"),
                            "yes_price": t.get("yes_price_dollars", t.get("yes_price")),
                            "count": t.get("count_fp", t.get("count")),
                            "taker_side": t.get("taker_side"), "taker_outcome_side": t.get("taker_outcome_side"),
                            "taker_book_side": t.get("taker_book_side")})
            cur = j.get("cursor")
            if not cur or not tr:
                break
            time.sleep(0.05)
        if out:
            break
    return out


tk = pd.read_csv(ROOT / "jobs" / "kx_wing_tickers.csv")
tk["day"] = pd.to_datetime(tk.close_time, utc=True).dt.date.astype(str)
days = sorted(tk.day.unique(), reverse=True)
for d in days:
    f = OUT_T / f"{d}.csv"
    if not f.exists():
        with ThreadPoolExecutor(max_workers=4) as ex:
            rows = [r for part in ex.map(prints, tk[tk.day == d].ticker) for r in part]
        pd.DataFrame(rows).to_csv(f, index=False)
        say(f"trades {d}: {tk[tk.day == d].ticker.nunique()} strikes, {len(rows)} prints")
    fb = OUT_B / f"{d}.csv.gz"
    if not fb.exists():
        r = get(f"https://data.binance.vision/data/spot/daily/klines/BTCUSDT/1s/BTCUSDT-1s-{d}.zip")
        if r is not None:
            z = zipfile.ZipFile(io.BytesIO(r.content))
            k = pd.read_csv(z.open(z.namelist()[0]), header=None, usecols=[0, 4])
            k.columns = ["open_time", "close"]
            k.to_csv(fb, index=False, compression="gzip")
            say(f"binance 1s {d}: {len(k)} seconds")
        else:
            say(f"binance 1s {d}: not available")

exp = ROOT / "_export"
exp.mkdir(exist_ok=True)
files = sorted(OUT_T.glob("*.csv")) + sorted(OUT_B.glob("*.csv.gz"))
part, size, n = None, 0, 0
for p in files:
    if part is None or size > 400e6:
        if part:
            part.close()
        n += 1
        part, size = tarfile.open(exp / f"maker_data_{n}.tar.gz", "w:gz"), 0
    part.add(p, arcname=f"{p.parent.name}/{p.name}")
    size += p.stat().st_size
if part:
    part.close()
say(f"packed {n} parts")
