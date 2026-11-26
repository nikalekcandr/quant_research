"""Cache near-spot minute bid/ask closes for both recorded and unrecorded hours."""
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lab.history import Minutes
from lab.core import utcnow
from src.download.kalshi import markets, candles, candles_hist, event_ticker
from src.utils import Http, ET

CACHE = ROOT / "data/raw/wing_gap_candles"
START = pd.Timestamp("2026-11-09", tz="UTC")

def main():
    CACHE.mkdir(exist_ok=True)
    http = Http(min_interval=.45, retries=8)
    close = Minutes().close
    def fetch(end):
        path = CACHE / (end.strftime("%Y%m%d%H") + ".csv.gz")
        if path.exists():
            return
        prior = end-pd.Timedelta(minutes=32)
        ref = close.asof(prior)
        event = event_ticker(end.tz_convert(ET))
        items, historical = markets(http, event)
        meta = []
        for m in items or []:
            strike = m.get("floor_strike")
            if m.get("strike_type") == "greater" and strike and abs(float(strike)/ref-1) <= .025 and pd.Timestamp(m["close_time"]) == end:
                meta.append({k:m.get(k) for k in ("ticker","result","expiration_value","close_time","strike_type")} | {"strike":strike})
        if not meta:
            return
        tickers = [m["ticker"] for m in meta]
        getter = candles_hist if historical else candles
        rows = getter(http, tickers, int(end.timestamp())-31*60, int(end.timestamp())-15*60)
        if rows:
            tmp = path.with_suffix(".tmp")
            pd.DataFrame(rows).merge(pd.DataFrame(meta), on="ticker").to_csv(tmp,index=False,compression="gzip")
            tmp.replace(path)
    end = utcnow().floor("h")-pd.Timedelta(hours=1)
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(fetch,pd.date_range(START-pd.Timedelta(hours=12),end,freq="h")))
    print("Cached expiry hours:",len(list(CACHE.glob("*.csv.gz"))))

if __name__ == "__main__":
    main()
