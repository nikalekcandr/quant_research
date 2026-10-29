"""Probe the Kalshi trades endpoints for one live and one archived 15-minute market (prints raw answers)."""
import json, sys, time
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from src.utils import Http
KX = "https://api.elections.kalshi.com/trade-api/v2"
mk = pd.read_csv(config.RAW / "kx15" / "markets.csv")
h = Http(min_interval=0.2, retries=2)
for hist in (False, True):
    tk = mk[mk["historical"] == hist]["ticker"].iloc[-5]
    for path in ("/markets/trades", "/historical/trades"):
        t0 = time.time()
        try:
            r = h.get(f"{KX}{path}", {"ticker": tk, "limit": 1000})
        except Exception as e:
            print(path, tk, "ERROR", str(e)[:200], flush=True); continue
        tr = (r or {}).get("trades") or []
        print(path, tk, f"{time.time()-t0:.1f}s", "n", len(tr), "cursor", bool((r or {}).get("cursor")),
              "tickers", sorted({x.get("ticker") for x in tr})[:3], json.dumps(tr[:1])[:400], flush=True)
