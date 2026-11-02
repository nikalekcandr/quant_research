"""Deribit official daily settlement prices (index btc_usd), all history, 100 per request."""
import sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from src.utils import Http  # noqa: E402
h = Http(min_interval=0.1, retries=4)
rows, off = [], 0
while off < 5000:
    r = h.get("https://www.deribit.com/api/v2/public/get_delivery_prices", {"index_name": "btc_usd", "offset": off, "count": 100})
    d = ((r or {}).get("result") or {}).get("data") or []
    rows.extend(d)
    print(off, len(d), d[-1]["date"] if d else None, flush=True)
    if not d:
        break
    off += len(d)
pd.DataFrame(rows).drop_duplicates("date").to_csv(config.RAW / "deribit" / "delivery_btc_usd.csv", index=False)
print("done", len(rows), flush=True)
