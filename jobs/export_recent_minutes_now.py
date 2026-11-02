"""Export Binance 1-minute closes since 2026-10-20 (minutes.parquet + the lab's live buffer) to a small csv."""
import sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
m = pd.read_parquet(config.INTERIM / "minutes.parquet", columns=["open_time", "close"])
m["open_time"] = pd.to_datetime(m["open_time"], utc=True)
m = m[m["open_time"] >= "2026-10-20"].set_index("open_time")["close"]
r = pd.read_parquet(config.DATA / "lab" / "klines_recent.parquet")
r.index = pd.to_datetime(r.index, utc=True)
s = pd.concat([m, r["close"]])
s = s[~s.index.duplicated(keep="last")].sort_index()
s.rename("close").rename_axis("open_time").reset_index().to_csv(config.DATA / "lab" / "minutes_recent.csv.gz", index=False)
print(len(s), s.index.min(), s.index.max(), flush=True)
