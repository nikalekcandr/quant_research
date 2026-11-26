"""Independent CAViaR test with the archive's observable candles; one contract only."""
import sys
from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from lab.core import RESULTS,atomic_write
from lab.wing_regime import regimes
from studies.wings_old_replay import archive
from studies.wings_odds_pool import score
from studies.wing_quant_replay import caviar

if __name__=="__main__":
    q,hours=archive()
    start=pd.Timestamp("2026-09-01",tz="UTC")
    hours=hours[(hours>=start)&(hours<pd.Timestamp("2026-10-28",tz="UTC"))]
    # Anchor the initial training window to the experiment, not to the replay date.
    minutes = pd.read_parquet(ROOT/"data/interim/minutes.parquet", columns=["open_time", "close"])
    close = minutes.set_index(pd.to_datetime(minutes.open_time, utc=True)).close.sort_index()
    close = close[~close.index.duplicated(keep="last")].loc[start-pd.Timedelta(days=90):hours.max()]
    if close.empty or close.index.min() > start-pd.Timedelta(days=89):
        raise ValueError("Missing start of archived CAViaR training history")
    g=q[q.end.isin(hours)&q.ts.dt.floor("min").map(regimes(close).blocked).eq(False)].copy()
    g["caviar"],_=caviar(close,g,start)
    g=g.dropna(subset=["caviar"])
    frame=pd.DataFrame(score(g,hours,["t4","caviar"],"01.09–27.10"))
    atomic_write(RESULTS/"wing_quant_archive.csv",frame.to_csv(index=False))
    print(frame.to_string(index=False))
