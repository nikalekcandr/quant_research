"""Read-only public downloads for the most recent 24 resolved W04/W06 tickers."""
import json
from pathlib import Path
import sys
import pandas as pd
import requests

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from lab.core import LAB,atomic_write,utcnow


def main():
    folder=ROOT/"data/raw/wing_quant_inputs"
    folder.mkdir(parents=True,exist_ok=True)
    ledger=pd.read_csv(LAB/"ledger.csv",low_memory=False)
    closed=ledger[ledger.hyp.isin(["W04","W06"])&ledger.event.eq("CLOSE")].sort_values("ts")
    count=0
    for r in closed.drop_duplicates("instrument",keep="last").tail(24).itertuples():
        path=folder/(r.instrument+".json")
        if path.exists():
            continue
        end=pd.Timestamp(r.ts).ceil("h")
        params=dict(ticker=r.instrument,min_ts=int((end-pd.Timedelta(minutes=35)).timestamp()),
                    max_ts=int(end.timestamp()),limit=1000)
        prints=[]
        while True:
            res=requests.get("https://api.elections.kalshi.com/trade-api/v2/markets/trades",params=params,timeout=15)
            res.raise_for_status()
            data=res.json()
            prints.extend(data["trades"])
            if not data.get("cursor"):
                break
            params["cursor"]=data["cursor"]
        atomic_write(path,json.dumps(prints))
        count+=len(prints)
    print("Kalshi prints downloaded:",count,flush=True)
    res=requests.get("https://www.deribit.com/api/v2/public/get_book_summary_by_currency",
                     params={"currency":"BTC","kind":"option"},timeout=20)
    res.raise_for_status()
    now=utcnow()
    atomic_write(folder/("deribit_"+now.strftime("%Y%m%d_%H%M%S")+".json"),
                 json.dumps({"received":str(now),"result":res.json()["result"]}))


if __name__=="__main__":
    main()
