"""Cache official settlement values for expired hourly ladders with paired research forecasts."""
import json
from pathlib import Path
import re
import sys

import pandas as pd
import requests

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from lab.core import LAB, RESULTS, atomic_write, utcnow


def main():
    dest = RESULTS/"wing_forecast_settlements.json"
    cache=LAB.parent/"raw/wing_quant_inputs/settlements.json"
    values = json.loads(cache.read_text(encoding="utf-8")) if cache.exists() else {}
    if dest.exists():
        values.update(json.loads(dest.read_text(encoding="utf-8")))
    paths=sorted((LAB/"wing_forecasts").glob("*.csv"))
    capacity=RESULTS/"wing_capacity_entries.csv"
    if capacity.exists():
        paths.append(capacity)
    for path in paths:
        if path==capacity:
            quotes=pd.read_csv(path,usecols=["end","market","pnl"]).query("pnl != pnl")
        else:
            quotes = pd.read_csv(path,usecols=["end","market"])
        expired = quotes[pd.to_datetime(quotes.end,utc=True)<utcnow()]
        for ticker in expired.market.drop_duplicates():
            event = ticker.rpartition("-T")[0]
            if event in values or not re.fullmatch(r"KXBTCD-[0-9A-Z]+-T\d+(?:\.\d+)?",ticker):
                continue
            response = requests.get("https://api.elections.kalshi.com/trade-api/v2/markets/"+ticker,timeout=10)
            response.raise_for_status()
            m = response.json()["market"]
            if m["event_ticker"] != event or m["ticker"] != ticker or m.get("strike_type")!="greater":
                continue
            if m.get("result") in ("yes","no") and m.get("expiration_value") is not None:
                value = float(m["expiration_value"])
                if (value>float(m["floor_strike"])) == (m["result"]=="yes"):
                    values[event] = {"value":value,"source":ticker}
                    atomic_write(cache,json.dumps(values,indent=2,allow_nan=False))
                    atomic_write(dest,json.dumps(values,indent=2,allow_nan=False))
    atomic_write(cache,json.dumps(values,indent=2,allow_nan=False))
    atomic_write(dest,json.dumps(values,indent=2,allow_nan=False))
    print(f"{len(values)} official expiry values")


if __name__=="__main__":
    main()
