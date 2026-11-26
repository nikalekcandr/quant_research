"""Top-of-book capacity on the actual closed W01–W12 signals; preserve missing coverage."""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from lab.core import LAB,RESULTS,atomic_write
from lab.report import void_mask
from studies.wing_capacity_ml import fee


def main():
    w=pd.read_csv(LAB/"ledger.csv",low_memory=False)
    w=w[w.event.eq("CLOSE")&w.hyp.str.fullmatch(r"W\d\d")].copy()
    w=w[~void_mask(w)].copy()
    w["ts"]=pd.to_datetime(w.ts,utc=True,format="mixed").dt.floor("s")
    rows=[]
    for day,g in w.groupby(w.ts.dt.date):
        path=LAB/"markets"/f"{day}.csv"
        quotes={}
        if path.exists():
            cols={"market","ts","bid","ask","book","bid_size","ask_size"}
            data=pd.read_csv(path,usecols=lambda k:k in cols)
            data=data[data.market.isin(g.instrument)]
            data["ts"]=pd.to_datetime(data.ts,utc=True,format="mixed").dt.floor("s")
            quotes={(r.ts,r.market):r for r in data.itertuples()}
        for r in g.itertuples():
            info=json.loads(r.info)
            entry=dict(rule=r.hyp,venue=r.venue,ts=r.ts,market=r.instrument,status="нет стакана",n=0,cost=0.,pnl=0.,fees=0.)
            rows.append(entry)
            if info.get("maker"):
                entry["status"]="мейкер: нет исполнимого объёма очереди"
                continue
            book=quotes.get((r.ts,r.instrument))
            if book is None or (r.venue=="kalshi" and getattr(book,"book",0)!=1):
                continue
            p=book.ask if r.side=="yes" else 1-book.bid
            size=book.ask_size if r.side=="yes" else book.bid_size
            if not np.isfinite(size) or abs(p-r.price)>1e-6 or not 0<p<1:
                continue
            n=int(size)
            if n<1:
                continue
            f=fee(n,p) if r.venue=="kalshi" else n*r.fee
            # Price is unchanged from the qualifying recorded rule decision.
            entry.update(status="сопоставлено",n=n,cost=n*p+f,fees=f,pnl=n*(r.exit_price-p)-f)
    detail=pd.DataFrame(rows)
    atomic_write(RESULTS/"wing_ledger_capacity_entries.csv",detail.to_csv(index=False))
    summary=[]
    for rule,g in detail.groupby("rule"):
        filled=g[g.n.gt(0)]
        cost=filled.cost.sum()
        summary.append(dict(rule=rule,closed=len(g),matched=len(filled),maker=int(g.status.str.startswith("мейкер").sum()),
                            contracts=int(filled.n.sum()),capital=round(cost,2),fees=round(filled.fees.sum(),2),
                            pnl=round(filled.pnl.sum(),2) if cost else None,roi_pct=round(100*filled.pnl.sum()/cost,3) if cost else None))
    atomic_write(RESULTS/"wing_ledger_capacity.csv",pd.DataFrame(summary).to_csv(index=False))


if __name__=="__main__":
    main()
