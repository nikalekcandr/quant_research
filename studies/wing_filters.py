"""Fixed gates on the neighbor model; matched hourly net returns for two fill conventions."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from lab.core import RESULTS, atomic_write
from lab.history import Minutes
from lab.wing_regime import regimes
from studies.wing_ten import BASE,END,START,CACHE,enrich,archive,books,deepen,replay,offers,picks,next_minute


def gates(q, close):
    r = np.log(close.asfreq("min")).diff()
    def prior(s):
        return (q.ts.dt.floor("min")-pd.Timedelta(minutes=2)).map(s)
    direction = np.where(q.cheap_yes,1,-1)
    rv = r.pow(2).rolling(30).sum()
    bv = np.pi/2*(r.abs()*r.shift().abs()).rolling(30).sum()
    efficiency = r.rolling(15).sum()/r.abs().rolling(15).sum()
    location = (close-close.rolling(60).min())/(close.rolling(60).max()-close.rolling(60).min())
    wing = q.bid.where(q.cheap_yes,1-q.ask)
    # Minute aggregates use only the immediately preceding, completed minute.
    keys = [q.market,q.cheap_yes,q.ts.dt.floor("min")]
    prev = wing.groupby(keys).last()
    idx = pd.MultiIndex.from_arrays([q.market,q.cheap_yes,q.ts.dt.floor("min")-pd.Timedelta(minutes=1)])
    previous = prev.reindex(idx).to_numpy()
    neighbors = q.curve.where(q.cheap_yes,1-q.curve)
    q[BASE] = (q.t4+neighbors.fillna(q.t4))/2
    side_count = q.market.groupby([q.end,q.ts,q.cheap_yes]).transform("nunique")
    checks = {
        "Малая доля скачков":prior(bv/rv).ge(.8),
        "Нет направленного хода":(prior(efficiency)*direction).le(.35),
        "Вдали от границы диапазона":((2*prior(location)-1)*direction).le(.6),
        "Спокойнее последних суток":prior(rv/rv.shift(30).rolling(1440).median()).le(1),
        "Нет ускорения к крылу":(prior(r.rolling(3).mean()-r.shift(3).rolling(12).mean())*direction).le(0),
        "Цена крыла не растёт":wing.le(previous+1e-9),
        "Несколько страйков крыла":side_count.ge(2),
        "Спред до одного цента":(q.ask-q.bid).le(.01000001),
        "Центральное окно":q.left.between(20,25),
        "Премия к риску 50%":wing.ge(1.5*q[BASE])
    }
    for name,ok in checks.items():
        q[name] = q[BASE].where(ok)
    return [BASE,*checks]


def evaluate(q,hours,close,depth):
    q = q.sort_values("ts")
    names = gates(q,close)
    records = []
    for name in names:
        if depth:
            for t in replay(q,name,"filters"):
                records.append(dict(end=t["end"],model=name,pnl=t["delayed_pnl"],n=t["delayed_n"],cost=t["delayed_cost"]))
        else:
            g = next_minute(picks(offers(q.assign(t4=q[name]),False)))
            for end,t in g.iterrows():
                records.append(dict(end=end,model=name,pnl=t.pnl,n=int(t.filled),cost=t.pay-t.pnl if t.filled else 0.))
    entries = pd.DataFrame(records)
    hours = hours.difference(entries.loc[entries.pnl.isna(),"end"])
    entries = entries[entries.end.isin(hours)]
    panel = entries.pivot(index="end",columns="model",values="pnl").reindex(index=hours,columns=names).fillna(0.)
    daily = panel.sub(panel[BASE],axis=0).groupby(hours.floor("D")).sum().iloc[:,1:]
    signs = np.random.default_rng(91).choice([-1,1],(50000,len(daily)))
    p = (1+(signs@daily.to_numpy()>=daily.sum().to_numpy()-1e-9).sum(axis=0))/50001
    rows = []
    for name,padj in zip(names,[np.nan,*multipletests(p,method="holm")[1]]):
        g = entries[entries.model.eq(name)]
        rows.append(dict(model=name,hours=len(hours),orders=len(g),fills=int(g.n.gt(0).sum()),contracts=int(g.n.sum()),
                         net=round(g.pnl.sum(),2),cost=round(g.cost.sum(),2),worst=round(panel[name].min(),2),
                         late=round(panel.loc["2026-11-20":,name].sum(),2),p_holm=round(padj,4)))
    return pd.DataFrame(rows),entries


def main():
    close = Minutes().close
    gate = regimes(close).blocked
    candle,hours = archive(enrich,files=CACHE.glob("*.csv.gz"),minute_close=close)
    live,_ = books(END,deepen)
    values = json.loads((CACHE.parent/"wing_quant_inputs/settlements.json").read_text())
    val = live.event.map({e:r["value"] for e,r in values.items()})
    yes = val.gt(live.strike).astype(float)
    live["y"] = live.y.fillna(yes.where(live.cheap_yes,1-yes).where(val.notna()))
    results,entries = [],[]
    for label,data,depth in [("Архив",candle,False),("Объём",live,True)]:
        q = data[data.end.between(START,END)&data.left.ge(16)&data.ts.dt.floor("min").map(gate).eq(False)].copy()
        cohort = pd.DatetimeIndex(sorted(q.end.unique())) if depth else hours[(hours>=START)&(hours<=END)]
        table,trades = evaluate(q,cohort,close,depth)
        results.append(table.assign(sample=label))
        entries.append(trades.assign(sample=label))
    table = pd.concat(results)
    atomic_write(RESULTS/"wing_filters.csv",table.to_csv(index=False))
    atomic_write(RESULTS/"wing_filter_entries.csv",pd.concat(entries).to_csv(index=False))
    print(table.to_string(index=False))

if __name__=="__main__":
    main()
