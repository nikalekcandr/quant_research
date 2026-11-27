"""Ten quantity policies on identical preselected wing signals."""
import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from lab.core import RESULTS,atomic_write
from studies.ladders import no_data
from lab.history import Minutes
from lab.wing_regime import regimes
from studies.wing_ten import CACHE,START,END,books,deepen,replay
from studies.wing_capacity_ml import sweep

RULES = ["Полная глубина","Лимит контракта на доллар риска","Корень глубины","Бюджет по edge",
         "Четверть Келли","Штраф за расхождение","Бюджет по спреду","После просадки",
         "Дневной лимит потерь","По прошлым исполнениям","Лимит хвостового риска"]


def target(rule,t,equity,peak,day_start,history):
    n, edge = t["n"],t["edge"]
    price = t["cost"]/n
    if rule==0: return n
    if rule==1: return 1000
    if rule==2: return np.sqrt(250*n)
    if rule==3: return 1000*np.clip(edge/.03,0,3)
    if rule==4: return .25*equity*edge/max(.001,price*(1-price))
    if rule==5: return n/(1+t["disagreement"]/.02)
    if rule==6: return n*min(1,.01/max(t["spread"],.001))
    if rule==7: return n*(.25 if peak-equity>=200 else 1)
    if rule==8: return max(0,500+min(0,equity-day_start))
    if rule==9: return n*(1+sum(history[-5:]))/(2+len(history[-5:]))
    return .01*equity/max(.01,price*np.minimum(1,(1-t["fair"])/.05))


def simulate(signals,rule):
    equity=peak=day_start=10000.
    day=None
    rows=[]
    history=[]
    for t in signals:
        if t["ts"].date()!=day:
            day=t["ts"].date()
            day_start=equity
        # One dollar fully collateralises one contract including fees.
        qty=int(max(0,min(t["n"],equity,target(rule,t,equity,peak,day_start,history))))
        n,c,f,_=sweep(t["later_levels"],t["fair"],limit=t["limit"],cap=qty)
        pnl=n*t["pay"]-c-f
        equity+=pnl
        peak=max(equity,peak)
        if qty:
            history.append(n/qty)
        rows.append(dict(end=t["end"],market=t["market"],rule=RULES[rule],requested=qty,
                         contracts=n,cost=c+f,net=pnl,drawdown=peak-equity))
    return rows


def main():
    gate=regimes(Minutes().close).blocked
    q,_=books(END,deepen)
    q=q[q.end.between(START,END)&q.left.ge(16)&q.ts.dt.floor("min").map(gate).eq(False)].copy()
    cache=json.loads((CACHE.parent/"wing_quant_inputs/settlements.json").read_text())
    val=q.event.map({e:r["value"] for e,r in cache.items()})
    yes=pd.to_numeric(val,errors="coerce").gt(q.strike).astype(float)
    q["y"]=q.y.fillna(yes.where(q.cheap_yes,1-yes).where(val.notna()))
    q["base"]=(q.t4+q.curve.where(q.cheap_yes,1-q.curve).fillna(q.t4))/2
    signals=replay(q,"base","books",include_sizing=True)
    signals.sort(key=lambda t:t["ts"])
    if any(a["end"]>=b["ts"] for a,b in zip(signals,signals[1:])) or any(pd.isna(t["pay"]) for t in signals):
        raise ValueError("Sizing requires distinct settled hours with verified payouts")
    ledger=pd.DataFrame([row for rule in range(len(RULES)) for row in simulate(signals,rule)])
    if ledger.empty or "rule" not in ledger.columns:
        no_data("ни одной сделки в окне — размеру не на чем учиться")
    panel=ledger.pivot(index="end",columns="rule",values="net").reindex(columns=RULES)
    delta=panel.sub(panel.iloc[:,0],axis=0).iloc[:,1:].groupby(panel.index.floor("D")).sum()
    signs=np.random.default_rng(62).choice([-1,1],(50000,len(delta)))
    p=(1+(signs@delta.to_numpy()>=delta.sum().to_numpy()-1e-9).sum(axis=0))/50001
    summary=ledger.groupby("rule",sort=False).agg(contracts=("contracts","sum"),cost=("cost","sum"),
                net=("net","sum"),drawdown=("drawdown","max"),worst=("net","min"))
    summary["p_holm"]=[np.nan,*multipletests(p,method="holm")[1]]
    summary["late"]=panel.loc["2026-11-20":].sum()
    summary=summary.round(3)
    atomic_write(RESULTS/"wing_sizing.csv",summary.to_csv())
    atomic_write(RESULTS/"wing_sizing_entries.csv",ledger.to_csv(index=False))
    print(summary.to_string())

if __name__=="__main__":
    main()
