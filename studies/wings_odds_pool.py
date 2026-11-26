"""Freeze a market/t4 log-odds pool before September, then replay chronological samples."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from lab.core import RESULTS, atomic_write, utcnow
from lab.wing_calibration import PARAMS, pooled_probability, predictions, sampled
from studies.wings_old_replay import archive
from studies.wings_calibration import books
from studies.wings_two_sided import offers, picks
from studies.wings_edge_timing import START,SPLIT,STOP,day_interval


def fit(g,market_only=False):
    def objective(beta):
        p = pooled_probability(g.t4,g.mid,g.weekend,beta)
        loss = np.dot(g.weight,-g.y*np.log(p)-(1-g.y)*np.log1p(-p))
        # Weak fixed shrinkage towards zero bias and equal log-odds weights.
        return loss+2*(beta[0]**2+beta[1]**2+(beta[2]-.5)**2)
    fitted = minimize(objective,[0.,0.,0. if market_only else .5],
                      bounds=[(-3,3),(-3,3),(0,0) if market_only else (0,1)])
    if not fitted.success:
        raise ValueError(fitted.message)
    return fitted.x


def score(q,hours,names,label):
    g = sampled(q)
    pnls, entries = {}, {}
    for name in names:
        chosen = picks(offers(q.assign(t4=q[name]),False))
        pnls[name] = (chosen.pay-chosen.price-chosen.fee).reindex(hours,fill_value=0.)
        entries[name] = chosen.index
    pnl = pd.DataFrame(pnls).dropna()
    rows = []
    for name in names:
        lo,hi = day_interval(pnl[name]-pnl[names[0]])
        rows.append({"Период":label,"Прогноз":name,"Часы для прибыли":len(pnl),"Экспирации для Brier":g.end.nunique(),
                     "Brier":round(np.average((g[name]-g.y)**2,weights=g.weight),6) if len(g) else None,
                     "Входы":len(entries[name].intersection(pnl.index)),"Убытки":int((pnl[name]<0).sum()),
                     "Нетто, $":round(pnl[name].sum(),2),
                     "95% Δ к t₄, ¢/час":f"{lo:.2f} … {hi:.2f}" if lo is not None else "—"})
    return rows


def calendar(q):
    mid = (q.ask+q.bid)/2
    return q.assign(mid=mid.where(q.cheap_yes,1-mid),weekend=q.end.dt.day_of_week>=5)


def main():
    old,hours = archive()
    old = calendar(old)
    split = pd.Timestamp("2026-09-01",tz="UTC")
    train = sampled(old[old.end<split-pd.Timedelta(days=1)])
    beta = fit(train)
    market_beta = fit(train,market_only=True)
    old["odds_pool"] = pooled_probability(old.t4,old.mid,old.weekend,beta)
    old["calendar_only"] = pooled_probability(old.t4,old.mid,old.weekend,market_beta)
    cutoff = pd.Timestamp("2026-10-28",tz="UTC")
    idx = hours[(hours>=split)&(hours<cutoff)]
    rows = score(old[old.end.isin(idx)],idx,["t4","mid","calendar_only","odds_pool"],"1 сентября – 27 октября")
    until = utcnow().floor("h")-pd.Timedelta(hours=1)
    live,hours = books(until)
    live = calendar(live)
    live["odds_pool"] = pooled_probability(live.t4,live.mid,live.weekend,beta)
    live["calendar_only"] = pooled_probability(live.t4,live.mid,live.weekend,market_beta)
    live["hybrid"] = predictions(live.t4,live.mid,live.weekend,PARAMS)[2]
    for a,b,label in [(START,SPLIT,"30 октября – 8 ноября"),(SPLIT,STOP,"9–19 ноября"),(STOP,until+pd.Timedelta(seconds=1),"С 20 ноября")]:
        idx = hours[(hours>=a)&(hours<b)]
        if len(idx):
            models = ["t4","mid","calendar_only","odds_pool"]
            if a>=SPLIT:
                models.append("hybrid")
            rows.extend(score(live[live.end.isin(idx)],idx,models,label))
    result = dict(coefficients=beta.tolist(),market_coefficients=market_beta.tolist(),train_hours=train.end.nunique(),until=str(until),rows=rows)
    atomic_write(RESULTS/"wings_odds_pool.json",json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False))
    print(beta,train.end.nunique())
    print(pd.DataFrame(rows).to_string(index=False))


if __name__=="__main__":
    main()
