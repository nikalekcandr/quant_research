"""Calibrate the cheap side of Kalshi hourly wings using archived outcomes and live book snapshots.

python studies/wings_calibration.py [--until 2026-11-21T09:00Z]
Training and validation boundaries are fixed. The last block extends through completed hours.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from lab.core import LAB, RESULTS, atomic_write, utcnow
from lab.edge_timing import candidates
from lab.wing_calibration import depth_hybrid, eligible, odds_adjust, predictions, sampled
from studies.wings_edge_timing import COLUMNS, START, SPLIT, STOP, day_interval, resolve_ladder, settlement_cache

MODELS = ["t4","normal","mid","market_cal","t4_cal","hybrid","depth"]


def history():
    parts = []
    for path in sorted((ROOT/"data/raw/kalshi_kxbtcd_60m").glob("*.csv*")):
        q = pd.read_csv(path,usecols=["ticker","end_ts","close_time","strike_type","result","yes_bid","yes_ask"])
        q = q[q.strike_type.eq("greater") & q.result.isin(["yes","no"])].rename(
            columns={"ticker":"market","yes_bid":"bid","yes_ask":"ask"})
        q["ts"] = pd.to_datetime(q.end_ts,unit="s",utc=True)
        q["end"] = pd.to_datetime(q.close_time,utc=True)
        q = q[q.end < pd.Timestamp("2026-10-28",tz="UTC")].copy()
        q["left"] = (q.end-q.ts).dt.total_seconds()/60
        q = eligible(q)
        q["y"] = q.result.eq("yes").eq(q.cheap_yes).astype(float)
        parts.append(q[["market","ts","end","left","mid","weekend","y"]])
    return sampled(pd.concat(parts,ignore_index=True))


def books(until,enrich=None,*,since=START):
    parts, hours = [], set()
    for path in sorted((LAB/"markets").glob("*.csv")):
        day = pd.Timestamp(path.stem,tz="UTC")
        if day < since-pd.Timedelta(days=1) or day > until:
            continue
        cols = COLUMNS+["fair"]
        if not set(cols).issubset(pd.read_csv(path,nrows=0).columns):
            continue
        allowed = set(cols) | ({"yes_depth","no_depth"} if enrich else set())
        q = pd.read_csv(path,usecols=lambda c:c in allowed,low_memory=False)
        q = q[q.kind.eq("kx_ladder_1h") & q.venue.eq("kalshi") & q.book.eq(1)].copy()
        q["ts"] = pd.to_datetime(q.ts,utc=True,format="mixed")
        q["end"] = pd.to_datetime(q.end,utc=True,format="mixed")
        q["left"] = (q.end-q.ts).dt.total_seconds()/60
        if enrich is not None:
            q = enrich(q[q.left.between(13,30)].drop_duplicates(["market","ts"]))
        q = q[q.end.between(since,until) & q.left.between(15,30) & q.bid.gt(0) & q.ask.lt(1) & q.ask.ge(q.bid)
              & q.fair_t4.between(0,1) & q.fair.between(0,1)].drop_duplicates(["market","ts"])
        hours.update(q.end)
        q = eligible(q)
        q = q[q.bid_size.where(q.cheap_yes,q.ask_size).ge(1)].copy()
        q["t4"] = q.fair_t4.where(q.cheap_yes,1-q.fair_t4)
        q["normal"] = q.fair.where(q.cheap_yes,1-q.fair)
        parts.append(q)
    d = pd.concat(parts,ignore_index=True)
    # numeric, not object: pandas 3 keeps a mapped column as strings when any key is missing, and then
    # `1 - outcome` raises instead of subtracting (this alone broke a dozen wing studies on pandas 3)
    outcome = pd.to_numeric(d.market.map(resolve_ladder(d.market.unique(),settlement_cache())),errors="coerce")
    # The downloader also archives each event's official BRTI settlement.
    # Adjacent resolved strikes alone do not always bracket an otherwise missing ticker.
    missing = d.loc[outcome.isna(),["event","market"]].drop_duplicates()
    archive = ROOT/"data/interim/kalshi.parquet"
    if len(missing) and archive.exists():
        settled = pd.read_parquet(archive,columns=["event","result","expiration_value","strike_type"])
        settled = settled[settled.result.isin(["yes","no"]) & settled.strike_type.eq("greater")].copy()
        settled["expiration_value"] = pd.to_numeric(settled.expiration_value,errors="coerce")
        values = settled.groupby("event").expiration_value.agg(["min","max"])
        official = missing.event.map(values.loc[values["min"].eq(values["max"]),"min"])
        strike = pd.to_numeric(missing.market.str.rsplit("-T",n=1).str[-1],errors="coerce")
        valid = official.notna() & strike.notna()
        extra = dict(zip(missing.market[valid],official[valid].gt(strike[valid]).astype(float)))
        outcome = outcome.fillna(d.market.map(extra))
    d["y"] = outcome.where(d.cheap_yes,1-outcome)
    return d,pd.DatetimeIndex(sorted(hours),name="end")


def fit_offset(s, column):
    # Thirty equivalent expiry hours centred on the original probabilities.
    p, w = s[column], s.weight
    if s.empty:
        raise ValueError(f"No training observations for {column}")
    target = (np.dot(w,s.y)+30*np.average(p,weights=w))/(w.sum()+30)
    return brentq(lambda a: np.average(odds_adjust(p,a),weights=w)-target,-25,25)


def fit(historical, train):
    params = {name:fit_offset(historical[historical.weekend.eq(weekend)],"mid")
              for name,weekend in [("weekday",False),("weekend",True)]}
    params["t4"] = fit_offset(train,"t4")
    market, model, _ = predictions(train.t4,train.mid,train.weekend,{**params,"weight":.5})
    gap = model-market
    den = np.dot(train.weight,gap**2)
    raw = np.clip(np.dot(train.weight,gap*(train.y-market))/den,0,1) if den > 0 else .5
    # Shrink the Brier-optimal mixture towards equal weights by twelve expiry hours.
    params["weight"] = float((raw*train.weight.sum()+6)/(train.weight.sum()+12))
    return params


def archive_check(hist):
    """Test the calendar correction on an earlier, disjoint time split before the final refit."""
    split = pd.Timestamp("2026-09-01",tz="UTC")
    rows = []
    for weekend,label in [(False,"Будни"),(True,"Выходные")]:
        group = hist[hist.weekend.eq(weekend)]
        intercept = fit_offset(group[group.end < split-pd.Timedelta(days=1)],"mid")
        g = group[group.end>=split]
        calibrated = odds_adjust(g.mid,intercept)
        rows.append({"Дни UTC":label,"Часы":g.end.nunique(),
                     "Цена стакана, %":round(100*np.average(g.mid,weights=g.weight),2),
                     "После поправки, %":round(100*np.average(calibrated,weights=g.weight),2),
                     "Выплата, %":round(100*np.average(g.y,weights=g.weight),2),
                     "Brier стакана":round(np.average((g.mid-g.y)**2,weights=g.weight),5),
                     "Brier с поправкой":round(np.average((calibrated-g.y)**2,weights=g.weight),5)})
    return rows


def evaluate(d,hours,label):
    sample = sampled(d)
    if sample.empty:
        return {"period":label,"hours":len(hours),"scored":0,"rows":[]}
    scores, trades = {}, {}
    for model in MODELS:
        p = sample[model].clip(1e-6,1-1e-6)
        by_hour = pd.DataFrame({"end":sample.end,"loss":sample.weight*(p-sample.y)**2}).groupby("end").loss.sum()
        scores[model] = by_hour
        # Replay the same W12 choice on all quoted strikes; attach outcomes after selection.
        q = candidates(d.assign(fair_t4=d[model].where(d.cheap_yes,1-d[model])))
        first = q[q.edge >= .01].sort_values("ts").drop_duplicates("end").set_index("end")
        pnl = (1-first.y-first.price-first.fee).reindex(hours)
        pnl.loc[~hours.isin(first.index)] = 0.
        trades[model] = (first,pnl)
    paired_hours = pd.DataFrame({m:pnl.notna() for m,(_,pnl) in trades.items()}).all(axis=1)
    rows = []
    for model in MODELS:
        p = sample[model].clip(1e-6,1-1e-6)
        first,pnl = trades[model]
        pnl = pnl[paired_hours]
        traded = first[first.index.isin(pnl.index)]
        gain = scores["t4"]-scores[model]
        ci = day_interval(gain)
        pnl_ci = day_interval(pnl-trades["t4"][1].loc[pnl.index])
        rows.append(dict(model=model, probability=float(np.average(p,weights=sample.weight)),
                         brier=float(scores[model].mean()), gain=float(gain.mean()), ci=[v/100 if v is not None else None for v in ci],
                         logloss=float(np.average(-sample.y*np.log(p)-(1-sample.y)*np.log1p(-p),weights=sample.weight)),
                         entries=len(traded), losses=int((pnl<0).sum()), net=float(pnl.sum()), pnl_ci=pnl_ci,
                         net_hour=float(100*pnl.mean()) if len(pnl) else None,
                         expected=float(100*traded.edge.mean()) if len(traded) else None))
    return dict(period=label,hours=len(hours),scored=sample.end.nunique(),contracts=sample.market.nunique(),
                quotes=len(sample),unresolved_markets=d.loc[d.y.isna(),"market"].nunique(),
                trade_hours=int(paired_hours.sum()), payout=float(np.average(sample.y,weights=sample.weight)),rows=rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--until",type=pd.Timestamp,help="Last settled UTC expiry to consider (inclusive)")
    args = ap.parse_args()
    last = utcnow().floor("h")-pd.Timedelta(hours=1)
    until = min(pd.to_datetime(args.until,utc=True),last) if args.until else last
    hist = history()
    live,hours = books(until)
    train = sampled(live[live.end < SPLIT-pd.Timedelta(days=1)])
    params = fit(hist,train)
    live["market_cal"],live["t4_cal"],live["hybrid"] = predictions(live.t4,live.mid,live.weekend,params)
    live["depth"] = depth_hybrid(live, params)
    periods = []
    for name,start,end in [("Проверка 9–19 ноября",SPLIT,STOP),("С 20 ноября",STOP,until+pd.Timedelta(seconds=1))]:
        periods.append(evaluate(live[live.end.between(start,end,inclusive="left")],
                                hours[(hours>=start)&(hours<end)],name))
    result = dict(until=str(until),params=params,history_hours=hist.end.nunique(),history_markets=hist.market.nunique(),archive_check=archive_check(hist),
                  train_hours=train.end.nunique(),periods=periods)
    atomic_write(RESULTS/"wing_calibration.json",json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False))
    print(json.dumps(result,ensure_ascii=False,indent=1))


if __name__=="__main__":
    main()
