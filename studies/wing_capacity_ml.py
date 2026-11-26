"""Capacity of eligible displayed levels, execution labels and causal HMM stress filter."""
import json
import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
from threadpoolctl import threadpool_limits

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from lab.core import LAB,RESULTS,atomic_write,utcnow
from lab.wing_calibration import PARAMS,predictions,pooled_probability,depth_hybrid
from studies.wings_calibration import books
from studies.wings_neighbors import enrich,venue_stress
from studies.wings_regime_ml import features,fit_wing_models,FEATURES
from studies.wings_edge_timing import SPLIT


def hmm(close,relative=False):
    # Right-labelled, non-overlapping 15-minute RV; close index denotes minute OPEN.
    r = np.log(close.asfreq("min")).diff()
    r.index += pd.Timedelta(minutes=1)
    rv = r.pow(2).resample("15min",label="right",closed="right").sum(min_count=15)
    x = np.log(rv.clip(lower=1e-12))
    if relative:
        x=x-x.shift(1).rolling(7*96,min_periods=6*96).median()
    x=x.loc["2026-07-01":]
    training = x[x.index<SPLIT-pd.Timedelta(days=1)].dropna()
    center,sd=training.mean(),training.std()
    x=(x-center)/sd
    training=(training-center)/sd
    def specification(series):
        return MarkovRegression(series.to_numpy(),k_regimes=2,trend="c",switching_variance=False)
    fit = specification(training).fit(start_params=[.95,.05,-.6,.6,.7],em_iter=0,disp=False,maxiter=150)
    if not fit.mle_retvals["converged"]:
        raise ValueError("HMM did not converge")
    params = dict(zip(fit.model.param_names,fit.params))
    params.update(log_rv_center=center,log_rv_sd=sd)
    high = int(params["const[1]"]>params["const[0]"])
    # Restart the filter across missing blocks; never propagate future observations backwards.
    output = pd.Series(np.nan,index=x.index)
    for _,g in x.dropna().groupby(x.isna().cumsum()):
        if len(g)>2:
            output.loc[g.index] = specification(g).filter(fit.params).filtered_marginal_probabilities[:,high]
    return output,params


def deepen(frame):
    q = enrich(frame)
    for col in ["yes_depth","no_depth"]:
        if col not in q:
            q[col] = None
        q["later_"+col] = q.sort_values("ts").groupby("market")[col].shift(-1)
    return q


def fee(n,p):
    return math.ceil(round(7*n*p*(1-p),6))/100


def levels(row,later=False):
    prefix = "later_" if later else ""
    raw = getattr(row,prefix+("yes_depth" if row.cheap_yes else "no_depth"))
    if isinstance(raw,str):
        return sorted([(1-float(b),int(s)) for b,s in json.loads(raw) if 0<b<1 and s>=1])
    b,a = getattr(row,prefix+"bid"),getattr(row,prefix+"ask")
    size = getattr(row,prefix+("bid_size" if row.cheap_yes else "ask_size"))
    if not (0<b<=a<1 and np.isfinite(size) and size>=1):
        return []
    return [(1-b if row.cheap_yes else a,int(size))]


def sweep(levels_,fair,margin=.01,limit=1.,cap=None):
    qty,cost,fees,last = 0,0.,0.,0.
    for p,n in levels_:
        if cap is not None:
            n = min(n,cap-qty)
        if n<=0 or p>limit+1e-9 or fair-p-.07*p*(1-p)<margin-1e-10:
            break
        while n and fair-p-fee(n,p)/n<margin-1e-10:
            n-=1
        if n:
            qty+=n
            cost+=p*n
            fees+=fee(n,p)
            last=p
    return qty,cost,fees,last


def replay(q,model,period,*,stress=False,include_sizing=False):
    records=[]
    # First qualifying snapshot, largest per-contract net edge, ticker tie break.
    z=q.dropna(subset=[model]).copy()
    z["p_model"]=z[model]
    wing=z.bid.where(z.cheap_yes,1-z.ask)
    z=z[wing-z.p_model>=.01-1e-9]
    for end,g in z.groupby("end"):
        winner=None
        for row in g.sort_values(["ts","market"]).itertuples():
            if winner and row.ts>winner[0].ts:
                break
            fair=1-row.p_model
            margin=.01/row.p_fill if model=="execution" else .01
            n,c,f,last=sweep(levels(row),fair,margin)
            if n and (winner is None or fair-(c+f)/n>winner[1]):
                winner=(row,fair-(c+f)/n,n,c,f,last)
        if winner:
            r,edge,n,c,f,limit=winner
            observed=pd.notna(r.later_ts) and 0<(r.later_ts-r.ts).total_seconds()<=90
            margin=.01/r.p_fill if model=="execution" else .01
            dn,dc,df,_=sweep(levels(r,True),1-r.p_model,margin,limit=limit,cap=n) if observed else (0,0.,0.,0.)
            records.append(dict(model=model,period=period,end=end,market=r.market,n=n,cost=c+f,fee=f,pnl=n*(1-r.y)-c-f,
                                delayed_n=dn,delayed_cost=dc+df,delayed_pnl=dn*(1-r.y)-dc-df if dn else 0.,observed=observed))
            if include_sizing:
                records[-1].update(ts=r.ts,fair=1-r.p_model,limit=limit,edge=edge,
                                   disagreement=abs(r.t4-r.p_model)*2,spread=r.ask-r.bid,
                                   later_levels=levels(r,True) if observed else [],pay=1-r.y)
            if stress:
                worse = [(round(p+.01, 6), size) for p, size in levels(r, True) if p+.01 < 1] if observed else []
                sn, sc, sf, _ = sweep(worse, 1-r.p_model, margin, limit=limit, cap=n)
                records[-1].update(stress_n=sn, stress_pnl=sn*(1-r.y)-sc-sf if sn else 0.,
                                   delayed_edge=dn*(1-r.p_model)-dc-df,
                                   stress_edge=sn*(1-r.p_model)-sc-sf)
    return records


def missing_bounds(summary,trades):
    table=summary.copy()
    for name,prefix in [("pnl",""),("delayed_pnl","delayed_")]:
        qty=trades[prefix+"n"]
        cost=trades[prefix+"cost"]
        by=[trades.period,trades.model]
        for suffix,fallback in [("min",-cost),("max",qty-cost)]:
            values=trades[name].fillna(fallback).groupby(by).sum()
            table[name+"_all_"+suffix]=values.reindex(pd.MultiIndex.from_frame(table[["period","model"]])).fillna(0).round(2).to_numpy()
    return table


def main():
    close,stats,hist,gate,_,_,_,_,coef,plain,trees,base_cols=fit_wing_models()
    h,params=hmm(close)
    relative,relative_params=hmm(close,relative=True)
    q,_=books(utcnow().floor("h")-pd.Timedelta(hours=1),deepen)
    official=RESULTS/"wing_forecast_settlements.json"
    if official.exists():
        values=json.loads(official.read_text(encoding="utf-8"))
        settled=q.event.map({key:rec["value"] for key,rec in values.items()})
        yes=settled.gt(q.strike).astype(float)
        q["y"]=q.y.fillna(yes.where(q.cheap_yes,1-yes).where(settled.notna()))
    q=features(venue_stress(q),stats,hist,gate)
    all_quotes=q.copy()
    q=q[q.blocked.eq(False)].dropna(subset=FEATURES).copy()
    q["hmm"]=q.ts.dt.floor("15min").map(h)
    q["hmm_gate"]=q.t4.where(q.hmm.le(.5))
    q["relative_gate"]=q.t4.where(q.ts.dt.floor("15min").map(relative).le(.5))
    q["calendar"]=pooled_probability(q.t4,q.mid,q.weekend,coef)
    pool=json.loads((RESULTS/"wings_odds_pool.json").read_text(encoding="utf-8"))
    q["odds_pool"]=pooled_probability(q.t4,q.mid,q.weekend,pool["coefficients"])
    q["market_cal"],q["t4_cal"],q["hybrid"]=predictions(q.t4,q.mid,q.weekend,PARAMS)
    q["depth"]=depth_hybrid(q,PARAMS)
    q["plain"],q["trees"]=plain.predict_proba(q[base_cols])[:,1],trees.predict_proba(q[FEATURES])[:,1]
    neighbor=q.curve.where(q.cheap_yes,1-q.curve)
    q["neighbor_mix"]=(q.t4+neighbor.fillna(q.t4))/2
    q["neighbor_confirm"]=np.maximum(q.t4,neighbor)
    q["neighbor_stress"]=np.maximum(q.stressed,neighbor)
    size=q.bid_size.where(q.cheap_yes,q.ask_size)
    future_size=q.later_bid_size.where(q.cheap_yes,q.later_ask_size)
    price=1-q.bid.where(q.cheap_yes,1-q.ask)
    next_price=1-q.later_bid.where(q.cheap_yes,1-q.later_ask)
    observed=(q.later_ts-q.ts).dt.total_seconds().between(0,90,inclusive="right")&q.later_bid.gt(0)&q.later_ask.lt(1)&q.later_ask.ge(q.later_bid)
    fill=observed & next_price.le(price+1e-9)&future_size.ge(1)
    q["log_size"]=np.log1p(size)
    cols=["t4","mid","spread","log_size","ratio","direction","left"]
    train=q.end.lt(SPLIT-pd.Timedelta(days=1))&observed
    def logistic(mask,label):
        return make_pipeline(StandardScaler(),LogisticRegression(max_iter=400)).fit(q.loc[mask,cols],label[mask])
    fill_model=logistic(train,fill)
    tail_model=logistic(train&fill&q.y.notna(),q.y)
    pf=fill_model.predict_proba(q[cols])[:,1]
    q["p_fill"]=pf
    pc=tail_model.predict_proba(q[cols])[:,1]
    # p(y|price survives) sets the price cap; the survival estimate gates expected profit.
    gross=1-pc-price-.07*price*(1-price)
    q["execution"]=np.where(pf*gross>=.01,pc,np.nan)
    test=q.end.ge(SPLIT)
    models=["t4","normal","mid","calendar","odds_pool","market_cal","t4_cal","hybrid","depth","plain","trees","fhs_all","fhs","stressed","neighbor_mix","neighbor_confirm","neighbor_stress","hmm_gate","execution"]
    rows=[]
    rv=np.log(close.asfreq("min")).diff().pow(2).rolling(30).sum().pow(.5)*100
    span=(close.asfreq("min").rolling(61).max()/close.asfreq("min").rolling(61).min()-1)*100
    quiet=(rv.lt(1)&span.lt(1.5)).rolling(15).sum().eq(15)
    quiet.index+=pd.Timedelta(minutes=1)
    restart=all_quotes.ts.dt.floor("min").map(quiet).eq(True)&all_quotes.ts.dt.floor("15min").map(h).le(.2)
    all_quotes["hmm_resume"]=all_quotes.t4.where(restart)
    rel_restart=all_quotes.ts.dt.floor("min").map(quiet).eq(True)&all_quotes.ts.dt.floor("15min").map(relative).le(.2)
    all_quotes["relative_resume"]=all_quotes.t4.where(rel_restart)
    models.extend(["hmm_resume","relative_gate","relative_resume"])
    for start,end,label in [(SPLIT,pd.Timestamp("2026-11-20",tz="UTC"),"09–19.11"),(pd.Timestamp("2026-11-20",tz="UTC"),q.end.max()+pd.Timedelta(seconds=1),"с 20.11")]:
        g=q[q.end.between(start,end,inclusive="left")]
        for model in models:
            data=all_quotes[all_quotes.end.between(start,end,inclusive="left")] if model in ("hmm_resume","relative_resume") else g
            rows.extend(replay(data,model,label))
    files=sorted((LAB/"wing_forecasts").glob("*.csv"))
    if files:
        pairs=pd.concat([pd.read_csv(f) for f in files]).dropna(subset=["w06"])
        pairs["ts"]=pd.to_datetime(pairs.ts,utc=True,format="mixed").dt.floor("s")
        matched=q[test].assign(ts=q.loc[test,"ts"].dt.floor("s")).merge(pairs[["ts","market","w06"]].drop_duplicates(["ts","market"]),on=["ts","market"])
        matched["pair"]=(matched.w06+matched.t4)/2
        for model in ["t4","w06","pair"]:
            rows.extend(replay(matched,model,"парные W06"))
    trades=pd.DataFrame(rows)
    atomic_write(RESULTS/"wing_capacity_entries.csv",trades.to_csv(index=False))
    summary=[]
    for period,group in trades.groupby("period",sort=False):
        unresolved=group.loc[group.pnl.isna(),"end"].unique()
        names=models if period!="парные W06" else ["t4","w06","pair"]
        for name in names:
            g=group[group.model.eq(name)&~group.end.isin(unresolved)]
            cost=g.cost.sum()
            summary.append(dict(period=period,model=name,trades=len(g),contracts=g.n.sum(),capital=round(cost,2),fees=round(g.fee.sum(),2),
                                pnl=round(g.pnl.sum(),2),roi_pct=round(100*g.pnl.sum()/cost,3) if cost else None,
                                worst=round(g.pnl.min(),2) if len(g) else None,delayed_contracts=g.delayed_n.sum(),
                                delayed_cost=round(g.delayed_cost.sum(),2),delayed_pnl=round(g.delayed_pnl.sum(),2),
                                delayed_roi=round(100*g.delayed_pnl.sum()/g.delayed_cost.sum(),3) if g.delayed_cost.sum() else None,
                                next_books=int(g.observed.sum()),excluded_hours=len(unresolved)))
    atomic_write(RESULTS/"wing_capacity.csv",missing_bounds(pd.DataFrame(summary),trades).to_csv(index=False))
    evaluation=test&observed
    conditional=evaluation&fill&q.y.notna()
    def weights(model):
        scale,reg=model.steps[0][1],model.steps[1][1]
        return dict(features=cols,center=scale.mean_.tolist(),scale=scale.scale_.tolist(),
                    coefficients=reg.coef_[0].tolist(),intercept=float(reg.intercept_[0]))
    diagnostics=dict(hmm=params,hmm_relative=relative_params,train_quotes=int(train.sum()),train_fills=int((train&fill).sum()),
                     fill_model=weights(fill_model),conditional_model=weights(tail_model),
                     hmm_test_hours=int(q.loc[test&q.hmm.notna(),"end"].nunique()),
                     hmm_median=float(q.loc[test,"hmm"].median()),
                     conditional_brier=float(np.mean((pc[conditional]-q.loc[conditional,"y"])**2)),
                     conditional_t4_brier=float(np.mean((q.loc[conditional,"t4"]-q.loc[conditional,"y"])**2)),
                     test_quotes=int(evaluation.sum()),fill_brier=float(np.mean((pf[evaluation]-fill[evaluation])**2)),
                     constant_brier=float(np.mean((float(fill[train].mean())-fill[evaluation])**2)),
                     until=str(q.end.max()),full_depth_rows=int(q.yes_depth.notna().sum()))
    atomic_write(RESULTS/"wing_capacity_fit.json",json.dumps(diagnostics,indent=2))
    print(diagnostics)


if __name__=="__main__":
    with threadpool_limits(limits=2):
        main()
