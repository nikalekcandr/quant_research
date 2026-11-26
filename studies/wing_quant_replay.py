"""Causal CAViaR, SVI digitals, snapshot OFI/queue states and two-tranche execution."""
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from scipy.optimize import least_squares,minimize
from scipy.signal import lfilter
from scipy.special import ndtr
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from lab.core import LAB,RESULTS,atomic_write,utcnow
from lab.history import Minutes
from lab.wing_regime import regimes
from lab.wing_calibration import sampled
from studies.wings_calibration import books
from studies.wings_neighbors import venue_stress
from studies.wings_edge_timing import SPLIT,day_interval
from lab.wing_calibration import PARAMS,predictions
from studies.wing_capacity_ml import deepen,replay,levels,sweep


def observations(frame):
    q=deepen(frame).sort_values(["market","ts"])
    lag=q.groupby("market")[["ts","bid","ask","bid_size","ask_size"]].shift()
    fresh=(q.ts-lag.ts).dt.total_seconds().between(0,90,inclusive="right")
    flow=(q.bid_size*q.bid.ge(lag.bid)-lag.bid_size*q.bid.le(lag.bid)
          -q.ask_size*q.ask.le(lag.ask)+lag.ask_size*q.ask.ge(lag.ask))
    q["ofi"]=(flow/(1+lag.bid_size+lag.ask_size)).clip(-3,3).where(fresh)
    return q


def caviar(close,q,split=SPLIT):
    lc=np.log(close.asfreq("min"))
    lc.index+=pd.Timedelta(minutes=1)
    ret=(100*lc.resample("20min",closed="right",label="right").last().diff()).dropna()
    train=ret.index<split-pd.Timedelta(days=1)
    r=ret.to_numpy()
    probs=np.array([.01,.05,.10,.25,.50,.75,.90,.95,.99])
    forecasts=[]
    parameters=[]
    for a in probs:
        def predict(b,x):
            forcing=b[0]+b[2]*np.maximum(x,0)+b[3]*np.maximum(-x,0)
            return lfilter([1],[1,-b[1]],forcing)
        def loss(b):
            err=r[train][1:]-predict(b,r[train][:-1])
            return np.mean(np.maximum(a*err,(a-1)*err))
        fit=minimize(loss,[np.quantile(r[train],a)*.2,.8,0,0],method="Powell",
                     bounds=[(-2,2),(0,.995),(-3,3),(-3,3)])
        if not fit.success:
            raise ValueError(fit.message)
        forecasts.append(predict(fit.x,r))
        parameters.append(fit.x.tolist())
    paths=pd.DataFrame(np.sort(np.asarray(forecasts).T,axis=1),index=ret.index)
    rows=paths.reindex(pd.DatetimeIndex(q.ts.dt.floor("20min"))).to_numpy()
    distance=100*np.log(q.strike/q.usd_mid)*np.sqrt(20/q.left)
    cdf=np.array([np.interp(k,row,probs) if np.isfinite(row).all() else np.nan for k,row in zip(distance,rows)])
    return np.where(q.cheap_yes,1-cdf,cdf),parameters


def svi_smile(k,iv):
    def value(p,x):
        a,b,r,m,s=p
        return a+b*(r*(x-m)+np.sqrt((x-m)**2+s*s))
    p=least_squares(lambda p:value(p,k)-iv**2,[.1,.2,-.2,0,.15],
                    bounds=([.0001,.0001,-.98,-.5,.001],[4,8,.98,.5,2]),max_nfev=100).x
    return p


def density_ok(p,lo,hi,years):
    a,b,r,m,s=p
    k=np.linspace(lo,hi,101)
    root=np.sqrt((k-m)**2+s*s)
    w=years*(a+b*(r*(k-m)+root))
    wp=years*b*(r+(k-m)/root)
    wpp=years*b*s**2/root**3
    return np.all((1-k*wp/(2*w))**2-.25*wp**2*(1/w+.25)+.5*wpp>=0)


def svi(q):
    pred=pd.Series(np.nan,index=q.index)
    fits=0
    for day,g in q.groupby(q.ts.dt.strftime("%Y-%m-%d")):
        path=ROOT/"data/live/deribit_chain"/(day+".csv.gz")
        if not path.exists():
            continue
        data=pd.read_csv(path,usecols=["snap_ts","instrument","mark_iv","underlying","bid","ask"])
        data["ts"]=pd.to_datetime(data.snap_ts,utc=True,format="mixed").astype(g.ts.dtype)
        parts=data.instrument.str.extract(r"^BTC-(\d+[A-Z]{3}\d{2})-(\d+(?:\.\d+)?)-[CP]$")
        data["exp"]=pd.to_datetime(parts[0],format="%d%b%y",errors="coerce",utc=True)+pd.Timedelta(hours=8)
        data["k"]=np.log(pd.to_numeric(parts[1])/data.underlying)
        data=data[(data.exp-data.ts).between(pd.Timedelta(hours=6),pd.Timedelta(days=10))&
                  data.bid.gt(0)&data.ask.ge(data.bid)&data.mark_iv.between(2,300)&data.k.abs().le(.35)]
        if data.empty:
            continue
        snaps=pd.DatetimeIndex(sorted(data.ts.unique()))
        lookup=snaps.get_indexer(g.ts,method="pad",tolerance=pd.Timedelta(minutes=15))
        surface=data.groupby("ts")
        for idx,h in g.groupby(lookup):
            if idx<0:
                continue
            block=surface.get_group(snaps[idx])
            for expiry_time,expiry in block.groupby("exp",sort=True):
                vol=expiry.groupby("k").mark_iv.median()/100
                if len(vol)<7 or min(vol.index)>=0 or max(vol.index)<=0:
                    continue
                a,b,r,m,s=svi_smile(vol.index.to_numpy(),vol.to_numpy())
                if not all(density_ok((a,b,r,m,s),min(vol.index),max(vol.index),t) for t in
                           ((expiry_time-snaps[idx]).total_seconds()/(365*86400),30/(365*1440))):
                    continue
                k=np.log(h.strike/h.usd_mid)
                v=np.sqrt(a+b*(r*(k-m)+np.sqrt((k-m)**2+s*s)))
                root=np.sqrt(h.left/(365*24*60))
                d2=-k/(v*root)-.5*v*root
                deriv=b*(r+(k-m)/np.sqrt((k-m)**2+s*s))/(2*v)
                digital=ndtr(d2)-np.exp(-.5*d2**2)/np.sqrt(2*np.pi)*root*deriv
                yes=pd.Series(digital,index=h.index).where(digital.between(0,1)&k.between(min(vol.index),max(vol.index)))
                pred.loc[h.index]=yes.where(h.cheap_yes,1-yes)
                fits+=1
                break
    return pred,fits


def maker_prints():
    folder=ROOT/"data/raw/wing_quant_inputs"
    ledger=pd.read_csv(LAB/"ledger.csv",low_memory=False)
    rows=[]
    for r in ledger[ledger.hyp.eq("W06")&ledger.event.eq("CLOSE")].itertuples():
        info=json.loads(r.info)
        path=folder/(r.instrument+".json")
        if not path.exists() or not info.get("maker") or "queue" not in info:
            continue
        trades=pd.DataFrame(json.loads(path.read_text()))
        if trades.empty:
            continue
        times=pd.to_datetime(trades.created_time,utc=True,format="mixed")
        cheap="yes" if r.side=="no" else "no"
        trades=trades[(times>pd.Timestamp(info["quoted_at"]))&(times<=pd.Timestamp(r.ts))&
                      trades.taker_side.eq(cheap)&~trades.is_block_trade].drop_duplicates("trade_id")
        px=trades[cheap+"_price_dollars"].astype(float)
        volume=trades.count_fp.astype(float)
        through=volume[px>info["wing_price"]+1e-8].sum()
        at=volume[(px-info["wing_price"]).abs()<1e-8].sum()
        estimate=int(through+max(0,at-info["queue"]))
        rows.append(dict(market=r.instrument,quoted=info["quoted_at"],queue=info["queue"],at=at,
                         through=through,scenario_contracts=estimate,one_contract_pnl=r.pnl))
    atomic_write(RESULTS/"wing_print_queue.csv",pd.DataFrame(rows).to_csv(index=False))
    return len(rows)


def split_execution(q,train_variance):
    base=replay(q,"t4","split")
    rows=[]
    for pick in base:
        g=q[q.end.eq(pick["end"])&q.market.eq(pick["market"])].sort_values("ts")
        for r in g.itertuples():
            fair=1-r.t4
            depth=levels(r)
            n,c,f,limit=sweep(depth,fair)
            if n:
                break
        if not n or pd.isna(r.y):
            continue
        eta=max(r.ask-r.bid,.001)
        # Normalised two-period AC objective eta*(x1²+x2²)+lambda*var*x2².
        # Fixed lambda=1; eta is a spread proxy, not a fitted impact coefficient.
        first=n-int(n*eta/(2*eta+train_variance))
        n1,c1,f1,_=sweep(depth,fair,cap=first)
        n2=c2=f2=0
        if pd.notna(r.later_ts) and 0<(r.later_ts-r.ts).total_seconds()<=90:
            # Subtract own first consumption at unchanged levels (zero assumed refill).
            consumed={}
            left=n1
            for p,count in depth:
                used=min(left,count)
                consumed[round(p,6)]=used
                left-=used
            later=[(p,max(0,count-consumed.get(round(p,6),0))) for p,count in levels(r,True)]
            n2,c2,f2,_=sweep([(p,k) for p,k in later if k],fair,limit=limit,cap=n-n1)
        rows.append(dict(end=r.end,n=n,split=n1+n2,full_pnl=n*(1-r.y)-c-f,
                         split_pnl=(n1+n2)*(1-r.y)-c1-f1-c2-f2))
    d=pd.DataFrame(rows)
    atomic_write(RESULTS/"wing_split_execution.csv",d.to_csv(index=False))
    return {col:round(float(d[col].sum()),2) for col in ["n","split","full_pnl","split_pnl"]}


def current_chain():
    files=sorted((ROOT/"data/raw/wing_quant_inputs").glob("deribit_*.json"))
    if not files:
        return {}
    data=json.loads(files[-1].read_text())
    g=pd.DataFrame(data["result"])
    parts=g.instrument_name.str.extract(r"^BTC-(\d+[A-Z]{3}\d{2})-(\d+(?:\.\d+)?)-[CP]$")
    g["expiry"]=pd.to_datetime(parts[0],format="%d%b%y",utc=True,errors="coerce")+pd.Timedelta(hours=8)
    g["k"]=np.log(pd.to_numeric(parts[1])/g.underlying_price)
    horizon=g.expiry-pd.Timestamp(data["received"])
    clean=g[horizon.between(pd.Timedelta(hours=6),pd.Timedelta(days=10))&
            g.bid_price.gt(0)&g.ask_price.ge(g.bid_price)&g.mark_iv.between(2,300)&g.k.abs().le(.35)]
    for expiry,block in clean.groupby("expiry"):
        vol=block.groupby("k").mark_iv.median()/100
        if len(vol)>=7 and vol.index.min()<0<vol.index.max():
            return dict(time=data["received"],expiry=str(expiry),rows=len(g),strikes=len(vol),
                        svi=svi_smile(vol.index.to_numpy(),vol.to_numpy()).tolist())
    return dict(time=data["received"],rows=len(g),strikes=0)


def main():
    close=Minutes().close
    q,_=books(utcnow().floor("h")-pd.Timedelta(hours=1),observations)
    q=venue_stress(q)
    q=q[q.ts.dt.floor("min").map(regimes(close).blocked).eq(False)].copy().reset_index(drop=True)
    path=RESULTS/"wing_forecast_settlements.json"
    if path.exists():
        ev=q.event.map({k:v["value"] for k,v in json.loads(path.read_text()).items()})
        yes=ev.gt(q.strike).astype(float)
        q["y"]=q.y.fillna(yes.where(q.cheap_yes,1-yes).where(ev.notna()))
    q["caviar"],params=caviar(close,q)
    q["svi"],fits=svi(q)
    size=q.bid_size.where(q.cheap_yes,q.ask_size)
    other=q.ask_size.where(q.cheap_yes,q.bid_size)
    price=(1-q.bid).where(q.cheap_yes,q.ask)
    future=(1-q.later_bid).where(q.cheap_yes,q.later_ask)
    future_size=q.later_bid_size.where(q.cheap_yes,q.later_ask_size)
    observed=(q.later_ts-q.ts).dt.total_seconds().between(0,90,inclusive="right")&q.later_bid.gt(0)&q.later_ask.lt(1)&q.later_ask.ge(q.later_bid)
    retained=(future_size/size).clip(0,1).where(future.le(price+1e-9),0.)
    q["imb"]=(size-other)/(1+size+other)
    q["flow"]=q.ofi.where(q.cheap_yes,-q.ofi)
    q["spread"]=q.ask-q.bid
    q["volume"]=np.log1p(size)
    cols=["flow","imb","spread","volume","left"]
    valid=q[cols].notna().all(axis=1)
    train=q.end.lt(SPLIT-pd.Timedelta(days=1))&observed&valid&retained.notna()
    model=make_pipeline(StandardScaler(),Ridge(alpha=10)).fit(q.loc[train,cols],retained[train])
    fraction=pd.Series(np.nan,index=q.index)
    fraction.loc[valid]=model.predict(q.loc[valid,cols]).clip(0,1)
    q["ofi_queue"]=q.t4.where(fraction*(1-q.t4-price-.07*price*(1-price))>=.01)
    # Conditional one-step transitions from an observable queue state to retained
    # volume. Small states shrink to the training mean (20 pseudo-observations).
    keys=[pd.cut(q.imb,[-np.inf,-.33,.33,np.inf],labels=False),
          pd.cut(q.volume,[-np.inf,np.log1p(100),np.log1p(1000),np.inf],labels=False),
          np.sign(q.flow)]
    states=pd.MultiIndex.from_arrays(keys)
    grouped=pd.Series(retained[train].to_numpy(),index=states[train]).groupby(level=[0,1,2]).agg(["sum","size"])
    est=(grouped["sum"]+20*retained[train].mean())/(grouped["size"]+20)
    state_fraction=pd.Series(est.reindex(states).fillna(retained[train].mean()).to_numpy(),index=q.index)
    q["queue_states"]=q.t4.where(valid & (state_fraction*(1-q.t4-price-.07*price*(1-price))>=.01))
    test=q.end.ge(SPLIT)
    mask=test&observed&valid&retained.notna()
    info=dict(caviar=params,svi_snapshot_fits=fits,queue_train=int(train.sum()),queue_test=int(mask.sum()),
              queue_mae=float((retained[mask]-fraction[mask]).abs().mean()),
              queue_state_mae=float((retained[mask]-state_fraction[mask]).abs().mean()),
              constant_mae=float((retained[mask]-retained[train].median()).abs().mean()))
    rows=[]
    scores={}
    hours={}
    q["hybrid"]=predictions(q.t4,q.mid, q.end.dt.day_of_week>=5,PARAMS)[2]
    for m in ["caviar","svi","ofi_queue","queue_states"]:
        available=valid if m in ("ofi_queue","queue_states") else q[m].notna()
        common=q[test&available]
        hours[m]=pd.DatetimeIndex(sorted(common.end.unique()))
        if m in ("caviar","svi"):
            sample=sampled(common)
            scores[m]={col:float(np.average((sample[col]-sample.y)**2,weights=sample.weight)) for col in ["t4",m]} if len(sample) else {}
        for model_name in ["t4","hybrid",m]:
            rows.extend(replay(common,model_name,m))
    trades=pd.DataFrame(rows)
    atomic_write(RESULTS/"wing_quant_entries.csv",trades.to_csv(index=False))
    summary=[]
    for cohort,g in trades.groupby("period"):
        missing=g.loc[g.pnl.isna(),"end"].unique()
        idx=hours[cohort].difference(missing)
        base=g[g.model.eq("t4")].set_index("end").delayed_pnl.reindex(idx,fill_value=0.)
        for name in ("t4","hybrid",cohort):
            t=g[g.model.eq(name)&~g.end.isin(missing)]
            cost=t.cost.sum()
            delta=t.set_index("end").delayed_pnl.reindex(idx,fill_value=0.)-base
            lo,hi=day_interval(delta)
            summary.append(dict(cohort=cohort,model=name,trades=len(t),contracts=int(t.n.sum()),
                                cost=round(cost,2),roi=round(100*t.pnl.sum()/cost,3) if cost else None,
                                net=round(t.pnl.sum(),2),delayed=round(t.delayed_pnl.sum(),2),excluded=len(missing),
                                delta_ci_dollars_hour=[None if v is None else round(v/100,2) for v in (lo,hi)]))
    info["results"]=summary
    info["first_test_expiry"]=str(q.loc[test,"end"].min())
    info["last_test_expiry"]=str(q.loc[test,"end"].max())
    info["brier"]=scores
    info["maker_samples"]=maker_prints()
    info["split_execution"]=split_execution(q[test],float((future[train]-price[train]).var()))
    info["deribit_download"]=current_chain()
    atomic_write(RESULTS/"wing_quant_replay.json",json.dumps(info,indent=2))
    print(pd.DataFrame(summary).to_string(index=False))


if __name__=="__main__":
    main()
