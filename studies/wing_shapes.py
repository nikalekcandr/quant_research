"""Ten leave-one-strike-out cross-sectional probability curves, with identical execution."""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import expit,logit,ndtr,ndtri,stdtr,stdtrit
from statsmodels.stats.multitest import multipletests

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from lab.core import RESULTS,atomic_write
from lab.history import Minutes
from lab.wing_regime import regimes
from studies.wing_gap_download import CACHE
from studies.wing_capacity_ml import deepen,replay
from studies.wings_calibration import books
from studies.wings_neighbors import enrich
from studies.wings_old_replay import archive
from studies.wings_two_sided import offers,picks
from studies.wing_sleep_replay import next_minute

BASE="Соседи + t₄"
NAMES=["Линейная вероятность","Пробит","Шкала Гумбеля","Обратная t₄","Второе кольцо",
       "Медиана трёх колец","Шесть страйков WLS","Два односторонних наклона","Границы bid/ask","Вся лестница"]

def curves(frame,live=False):
    q=deepen(frame) if live else enrich(frame)
    b=q[q.bid.gt(0)&q.ask.lt(1)&(q.ask-q.bid).round(6).between(0,.03)&q.strike.gt(0)].sort_values("strike").copy()
    b["x"]=np.log(b.strike)
    b["p"]=(b.bid+b.ask)/2
    b["z"]=logit(b.p)
    group=b.groupby(["end","ts"])
    lags={k:group[["x","p","z","bid","ask"]].shift(k) for k in [1,-1,2,-2,3,-3]}
    l,h=lags[1],lags[-1]
    u=(b.x-l.x)/(h.x-l.x)
    def interp(a,c,w=u):
        return a+w*(c-a)
    def ring(k):
        a,c=lags[k],lags[-k]
        return expit(interp(a.z,c.z,(b.x-a.x)/(c.x-a.x))).where(a.z.ge(c.z))
    estimates=[interp(l.p,h.p),ndtr(interp(ndtri(l.p),ndtri(h.p))),
               1-np.exp(-np.exp(interp(np.log(-np.log1p(-l.p)),np.log(-np.log1p(-h.p))))),
               stdtr(4,interp(stdtrit(4,l.p),stdtrit(4,h.p))),ring(2),
               pd.concat([ring(k) for k in (1,2,3)],axis=1).median(axis=1).where(ring(2).notna())]
    sw=sx=sy=sxx=sxy=0.
    for a in lags.values():
        x=(a.x-b.x)*100
        w=1/((a.ask-a.bid+.01)*(1+x.abs()))
        sw=sw+w.fillna(0); sx=sx+(w*x).fillna(0); sy=sy+(w*a.z).fillna(0)
        sxx=sxx+(w*x*x).fillna(0); sxy=sxy+(w*x*a.z).fillna(0)
    intercept=(sy*sxx-sx*sxy)/(sw*sxx-sx*sx)
    left=l.z+(b.x-l.x)*(l.z-lags[2].z)/(l.x-lags[2].x)
    right=h.z+(b.x-h.x)*(h.z-lags[-2].z)/(h.x-lags[-2].x)
    interval=(expit(interp(logit(l.bid),logit(h.bid)))+expit(interp(logit(l.ask),logit(h.ask))))/2
    # Remove the target's contribution from all five global sufficient statistics.
    x=100*(b.x-11)
    def others(s):
        return s.groupby([b.end,b.ts]).transform("sum")-s
    n=group.x.transform("size")-1
    xx,zz=others(x),others(b.z)
    slope=(n*others(x*b.z)-xx*zz)/(n*others(x*x)-xx*xx).where(n>=3)
    global_fit=expit((zz-slope*xx)/n+slope*x).where(slope.lt(0))
    estimates.extend([expit(intercept),(expit(left)+expit(right))/2,interval,global_fit])
    for name,est in zip(NAMES,estimates):
        q[name]=pd.Series(est,index=b.index).clip(0,1).where(u.between(0,1)&l.p.ge(h.p))
    return q

def evaluate(q,hours,gate,depth,policy=None):
    q=q[q.left.ge(16)&q.ts.dt.floor("min").map(gate).eq(False)].copy()
    def blend(p):
        return (q.t4+p.where(q.cheap_yes,1-p).fillna(q.t4))/2
    q[BASE]=blend(q.curve)
    names=policy(q) if policy is not None else NAMES
    entries=[]
    for name in [BASE,*names]:
        if name!=BASE and policy is None: q[name]=blend(q[name].fillna(q.curve))
        if depth:
            entries.extend(dict(end=t["end"],model=name,net=t["delayed_pnl"],n=t["delayed_n"]) for t in replay(q,name,"depth"))
        else:
            g=next_minute(picks(offers(q.assign(t4=q[name]),False)))
            entries.extend(dict(end=t.Index,model=name,net=t.pnl,n=int(t.filled)) for t in g.itertuples())
    df=pd.DataFrame(entries)
    hours=hours.difference(df.loc[df.net.isna(),"end"])
    df=df[df.end.isin(hours)]
    panel=df.pivot(index="end",columns="model",values="net").reindex(index=hours,columns=[BASE,*names]).fillna(0.)
    delta=panel[names].sub(panel[BASE],axis=0).groupby(hours.floor("D")).sum()
    signs=np.random.default_rng(818).choice([-1,1],(30000,len(delta)))
    p=(1+(signs@delta.to_numpy()>=delta.sum().to_numpy()-1e-9).sum(axis=0))/30001
    out=df.groupby("model").agg(signals=("n","size"),fills=("n",lambda x:int((x>0).sum())),
                                contracts=("n","sum")).reindex([BASE,*names],fill_value=0)
    out["net"]=panel.sum();out["nov"]=panel.loc["2026-11":].sum();out["delta"]=out.net-out.net[BASE]
    out["p_holm"]=pd.Series(multipletests(p,method="holm")[1],index=names)
    out["hours"]=len(hours)
    return out.reindex([BASE,*names]).round(4),df

def main(policy=None,prefix="wing_shapes"):
    m=pd.read_parquet(ROOT/"data/interim/minutes.parquet",columns=["open_time","close"])
    close=pd.concat([m.set_index(pd.to_datetime(m.open_time,utc=True)).close,Minutes().close])
    close=close[~close.index.duplicated(keep="last")].sort_index()
    gate=regimes(close).blocked
    files=[*(ROOT/"data/raw/kalshi_kxbtcd_60m").glob("*.csv.gz"),*CACHE.glob("*.csv.gz")]
    q,hours=archive(curves if policy is None else enrich,files=files,minute_close=close)
    a,ae=evaluate(q,hours,gate,False,policy)
    live,_=books(close.index.max().floor("h"),(lambda f:curves(f,True)) if policy is None else deepen,since=close.index.min())
    values=json.loads((CACHE.parent/"wing_quant_inputs/settlements.json").read_text())
    v=live.event.map({e:r["value"] for e,r in values.items()})
    y=v.gt(live.strike).astype(float)
    live["y"]=live.y.fillna(y.where(live.cheap_yes,1-y).where(v.notna()))
    usable=live[live.left.ge(16)&live.ts.dt.floor("min").map(gate).eq(False)]
    b,be=evaluate(live,pd.DatetimeIndex(sorted(usable.end.unique())),gate,True,policy)
    out=pd.concat([a.add_prefix("archive_"),b.add_prefix("book_")],axis=1)
    atomic_write(RESULTS/(prefix+".csv"),out.to_csv())
    atomic_write(RESULTS/(prefix+"_entries.csv"),pd.concat([ae.assign(sample="archive"),be.assign(sample="book")]).to_csv(index=False))
    print(out.to_string())

if __name__=="__main__":
    main()
