"""Compare conditional empirical tails and a direct wing classifier with the same volatility gate."""
from pathlib import Path
import argparse
import json
import sys

import numpy as np
import pandas as pd
from scipy.special import stdtrit
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.base import clone
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from lab.core import RESULTS,atomic_write,utcnow
import config
from lab.history import RECENT
from lab.wing_calibration import pooled_probability,sampled
from lab.wing_regime import regimes
from studies.wings_old_replay import archive
from studies.wings_calibration import books
from studies.wings_odds_pool import calendar,fit,score
from studies.wings_edge_timing import START,SPLIT,STOP

FEATURES = ["t4","mid","weekend","left","spread","fhs","ratio","downside","direction","cheap_yes"]


def minute_data(close):
    close = close.asfreq("min")
    close.index = close.index+pd.Timedelta(minutes=1)
    r = np.log(close).diff()
    v = r.pow(2)
    fast,slow = v.rolling(30).mean(),v.rolling(1440).mean()
    ratio = np.sqrt(fast/slow.where(slow.gt(0)))
    scale = np.sqrt((.5*fast+.5*slow)*20)
    data = pd.DataFrame({"ratio":ratio,"downside":r.clip(upper=0).pow(2).rolling(30).mean()/fast.where(fast.gt(0)),
                         "trend":r.rolling(15).sum()/np.sqrt(15*slow.where(slow.gt(0)))})
    # Each residual is indexed by its END. Both scale and class belong to the START.
    residual = np.log(close/close.shift(20))/scale.shift(20).where(scale.shift(20).gt(0))
    history = pd.DataFrame({"residual":residual,"ratio":ratio.shift(20)}).dropna()
    history = history[history.index.minute%5==0]
    return data,history


def features(q,stats,history,gate):
    q = calendar(q).reset_index(drop=True)
    minute = q.ts.dt.floor("min")
    q = q.join(stats,on=minute).drop(columns=["key_0"],errors="ignore")
    q["blocked"] = minute.map(gate)
    q["direction"] = q.trend.where(q.cheap_yes,-q.trend)
    q["spread"] = q.ask-q.bid
    q["fhs"] = q["fhs_all"] = np.nan
    for (day,high),g in q.groupby([minute.dt.floor("D"),q.ratio.ge(1.25)]):
        hist = history.loc[(history.index>=day-pd.Timedelta(days=30))&(history.index<day)]
        distance = -stdtrit(4,g.t4.clip(1e-6,1-1e-6))/np.sqrt(2)
        for name,series in [("fhs_all",hist.residual),("fhs",hist.loc[hist.ratio.ge(1.25).eq(high),"residual"])]:
            x = np.sort(series.values)
            if len(x)>=100:
                hits = np.where(g.cheap_yes,len(x)-np.searchsorted(x,distance,side="right"),
                                np.searchsorted(x,-distance,side="right"))
                q.loc[g.index,name] = (hits+100*g.t4)/(len(x)+100)
    return q.replace([np.inf,-np.inf],np.nan)


def fit_wing_models():
    minute_file = pd.read_parquet(config.INTERIM/"minutes.parquet",columns=["open_time","close"])
    close = minute_file.set_index(pd.to_datetime(minute_file.open_time,utc=True)).close
    if RECENT.exists():
        close = pd.concat([close,pd.read_parquet(RECENT,columns=["close"]).close])
    close = close[~close.index.duplicated(keep="last")].sort_index().loc["2026-05-20":]
    stats,history = minute_data(close)
    gate = regimes(close).blocked
    q,hours = archive()
    q = features(q,stats,history,gate)
    split = pd.Timestamp("2026-09-01",tz="UTC")
    train = sampled(q[q.end.lt(split-pd.Timedelta(days=1)) & q.blocked.eq(False)].dropna(subset=FEATURES))
    coef = fit(train,market_only=True)
    model = HistGradientBoostingClassifier(max_iter=60,max_leaf_nodes=5,learning_rate=.05,
                                          min_samples_leaf=80,l2_regularization=4,early_stopping=False,random_state=42)
    base_features = ["t4","mid","weekend","left","spread","cheap_yes"]
    plain = clone(model).fit(train[base_features],train.y,sample_weight=train.weight)
    model.fit(train[FEATURES],train.y,sample_weight=train.weight)
    return close,stats,history,gate,q,hours,split,train,coef,plain,model,base_features


def main(until=None):
    close,stats,history,gate,q,hours,split,train,coef,plain,model,base_features = fit_wing_models()
    rows = []
    latest = utcnow().floor("h")-pd.Timedelta(hours=1)
    until = min(latest,pd.to_datetime(until,utc=True)) if until else latest
    def evaluate(q,hours,start,end,label):
        idx = hours[(hours>=start)&(hours<end)]
        if not len(idx):
            return
        g = q[q.end.isin(idx)&q.blocked.eq(False)].dropna(subset=FEATURES).copy()
        if g.empty or g.y.notna().sum()==0:
            return
        g["calendar"] = pooled_probability(g.t4,g.mid,g.weekend,coef)
        g["trees"] = model.predict_proba(g[FEATURES])[:,1]
        g["plain"] = plain.predict_proba(g[base_features])[:,1]
        rows.extend(score(g,idx,["t4","calendar","fhs_all","fhs","plain","trees"],label))
    evaluate(q,hours,split,pd.Timestamp("2026-10-28",tz="UTC"),"1 сентября – 27 октября")
    live,hours = books(until)
    live = features(live,stats,history,gate)
    for a,b,label in [(START,SPLIT,"30 октября – 8 ноября"),(SPLIT,STOP,"9–19 ноября"),
                       (STOP,until+pd.Timedelta(seconds=1),"С 20 ноября")]:
        evaluate(live,hours,a,b,label)
    result = dict(until=str(until),training_hours=train.end.nunique(),training_rows=len(train),
                  calendar_coefficients=coef.tolist(),rows=rows)
    atomic_write(RESULTS/"wing_regime_ml.json",json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False))
    print(result["training_hours"],result["training_rows"])
    print(pd.DataFrame(rows).to_string(index=False))


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--until",help="Inclusive last recent expiry, UTC")
    with threadpool_limits(limits=2):
        main(parser.parse_args().until)
