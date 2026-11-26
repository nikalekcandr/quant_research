"""Ten exploratory alternatives on the frozen November hourly candle cohort."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import stdtr, stdtrit
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import SplineTransformer, StandardScaler
from statsmodels.stats.multitest import multipletests
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from lab.core import RESULTS, atomic_write
from lab.history import Minutes
from lab.wing_calibration import sampled
from lab.wing_regime import regimes
from studies.wing_gap_download import CACHE, START
from studies.wing_sleep_replay import features, next_minute
from studies.wings_neighbors import enrich
from studies.wings_old_replay import archive
from studies.wings_two_sided import offers, picks
from studies.wings_calibration import books
from studies.wing_capacity_ml import deepen, replay
import json

BASE = "Соседи + t₄"
NAMES = ["Локальные аналоги","Сплайновая логистическая","Ядерная логистическая","Случайные деревья",
         "Байесовские группы","Асимметричная дисперсия","Смесь скачков","Затухающий импульс",
         "Адаптивные веса","Лимит на цент лучше"]
END = pd.Timestamp("2026-11-25T16:00Z")


def groups(x):
    return (np.digitize(x.t4,[.04,.08,.12,.20])*4 + x.yes*2 + (x.toward>0)).astype(int)


def training_models(train,q):
    a,b = features(train),features(q)
    scale = StandardScaler().fit(a,sample_weight=train.weight)
    x,z = scale.transform(a),scale.transform(b)
    near = NearestNeighbors(n_neighbors=100).fit(x)
    distance, idx = near.kneighbors(z)
    w = train.weight.to_numpy()[idx]/(.25+distance)
    q[NAMES[0]] = ((w*train.y.to_numpy()[idx]).sum(axis=1)+2*b.t4)/(w.sum(axis=1)+2)
    for name, transform in [(NAMES[1],SplineTransformer(n_knots=4,degree=2)),
                            (NAMES[2],RBFSampler(gamma=.15,n_components=128,random_state=19))]:
        model = LogisticRegression(C=1,max_iter=500).fit(transform.fit_transform(x),train.y,sample_weight=train.weight)
        q[name] = model.predict_proba(transform.transform(z))[:,1]
        if name == NAMES[1]:
            spline = (scale,transform,model)
    trees = ExtraTreesClassifier(n_estimators=100,min_samples_leaf=75,max_features=.8,random_state=19,n_jobs=2)
    trees.fit(x, train.y, sample_weight=train.weight)
    q[NAMES[3]] = trees.predict_proba(z)[:,1]
    sums = pd.DataFrame(dict(group=groups(a),wins=train.y*train.weight,n=train.weight)).groupby("group").sum()
    key = groups(b)
    q[NAMES[4]] = (key.map(sums.wins).fillna(0)+25*b.t4)/(key.map(sums.n).fillna(0)+25)
    return spline


def capacity(spline,close):
    q,_ = books(END,deepen)
    gate = regimes(close).blocked
    m = q.ts.dt.floor("min")
    lag = m-pd.Timedelta(minutes=2)
    q["move5"] = np.log(lag.map(close)/ (lag-pd.Timedelta(minutes=5)).map(close))
    q = q[q.end.ge(START)&q.left.ge(16)&m.map(gate).eq(False)].dropna(subset=["move5"]).copy()
    cache = ROOT/"data/raw/wing_quant_inputs/settlements.json"
    if cache.exists():
        value = q.event.map({e:r["value"] for e,r in json.loads(cache.read_text()).items()})
        yes = value.gt(q.strike).astype(float)
        q["y"] = q.y.fillna(yes.where(q.cheap_yes,1-yes).where(value.notna()))
    q[BASE] = (q.t4+q.curve.where(q.cheap_yes,1-q.curve).fillna(q.t4))/2
    scale,spl,model = spline
    q[NAMES[1]] = model.predict_proba(spl.transform(scale.transform(features(q))))[:,1]
    entries = pd.DataFrame(replay(q,BASE,"books")+replay(q,NAMES[1],"books"))
    missing = entries.loc[entries.pnl.isna(),"end"].unique()
    usable = entries[~entries.end.isin(missing)]
    result = usable.groupby("model").agg(signals=("n","size"), contracts=("delayed_n","sum"),
                        cost=("delayed_cost","sum"),net=("delayed_pnl","sum"),worst=("delayed_pnl","min")).round(2)
    result["excluded_hours"] = len(missing)
    atomic_write(RESULTS/"wing_ten_capacity.csv",result.to_csv())
    print(result.to_string())


def structural(q,close):
    r = np.log(close.asfreq("min")).diff()
    r.index += pd.Timedelta(minutes=1)
    share = r.clip(lower=0).pow(2).rolling(60).sum()/r.pow(2).rolling(60).sum()
    up = q.ts.map(share.shift(1)).fillna(.5)
    factor = np.sqrt(2*up.where(q.cheap_yes,1-up)).clip(.65,1.5)
    z = stdtrit(4,q.t4.clip(1e-6,1-1e-6))
    q[NAMES[5]] = stdtr(4,z/factor)
    # Match unconditional scale: 95% quiet component, 5% double scale.
    q[NAMES[6]] = .95*stdtr(4,z/np.sqrt(.8/.95))+.05*stdtr(4,z/2)
    vol = q.ts.map(r.rolling(60).std().shift(1))*np.sqrt(q.left)
    drift = (.25*q.move5*np.where(q.cheap_yes,1,-1)/vol.replace(0,np.nan)).clip(-.5,.5).fillna(0)
    q[NAMES[7]] = stdtr(4,z+drift*np.sqrt(2))
    curve = q.curve.where(q.cheap_yes,1-q.curve).fillna(q.t4)
    q[BASE] = (q.t4+curve)/2
    s = sampled(q.assign(neighbor=curve))
    diff = s.weight*((s.t4-s.y)**2-(s.neighbor-s.y)**2)
    # Use only previous UTC days so every training label has settled.
    daily = diff.groupby(s.end.dt.floor("D")).sum()
    days = pd.date_range(q.ts.min().floor("D"),q.end.max().floor("D"),freq="D")
    past = daily.reindex(days,fill_value=0).rolling(7,min_periods=1).sum().shift(1).fillna(0)
    weight = 1/(1+np.exp(-4*q.ts.dt.floor("D").map(past).clip(-10,10)))
    q[NAMES[8]] = (1-weight)*q.t4+weight*curve
    q[NAMES[9]] = q[BASE]


def main():
    old,_ = archive()
    train = sampled(old[old.end<START-pd.Timedelta(days=1)])
    close = Minutes().close
    q,hours = archive(enrich,files=CACHE.glob("*.csv.gz"),minute_close=close)
    hours = hours[(hours>=START)&(hours<=END)]
    q = q[q.end.isin(hours)&q.left.ge(16)&q.ts.map(regimes(close).blocked).eq(False)].copy()
    spline = training_models(train,q)
    structural(q,close)
    records = []
    for name in [BASE,*NAMES]:
        chosen = picks(offers(q.assign(t4=q[name]),False)).copy()
        if name==NAMES[9]:
            chosen["price"] -= .01
            chosen["price"] = chosen.price.round(2)
        records.append(next_minute(chosen).assign(model=name)[["model","market","ts","filled","pnl"]])
    trades = pd.concat(records)
    hours = hours.difference(trades.index[trades.pnl.isna()])
    pnl = trades.pivot(columns="model",values="pnl").reindex(hours).fillna(0.)
    ev = sampled(q[q.end.isin(hours)])
    daily = pnl.sub(pnl[BASE],axis=0).groupby(hours.floor("D")).sum()
    # Paired one-sided randomisation by days; same signs for all ten comparisons.
    signs = np.random.default_rng(19).choice([-1,1],size=(50000,len(daily)))
    delta = daily[NAMES].to_numpy()
    raw = (1+(signs@delta>=delta.sum(axis=0)-1e-9).sum(axis=0))/(len(signs)+1)
    adjusted = dict(zip(NAMES,multipletests(raw, method="holm")[1]))
    rows = []
    for name in [BASE,*NAMES]:
        g = trades[trades.model.eq(name)&trades.index.isin(hours)]
        v = pnl[name]
        rows.append({"Модель":name,"Часы":len(hours),"Исполнено":int(g.filled.sum()),"Убытки":int(v.lt(0).sum()),
                     "Нетто, $":round(v.sum(),2),"Δ к соседям, $":round((v-pnl[BASE]).sum(),2),
                     "С 20 ноября, $":round(v.loc["2026-11-20":].sum(),2),
                     "Brier":round(np.average((ev[name]-ev.y)**2,weights=ev.weight),5),
                     "p Holm":round(adjusted[name],4) if name in adjusted else None})
    result = pd.DataFrame(rows)
    atomic_write(RESULTS/"wing_ten.csv",result.to_csv(index=False))
    atomic_write(RESULTS/"wing_ten_entries.csv",trades.reset_index().to_csv(index=False))
    print(result.to_string(index=False))
    capacity(spline,close)

if __name__=="__main__":
    with threadpool_limits(limits=2):
        main()
