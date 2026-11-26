"""Fixed October training cutoff and a one-contract, next-minute November replay."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lab.core import LAB, RESULTS, atomic_write, kalshi_fee, utcnow
from lab.history import Minutes
from lab.wing_calibration import sampled
from lab.wing_regime import regimes
from studies.wing_gap_download import START, CACHE
from studies.wings_old_replay import archive
from studies.wings_neighbors import enrich
from studies.wings_two_sided import offers, picks
from studies.wings_edge_timing import day_interval


def features(q):
    mid = ((q.bid+q.ask)/2).where(q.cheap_yes, 1-(q.bid+q.ask)/2)
    direction = np.where(q.cheap_yes, 1, -1)
    return pd.DataFrame(dict(t4=q.t4, mid=mid, toward=100*q.move5*direction,
                             minutes=q.left, spread=q.ask-q.bid, yes=q.cheap_yes.astype(int)))


def beta_design(x):
    p = x[["t4","mid"]].clip(1e-6,1-1e-6)
    return np.column_stack([np.log(p), -np.log1p(-p), x.toward, x.toward*x.yes, x.yes])


def next_minute(chosen):
    price = (1-chosen.later_bid).where(chosen.cheap_yes,chosen.later_ask)
    valid_book = chosen.later_bid.gt(0)&chosen.later_ask.lt(1)&chosen.later_ask.ge(chosen.later_bid)
    seen = (chosen.later_ts-chosen.ts).eq(pd.Timedelta(minutes=1))&valid_book
    filled = seen & price.le(chosen.price+1e-9)
    pnl = (chosen.pay-price-price.fillna(.5).map(kalshi_fee)).where(filled,0.)
    return chosen.assign(seen=seen,filled=filled,pnl=pnl)


def overlap_seconds(start, stop, periods):
    cursor = start
    seconds = 0.
    for sleep,wake in periods.itertuples(index=False):
        left,right = max(cursor,sleep),min(stop,wake)
        if right>left:
            seconds += (right-left).total_seconds()
            cursor = right
    return seconds


def coverage(hours, periods):
    times = []
    for path in sorted((LAB/"markets").glob("*.csv")):
        if path.stem < str(START.date()):
            continue
        log = pd.read_csv(path,usecols=["end","ts","kind","venue","book"])
        log = log[log.venue.eq("kalshi") & log.kind.eq("kx_ladder_1h") & log.book.eq(1)]
        end = pd.to_datetime(log.end,utc=True,format="mixed")
        ts = pd.to_datetime(log.ts,utc=True,format="mixed")
        keep = (end-ts).dt.total_seconds().between(15*60,30*60)
        times.append(pd.DataFrame(dict(end=end[keep],minute=ts[keep].dt.floor("min"))))
    counted = pd.concat(times).groupby("end").minute.nunique() if times else {}
    rows = []
    for end in hours:
        a,b = end-pd.Timedelta(minutes=30),end-pd.Timedelta(minutes=15)
        seconds = overlap_seconds(a,b,periods)
        logged = counted.get(end,0)
        status = ("Сон всё окно" if seconds>=900 else "Частичный сон" if seconds>0
                  else "Есть запись" if logged else "Прочие пропуски")
        rows.append(dict(end=end,group=status,logged_minutes=logged,sleep_seconds=seconds))
    return pd.DataFrame(rows).set_index("end")


def main():
    close = Minutes().close
    old,_ = archive()
    train = sampled(old[old.end.lt(START-pd.Timedelta(days=1))])
    x = features(train)
    beta = LogisticRegression(C=1.,max_iter=500).fit(beta_design(x),train.y,sample_weight=train.weight)
    tree = HistGradientBoostingClassifier(max_iter=80,max_leaf_nodes=7,l2_regularization=3,
              monotonic_cst=[1,1,0,0,0,0],early_stopping=False,random_state=15)
    tree.fit(x,train.y,sample_weight=train.weight)
    files = sorted(CACHE.glob("*.csv.gz"))
    q,valid = archive(enrich,files=files,minute_close=close)
    last = min(close.index.max().floor("h"),utcnow().floor("h")-pd.Timedelta(hours=1))
    periods = pd.read_csv(CACHE/"sleep.csv").apply(lambda col: pd.to_datetime(col, utc=True)).sort_values("sleep")
    audit = coverage(pd.date_range(START,last,freq="h"), periods)
    audit["downloaded"] = audit.index.strftime("%Y%m%d%H").isin({p.name[:10] for p in files})
    audit["priced"] = audit.index.isin(valid)
    gate = regimes(close).blocked
    q = q[q.end.isin(audit.index) & q.left.ge(16) & q.ts.map(gate).eq(False)].copy()
    x = features(q)
    q["beta_pool"] = beta.predict_proba(beta_design(x))[:,1]
    q["monotone_boost"] = tree.predict_proba(x)[:,1]
    q["ensemble"] = (q.beta_pool+q.monotone_boost+q.t4)/3
    neighbor = q.curve.where(q.cheap_yes,1-q.curve)
    q["neighbor_mix"] = (q.t4+neighbor.fillna(q.t4))/2
    q["W04_proxy"] = q.t4
    asleep = pd.Series(False,index=q.index)
    for sleep,wake in periods.itertuples(index=False):
        asleep |= q.ts.between(sleep,wake,inclusive="left")
    audit["candidates"] = q.groupby("end").size().reindex(audit.index,fill_value=0)
    entries = []
    names = ["t4","neighbor_mix","W04_proxy","beta_pool","monotone_boost","ensemble"]
    for name in names:
        signals = offers(q.assign(t4=q[name]), False, margin=0 if name=="W04_proxy" else .01)
        chosen = picks(signals).copy()
        awake = picks(signals[~asleep.reindex(signals.index)]).copy()
        chosen["policy"], awake["policy"] = "continuous","awake"
        chosen = pd.concat([chosen,awake])
        chosen = next_minute(chosen)
        chosen["model"] = name
        chosen["asleep_at_signal"] = False
        for sleep,wake in periods.itertuples(index=False):
            chosen["asleep_at_signal"] |= chosen.ts.between(sleep,wake,inclusive="left")
        entries.append(chosen[["market","ts","model","policy","seen","filled","asleep_at_signal","pnl"]])
    trades = pd.concat(entries).join(audit[["group"]])
    audit["scored"] = audit.priced & ~audit.index.isin(trades.index[trades.pnl.isna()])
    rows = []
    for label, hours in [("Все часы",audit),*audit.groupby("group")]:
        h = hours.index[hours.scored]
        evaluation = sampled(q[q.end.isin(h)])
        cohort = trades[trades.index.isin(h)]
        continuous = cohort[cohort.policy.eq("continuous")]
        baseline = continuous[continuous.model.eq("t4")].pnl.reindex(h,fill_value=0.)
        for name in names:
            t = continuous[continuous.model.eq(name)]
            awake_net = cohort.loc[cohort.policy.eq("awake") & cohort.model.eq(name),"pnl"].sum()
            lo,hi = day_interval(t.pnl.reindex(h,fill_value=0.)-baseline)
            rows.append({"Период":label,"Модель":name,"Часы":len(hours),"Оценены":len(h),
                         "Сигналы":len(t),"Повторная цена":int(t.seen.sum()),"Входы":int(t.filled.sum()),
                         "Убытки":int(t.pnl.lt(0).sum()),"Нетто, $":round(t.pnl.sum(),2),
                         "Только бодрствование, $":round(awake_net,2),
                         "Эффект 24/7, $":round(t.pnl.sum()-awake_net,2),
                         "Нетто сигналов во сне, $":round(t.loc[t.asleep_at_signal,"pnl"].sum(),2),
                         "Brier":round(np.average((evaluation[name]-evaluation.y)**2,weights=evaluation.weight),5) if len(evaluation) else None,
                         "95% Δ к t₄, ¢/час":f"{lo:.2f} … {hi:.2f}" if lo is not None else "—"})
    for name,frame in [("hours",audit.reset_index()),("entries",trades.reset_index()),("summary",pd.DataFrame(rows))]:
        atomic_write(RESULTS/f"wing_sleep_{name}.csv",frame.to_csv(index=False))
    print(pd.DataFrame(rows).to_string(index=False))

if __name__ == "__main__":
    with threadpool_limits(limits=2):
        main()
