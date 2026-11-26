"""Leave-one-strike-out pricing and a next-snapshot limit fill sensitivity."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from scipy.special import expit,logit,stdtr,stdtrit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from lab.core import LAB,RESULTS,atomic_write,kalshi_fee,utcnow
from studies.wings_calibration import books
from studies.wings_old_replay import archive
from studies.wings_two_sided import offers,picks
from studies.wings_edge_timing import SPLIT,STOP,day_interval


def enrich(frame):
    q = frame.copy().reset_index(drop=True)
    q["strike"] = pd.to_numeric(q.market.str.rsplit("-T",n=1).str[-1],errors="coerce")
    book = q[q.bid.gt(0)&q.ask.lt(1)&(q.ask-q.bid).round(6).between(0,.03)&q.strike.gt(0)].copy()
    book["odds"] = logit((book.bid+book.ask)/2)
    group = book.sort_values("strike").groupby(["end","ts"])[["strike","odds"]]
    lower,upper = group.shift(),group.shift(-1)
    u = np.log(book.strike/lower.strike)/np.log(upper.strike/lower.strike)
    curve = expit(lower.odds+u*(upper.odds-lower.odds))
    q["curve"] = curve.where(lower.odds.ge(upper.odds)&u.between(0,1))
    if "bid_size" not in q:
        q["bid_size"] = q["ask_size"] = 1.
    cols = ["ts","bid","ask","bid_size","ask_size"]
    later = q.sort_values("ts").groupby(["market","end"])[cols].shift(-1)
    return q.join(later.add_prefix("later_"))


def delayed_execution(order):
    price = 1-order.later_bid.where(order.cheap_yes,1-order.later_ask)
    size = order.later_bid_size.where(order.cheap_yes,order.later_ask_size)
    elapsed = (order.later_ts-order.ts).dt.total_seconds()
    seen = elapsed.between(0,90,inclusive="right") & order.later_bid.gt(0) & order.later_ask.lt(1) & order.later_ask.ge(order.later_bid)
    fill = seen & size.ge(1) & price.round(6).le(order.price.round(6))
    value = order.pay[fill]-price[fill]-price[fill].map(kalshi_fee)
    return seen,fill,value


def replay(q,hours,label):
    wing = q.curve.where(q.cheap_yes,1-q.curve)
    methods = {"База t₄":q.t4,"Среднее с соседями":(q.t4+wing.fillna(q.t4))/2,
               "Подтверждение соседями":np.maximum(q.t4,wing)}
    if "stressed" in q:
        methods["Стресс Coinbase/Kraken"] = q.stressed
        methods["Соседи и стресс курса"] = np.maximum(q.stressed,wing)
    picks_by_model,immediate,delayed,counts = {},{}, {}, {}
    attempts = {}
    for name,p in methods.items():
        first = picks(offers(q.assign(t4=p),False))
        picks_by_model[name] = first.index
        seen,fill,value = delayed_execution(first)
        immediate[name] = (first.pay-first.price-first.fee).reindex(hours,fill_value=0.)
        delayed[name] = value.reindex(hours,fill_value=0.)
        counts[name] = (seen,fill)
        attempts[name] = first.index
        if name=="База t₄":
            retry_name = "База t₄ · повторная попытка"
            deadlines = first.ts+pd.Timedelta(seconds=90)
            # A non-fill is known only at the observed response or the timeout.
            ready = first.later_ts.where(first.later_ts.le(deadlines),deadlines)
            failed = ready[~fill]
            second = picks(offers(q[q.ts.ge(q.end.map(failed))],False))
            seen2,fill2,value2 = delayed_execution(second)
            delayed[retry_name] = pd.concat([value,value2]).reindex(hours,fill_value=0.)
            immediate[retry_name] = immediate[name]
            counts[retry_name] = (seen | seen2.reindex(first.index,fill_value=False),
                                  fill | fill2.reindex(first.index,fill_value=False))
            picks_by_model[retry_name] = first.index
            attempts[retry_name] = first.index.append(second.index)
    now,later = pd.DataFrame(immediate),pd.DataFrame(delayed)
    common = now.notna().all(axis=1)&later.notna().all(axis=1)
    rows = []
    for name in delayed:
        seen,fill = counts[name]
        idx = now.index[common]
        lo,hi = day_interval((later[name]-later["База t₄"]).loc[idx])
        rows.append({"Период":label,"Правило":name,"Часы":len(idx),"Сигналы":len(idx.intersection(picks_by_model[name])),
                     "Попытки":int(attempts[name].isin(idx).sum()),
                     "Повторный стакан":len(idx.intersection(seen[seen].index)),"Исполнимо":len(idx.intersection(fill[fill].index)),
                     "Сразу, $":round(now.loc[idx,name].sum(),2),"С задержкой, $":round(later.loc[idx,name].sum(),2),
                     "Убытки с задержкой":int((later.loc[idx,name]<0).sum()),
                     "95% Δ с задержкой, ¢/час":f"{lo:.2f} … {hi:.2f}" if lo is not None else "—"})
    return rows


def venue_stress(q):
    parts = []
    cols = ["ts","usd_mid","cb_mid","kr_mid"]
    for day in q.ts.dt.strftime("%Y-%m-%d").unique():
        path = LAB/"ticks"/(day+".csv")
        if path.exists() and set(cols).issubset(pd.read_csv(path,nrows=0).columns):
            parts.append(pd.read_csv(path,usecols=cols))
    if not parts:
        return q.assign(stressed=np.nan)
    ticks = pd.concat(parts,ignore_index=True)
    ticks["ts"] = pd.to_datetime(ticks.ts,utc=True,format="mixed").astype(q.ts.dtype)
    q = pd.merge_asof(q.sort_values("ts"),ticks.sort_values("ts"),on="ts",direction="backward",
                      tolerance=pd.Timedelta(seconds=5))
    yes = q.t4.where(q.cheap_yes,1-q.t4)
    distance = np.log(q.usd_mid/q.strike)
    scale = stdtrit(4,yes)/distance.where(distance.abs()>1e-9)
    mids = q[["cb_mid","kr_mid"]]
    adverse = mids.max(axis=1).where(q.cheap_yes,mids.min(axis=1))
    adjusted = stdtr(4,scale*np.log(adverse/q.strike))
    ok = mids.gt(0).all(axis=1)&scale.gt(0)&np.isfinite(scale)
    q["stressed"] = np.maximum(q.t4,adjusted.where(q.cheap_yes,1-adjusted)).where(ok)
    return q


def main():
    rows = []
    for source in ["archive","books"]:
        if source=="archive":
            q,hours = archive(enrich)
            periods = [("2026-07-01","2026-09-01"),("2026-09-01","2026-10-28")]
        else:
            until = utcnow().floor("h")-pd.Timedelta(hours=1)
            q,hours = books(until,enrich)
            q = venue_stress(q)
            periods = [("2026-10-30",SPLIT),(SPLIT,STOP),(STOP,until+pd.Timedelta(seconds=1))]
        for start,end in periods:
            idx = hours[(hours>=pd.to_datetime(start,utc=True))&(hours<pd.to_datetime(end,utc=True))]
            if len(idx):
                rows.extend(replay(q[q.end.isin(idx)],idx,f"{idx.min():%d.%m} – {idx.max():%d.%m}"))
    result = pd.DataFrame(rows)
    atomic_write(RESULTS/"wing_neighbors.csv",result.to_csv(index=False))
    print(result.to_string(index=False))


if __name__=="__main__":
    main()
