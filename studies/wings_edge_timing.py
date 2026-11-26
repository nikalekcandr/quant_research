"""Compare causal edge stopping policies on the full hourly Kalshi ladder, one contract per hour.

python studies/wings_edge_timing.py
The Nov 9 split and Nov 20 cutoff fix the retrospective exercise. Fit only on earlier quotes.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from lab.core import LAB, RESULTS, atomic_write, kalshi_fee, read_ledger
from lab.edge_timing import candidates

START = pd.Timestamp("2026-10-30",tz="UTC")
SPLIT = pd.Timestamp("2026-11-09",tz="UTC")
STOP = pd.Timestamp("2026-11-20",tz="UTC")
COLUMNS = ["ts","end","event","market","kind","venue","book","fair_t4","bid","ask","bid_size","ask_size"]
FIXED = {"0¢":0., "1¢":.01, "2¢":.02, "3¢":.03}


def observations():
    parts, monitored = [], set()
    for file in sorted((LAB/"markets").glob("*.csv")):
        day = pd.Timestamp(file.stem,tz="UTC")
        if day < START-pd.Timedelta(days=1) or day >= STOP:
            continue
        if not set(COLUMNS).issubset(pd.read_csv(file,nrows=0).columns):
            continue
        q = pd.read_csv(file,usecols=COLUMNS,low_memory=False)
        q = q[q.kind.eq("kx_ladder_1h") & q.venue.eq("kalshi")].copy()
        q["ts"] = pd.to_datetime(q.ts,utc=True,format="mixed")
        q["end"] = pd.to_datetime(q.end,utc=True,format="mixed")
        q["left"] = (q.end-q.ts).dt.total_seconds()/60
        q = q[q.end.between(START,STOP,inclusive="left") & q.left.between(12,30)].sort_values("ts")
        q = q.drop_duplicates(["market","ts"])
        book_ok = q.book.eq(1) & q.bid.gt(0) & q.ask.lt(1) & q.ask.ge(q.bid)
        monitored.update(q.loc[book_ok & q.fair_t4.between(0,1) & q.left.ge(15),"end"])
        # The next logged quote is used only for latency diagnostics after selection.
        future = q.groupby("market")[["ts","book","bid","ask","bid_size","ask_size","fair_t4"]].shift(-1).add_prefix("next_")
        parts.append(candidates(q.join(future)))
    if not parts or not monitored:
        raise ValueError("No Kalshi book snapshots in the study window")
    c = pd.concat(parts,ignore_index=True).sort_values(["end","ts"])
    c["minute"] = c.left.astype(int)
    return c, pd.DatetimeIndex(sorted(monitored),name="end")


def fit_schedule(c, hours):
    """Compare taking today's best quote with the expected value of waiting one more minute.

    Continuation values are estimated on training hours with a candidate at this minute;
    five pooled pseudo-hours stabilise thin bins. A missed trade has value zero.
    """
    future = pd.Series(0.,index=hours)
    schedule = {}
    for minute in range(15,31):
        block = c[c.minute.eq(minute)]
        available = block.end.unique()
        floor = (future.loc[available].sum()+5*future.mean())/(len(available)+5)
        schedule[minute] = round(float(floor),5)
        first = block[block.edge.ge(schedule[minute])].sort_values("ts").drop_duplicates("end")
        future.loc[first.end] = first.edge.to_numpy()
    return schedule


def first_entries(c, floor):
    threshold = c.minute.map(floor) if isinstance(floor,dict) else floor
    picks = c.assign(margin=threshold)
    return picks[picks.edge.ge(picks.margin)].sort_values(["end","ts"]).drop_duplicates("end").set_index("end")


def resolve_ladder(markets, outcomes):
    """For a missing strike, a settled YES above it or NO below it implies its outcome."""
    bounds = {}
    resolved = {}
    for key,value in outcomes.items():
        if not key.startswith("kalshi|KXBTCD-") or value not in (0,1):
            continue
        ticker = key.split("|",1)[1]
        event, marker, raw = ticker.rpartition("-T")
        if not marker:
            continue
        strike = float(raw)
        low,high = bounds.get(event,(-np.inf,np.inf))
        bounds[event] = (max(low,strike),high) if value == 1 else (low,min(high,strike))
        resolved[ticker] = value
    for ticker in markets:
        if ticker in resolved:
            continue
        event, marker, raw = ticker.rpartition("-T")
        low,high = bounds.get(event,(-np.inf,np.inf))
        if marker and low < high:
            if float(raw) <= low:
                resolved[ticker] = 1
            elif float(raw) >= high:
                resolved[ticker] = 0
    return resolved


def settlement_cache():
    outcomes = json.loads((LAB/"resolutions.json").read_text(encoding="utf-8"))
    archive = ROOT/"data/interim/kalshi.parquet"
    if archive.exists():
        d = pd.read_parquet(archive,columns=["ticker","result","strike_type"])
        d = d[d.strike_type.eq("greater") & d.result.isin(["yes","no"])].drop_duplicates(["ticker","result"])
        if d.ticker.duplicated().any():
            raise ValueError("Conflicting archived Kalshi resolutions")
        for row in d.itertuples():
            key, value = "kalshi|"+row.ticker, int(row.result=="yes")
            if key in outcomes and outcomes[key] != value:
                raise ValueError(f"Conflicting settlement for {row.ticker}")
            outcomes[key] = value
    return outcomes


def score(entries,hours,resolutions):
    e = entries.reindex(hours)
    taken = e.market.notna()
    yes = e.market.map(resolutions)
    payout = yes.where(e.side.eq("yes"),1-yes)
    e["pnl"] = (payout-e.price-e.fee).where(taken,0.)
    e["taken"] = taken
    # Is the same limit still executable with positive edge at the next observation?
    seconds = (e.next_ts-e.ts).dt.total_seconds()
    price = e.next_ask.where(e.side.eq("yes"),1-e.next_bid)
    fair = e.next_fair_t4.where(e.side.eq("yes"),1-e.next_fair_t4)
    size = e.next_ask_size.where(e.side.eq("yes"),e.next_bid_size)
    observed = (seconds.between(1,90) & e.next_book.eq(1) & e.next_bid.gt(0) & e.next_ask.lt(1)
                & (e.next_ask-e.next_bid).ge(0) & fair.between(0,1) & size.notna())
    fee = price.where(observed).dropna().map(kalshi_fee).reindex(hours)
    e["retained"] = ((price <= e.price+1e-8) & (fair-price-fee >= e.margin) & size.ge(1)
                     & (e.next_ask-e.next_bid).round(6).le(.03)).astype(float).where(observed)
    return e


def day_interval(delta):
    daily = delta.groupby(delta.index.floor("D")).agg(["sum","size"])
    if len(daily)<2:
        return [None,None]
    draws = np.random.default_rng(404).integers(0,len(daily),size=(10000,len(daily)))
    estimates = daily["sum"].to_numpy()[draws].sum(axis=1)/daily["size"].to_numpy()[draws].sum(axis=1)
    return np.quantile(100*estimates,[.025,.975]).round(3).tolist()


def comparison(a,b):
    """Bounds on the paired difference across ALL observed hours, including unknown results."""
    lower = a.pnl.fillna(-a.price-a.fee) - b.pnl.fillna(1-b.price-b.fee)
    upper = a.pnl.fillna(1-a.price-a.fee) - b.pnl.fillna(-b.price-b.fee)
    same = a.market.eq(b.market) & a.side.eq(b.side)
    cost_diff = b.price+b.fee-a.price-a.fee
    return [round(100*x.where(~same,cost_diff).mean(),3) for x in (lower,upper)]


def main():
    c,hours = observations()
    train = hours[hours < SPLIT]
    if train.empty or not (hours >= SPLIT).any():
        raise ValueError("Both chronological samples are required")
    schedule = fit_schedule(c[c.end < SPLIT],train)
    outcomes = resolve_ladder(c.market.unique(),settlement_cache())
    paths = {name:score(first_entries(c,floor),hours,outcomes) for name,floor in {**FIXED,"По времени":schedule}.items()}
    known = pd.DataFrame({k:p.pnl.notna() for k,p in paths.items()}).all(axis=1)
    rows = []
    for name,p in paths.items():
        for period,mask in [("Ранний", hours < SPLIT),("Поздний", hours >= SPLIT)]:
            g = p.loc[known & mask]
            delta = g.pnl-paths["0¢"].loc[g.index,"pnl"]
            trades = g[g.taken]
            lo,hi = day_interval(delta)
            low_all,high_all = comparison(p.loc[mask],paths["0¢"].loc[mask])
            rows.append({"sample":period,"policy":name,"observed":int(mask.sum()),"hours":len(g),"entries":len(trades),
                         "net":g.pnl.sum(),"net_c_hour":100*g.pnl.mean(),"model_c_trade":100*trades.edge.mean(),
                         "model_c_hour":100*trades.edge.sum()/len(g) if len(g) else None,
                         "net_c_trade":100*trades.pnl.mean(),"delta_c_hour":100*delta.mean(),"lo":lo,"hi":hi,
                         "full_low":low_all,"full_high":high_all,
                         "losses":int((trades.pnl<0).sum()),"next_obs":int(trades.retained.count()),
                         "next_ok":int(trades.retained.sum()),"worst":trades.pnl.min()})
    entries = pd.concat({name:p.loc[p.taken,["ts","market","side","edge","price","fee","pnl","retained","margin"]] for name,p in paths.items()},
                        names=["policy","end"])
    RESULTS.mkdir(parents=True,exist_ok=True)
    atomic_write(RESULTS/"edge_timing_entries.csv",entries.to_csv())
    atomic_write(RESULTS/"edge_timing.csv",pd.DataFrame(rows).to_csv(index=False))
    from lab.wing_report import prepare
    paper,_ = prepare(read_ledger())
    paper = paper[paper.hyp.eq("W04") & paper.hour.between(START,STOP,inclusive="left")]
    baseline = paths["0¢"].loc[lambda x:x.taken]
    linked = paper.join(baseline[["ts","market","side"]].rename(columns={"ts":"ts_replay","side":"side_replay"}),on="hour")
    matches = (linked.instrument.eq(linked.market) & linked.side.eq(linked.side_replay)
               & linked.ts.dt.floor("s").eq(linked.ts_replay))
    meta = {"start":str(START),"split":str(SPLIT),"stop":str(STOP),"floors":schedule,
            "paper":len(paper),"baseline":len(baseline),"matches":int(matches.sum())}
    a = paths["0¢"].loc[known & (hours >= SPLIT) & paths["0¢"].taken]
    b = paths["1¢"].loc[a.index]
    reason = pd.Series("Сменился страйк или сторона",index=a.index)
    same = a.market.eq(b.market) & a.side.eq(b.side)
    reason.loc[same] = "Тот же контракт позже"
    reason.loc[same & a.ts.eq(b.ts)] = "Тот же вход"
    reason.loc[~b.taken] = "Пропуск"
    paired = pd.DataFrame({"Случай":reason, "W04, $":a.pnl,"W12, $":b.pnl,"Разница, $":b.pnl-a.pnl,"Часы":1})
    meta["attribution"] = paired.groupby("Случай",as_index=False).sum().round(4).to_dict("records")
    atomic_write(RESULTS/"edge_timing_meta.json",json.dumps(meta,indent=2))
    print("Training floors, dollars, minutes 15 to 30:",list(schedule.values()))
    print(pd.DataFrame(rows).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
