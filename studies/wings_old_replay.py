"""Replay the two-sided rules on archived minute-close quotes and test a fixed momentum gate."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import ndtr, stdtr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from lab.core import atomic_write, RESULTS
from studies.wings_two_sided import compare
from studies.wings_calibration import books
from studies.wings_edge_timing import START,SPLIT,STOP


def archive(enrich=None, *, files=None, minute_close=None):
    parts, settlements = [], []
    for f in sorted(files if files is not None else (ROOT/"data/raw/kalshi_kxbtcd_60m").glob("*.csv*")):
        g = pd.read_csv(f,usecols=["ticker","yes_bid","yes_ask","end_ts","close_time","strike",
                                   "strike_type","expiration_value","result"])
        g = g[g.strike_type.eq("greater")].rename(columns={"ticker":"market","yes_bid":"bid","yes_ask":"ask"})
        g["end"] = pd.to_datetime(g.close_time,utc=True)
        settlements.append(g[["end","expiration_value"]].drop_duplicates())
        g["ts"] = pd.to_datetime(g.end_ts,unit="s",utc=True)
        g["left"] = (g.end-g.ts).dt.total_seconds()/60
        if enrich is not None:
            g = enrich(g[g.left.between(13,30)].drop_duplicates(["market","ts"]))
        parts.append(g[g.left.between(15,30) & g.bid.gt(0) & g.ask.lt(1) & (g.ask-g.bid).round(6).between(0,.03)])
    q = pd.concat(parts,ignore_index=True).drop_duplicates(["market","ts"])
    if minute_close is None:
        m = pd.read_parquet(ROOT/"data/interim/minutes.parquet",columns=["open_time","close"])
        m = m.set_index(pd.to_datetime(m.open_time,utc=True)+pd.Timedelta(minutes=1)).close.sort_index()
    else:
        m = minute_close.shift(freq="min").sort_index()
    m = m[~m.index.duplicated()]
    m = m.loc[q.ts.min()-pd.Timedelta(days=95):q.end.max()]
    r2 = np.log(m).diff().pow(2)
    # Hour-of-week profile from prior complete days only, rebuilt once per day.
    hour = m.index.day_of_week*24+m.index.hour
    daily = r2.groupby([m.index.floor("D"),hour]).mean().unstack()
    seasonal = daily.rolling("90D",closed="left",min_periods=4).mean()
    seasonal = seasonal.div(seasonal.mean(axis=1),axis=0).fillna(1.)
    s = pd.Series(seasonal.to_numpy()[seasonal.index.get_indexer(m.index.floor("D")),hour],index=m.index)
    z = r2/s
    lv = .5*z.ewm(halflife=30,min_periods=600).mean()+.3*z.rolling("24h").mean()+.2*z.rolling("8D").mean()
    ev = pd.concat(settlements).groupby("end").expiration_value.agg(["min","max"])
    basis = (ev["min"].where(ev["min"].eq(ev["max"]))/m.reindex(ev.index)).shift(1).rolling(48,min_periods=5).median()
    prev = q.ts-pd.Timedelta(minutes=1)
    spot = m.reindex(prev).to_numpy()
    q["usd_mid"] = spot*q.end.map(basis)
    var = (lv*s).reindex(prev).to_numpy()*(q.left-2/3)
    x = np.log(spot*q.end.map(basis)/q.strike)/np.sqrt(var)
    q["move5"] = np.log(spot/m.reindex(prev-pd.Timedelta(minutes=5)).to_numpy())
    q["normal"],q["t4"] = ndtr(x),stdtr(4,x*np.sqrt(2))
    q = q.dropna(subset=["normal","t4","move5"]).copy()
    hours = pd.DatetimeIndex(sorted(q.end.unique()),name="end")
    q["cheap_yes"] = q.bid.lt(.5)
    q = q[q.bid.where(q.cheap_yes,1-q.ask).round(6).between(.06,.10)].copy()
    for model in ["normal","t4"]:
        q[model] = q[model].where(q.cheap_yes,1-q[model])
    q["y"] = q.result.eq("yes").eq(q.cheap_yes).astype(float).where(q.result.isin(["yes","no"]))
    # Candles contain no displayed size; this replay assumes one contract at the close.
    q["bid_size"] = q["ask_size"] = 1
    return q,hours


def recent():
    minutes = pd.read_parquet(ROOT/"data/interim/minutes.parquet",columns=["open_time","close"])
    close = minutes.set_index(pd.to_datetime(minutes.open_time,utc=True)+pd.Timedelta(minutes=1)).close
    close = close[~close.index.duplicated()].sort_index()
    until = min(close.index.max().floor("h"),pd.Timestamp("2026-11-21T10:00Z"))
    quotes,hours = books(until)
    prev = quotes.ts.dt.floor("min")-pd.Timedelta(minutes=1)
    quotes["move5"] = np.log(close.reindex(prev).to_numpy()/close.reindex(prev-pd.Timedelta(minutes=5)).to_numpy())
    # Missing spot data must not make a skipped trade look like an advantage.
    hours = hours.difference(pd.DatetimeIndex(quotes.loc[quotes.move5.isna(),"end"].unique()))
    rows = []
    for start,end in [(START,SPLIT),(SPLIT,STOP),(STOP,until+pd.Timedelta(seconds=1))]:
        idx = hours[(hours>=start)&(hours<end)]
        if len(idx):
            rows.extend(compare(quotes[quotes.end.isin(idx)],idx,f"{idx.min():%d.%m} – {idx.max():%d.%m}",momentum=True))
    return rows


if __name__ == "__main__":
    q,hours = archive()
    rows = []
    for start,end in [("2026-07-01","2026-09-01"),("2026-09-01","2026-10-28")]:
        idx = hours[(hours>=pd.Timestamp(start,tz="UTC"))&(hours<pd.Timestamp(end,tz="UTC"))]
        rows.extend(compare(q[q.end.isin(idx)],idx,f"{idx.min():%d.%m} – {idx.max():%d.%m}",momentum=True))
    frame = pd.DataFrame(rows+recent())
    atomic_write(RESULTS/"wing_archive_replay.csv",frame.to_csv(index=False))
    print(frame.to_string(index=False))
