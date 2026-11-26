"""Causal one-minute continuation stress on observed W04 paper entries, one contract."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import stdtr, stdtrit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from lab.core import LAB, RESULTS, atomic_write
from lab.report import void_mask
from studies.wings_edge_timing import day_interval


def stressed_fair(side, fair, spot, previous, strike):
    """Recover the frozen t4 scale from the observed forecast; hold horizon fixed."""
    yes = fair if side=="yes" else 1-fair
    m = np.log(spot/strike)
    if abs(m)<1e-9 or not 0<yes<1:
        return np.nan
    scale = stdtrit(4,yes)/m
    if not np.isfinite(scale) or scale<=0:
        return np.nan
    move = np.log(spot/previous)
    adverse = min(move,0) if side=="yes" else max(move,0)
    p = stdtr(4,scale*(m+adverse))
    return p if side=="yes" else 1-p


def main():
    ledger = pd.read_csv(LAB/"ledger.csv",low_memory=False)
    w = ledger[ledger.hyp.eq("W04") & ledger.event.eq("CLOSE")].copy()
    w = w[~void_mask(w)]
    w["ts"] = pd.to_datetime(w.ts,utc=True,format="mixed")
    rows = []
    for day, trades in w.groupby(w.ts.dt.strftime("%Y-%m-%d")):
        path = LAB/"ticks"/(day+".csv")
        if not path.exists():
            continue
        tick = pd.read_csv(path,usecols=["ts","usd_mid"]).dropna()
        tick.index = pd.to_datetime(tick.ts,utc=True,format="mixed")
        spot = tick.usd_mid.sort_index()
        for r in trades.itertuples():
            # Ledger has microseconds, logger timestamp has second resolution.
            now = r.ts.floor("s")
            targets = [now,now-pd.Timedelta(minutes=1)]
            idx = spot.index.get_indexer(targets,method="pad",tolerance=pd.Timedelta(seconds=3))
            if min(idx)<0:
                continue
            current,previous = spot.iloc[idx]
            strike = float(r.instrument.split("-T")[-1])
            fair = stressed_fair(r.side,r.fair,current,previous,strike)
            if np.isfinite(fair):
                rows.append(dict(ts=r.ts,market=r.instrument,spot=current,previous=previous,
                                 edge=r.edge,stress_edge=fair-r.price-r.fee,pnl=r.pnl))
    d = pd.DataFrame(rows).set_index("ts").sort_index()
    atomic_write(RESULTS/"w04_stress_entries.csv",d.to_csv())
    rows = []
    for label,g in [("До 9 ноября",d[d.index<pd.Timestamp("2026-11-09",tz="UTC")]),
                    ("9–20 ноября",d[(d.index>=pd.Timestamp("2026-11-09",tz="UTC"))&(d.index<pd.Timestamp("2026-11-21",tz="UTC"))]),
                    ("С 21 ноября",d[d.index>=pd.Timestamp("2026-11-21",tz="UTC")])]:
        for name,keep in [("Edge ≥ 1¢",g.edge.ge(.01)),("Стресс ≥ 0¢",g.stress_edge.ge(0)),
                          ("Стресс ≥ 1¢",g.stress_edge.ge(.01))]:
            delta = g.pnl.where(keep,0)-g.pnl
            low,high = day_interval(delta)
            rows.append({"Период UTC":label,"Фильтр":name,"Сопоставлено":len(g),"Пропуски":int((~keep).sum()),
                         "Убыточных пропущено":int(((~keep)&g.pnl.lt(0)).sum()),
                         "Без фильтра, $":round(g.pnl.sum(),2),"С фильтром, $":round(g.pnl[keep].sum(),2),
                         "95% интервал Δ, ¢/вход":f"{low:.2f} … {high:.2f}" if low is not None else "—"})
    result = pd.DataFrame(rows)
    atomic_write(RESULTS/"w04_stress.csv",result.to_csv(index=False))
    print(f"{len(d)}/{len(w)} matched")
    print(result.to_string(index=False))
    print(d.tail(3).to_string())


if __name__=="__main__":
    main()
