"""One-contract replay of the volatility pause and audit of actual W04/W06 paper entries."""
from pathlib import Path
import sys
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from lab.core import LAB, RESULTS, atomic_write, utcnow
from lab.history import Minutes
from lab.report import void_mask
from lab.wing_regime import regimes
from studies.wings_old_replay import archive
from studies.wings_calibration import books
from studies.wings_two_sided import offers,picks
from studies.wings_edge_timing import SPLIT, STOP, day_interval


def main():
    state = regimes(Minutes().close)
    ledger = pd.read_csv(LAB/"ledger.csv",low_memory=False)
    w = ledger[ledger.event.eq("CLOSE") & ledger.hyp.isin(["W04","W06"])].copy()
    w = w[~void_mask(w)]
    # Makers are disabled for routing. Use taker entry times as decision times.
    w = w[~w["info"].str.contains(r'"maker"\s*:\s*true',regex=True,na=False)].copy()
    t = pd.to_datetime(w.ts,utc=True,format="mixed").dt.floor("min")
    audit = w[["ts","hyp","instrument","edge","pnl"]].join(state,on=t)
    atomic_write(RESULTS/"wing_regime_entries.csv",audit.to_csv(index=False))
    rows = []
    for source in ("archive","books"):
        if source=="archive":
            q,hours = archive()
            splits = [("2026-07-01","2026-09-01"),("2026-09-01","2026-10-28")]
        else:
            until = utcnow().floor("h")-pd.Timedelta(hours=1)
            q,hours = books(until)
            splits = [("2026-10-30",SPLIT),(SPLIT,STOP),(STOP,until+pd.Timedelta(seconds=1))]
        q = q.join(state[["blocked"]],on=q.ts.dt.floor("min"))
        # Capture only the measured regime effect on observed minutes.
        hours = hours.difference(pd.DatetimeIndex(q.loc[q.blocked.isna(),"end"].unique()))
        for start,end in splits:
            idx = hours[(hours>=pd.to_datetime(start,utc=True))&(hours<pd.to_datetime(end,utc=True))]
            if not len(idx):
                continue
            candidates = offers(q[q.end.isin(idx)],False)
            eligible = candidates[candidates.blocked.eq(False)]
            pnls, entries = [], []
            for frame in [candidates,eligible]:
                selected = picks(frame)
                entries.append(selected.index)
                pnls.append((selected.pay-selected.price-selected.fee).reindex(idx,fill_value=0))
            paired = pd.concat(pnls,axis=1,keys=["base","guard"]).dropna()
            lo,hi = day_interval(paired.guard-paired.base)
            rows.append({"Период":f"{idx.min():%d.%m} – {idx.max():%d.%m}","Часы":len(paired),
                         "Входы до / после":f"{len(paired.index.intersection(entries[0]))} / {len(paired.index.intersection(entries[1]))}",
                         "Убытки до / после":f"{sum(paired.base<0)} / {sum(paired.guard<0)}",
                         "База, $":round(paired.base.sum(),2),"С паузой, $":round(paired.guard.sum(),2),
                         "95% Δ, ¢/час":f"{lo:.2f} … {hi:.2f}" if lo is not None else "—"})
    result = pd.DataFrame(rows)
    atomic_write(RESULTS/"wing_regime_replay.csv",result.to_csv(index=False))
    print(result.to_string(index=False))
    print(audit.tail(7).to_string(index=False))


if __name__=="__main__":
    main()
