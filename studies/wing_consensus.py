"""Compare conservative wing forecasts on an identical, settled opportunity set."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lab.core import RESULTS, atomic_write, utcnow
from lab.history import Minutes
from lab.wing_regime import regimes
from studies.wing_capacity_ml import deepen, replay
from studies.wings_calibration import books
from studies.wings_edge_timing import SPLIT, day_interval


def forecasts(quotes):
    q = quotes.sort_values(["market", "ts"]).copy()
    neighbor = q.curve.where(q.cheap_yes, 1 - q.curve)
    q["mix"] = (q.t4 + neighbor) / 2
    q["conservative"] = q.mix + (q.t4 - neighbor).abs() / 4
    group = q.groupby(["market", "cheap_yes"])
    previous = group.mix.shift()
    elapsed = (q.ts - group.ts.shift()).dt.total_seconds()
    q["persistent"] = np.maximum(q.mix, previous.where(elapsed.between(0, 90, inclusive="right")))
    return q


def robustness(trades, label, name, hours):
    rows = []
    for scenario, prefix in [("Повторный снимок", "delayed"), ("На 1¢ дороже", "stress")]:
        filled = trades[trades[prefix+"_n"].gt(0)]
        qty = int(filled[prefix+"_n"].sum())
        pnl = filled[prefix+"_pnl"].astype(float)
        net = pnl.sum()
        best = pnl.idxmax() if len(pnl) and pnl.max() > 0 else None
        rows.append({"Период":label, "Модель":name, "Цена покупки":scenario, "Часы":len(hours),
                     "Исполненные входы":len(filled), "Контракты":qty,
                     "Средний edge, ¢":round(100*filled[prefix+"_edge"].sum()/qty, 3) if qty else None,
                     "Нетто, $":round(net, 2),
                     "Лучшая прибыльная экспирация UTC":str(filled.at[best, "end"]) if best is not None else None,
                     "Нетто без лучшей прибыли, $":round(net-pnl[best], 2) if best is not None else round(net, 2)})
    return rows


def main():
    q, _ = books(utcnow().floor("h") - pd.Timedelta(hours=1), deepen)
    q = forecasts(q)
    file = RESULTS / "wing_forecast_settlements.json"
    if file.exists():
        official = json.loads(file.read_text(encoding="utf-8"))
        value = q.event.map({event: rec["value"] for event, rec in official.items()})
        yes = value.gt(q.strike).astype(float)
        q["y"] = q.y.fillna(yes.where(q.cheap_yes, 1 - yes).where(value.notna()))
    gate = regimes(Minutes().close).blocked
    q = q[q.ts.dt.floor("min").map(gate).eq(False)].dropna(subset=["persistent"])
    cutoff = pd.Timestamp("2026-11-22T17:00Z")
    rows = []
    sensitivity = []
    for label, mask in [("До прошлого среза", q.end.between(SPLIT, cutoff)),
                        ("После прошлого среза", q.end.gt(cutoff))]:
        sample = q[mask]
        if sample.empty:
            continue
        models = ["t4", "mix", "conservative", "persistent"]
        attempts = {name: pd.DataFrame(replay(sample, name, label, stress=True)) for name in models}
        unknown = {hour for trades in attempts.values() if len(trades)
                   for hour in trades.loc[trades.pnl.isna(), "end"]}
        hours = pd.DatetimeIndex(sorted(set(sample.end) - unknown))
        if hours.empty:
            continue
        for name in models:
            trades = attempts[name]
            if trades.empty:
                trades = pd.DataFrame(columns=["end","n","pnl","delayed_n","delayed_pnl","observed",
                                               "stress_n","stress_pnl","stress_edge","delayed_edge"])
            g = trades[trades.end.isin(hours)]
            sensitivity.extend(robustness(g, label, name, hours))
            pnl = g.set_index("end").delayed_pnl.astype(float).reindex(hours, fill_value=0.)
            if name == "t4":
                base = pnl
            low, high = day_interval(pnl - base)
            rows.append({"Период":label, "Первая UTC":str(hours.min()), "Последняя UTC":str(hours.max()),
                         "Модель":name, "Часы":len(hours), "Без исхода":len(unknown), "Входы":len(g),
                         "Контракты":int(g.n.sum()), "Нетто, $":round(g.pnl.sum(),2),
                         "С повторным снимком":int(g.observed.sum()),
                         "Входы с объёмом позже":int(g.delayed_n.gt(0).sum()),
                         "Контракты позже":int(g.delayed_n.sum()), "Нетто позже, $":round(pnl.sum(),2),
                         "95% Δ к t₄, $/час": "—" if low is None else f"{low/100:.2f} … {high/100:.2f}"})
    frame = pd.DataFrame(rows)
    atomic_write(RESULTS / "wing_consensus.csv", frame.to_csv(index=False))
    atomic_write(RESULTS / "wing_robustness.csv", pd.DataFrame(sensitivity).to_csv(index=False))
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
