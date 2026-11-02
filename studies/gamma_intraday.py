"""Intraday gamma on Deribit dailies: buy or sell the ATM straddle of the next 08:00 UTC expiry and delta-hedge it
with the BTC perpetual until expiry. Which entry hour, if any, makes money after spreads and fees?

Why it could work. A delta-hedged option earns (Black–Scholes P&L attribution, hedging at the implied vol σ):
        dP&L ≈ ½ Γ S² (r² − σ² dt)          summed over the hedging intervals
so a long straddle hedged to expiry earns ≈ ½ Γ S² · (realised variance − implied variance) over its life. The
options market prices one implied vol for the whole day, but BTC's realised variance has a strong intraday shape
(the US open 13:30–16:00 UTC is 2–3× the Asian night). An entry at 09:00 UTC covers the US session; one at 16:00
covers the quiet night; one at 00:00 covers Asia only. If the implied vol does not move with that shape, gamma is
cheap in the morning and dear in the evening.

Simulation (no look-ahead): the option panel from studies/o04_straddle.py (ATM IV from Deribit marks before t,
strike K nearest 500 to the forward, straddle mid by Black-76, ask/bid = mid × (1 ± h) with h from the live Deribit
chain), Binance 1-minute closes × the Deribit-index basis at entry for the hedge path, settlement = Deribit
delivery price (or the 30-min TWAP before 08:00). Delta of the straddle = 2N(d1) − 1 at the entry IV; the hedge is
re-set every `every` minutes (or when it drifts by more than a band) until 07:30 UTC, then held to settlement.
Costs: option spread and 0.03%/leg fee at entry; perp 5 bp per side on each re-hedge (taker; 2 bp = maker variant).
    python studies/gamma_intraday.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "studies"))
import config  # noqa: E402
import o04_straddle as O  # noqa: E402

OUT = config.RESULTS / "studies"


def option_leg(row, side: int, spread: float = 1.0) -> float:
    """P&L of the straddle itself, bought at ask / sold at bid, held to settlement, Deribit fees."""
    h = O.half_spread(row.tau_h) * spread
    px = row.mid * (1 + side * h)
    fee = sum(min(O.FEE_RATE * row.F, O.FEE_CAP * x * (1 + side * h)) for x in (row.call, row.put))
    return side * (row.payoff - px) - fee


def hedge_leg(row, path: pd.Series, side: int, every: int = 1, band: float | None = None):
    """Delta hedge with the perp: (gross P&L, turnover in BTC·$). Re-hedge every `every` minutes, or — with
    `band` — only when the straddle delta drifted more than `band` BTC from the hedge (checked every minute).
    Hedging stops at 07:30 UTC (the settlement average starts); the last hedge is closed at the settlement price."""
    stop = row.expiry - pd.Timedelta(minutes=30)
    seg = path[(path.index >= row.entry) & (path.index <= stop)]
    if len(seg) < 2:
        return 0.0, 0.0
    if band is None:
        seg = seg.iloc[::every]
    s = seg.values
    tau = (row.expiry - seg.index).total_seconds().values / (365 * 86400)
    d1 = (np.log(s / row.K) + 0.5 * row.iv ** 2 * tau) / (row.iv * np.sqrt(tau))
    target = -side * (2 * norm.cdf(d1) - 1)              # long straddle → short its delta
    if band is not None:
        pos = np.empty_like(target)
        cur = target[0]
        for i, x in enumerate(target):
            if abs(x - cur) > band:
                cur = x
            pos[i] = cur
    else:
        pos = target
    gross = float(np.sum(pos[:-1] * np.diff(s)) + pos[-1] * (row.S_T - s[-1]))
    turnover = float(np.sum(np.abs(np.diff(np.concatenate([[0.0], pos, [0.0]]))) * np.concatenate([s, [row.S_T]])))
    return gross, turnover


HEDGES = {"nohedge": None, "every 5 min": dict(every=5), "every 15 min": dict(every=15), "every 60 min": dict(every=60),
          "band 0.10": dict(band=0.10), "band 0.25": dict(band=0.25), "every 1 min (theory)": dict(every=1)}


def main():
    b = pd.read_csv(OUT / "o04_panel.csv", parse_dates=["entry", "expiry"])
    close = O.load_minutes(sys.argv[1] if len(sys.argv) > 1 else None)
    close.index = close.index + pd.Timedelta(minutes=1)                  # close time
    b["hour"] = b["entry"].dt.hour
    rows = []
    for _, r in b.iterrows():
        basis = r.F / close.asof(r.entry)
        path = close[(close.index >= r.entry) & (close.index <= r.expiry)] * basis
        if len(path) < 0.9 * r.tau_h * 60:
            continue                                                     # gaps in the minute data
        rec = {"entry": r.entry, "expiry": r.expiry, "hour": r.hour, "iv": r.iv, "tau_h": r.tau_h, "fc_vol": r.fc_vol,
               "rv_ann": math.sqrt(r.rv / (r.tau_h / 8760)), "mid": r.mid, "payoff": r.payoff, "F": r.F}
        for side, sname in ((1, "long"), (-1, "short")):
            rec[f"{sname}|option"] = option_leg(r, side)
            rec[f"{sname}|option_tight"] = option_leg(r, side, spread=0.3)
            for tag, kw in HEDGES.items():
                if kw is None:
                    continue
                g, to = hedge_leg(r, path, side, **kw)
                rec[f"{sname}|{tag}|gross"] = g
                rec[f"{sname}|{tag}|turnover"] = to
        rows.append(rec)
    g = pd.DataFrame(rows)
    g.to_csv(OUT / "gamma_intraday_panel.csv", index=False)
    return g


def pnl(g, side, hedge, perp_bp=5.0, tight=False):
    opt = g[f"{side}|option_tight" if tight else f"{side}|option"]
    if hedge == "nohedge":
        return opt
    return opt + g[f"{side}|{hedge}|gross"] - g[f"{side}|{hedge}|turnover"] * perp_bp / 1e4


def summarise(g: pd.DataFrame, by_hour: bool = True) -> pd.DataFrame:
    out = []
    groups = [("all", g)] + ([(h, x) for h, x in g.groupby("hour")] if by_hour else [])
    for hour, x in groups:
        for side in ("long", "short"):
            for hedge in HEDGES:
                for bp, tight, cost in ((5.0, False, "taker 5bp, quoted spread"), (2.0, False, "maker 2bp"),
                                        (0.0, True, "no perp cost, print spread")):
                    if hedge == "nohedge" and bp != 5.0 and not tight:
                        continue
                    p = pnl(x, side, hedge, bp, tight)
                    out.append({"entry UTC": hour, "side": side, "hedge": hedge, "costs": cost, "n": len(p),
                                "$/trade": p.mean(), "% of mid": 100 * (p / x["mid"]).mean(), "t HAC": O.hac_t(p.values),
                                "hit %": 100 * (p > 0).mean(), "total $": p.sum(), "worst $": p.min(),
                                "perp cost $": (x[f"{side}|{hedge}|turnover"] * bp / 1e4).mean() if hedge != "nohedge" else 0.0})
    return pd.DataFrame(out)


if __name__ == "__main__":
    g = main()
    s = summarise(g)
    s.to_csv(OUT / "gamma_intraday_summary.csv", index=False)
    print(s.round(2).to_string(index=False))
