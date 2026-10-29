"""Kalshi 15-minute vs Polymarket 15-minute "BTC up?" on the same quarter-hour, 14.09 – 28.10.2026.

Kalshi: 1-minute book candles (best YES bid / ask), strike = BRTI at the start, settles on BRTI.
Polymarket: 1-minute price history of the Up token (no book in history: we assume a 1¢ spread, the typical
quote of these markets in the live logs), price to beat = Chainlink BTC/USD at the start.
Same question, different reference prices: the two outcomes can differ when BTC ends near the strike.

Cross-venue trade (A12 in the lab): buy the cheaper YES-equivalent on one venue and the opposite side on the
other when the total cost after both taker fees is below 1 − margin. Payoff 1 if outcomes agree, 0 or 2 if not.
Every minute with ≥ 2 minutes left is a decision point; one trade per window (first signal).
    python studies/kx_pm_15m_arb.py <raw dir>
"""
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/raw")
PM_HALF_SPREAD = 0.005


def kx_fee(p):
    return np.ceil(np.round(0.07 * p * (1 - p) * 100, 6)) / 100


def pm_fee(p):
    return 0.07 * p * (1 - p)


def main():
    mk = pd.read_csv(RAW / "kx15" / "markets.csv")
    mk["close_time"] = pd.to_datetime(mk["close_time"], utc=True)
    c = pd.concat([pd.read_csv(f) for f in sorted((RAW / "kx15" / "candles").glob("2026-09-1*.csv")) +
                   sorted((RAW / "kx15" / "candles").glob("2026-09-[23]*.csv")) +
                   sorted((RAW / "kx15" / "candles").glob("2026-10-*.csv")) if f.stat().st_size > 10])
    c = c.merge(mk[["ticker", "close_time", "floor_strike", "result", "expiration_value"]], on="ticker")
    c["ts"] = pd.to_datetime(c["end_ts"], unit="s", utc=True)
    c = c[(c.bid > 0) & (c.ask < 1)]
    pm = pd.concat([pd.read_csv(f) for f in sorted((RAW / "pm_updown" / "15m").glob("*.csv"))])
    pm["start"] = pd.to_datetime(pm["start"], utc=True)
    pm["end"] = pm["start"] + pd.Timedelta(minutes=15)
    # Polymarket history points carry seconds :20–:24; Kalshi candles close at :00. Nearest minute = 20–40 s apart,
    # which is NOT simultaneous (AGENTS.md rule 7): BTC moves in 30 s; the result is an upper bound, the live A12 /
    # A12F (both books read in the same second) are the real test.
    pm["ts"] = pd.to_datetime(pm["t"], unit="s", utc=True).dt.round("min")
    pm = pm.dropna(subset=["p"]).drop_duplicates(["slug", "ts"], keep="last")
    w = pm.groupby("end").agg(outcome=("outcome", "first"), ptb=("price_to_beat", "first")).reset_index()
    k = mk[["close_time", "floor_strike", "result", "expiration_value"]].rename(columns={"close_time": "end"})
    j = w.merge(k, on="end")
    j["kx_yes"] = (j.result == "yes").astype(float)
    agree = (j.kx_yes == j.outcome).mean()
    print(f"windows on both venues: {len(j)}; outcomes agree {agree:.1%}; strike gap Kalshi − Polymarket: median "
          f"{(j.floor_strike - j.ptb).median():.1f} $, IQR {(j.floor_strike - j.ptb).quantile(.25):.1f} … "
          f"{(j.floor_strike - j.ptb).quantile(.75):.1f} $")
    d = c.rename(columns={"close_time": "end"}).merge(pm[["end", "ts", "p", "outcome"]], on=["end", "ts"])
    d["left"] = (d.end - d.ts).dt.total_seconds() / 60
    d = d[d.left >= 2]
    d["kx_yes"] = (d.result == "yes").astype(float)
    pa, pb = (d.p + PM_HALF_SPREAD).clip(upper=0.995), (d.p - PM_HALF_SPREAD).clip(lower=0.005)
    # A: Kalshi YES (ask) + Polymarket Down (1 − Up bid);  B: Polymarket Up (ask) + Kalshi NO (1 − bid)
    costA = d.ask + kx_fee(d.ask) + (1 - pb) + pm_fee(1 - pb)
    costB = pa + pm_fee(pa) + (1 - d.bid) + kx_fee(1 - d.bid)
    d["costA"], d["costB"] = costA, costB
    d["gap_mid"] = (d.bid + d.ask) / 2 - d.p
    print(f"\nminutes compared: {len(d)} in {d.end.nunique()} windows; mid gap Kalshi − Polymarket: mean "
          f"{100 * d.gap_mid.mean():.2f}¢, sd {100 * d.gap_mid.std():.2f}¢, |gap| > 5¢ in {(d.gap_mid.abs() > .05).mean():.1%}")
    split = d.end.quantile(0.5)
    rows = []
    for margin in (0.0, 0.01, 0.02, 0.04):
        s = d[(d.costA < 1 - margin) | (d.costB < 1 - margin)].sort_values("ts").groupby("end").head(1).copy()
        a = s.costA <= s.costB
        pay = np.where(a, s.kx_yes + (1 - s.outcome), s.outcome + (1 - s.kx_yes))
        s["pnl"] = pay - np.where(a, s.costA, s.costB)
        for half, g in (("first", s[s.end < split]), ("second", s[s.end >= split]), ("all", s)):
            x = g.pnl.values
            rows.append({"margin ¢": 100 * margin, "half": half, "trades": len(x),
                         "mean ¢": 100 * x.mean() if len(x) else np.nan,
                         "t": x.mean() / (x.std(ddof=1) / math.sqrt(len(x))) if len(x) > 2 and x.std() > 0 else np.nan,
                         "disagreements": int((np.abs(pay[g.index.map(s.index.get_loc)] - 1) > 0.5).sum()) if len(x) else 0,
                         "worst ¢": 100 * x.min() if len(x) else np.nan})
    t = pd.DataFrame(rows)
    print("\n", t.round(2).to_string(index=False))
    out = RAW.parent.parent / "results" / "studies"
    out.mkdir(parents=True, exist_ok=True)
    t.to_csv(out / "kx_pm_15m_arb.csv", index=False)


if __name__ == "__main__":
    main()
