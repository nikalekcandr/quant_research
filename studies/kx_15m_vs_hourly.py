"""Two Kalshi products, one settlement: the 15-minute market that closes on the hour (KXBTC15M, strike = BRTI at :45)
and the hourly ladder (KXBTCD, "above K" for ~190 strikes) both settle on the same 60-second BRTI average.
So for ladder strikes K_lo ≤ K15 ≤ K_hi:   P(>K_hi) ≤ P15 ≤ P(>K_lo).
  Arb A: 15m YES ask + fee < ladder YES bid at K_hi − fee  → buy 15m YES, sell ladder K_hi YES  (payoff ≥ 0)
  Arb B: 15m YES bid − fee > ladder YES ask at K_lo + fee  → sell 15m YES, buy ladder K_lo YES  (payoff ≥ 0)
Both books from the same venue's 1-minute candles (closing book of the same minute).
    python studies/kx_15m_vs_hourly.py <raw dir>
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_A = [a for a in sys.argv[1:] if not a.startswith("--")]
RAW = Path(_A[0]) if len(_A) > 0 else Path("data/raw")
LADDER = Path(_A[1]) if len(_A) > 1 else RAW / "kalshi_kxbtcd_60m"


def fee(p):
    return np.ceil(np.round(0.07 * p * (1 - p) * 100, 6)) / 100


def main():
    mk = pd.read_csv(RAW / "kx15" / "markets.csv")
    mk["close_time"] = pd.to_datetime(mk["close_time"], utc=True)
    mk = mk[mk.close_time.dt.minute == 0]
    lad = pd.concat([pd.read_csv(f) for f in sorted(LADDER.glob("*.csv*"))])
    lad["close_time"] = pd.to_datetime(lad["close_time"], utc=True)
    ev = lad.groupby("close_time")["expiration_value"].first()
    j = mk.set_index("close_time").join(ev.rename("ev_ladder"), how="inner")
    print(f"15m markets closing on the hour with a ladder: {len(j)}; settlement value identical in "
          f"{(np.isclose(j.expiration_value, j.ev_ladder, atol=0.011)).mean():.1%} "
          f"(median |diff| {np.nanmedian(np.abs(j.expiration_value - j.ev_ladder)):.4f}, {j.expiration_value.isna().sum()} without a value)")
    same = j[np.isclose(j.expiration_value, j.ev_ladder, atol=0.011) | j.expiration_value.isna()]
    c = pd.concat([pd.read_csv(f) for f in sorted((RAW / "kx15" / "candles").glob("*.csv")) if f.stat().st_size > 10])
    c = c[c.ticker.isin(set(same.ticker))]
    c = c.merge(mk[["ticker", "close_time", "floor_strike", "result"]], on="ticker")
    lad = lad[lad.close_time.isin(set(same.index)) & (lad.strike_type == "greater")]
    lad = lad[(lad.yes_bid > 0) | (lad.yes_ask < 1)]
    L = lad.set_index(["close_time", "end_ts"]).sort_index()
    rows = []
    for (ct, ts), g in c.groupby(["close_time", "end_ts"]):
        try:
            l = L.loc[(ct, ts)]
        except KeyError:
            continue
        if isinstance(l, pd.Series):
            l = l.to_frame().T
        l = l.reset_index(drop=True)
        r = g.iloc[0]
        K = r.floor_strike
        a15, b15 = r.ask, r.bid
        if not (0 < b15 < a15 < 1):
            continue
        hi = l[l.strike >= K]
        lo = l[l.strike <= K]
        best_a = best_b = None
        if len(hi):
            bh = hi[hi.yes_bid > 0]
            if len(bh):
                x = bh.loc[bh.yes_bid.idxmax()]
                best_a = (x.yes_bid - fee(x.yes_bid)) - (a15 + fee(a15)), x.strike, x.yes_bid
        if len(lo):
            al = lo[(lo.yes_ask < 1) & (lo.yes_ask > 0)]
            if len(al):
                x = al.loc[al.yes_ask.idxmin()]
                best_b = (b15 - fee(b15)) - (x.yes_ask + fee(x.yes_ask)), x.strike, x.yes_ask
        rows.append({"close_time": ct, "ts": ts, "left": (ct - pd.Timestamp(ts, unit="s", tz="UTC")).total_seconds() / 60,
                     "K15": K, "bid15": b15, "ask15": a15,
                     "arbA": best_a[0] if best_a else np.nan, "K_hi": best_a[1] if best_a else np.nan,
                     "arbB": best_b[0] if best_b else np.nan, "K_lo": best_b[1] if best_b else np.nan,
                     "gap_hi": best_a[1] - K if best_a else np.nan, "gap_lo": K - best_b[1] if best_b else np.nan})
    d = pd.DataFrame(rows)
    d["best"] = d[["arbA", "arbB"]].max(axis=1)
    print(f"minutes compared: {len(d)}, markets {d.close_time.nunique()}")
    for th in (0, 0.01, 0.02, 0.05):
        hit = d[d.best > th]
        print(f"  arbitrage > {th * 100:.0f}¢ after fees: {len(hit)} minutes in {hit.close_time.nunique()} markets; "
              f"mean {100 * hit.best.mean() if len(hit) else 0:.2f}¢")
    print("\nnear-misses: distribution of the best gap (¢):", (100 * d.best).describe().round(2).to_dict())
    print("\nmost negative gap = how far from arbitrage, by minutes left:")
    print(d.groupby(d.left.round()).best.agg(lambda x: 100 * x.median()).round(2).to_dict())
    d.to_csv("kx_15m_vs_hourly.csv", index=False)


if __name__ == "__main__" and "--rv" not in sys.argv:
    main()


def relative_value(raw=RAW, ladder_dir=LADDER):
    """Not a pure arbitrage (ladder strikes are $100–200 apart): interpolate the ladder's mid at the 15m strike and
    compare with the 15m mid. Which one is closer to the outcome, and is the gap tradeable on the 15m book?"""
    import math
    mk = pd.read_csv(raw / "kx15" / "markets.csv")
    mk["close_time"] = pd.to_datetime(mk["close_time"], utc=True)
    mk = mk[mk.close_time.dt.minute == 0]
    lad = pd.concat([pd.read_csv(f) for f in sorted(ladder_dir.glob("*.csv*"))])
    lad["close_time"] = pd.to_datetime(lad["close_time"], utc=True)
    lad = lad[(lad.strike_type == "greater") & (lad.yes_bid > 0) & (lad.yes_ask < 1) & (lad.yes_ask - lad.yes_bid <= 0.05)]
    lad["mid"] = (lad.yes_bid + lad.yes_ask) / 2
    c = pd.concat([pd.read_csv(f) for f in sorted((raw / "kx15" / "candles").glob("*.csv")) if f.stat().st_size > 10])
    c = c.merge(mk[["ticker", "close_time", "floor_strike", "result"]], on="ticker")
    c = c[(c.bid > 0) & (c.ask < 1) & (c.ask - c.bid <= 0.05)]
    G = {k: g for k, g in lad.groupby(["close_time", "end_ts"])}
    rows = []
    for r in c.itertuples():
        g = G.get((r.close_time, r.end_ts))
        if g is None or len(g) < 3:
            continue
        g = g.sort_values("strike")
        K = r.floor_strike
        if not (g.strike.min() < K < g.strike.max()):
            continue
        i = np.searchsorted(g.strike.values, K)
        k0, k1 = g.strike.values[i - 1], g.strike.values[i]
        if k1 - k0 > 300:
            continue
        m0, m1 = g.mid.values[i - 1], g.mid.values[i]
        p_l = m0 + (m1 - m0) * (K - k0) / (k1 - k0)
        rows.append({"ticker": r.ticker, "ts": r.end_ts, "left": round((r.close_time.timestamp() - r.end_ts) / 60),
                     "y": float(r.result == "yes"), "mid15": (r.bid + r.ask) / 2, "bid": r.bid, "ask": r.ask,
                     "p_ladder": p_l, "dk": k1 - k0})
    d = pd.DataFrame(rows)
    d = d[d.left.between(1, 14)]
    per = d.groupby("ticker")
    b15 = ((d.mid15 - d.y) ** 2).groupby(d.ticker).mean()
    bl = ((d.p_ladder - d.y) ** 2).groupby(d.ticker).mean()
    diff = b15 - bl
    print(f"\nrelative value: {len(d)} minutes, {d.ticker.nunique()} markets; Brier 15m mid {b15.mean():.4f}, "
          f"ladder interpolated {bl.mean():.4f}, ladder better by {diff.mean():.4f} (t {diff.mean() / (diff.std() / math.sqrt(len(diff))):.2f})")
    d["gap"] = d.p_ladder - d.mid15
    print("gap ladder − 15m (¢):", (100 * d.gap).describe().round(2).to_dict())
    d = d.sort_values(["ticker", "ts"])
    split = d.ts.quantile(0.5)
    for th in (0.02, 0.04, 0.06, 0.10):
        e_yes = d.p_ladder - d.ask - fee(d.ask)
        e_no = (1 - d.p_ladder) - (1 - d.bid) - fee(1 - d.bid)
        s = d.assign(e_yes=e_yes, e_no=e_no)
        s = s[(s.e_yes > th) | (s.e_no > th)].groupby("ticker").head(1)
        yes = s.e_yes > s.e_no
        pnl = np.where(yes, s.y - s.ask - fee(s.ask), (1 - s.y) - (1 - s.bid) - fee(1 - s.bid))
        for half, m in (("first", s.ts < split), ("second", s.ts >= split)):
            x = pnl[m.values]
            if len(x) > 2:
                print(f"  take the 15m book toward the ladder, edge > {th * 100:.0f}¢, {half} half: {len(x)} trades, "
                      f"{100 * x.mean():.2f}¢ (t {x.mean() / (x.std() / math.sqrt(len(x))):.2f}), hit {np.mean(x > 0):.2f}")
    return d


if __name__ == "__main__" and "--rv" in sys.argv:
    relative_value()
