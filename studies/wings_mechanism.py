"""Why do hourly wings (W03/W04) earn? Stale quotes vs a tail-risk premium (04.11.2026).

Data: Kalshi hourly-ladder book candles Jul–Oct 2026 (best YES bid/ask per strike and minute), Binance × lagged
BRTI basis, the frozen W03/W04 selection (studies/wings_ladder.py logic via studies/brti_proxy.wings_frame).

1. Markout: the cheap side's bid/mid 1, 2, 3, 5 min after our sale, minus the same for unselected cheap sides with the
   same band and minutes left (wings decay as time passes — the baseline removes that). Stale quote ⇒ the price drops
   after the signal more than the baseline.
2. Entry delay: sell 1, 2, 3, 5 min after the signal at that minute's bid (same strike). Stale ⇒ the edge vanishes.
3. Spot move before the signal (last 1/3/5 min, signed: + = away from the strike = the wing should be cheaper).
   Stale ⇒ P&L concentrated after moves away.
4. Regimes: P&L by variance level (terciles), weekend, US hours, distance in σ. Premium ⇒ larger when riskier.
5. The unfiltered wing (every cheap side, W01-like, one per strike) by regime: the raw favourite–longshot premium.
Clusters: hours.   python studies/wings_mechanism.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.argv = sys.argv[:1] + ["data/raw/usd_minutes"]
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from studies import brti_proxy as bp  # noqa: E402
from studies import wings_ladder as wl  # noqa: E402

wl.LAD, wl.MIN = Path("data/raw/kalshi_kxbtcd_60m"), Path("data/interim/minutes.parquet")
OUT = Path("results/studies")
RULES = {"W03": (0.10, 0.20, 5, 15, "normal", 0.01, 1), "W04": (0.06, 0.10, 15, 30, "t4", 0.0, 1)}


def ct(x: pd.Series, cl: pd.Series):
    """mean and t clustered by hour."""
    d = pd.DataFrame({"x": x.values, "c": cl.values}).dropna()
    if len(d) < 3:
        return np.nan, np.nan, len(d)
    g = d.groupby("c").x.agg(["sum", "size"])
    m = d.x.mean()
    u = g["sum"] - m * g["size"]
    se = math.sqrt((u ** 2).sum()) / len(d)
    return m, (m / se if se > 0 else np.nan), len(d)


def fmt(m, t, n, scale=100):
    return f"{scale * m:+.2f} (t {t:.1f}, n {n})" if np.isfinite(m) else "—"


def main():
    E = bp.settlements()
    S = bp.binance()
    c = bp.wings_frame(S, E)
    c["dist_sd"] = np.abs(np.log(c.S * c.basis / c.strike)) / np.sqrt(c.lvl * c.sea * np.maximum(c.left - 2 / 3, 1 / 3))
    # quotes of each strike by minute
    q = c.drop_duplicates(["ticker", "ts"]).set_index(["ticker", "ts"])[["yes_bid", "yes_ask", "S", "basis", "strike"]]
    q = q[~q.index.duplicated()]

    def side_quote(tick, ts, cheap):
        idx = pd.MultiIndex.from_arrays([np.asarray(tick), pd.to_datetime(np.asarray(ts), utc=True)])
        r = q.reindex(idx)
        yb, ya = r.yes_bid.values, r.yes_ask.values
        bid = np.where(cheap == "yes", yb, 1 - ya)
        ask = np.where(cheap == "yes", ya, 1 - yb)
        return bid, (bid + ask) / 2

    lines = ["# Why the hourly wings earn: stale quotes or a risk premium? (studies/wings_mechanism.py)", "",
             f"Kalshi hourly ladder candles {c['T'].min():%Y-%m-%d} .. {c['T'].max():%Y-%m-%d}; spot Binance × BRTI basis; "
             "frozen W03/W04 selection; t clustered by hour.", ""]
    trades = {h: wl.run(c, *p).copy() for h, p in RULES.items()}

    # ------------------------------------------------------------- 1. markout vs baseline
    rows = []
    for h, p in RULES.items():
        lo, hi, l0, l1 = p[:4]
        g = trades[h]
        base = c[c.c_bid.between(lo, hi) & c.left.between(l0, l1)].copy()
        base = base.sample(min(len(base), 60000), random_state=1)
        for k in (1, 2, 3, 5):
            for name, df in (("signal", g), ("baseline", base)):
                b1, m1 = side_quote(df.ticker.values, (df.ts + pd.Timedelta(minutes=k)).values, df.cheap.values)
                b0, mid0 = side_quote(df.ticker.values, df.ts.values, df.cheap.values)
                df[f"mo_bid{k}"] = df.c_bid.values - b1           # + = the bid fell after we sold
                df[f"mo_mid{k}"] = mid0 - m1
            # matched excess: baseline mean by (band ¢, minute-left) cell, subtracted from each signal
            for col in (f"mo_bid{k}", f"mo_mid{k}"):
                base["cell"] = list(zip((100 * base.c_bid).round(), base.left.round()))
                g["cell"] = list(zip((100 * g.c_bid).round(), g.left.round()))
                cm = base.groupby("cell")[col].mean()
                g[f"x_{col}"] = g[col] - g["cell"].map(cm)
            rows.append({"rule": h, "minutes after": k,
                         "signal: bid fell by ¢": fmt(*ct(g[f"mo_bid{k}"], g["T"])),
                         "baseline: bid fell by ¢": fmt(*ct(base[f"mo_bid{k}"], base["T"])),
                         "excess (bid) ¢": fmt(*ct(g[f"x_mo_bid{k}"], g["T"])),
                         "excess (mid) ¢": fmt(*ct(g[f"x_mo_mid{k}"], g["T"]))})
    lines += ["## 1. Markout: how much the cheap side's price falls after our sale",
              "Excess = signal minus unselected cheap sides with the same price (¢) and minutes left. Positive excess = "
              "the quote we hit was stale and caught up afterwards.", "", pd.DataFrame(rows).to_markdown(index=False), ""]

    # ------------------------------------------------------------- 2. entry delay
    rows = []
    for h, g in trades.items():
        rec = {"rule": h, "at signal": fmt(*ct(g.pnl, g["T"]))}
        for k in (1, 2, 3, 5):
            b1, _ = side_quote(g.ticker.values, (g.ts + pd.Timedelta(minutes=k)).values, g.cheap.values)
            px1 = 1 - b1
            pnl1 = b1 - g.c_won.values - wl.fee(px1)
            rec[f"+{k} min"] = fmt(*ct(pd.Series(pnl1, index=g.index), g["T"]))
        rows.append(rec)
    lines += ["## 2. Entry delay: the same trade sold 1–5 minutes later at that minute's bid (¢ per trade)",
              "Stale quotes ⇒ the edge vanishes within minutes; a premium ⇒ it stays.", "",
              pd.DataFrame(rows).to_markdown(index=False), ""]

    # ------------------------------------------------------------- 3. spot move before the signal
    Sb = S
    rows = []
    for h, g in trades.items():
        for k in (1, 3, 5):
            s0 = Sb.reindex(g.ts).values
            s1 = Sb.reindex(g.ts - pd.Timedelta(minutes=k)).values
            ret = np.log(s0 / s1)
            # + = spot moved away from the strike (the cheap side became less likely)
            above = g.S * g.basis > g.strike                 # spot above the strike
            cheap_is_no = g.cheap == "no"                    # cheap NO ⇔ YES likely ⇔ spot above
            away = np.where(cheap_is_no, ret, -ret)         # cheap NO: spot rising = away from strike
            g[f"away{k}"] = away / np.sqrt(g.lvl * g.sea * k)  # in σ of k minutes
            g[f"bk{k}"] = pd.cut(g[f"away{k}"], [-np.inf, -1, -0.3, 0.3, 1, np.inf],
                                 labels=["toward > 1σ", "toward 0.3–1σ", "flat", "away 0.3–1σ", "away > 1σ"])
            for bk, gg in g.groupby(f"bk{k}", observed=True):
                m, t, n = ct(gg.pnl, gg["T"])
                rows.append({"rule": h, "window min": k, "spot move before signal": bk, "n": n,
                             "share %": 100 * n / len(g), "¢/trade": 100 * m, "t": t,
                             "mean edge at entry ¢": 100 * (gg.c_bid - (gg.c_fair_n if h == "W03" else gg.c_fair_t)
                                                           - wl.fee(gg.px)).mean()})
    lines += ["## 3. P&L by the spot move just before the signal (σ of the window; away = the wing got less likely)",
              "", pd.DataFrame(rows).round(2).to_markdown(index=False), ""]

    # ------------------------------------------------------------- 4. regimes (signals) and 5. unfiltered wings
    def regimes(df, label):
        df = df.copy()
        df["vol tercile"] = pd.qcut(df.lvl, 3, labels=["low vol", "mid vol", "high vol"])
        df["weekend"] = np.where(df.ts.dt.dayofweek >= 5, "weekend", "weekday")
        hr = df.ts.dt.hour
        df["session"] = np.where(hr.between(13, 20), "US 13–21 UTC", np.where(hr.between(0, 7), "Asia 00–08 UTC",
                                                                                     "Europe 08–13 / late"))
        df["distance"] = pd.cut(df.dist_sd, [0, 1, 1.5, 2, np.inf], labels=["<1σ", "1–1.5σ", "1.5–2σ", ">2σ"])
        out = []
        for dim in ("vol tercile", "weekend", "session", "distance"):
            for v, gg in df.groupby(dim, observed=True):
                m, t, n = ct(gg.pnl, gg["T"])
                out.append({"sample": label, "split": dim, "value": v, "n": n, "¢/trade": 100 * m, "t": t,
                            "hit %": 100 * (gg.pnl > 0).mean(), "mean bid ¢": 100 * gg.c_bid.mean(),
                            "paid %": 100 * gg.c_won.mean()})
        return out
    rows = []
    for h, g in trades.items():
        rows += regimes(g, h)
    lines += ["## 4. Signals by regime", "", pd.DataFrame(rows).round(2).to_markdown(index=False), ""]
    rows = []
    for (lo, hi), (l0, l1), lab in (((0.10, 0.20), (5, 15), "all 10–20¢, 5–15 min"),
                                     ((0.06, 0.10), (15, 30), "all 6–10¢, 15–30 min")):
        u = c[c.c_bid.between(lo, hi) & c.left.between(l0, l1)].sort_values("ts").drop_duplicates(["T", "ticker"])
        rows += regimes(u, lab)
    lines += ["## 5. Unfiltered wings (every cheap side, first minute in the window, one per strike) by regime",
              "The raw favourite–longshot premium the market pays a seller with no model.", "",
              pd.DataFrame(rows).round(2).to_markdown(index=False), ""]

    # ------------------------------------------------------------- 6. does the model edge close?
    rows = []
    for h, g in trades.items():
        col = "c_fair_n" if h == "W03" else "c_fair_t"
        fr = c.drop_duplicates(["ticker", "ts", "cheap"]).set_index(["ticker", "ts", "cheap"])
        fr = fr[~fr.index.duplicated()]
        for k in (0, 1, 3, 5):
            idx = pd.MultiIndex.from_arrays([g.ticker.values, pd.to_datetime((g.ts + pd.Timedelta(minutes=k)).values,
                                                                             utc=True), g.cheap.values])
            r = fr.reindex(idx)
            e = (r.c_bid - r[col] - wl.fee(r.px)).values
            m, t, n = ct(pd.Series(e, index=g.index), g["T"])
            rows.append({"rule": h, "minutes after": k, "model edge bid − fair − fee ¢": 100 * m, "t": t, "n": n})
    lines += ["## 6. The model edge on the same strike 0–5 minutes later (it closes ⇒ the market caught up)", "",
              pd.DataFrame(rows).round(2).to_markdown(index=False), ""]
    (OUT / "wings_mechanism.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
