"""The wings backtest vs the live lab: which variance model do W03/W04 actually run on? (09.11.2026)

Finding that started this: in studies/wings_ladder.variance_level (and the copy in lab/signals.storm_ratio) the
hour-of-week profile is computed as daily.rolling(28 rows, min_periods=7) over a (day x hour-of-week) table in which
each column has one value per WEEK — so every window holds 4 values < 7, the profile is NaN and is filled with 1.0.
The backtest that froze W03/W04 therefore ran with NO seasonality (a pure EWMA-60 of squared returns), while the live
lab (lab/history.py) uses a 90-day hour-of-week profile and a blended level (0.5 EWMA-30 + 0.3 last 24 h + 0.2 last
7 d), on the USD spot without a basis. The live rule is not the rule that was backtested.

Models of the per-minute variance compared, all strictly lagged:
  A  frozen backtest: EWMA-60 of r², no profile (= what W03/W04 were chosen on)
  B  A with the profile it was meant to have: mean r² of the same hour-of-week over the previous 4 weeks
  L  the live lab: 90-day profile (as of the previous day), level 0.5·EWMA30 + 0.3·24h + 0.2·7d of deseasonalised r²
Spot: Binance × lagged BRTI basis (the backtest) or Coinbase USD close (the lab), fair at the settlement minute − 30 s.

Outputs: calibration (log-loss / Brier of the cheap side's probability against the outcome, all cheap sides 3–20¢,
5–45 min left, one row per strike and 5-minute bucket) and the W03/W04 rules re-run on each model and spot.
    python studies/wings_varfix.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm, t as student_t

ROOT = Path(__file__).resolve().parents[1]
sys.argv = sys.argv[:1] + [str(ROOT / "data/raw/usd_minutes")]
sys.path.insert(0, str(ROOT))
from studies import brti_proxy as bp  # noqa: E402
from studies import wings_ladder as wl  # noqa: E402

wl.LAD, wl.MIN = ROOT / "data/raw/kalshi_kxbtcd_60m", ROOT / "data/interim/minutes.parquet"
OUT = ROOT / "results/studies"
SPLIT = wl.SPLIT
RULES = {"W03": (0.10, 0.20, 5, 15, "normal", 0.01, 1), "W04": (0.06, 0.10, 15, 30, "t4", 0.0, 1)}


def r2_frame(close: pd.Series) -> pd.DataFrame:
    close = close[~close.index.duplicated()].sort_index()
    r2 = (np.log(close).diff() ** 2).rename("r2")
    df = r2.to_frame()
    df["how"] = df.index.dayofweek * 24 + df.index.hour
    df["day"] = df.index.floor("D")
    return df.dropna()


def model_A(close):
    """the frozen backtest (sea == 1 in practice)."""
    return wl.variance_level(close)[["sea", "lvl"]]


def model_B(close, weeks=4):
    """EWMA-60 of deseasonalised r², profile = mean r² of the same hour-of-week over the previous `weeks` weeks."""
    df = r2_frame(close)
    hw = df.groupby(["day", "how"]).r2.mean().reset_index()
    hw = hw.sort_values(["how", "day"])
    hw["prof"] = hw.groupby("how").r2.transform(lambda s: s.shift(1).rolling(weeks, min_periods=2).mean())
    # normalise: divide by the mean over the 168 hours of the profile as of that day
    allday = hw.groupby("day").prof.mean().rolling(7, min_periods=1).mean()   # smooth the normaliser over a week
    hw["sea"] = hw.prof / hw.day.map(allday)
    key = pd.MultiIndex.from_arrays([df.day, df.how])
    df["sea"] = hw.set_index(["day", "how"])["sea"].reindex(key).values
    df["sea"] = df["sea"].fillna(1.0).clip(0.2, 5)
    df["lvl"] = (df.r2 / df.sea).ewm(halflife=60, min_periods=120).mean()
    return df[["sea", "lvl"]]


def model_L(close):
    """the live lab (lab/history.py): profile from the previous 90 days (recomputed daily), clipped r²,
    level 0.5·EWMA30 + 0.3·last 24 h + 0.2·last 7 d of deseasonalised r²."""
    df = r2_frame(close)
    days = df.day.unique()
    profs = {}
    for d in days:
        w = df[(df.day < d) & (df.day >= d - pd.Timedelta(days=90))]
        if len(w) < 7 * 1440:
            continue
        x = w.r2.clip(upper=w.r2.quantile(0.999))
        s = x.groupby(w.how).mean()
        s = (s / s.mean()).reindex(range(168)).fillna(1.0)
        # centred 3-hour smoothing, as the lab (no wrap-around, min_periods=1)
        profs[d] = s.rolling(3, center=True, min_periods=1).mean()
    P = pd.DataFrame(profs).T
    sea = np.array([P.loc[d, h] if d in P.index else np.nan for d, h in zip(df.day, df.how)], dtype=float)
    df["sea"] = np.nan_to_num(sea, nan=1.0)
    z = df.r2 / df.sea
    z = z.clip(upper=z.rolling(8 * 1440, min_periods=600).quantile(0.9995).shift(1))
    ew = z.ewm(halflife=30).mean()
    d1 = z.rolling("24h").mean()
    d7 = z.rolling("7D").mean()
    df["lvl"] = 0.5 * ew + 0.3 * d1 + 0.2 * d7
    return df[["sea", "lvl"]]


def ladder():
    d = pd.concat([pd.read_csv(f) for f in sorted(p for p in wl.LAD.glob("*.csv*") if p.name.endswith((".csv", ".csv.gz")))], ignore_index=True)
    d = d[(d.strike_type == "greater") & (d.yes_bid > 0) & (d.yes_ask < 1) & (d.yes_ask - d.yes_bid <= 0.03)]
    d["ts"] = pd.to_datetime(d.end_ts, unit="s", utc=True)
    d["T"] = pd.to_datetime(d.close_time, utc=True)
    d["left"] = (d["T"] - d["ts"]).dt.total_seconds() / 60
    d["y"] = (d.result == "yes").astype(float)
    return d[d.left.between(0.5, 60)].copy()


def cheap_frame(d, S, basis, vm, end_adj):
    """cheap-side rows with fair values; S/basis/vm aligned on the minute (vm lagged one minute: no look-ahead)."""
    x = d.copy()
    x["S"] = S.reindex(x.ts).values
    x["basis"] = 1.0 if basis is None else x["T"].map(basis).values
    vml = vm          # indexed by the minute's close time: known at that time
    x["lvl"] = vml["lvl"].reindex(x.ts).values
    x["sea"] = vml["sea"].reindex(x.ts).values
    x = x.dropna(subset=["S", "lvl", "sea", "basis"])
    var = x.lvl * x.sea * np.maximum(x.left - end_adj, 1 / 3)
    z = np.log(x.S * x.basis / x.strike) / np.sqrt(var)
    x["p_n"], x["p_t"] = norm.cdf(z), student_t.cdf(z * math.sqrt(2), 4)
    cy, cn = x.yes_bid.between(0.005, 0.20), (1 - x.yes_ask).between(0.005, 0.20)
    a = x[cy].assign(cheap="yes", c_bid=x.yes_bid, c_fair_n=x.p_n, c_fair_t=x.p_t, c_won=x.y)
    b = x[cn].assign(cheap="no", c_bid=1 - x.yes_ask, c_fair_n=1 - x.p_n, c_fair_t=1 - x.p_t, c_won=1 - x.y)
    c = pd.concat([a, b], ignore_index=True)
    c["px"] = 1 - c.c_bid
    c["pnl"] = c.c_bid - c.c_won - wl.fee(c.px)
    return c


def calib(c):
    g = c[c.c_bid.between(0.03, 0.20) & c.left.between(5, 45)].copy()
    g["b5"] = (g.left // 5).astype(int)
    g = g.sort_values("ts").drop_duplicates(["ticker", "b5"])
    out = {}
    for nm, p in (("normal", g.c_fair_n), ("t4", g.c_fair_t), ("book mid", g.c_bid + 0.005)):
        p = p.clip(1e-4, 1 - 1e-4)
        y = g.c_won
        out[f"{nm} logloss"] = float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())
        out[f"{nm} mean p %"] = 100 * float(p.mean())
    out["paid %"] = 100 * float(g.c_won.mean())
    out["n"] = len(g)
    return out


def rule_stats(c):
    out = {}
    for h, p in RULES.items():
        g = wl.run(c, *p)
        for half, gg in (("Jul–Aug", g[g["T"] < SPLIT]), ("Sep–Oct", g[g["T"] >= SPLIT])):
            hh = gg.groupby("T").pnl.sum()
            out[f"{h} {half}"] = (f"{100 * gg.pnl.mean():+.2f}¢ (t {hh.mean() / (hh.std(ddof=1) / math.sqrt(len(hh))):.1f}, "
                                  f"n {len(gg)})") if len(hh) > 2 else "—"
        we = g[g["T"].dt.dayofweek >= 5]
        out[f"{h} weekends ¢"] = round(100 * we.pnl.mean(), 2)
        out[f"{h} weekdays ¢"] = round(100 * g[g["T"].dt.dayofweek < 5].pnl.mean(), 2)
    return out


def main():
    E = bp.settlements()
    bn = bp.binance()
    raw = bp.candles("cb")
    if raw is None or "c" not in getattr(raw, "columns", []):
        raise ValueError("нет данных в окне: нет свечей Coinbase (data/raw/coinbase_*)")
    cb = raw["c"]
    d = ladder()
    print(f"ladder rows {len(d)}, hours {d['T'].nunique()}", flush=True)
    models = {}
    for nm, src in (("binance", bn), ("coinbase", cb)):
        src = src[src.index >= d.ts.min() - pd.Timedelta(days=100)]
        models[(nm, "A")] = model_A(src)
        models[(nm, "B")] = model_B(src)
        models[(nm, "L")] = model_L(src)
        print("models done for", nm, flush=True)
    rows = []
    seas = {}
    for (spot, m), vm in models.items():
        S = bn if spot == "binance" else cb
        basis = bp.lagged_basis(E, S) if spot == "binance" else None
        end_adj = 2 / 3 if m != "L" else 0.5
        c = cheap_frame(d, S, basis, vm, end_adj)
        rec = {"spot": spot + (" × basis" if basis is not None else " (no basis)"), "variance": m}
        rec.update(calib(c))
        rec.update(rule_stats(c))
        rows.append(rec)
        seas[(spot, m)] = vm["sea"]
        print(rec, flush=True)
    R = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    R.to_csv(OUT / "wings_varfix.csv", index=False)
    # what the profiles look like: weekend vs weekday, hour of day
    prof = []
    for (spot, m), s in seas.items():
        s = s[s.index >= pd.Timestamp("2026-07-01", tz="UTC")]
        prof.append({"spot": spot, "variance": m, "sea weekday": s[s.index.dayofweek < 5].mean(),
                     "sea weekend": s[s.index.dayofweek >= 5].mean(), "sea 13–16 UTC": s[s.index.hour.isin([13, 14, 15])].mean(),
                     "sea 00–06 UTC": s[s.index.hour < 6].mean(), "share sea == 1": float((s == 1).mean())})
    P = pd.DataFrame(prof)
    lines = ["# The wings model in the backtest vs the live lab (studies/wings_varfix.py)", "",
             __doc__.split("Models of")[0].strip(), "",
             "## Seasonal profile actually applied (Jul–Oct)", "", P.round(3).to_markdown(index=False), "",
             "## Calibration of the cheap side (lower log-loss = better; 'book mid' = the market)", "",
             R[["spot", "variance", "normal logloss", "t4 logloss", "book mid logloss", "normal mean p %", "t4 mean p %",
                "book mid mean p %", "paid %", "n"]].round(4).to_markdown(index=False), "",
             "## W03/W04 re-run on each model (1 contract, taker, ¢ per trade, t by hour)", "",
             R[["spot", "variance"] + [c for c in R.columns if c.startswith("W0")]].to_markdown(index=False), ""]
    (OUT / "wings_varfix.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
