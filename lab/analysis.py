"""Hourly diagnostics of the lab (reports/lab_analysis.md, results/lab/analysis_*.csv):

1. markouts  — where does the quote go after our entry (+15 s, +1, +2, +5 min)? Positive = the market moved our way,
               i.e. the edge came from a stale or slow quote; zero with a positive P&L = our model knew better.
2. buckets   — P&L of every settled binary by time left, edge size, venue, US session
3. models    — Brier of every fair-value model vs the market mid on every quoted, settled short binary
4. capacity  — dollars of edge actually available at the top of the book
5. lead-lag  — do Polymarket 5-minute Up/Down quotes lead Binance (or follow it)?
"""
from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd

import config
from lab.core import FAST, LAB, MARKETS, RESULTS, day_file, read_ledger, utcnow

MARKETS_FAST = LAB / "markets_fast"
OUT_MD = config.REPORTS / "lab_analysis.md"
BIN_FAMS = ("P", "V", "A", "W", "M", "T")


def _read(folder, days=3, cols=None):
    parts = []
    now = utcnow()
    for d in range(days, -1, -1):
        f = day_file(folder, now - pd.Timedelta(days=d))
        if f.exists():
            try:
                parts.append(pd.read_csv(f, usecols=lambda c: cols is None or c in cols))
            except Exception:
                pass
    if not parts:
        return pd.DataFrame()
    d = pd.concat(parts, ignore_index=True)
    d["ts"] = pd.to_datetime(d["ts"], utc=True, format="mixed")
    return d


def tstat(x):
    x = pd.Series(x).dropna()
    if len(x) < 3 or x.std() == 0:
        return np.nan
    return x.mean() / (x.std() / math.sqrt(len(x)))


def quotes():
    cols = ["ts", "venue", "kind", "market", "bid", "ask", "bid_size", "ask_size", "end", "start", "fair", "fair_t4",
            "fair_kxiv", "fair_dvol", "fair_rv1h"]
    a = _read(MARKETS, cols=cols)
    b = _read(MARKETS_FAST, cols=cols)
    q = pd.concat([a, b], ignore_index=True)
    if q.empty:
        return q
    q["mid"] = (q["bid"] + q["ask"]) / 2
    return q.dropna(subset=["mid"]).sort_values("ts")


def markouts(led, q):
    op = led[(led["event"] == "OPEN") & led["family"].isin(BIN_FAMS)].copy()
    if op.empty or q.empty:
        return pd.DataFrame()
    q = q.set_index("ts")
    by_m = {m: g["mid"] for m, g in q.groupby("market")}
    rows = []
    for _, r in op.iterrows():
        try:
            info = json.loads(r["info"])
        except Exception:
            continue
        legs = info.get("legs") or []
        if len(legs) != 1:
            continue
        leg = legs[0]
        s = by_m.get(leg["market"])
        if s is None:
            continue
        t0 = r["ts"]
        mid0 = info.get("mid")
        if mid0 is None:
            continue
        sgn = 1 if leg["side"] == "yes" else -1
        row = {"hyp": r["hyp"], "ts": t0}
        for lab, sec in (("15s", 15), ("1m", 60), ("2m", 120), ("5m", 300)):
            after = s[(s.index >= t0 + pd.Timedelta(seconds=sec)) & (s.index <= t0 + pd.Timedelta(seconds=sec + 75))]
            row[lab] = 100 * sgn * (after.iloc[0] - mid0) if len(after) else np.nan
        rows.append(row)
    d = pd.DataFrame(rows)
    if d.empty:
        return d
    out = []
    for h, g in d.groupby("hyp"):
        rec = {"hyp": h, "entries": len(g)}
        for lab in ("15s", "1m", "2m", "5m"):
            rec[f"markout {lab}, ¢"] = g[lab].mean()
            rec[f"t {lab}"] = tstat(g[lab])
        out.append(rec)
    return pd.DataFrame(out)


def buckets(led):
    c = led[(led["event"] == "CLOSE") & led["family"].isin(BIN_FAMS)].copy()
    c["pnl"] = pd.to_numeric(c["pnl"], errors="coerce")
    c = c.dropna(subset=["pnl"])
    if c.empty:
        return pd.DataFrame()
    info = c["info"].map(lambda s: json.loads(s) if isinstance(s, str) else {})
    c["left"] = info.map(lambda i: i.get("left_min"))
    c["edge_c"] = 100 * pd.to_numeric(c["edge"], errors="coerce")
    et = c["ts"].dt.tz_convert("America/New_York")
    c["us_session"] = np.where((et.dt.dayofweek < 5) & (et.dt.hour * 60 + et.dt.minute >= 570) & (et.dt.hour < 16), "US hours", "other")
    rows = []
    for name, key in (("time left, min", pd.cut(c["left"], [0, 3, 7, 15, 60, 1e4])),
                      ("edge, ¢", pd.cut(c["edge_c"], [0, 5, 10, 20, 100])),
                      ("session", c["us_session"]), ("venue", c["venue"])):
        for (h, b), g in c.groupby([c["hyp"], key], observed=True):
            if len(g) >= 3:
                rows.append({"cut": name, "hyp": h, "bucket": str(b), "trades": len(g), "mean P&L, ¢": 100 * g["pnl"].mean(),
                             "t": tstat(g["pnl"]), "hit": (g["pnl"] > 0).mean()})
    return pd.DataFrame(rows)


def models(q, res):
    if q.empty:
        return pd.DataFrame()
    q = q[q["kind"].isin(["pm_updown_5m", "pm_updown_15m", "pm_updown_1h", "kx_15m", "kx_ladder_1h", "pm_ladder_1h"])].copy()
    q["o"] = [res.get(f"{v}|{m}") for v, m in zip(q["venue"], q["market"])]
    q = q.dropna(subset=["o"])
    q = q[(q["ask"] - q["bid"]) <= 0.10]
    if q.empty:
        return pd.DataFrame()
    rows = []
    cols = [c for c in ["fair", "fair_t4", "fair_kxiv", "fair_dvol", "fair_rv1h"] if c in q]
    for kind, g in q.groupby("kind"):
        per = g.groupby("market").agg(o=("o", "first"), mid=("mid", "mean"), **{c: (c, "mean") for c in cols})
        rec = {"kind": kind, "markets": len(per), "Brier market": ((per["mid"] - per["o"]) ** 2).mean()}
        for c in cols:
            ok = per[c].notna()
            if ok.sum() >= 3:
                d = (per.loc[ok, "mid"] - per.loc[ok, "o"]) ** 2 - (per.loc[ok, c] - per.loc[ok, "o"]) ** 2
                rec[f"Brier {c}"] = ((per.loc[ok, c] - per.loc[ok, "o"]) ** 2).mean()
                rec[f"t vs market ({c})"] = tstat(d)
        rows.append(rec)
    return pd.DataFrame(rows)


def capacity(led):
    op = led[(led["event"] == "OPEN") & led["family"].isin(BIN_FAMS)].copy()
    if op.empty:
        return pd.DataFrame()
    info = op["info"].map(lambda s: json.loads(s) if isinstance(s, str) else {})
    size = info.map(lambda i: i.get("size") if i.get("size") is not None else
                    (min(x for x in (i.get("pm_size"), i.get("kx_size")) if x is not None)
                     if (i.get("pm_size") is not None or i.get("kx_size") is not None) else None))
    op["size"] = pd.to_numeric(size, errors="coerce")
    op["edge"] = pd.to_numeric(op["edge"], errors="coerce")
    op["usd_edge"] = op["size"] * op["edge"]
    g = op.dropna(subset=["size"]).groupby("hyp")
    if not len(g):
        return pd.DataFrame()
    days = max((op["ts"].max() - op["ts"].min()).total_seconds() / 86400, 1 / 24)
    return g.agg(entries=("size", "size"), median_size=("size", "median"), median_edge_c=("edge", lambda x: 100 * x.median()),
                 usd_edge_total=("usd_edge", "sum")).assign(usd_edge_per_day=lambda x: x["usd_edge_total"] / days).reset_index()


def lead_lag(q):
    f = _read(FAST, days=2)
    if q.empty or f.empty or "bn" not in f:
        return pd.DataFrame()
    s = f.set_index("ts")["bn"].sort_index()
    s = s[~s.index.duplicated()]
    pm = q[q["kind"] == "pm_updown_5m"][["ts", "market", "mid"]].copy()
    pm = pm[pm["ts"].dt.second.between(0, 59)]
    rows = []
    for m, g in pm.groupby("market"):
        g = g.sort_values("ts")
        if len(g) < 5:
            continue
        g["t"] = g["ts"].dt.floor("15s")
        g = g.drop_duplicates("t").set_index("t")
        spot = np.log(s.reindex(g.index, method="ffill"))
        dm = g["mid"].diff()
        rs = spot.diff()
        for lag in (-2, -1, 0, 1, 2):
            rows.append(pd.DataFrame({"lag": lag, "dmid": dm.values, "r": rs.shift(-lag).values}))
    if not rows:
        return pd.DataFrame()
    d = pd.concat(rows).dropna()
    out = []
    for lag, g in d.groupby("lag"):
        r = g["dmid"].corr(g["r"])
        out.append({"lag (15-s steps)": lag, "meaning": "spot later" if lag > 0 else ("spot earlier" if lag < 0 else "same 15 s"),
                    "corr(Δ Polymarket mid, Binance return)": r, "t": r * math.sqrt(max(len(g) - 2, 1)) / math.sqrt(max(1e-9, 1 - r * r)),
                    "n": len(g)})
    return pd.DataFrame(out)


def md(df, d=3):
    if df is None or len(df) == 0:
        return "_(not enough data yet)_\n"

    def f(v):
        if isinstance(v, (float, np.floating)):
            return "—" if not np.isfinite(v) else f"{v:.{d}f}"
        return str(v)
    out = "| " + " | ".join(map(str, df.columns)) + " |\n|" + "---|" * len(df.columns) + "\n"
    for row in df.itertuples(index=False):
        out += "| " + " | ".join(f(v) for v in row) + " |\n"
    return out


def run(lab_obj):
    RESULTS.mkdir(parents=True, exist_ok=True)
    led = read_ledger()
    q = quotes()
    parts = {"markouts": markouts(led, q), "buckets": buckets(led), "models": models(q, lab_obj.res),
             "capacity": capacity(led), "lead_lag": lead_lag(q)}
    for k, v in parts.items():
        if v is not None and len(v):
            v.to_csv(RESULTS / f"analysis_{k}.csv", index=False)
    L = [f"# Lab diagnostics\n\n_Updated {utcnow():%Y-%m-%d %H:%M} UTC._\n",
         "## 1. Markouts after entry (mid moves in our favour, cents)\n",
         "Positive and growing = the quote was stale and the market caught up; ≈ 0 with a profit at settlement = our "
         "model knew better than the market.\n", md(parts["markouts"], 2),
         "\n## 2. Fair-value models vs market (Brier per market, lower is better; t > 2 = model better than market)\n",
         md(parts["models"], 4), "\n## 3. Where the P&L comes from\n", md(parts["buckets"], 2),
         "\n## 4. Capacity at the top of the book\n", md(parts["capacity"], 2),
         "\n## 5. Do Polymarket 5-minute quotes lead Binance?\n",
         "Correlation of the 15-s change of the Up mid with Binance returns 30 s before … 30 s after.\n", md(parts["lead_lag"], 3)]
    OUT_MD.write_text("\n".join(L), encoding="utf-8")
    return parts
