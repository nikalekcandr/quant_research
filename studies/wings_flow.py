"""Who pays the wing premium? Kalshi hourly-ladder trade prints, 15.08–28.10.2026 (09.11.2026).

Data: every trade print of the KXBTCD strikes that had a 3–20¢ cheap side in the last 45 minutes
(jobs/hist_kx_wing_trades_now.py → data/raw/kx_wing_trades), compacted by jobs/claude_compact_wings_now.py to
_export/compact/prints/<day>.parquet: last 65 minutes before the close, aggregated by (ticker, second, YES price,
taker side). Outcome of each strike from the settlement value (CF BRTI, studies/brti_proxy.settlements).

1. Flow: every taker who BOUGHT a wing (paid 3–20¢ for his side) — P&L of the resting seller per contract after the
   maker fee, and every taker who SOLD a wing (bought the 80–97¢ favourite) — his own P&L after the taker fee;
   by weekend / weekday, price band, minutes left, month. Contract-weighted; t clustered by settlement hour.
2. Strict maker test (the one wings_maker.py could only bound on candles): at 30 (45, 20) minutes left post one
   offer of the cheap side at its ask on the dearest 6–10¢ (10–20¢) strike (per hour, or per side of spot), keep it
   to 2 minutes before the close. Fill rules from the prints of the cheap side bought by takers:
     strict  a print strictly through our price (our queue position cannot matter)
     ≥ 50    at least 50 contracts printed at or through our price
     loose   any print at or through our price
   P&L per order = offer − payout − maker fee if filled, 0 otherwise; compared with selling at the bid as a taker
   at the same minute on the same strikes.
    python studies/wings_flow.py [prints dir]
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PRINTS = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "_export/compact/prints"
sys.argv = sys.argv[:1] + [str(ROOT / "data/raw/usd_minutes")]
sys.path.insert(0, str(ROOT))
from studies import brti_proxy as bp  # noqa: E402
from studies import wings_ladder as wl  # noqa: E402

wl.LAD, wl.MIN = ROOT / "data/raw/kalshi_kxbtcd_60m", ROOT / "data/interim/minutes.parquet"
OUT = ROOT / "results/studies"


def markdown_table(frame):
    rows = [frame.columns, ["---"] * len(frame.columns), *frame.itertuples(index=False, name=None)]
    return "\n".join("| " + " | ".join(map(str, row)) + " |" for row in rows)


def fee(p, rate):
    return np.ceil(np.round(rate * p * (1 - p) * 100, 6)) / 100


def load_prints():
    tk = pd.read_csv(ROOT / "jobs/kx_wing_tickers.csv")
    tk["T"] = pd.to_datetime(tk.close_time, utc=True)
    tk["K"] = tk.ticker.str.split("-T").str[-1].astype(float)
    E = bp.settlements()
    tk["y"] = np.where(tk["T"].map(E).notna(), (tk["T"].map(E) > tk.K).astype(float), np.nan)
    tk["tid"] = np.arange(len(tk), dtype="int32")
    tid = dict(zip(tk.ticker, tk.tid))
    parts = []
    for f in sorted(PRINTS.glob("*.parquet")):
        d = pd.read_parquet(f)
        parts.append(pd.DataFrame({"tid": d.ticker.astype(str).map(tid).astype("int32").values, "sec": d.sec.values,
                                   "yes_c": d.yes_c.values.astype("int16"), "taker_yes": d.taker_yes.values,
                                   "count": d["count"].values}))
    p = pd.concat(parts, ignore_index=True)
    p["sec"] = pd.to_datetime(p["sec"], utc=True)
    T = tk.set_index("tid")
    p["T"] = pd.to_datetime(T["T"].reindex(p.tid).values, utc=True)
    y = T["y"].reindex(p.tid).values
    p = p[~np.isnan(y)].copy()
    y = y[~np.isnan(y)]
    p["left"] = ((p["T"] - p["sec"]).dt.total_seconds() / 60).astype("float32")
    p["ps"] = np.where(p.taker_yes, p.yes_c, 100 - p.yes_c).astype("int16")      # price the taker paid, ¢
    p["won"] = np.where(p.taker_yes, y, 1 - y).astype("int8")                       # the taker's side paid
    return p, tk


def ratio(g, col):
    h = g.groupby("T").agg(s=(col, "sum"), n=("count", "sum"))
    m = h.s.sum() / h.n.sum()
    u = h.s - m * h.n
    se = math.sqrt((u ** 2).sum()) / h.n.sum()
    return pd.Series({"contracts (k)": h.n.sum() / 1e3, "hours": len(h), "¢/contract": 100 * m, "t": m / se,
                      "paid %": 100 * np.average(g.won, weights=g["count"]),
                      "avg price ¢": np.average(g.ps, weights=g["count"])})


def flow(p):
    P = p.ps / 100
    p["wk"] = np.where(p["T"].dt.day_of_week >= 5, "weekend", "weekday")
    p["maker"] = (P - p.won - fee(P, 0.0175)) * p["count"]
    p["taker"] = (p.won - P - fee(P, 0.07)) * p["count"]
    w = p[(p.ps >= 3) & (p.ps <= 20)].copy()
    s = p[(p.ps >= 80) & (p.ps <= 97)].copy()
    w["band"] = pd.cut(w.ps, [2, 5, 10, 20], labels=["3–5¢", "6–10¢", "11–20¢"])
    s["band"] = pd.cut(100 - s.ps, [2, 5, 10, 20], labels=["3–5¢", "6–10¢", "11–20¢"])
    for x in (w, s):
        x["min left"] = pd.cut(x.left, [0, 5, 15, 30, 45, 65], labels=["0–5", "5–15", "15–30", "30–45", "45–65"])
        x["month"] = x["T"].dt.strftime("%m")
    out = []
    for name, x, col in (("takers BUY wings → resting seller's P&L", w, "maker"),
                         ("takers SELL wings (buy the favourite) → their own P&L", s, "taker")):
        out += [f"### {name}", ""]
        for key in (["wk"], ["wk", "band"], ["wk", "min left"], ["wk", "month"]):
            R = x.groupby(key, observed=True).apply(ratio, col, include_groups=False).reset_index()
            out += [markdown_table(R.round(2)), ""]
    return out


def maker_sim(p, tk):
    c = bp.wings_frame(bp.binance(), bp.settlements())
    c = c[c["T"] >= p["T"].min()]
    c["ask_c"] = np.round(np.where(c.cheap == "yes", c.yes_ask, 1 - c.yes_bid), 2)
    tid = dict(zip(tk.ticker, tk.tid))
    rows = []
    for start, (lo, hi) in ((30, (0.06, 0.10)), (45, (0.06, 0.10)), (20, (0.06, 0.10)), (30, (0.10, 0.20))):
        g = c[c.left.between(start - 1, start) & c.c_bid.between(lo, hi)].sort_values(["T", "c_bid"], ascending=[True, False])
        for scope, gg in (("1 per hour", g.drop_duplicates("T")), ("1 per side", g.drop_duplicates(["T", "cheap"]))):
            gg = gg.assign(tid=gg.ticker.map(tid)).dropna(subset=["tid"])
            gg["tid"] = gg.tid.astype(int)
            pp = p[p.tid.isin(gg.tid) & (p.left >= 2) & (p.left < start)]
            m = pp.merge(gg[["tid", "cheap", "ask_c"]], on="tid")
            m = m[m.taker_yes == (m.cheap == "yes")]                 # takers bought the cheap side
            pc = m.ps / 100
            thru = m[pc > m.ask_c + 1e-9].groupby("tid").size()
            at = m[pc >= m.ask_c - 1e-9].groupby("tid")["count"].sum()
            gg["thru"] = gg.tid.map(thru).fillna(0) > 0
            gg["vol"] = gg.tid.map(at).fillna(0)
            gg["wk"] = gg["T"].dt.day_of_week >= 5
            taker = gg.c_bid - gg.c_won - fee(1 - gg.c_bid, 0.07)
            for rule, filled in (("strict", gg.thru), ("≥ 50", gg.vol >= 50), ("loose", gg.vol > 0)):
                pnl = np.where(filled, gg.ask_c - gg.c_won - fee(gg.ask_c, 0.0175), 0.0)
                for days, sel in (("weekend", gg.wk), ("weekday", ~gg.wk)):
                    x = pd.Series(pnl[sel.values], index=gg["T"][sel].values)
                    h = x.groupby(level=0).sum()
                    th = taker[sel].groupby(gg["T"][sel]).sum()
                    rows.append({"offer at min left": start, "band": f"{100 * lo:.0f}–{100 * hi:.0f}¢", "scope": scope,
                                 "fill": rule, "days": days, "orders": int(sel.sum()),
                                 "filled %": 100 * filled[sel].mean(),
                                 "paid % filled / all": f"{100 * gg.c_won[sel & filled].mean():.1f} / {100 * gg.c_won[sel].mean():.1f}",
                                 "maker ¢/order": 100 * x.mean(),
                                 "maker t": h.mean() / (h.std(ddof=1) / math.sqrt(len(h))) if len(h) > 2 else np.nan,
                                 "taker at bid ¢/order": 100 * taker[sel].mean(),
                                 "taker t": th.mean() / (th.std(ddof=1) / math.sqrt(len(th))) if len(th) > 2 else np.nan})
    return pd.DataFrame(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    p, tk = load_prints()
    lines = ["# Who pays the wing premium? Kalshi hourly-ladder trade prints (studies/wings_flow.py)", "",
             f"{len(p):,} aggregated print rows, {p.tid.nunique():,} strikes, {p['T'].nunique():,} settlement hours, "
             f"{p['T'].min():%Y-%m-%d} .. {p['T'].max():%Y-%m-%d}. ¢ per contract, contract-weighted, t clustered by hour.",
             "", "## 1. Flow", ""]
    lines += flow(p)
    S = maker_sim(p, tk)
    S.to_csv(OUT / "wings_flow_maker.csv", index=False)
    lines += ["## 2. Strict maker test vs taker on the same strikes (1 contract per order)", "",
              markdown_table(S.round(2)), ""]
    (OUT / "wings_flow.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
