"""Lab report: leaderboard of every hypothesis (Holm, deflated Sharpe), model-vs-market calibration of the
short-dated binaries, the volatility zoo, ML information coefficients, data health. Markdown + one HTML page."""
from __future__ import annotations

import base64
import io
import json
import math

import numpy as np
import pandas as pd
from scipy import stats

import config
from lab import signals
from lab.core import FROZEN, HEARTBEAT, LAB, MARKETS, REPORT_HTML, REPORT_MD, RESULTS, TICKS, day_file, read_ledger, utcnow
from lab.ml import ML
from lab.volzoo import VolZoo


def holm(p: pd.Series) -> pd.Series:
    p = p.dropna().sort_values()
    m = len(p)
    if m == 0:
        return p
    adj = np.maximum.accumulate([(m - i) * v for i, v in enumerate(p.values)])
    return pd.Series(np.minimum(adj, 1), index=p.index)


def dsr(sr, n, skew, kurt, n_trials):
    if not np.isfinite(sr) or n < 10:
        return np.nan
    den = math.sqrt(max(1e-12, 1 - skew * sr + (kurt - 1) / 4 * sr ** 2))
    N = max(2, n_trials)
    g = 0.5772156649
    sr0 = math.sqrt(1 / (n - 1)) * ((1 - g) * stats.norm.ppf(1 - 1 / N) + g * stats.norm.ppf(1 - 1 / (N * math.e)))
    return float(stats.norm.cdf((sr - sr0) * math.sqrt(n - 1) / den))


# Trades made by a rule version with a bug are void (kept in the ledger, left out of the statistics):
#   O01, O02  29.10 00:00–09:40 UTC: compared options settling on a BTC-USD index with ones on a BTC-USDT index, and
#             the parity hedge ignored funding — the "arbitrage" was the USDT/USD basis and carry
#   D17       gross P&L used the Polymarket mid instead of the BTC mid (net P&L unaffected, voided for clarity)
#   A11       up to 29.10 14:10 MSK: Kalshi's empty ask (reported as 0) was read as a price of 0
VOID = {"O01": {"960e7aaff561", "1af1b0747705"}, "O02": {"960e7aaff561", "1af1b0747705"},
        "D17": {"960e7aaff561", "1af1b0747705"},
        "A11": {"960e7aaff561", "1af1b0747705", "a407e66e6bc5", "44e7f70ebc1e", "81270e2e4c26", "dff21348f43f"},
        # the hourly Kalshi ladder still came from the listing in version dff21348f43f (14:10–14:45 MSK)
        "P07": {"dff21348f43f"}, "A13": {"dff21348f43f"}, "M02": {"dff21348f43f"}}


# Every trade with a Kalshi leg made before 29.10 14:10 MSK used the /markets listing, which lags the live book by up to
# a minute (6¢ on average on the 15-minute market): those prices were often no longer available. Void.
KALSHI_LISTING_VERSIONS = {"960e7aaff561", "1af1b0747705", "a407e66e6bc5", "44e7f70ebc1e", "81270e2e4c26"}


def void_mask(led: pd.DataFrame) -> pd.Series:
    by_rule = pd.Series([str(c) in VOID.get(h, ()) for h, c in zip(led["hyp"], led["code"])], index=led.index, dtype=bool)
    kalshi = led["venue"].astype(str).str.contains("kalshi") & led["code"].astype(str).isin(KALSHI_LISTING_VERSIONS)
    return by_rule | kalshi


def leaderboard(led: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    reg = {h.id: h for h in signals.REGISTRY}
    void = void_mask(led)
    led = led[~void]
    done = led[led["event"].isin(["CLOSE", "INSTANT"])].copy()
    done["pnl"] = pd.to_numeric(done["pnl"], errors="coerce")
    done["pnl_gross"] = pd.to_numeric(done["pnl_gross"], errors="coerce")
    opened = led[led["event"] == "OPEN"]
    open_ids = set(opened["id"]) - set(done["id"])
    rows = []
    for hid, h in reg.items():
        g = done[(done["hyp"] == hid) & done["pnl"].notna()]
        n = len(g)
        x = g["pnl"].values
        mean = x.mean() if n else np.nan
        sd = x.std(ddof=1) if n > 1 else np.nan
        # trades that settle in the same hour are correlated (several strikes of one ladder, one BTC move):
        # the t-statistic uses hourly clusters of trades
        # cluster by the hour the trade was OPENED (settlement processing can batch many markets into one minute)
        cl = g.groupby(g["ts"].dt.floor("h"))["pnl"].sum() if n else pd.Series(dtype=float)
        k = len(cl)
        t = (cl.mean() / (cl.std(ddof=1) / math.sqrt(k))) if k > 2 and cl.std(ddof=1) > 0 else np.nan
        rows.append({"id": hid, "family": h.family, "hypothesis": h.title, "unit": g["unit"].iloc[0] if n else "",
                     "trades": n, "clusters": k if n else 0, "open": sum(1 for i in open_ids if (opened[opened["id"] == i]["hyp"] == hid).any()),
                     "mean net": mean, "t": t, "p": 2 * stats.norm.sf(abs(t)) if np.isfinite(t) else np.nan,
                     "hit rate": (x > 0).mean() if n else np.nan, "total net": x.sum() if n else 0.0,
                     "mean gross": g["pnl_gross"].mean() if n else np.nan,
                     "SR/trade": mean / sd if n > 1 and sd > 0 else np.nan,
                     "skew": stats.skew(x) if n > 2 else 0.0, "kurt": stats.kurtosis(x, fisher=False) if n > 3 else 3.0,
                     "first": g["ts"].min() if n else pd.NaT})
    lb = pd.DataFrame(rows).set_index("id")
    tested = lb[(lb["trades"] >= 10) & (lb["clusters"] >= 3)]
    lb["Holm p"] = holm(tested["p"]).reindex(lb.index)
    lb["DSR"] = [dsr(r["SR/trade"], r["trades"], r["skew"], r["kurt"], len(tested)) for _, r in lb.iterrows()]

    def verdict(r):
        if r["trades"] < 10 or r["clusters"] < 3:
            return "collecting" if r["trades"] or r["open"] else "no signal yet"
        if r["Holm p"] < 0.05 and r["mean net"] > 0:
            return "ALPHA? survives Holm"
        if r["p"] < 0.05 and r["mean net"] > 0:
            return "promising (not after Holm)"
        if r["p"] < 0.05 and r["mean net"] < 0:
            return "loses money"
        return "nothing"
    lb["verdict"] = lb.apply(verdict, axis=1)
    fam = lb.groupby("family").agg(hypotheses=("trades", "size"), trades=("trades", "sum"), open=("open", "sum"))
    return lb, fam


def calibration(lab_obj, now) -> pd.DataFrame:
    """Model vs market on every short-dated binary we quoted: Brier score of our fair value and of the market mid,
    paired by market (each market's average over its minutes), for markets the venue has settled."""
    parts = []
    for d in range(0, 4):
        f = day_file(MARKETS, now - pd.Timedelta(days=d))
        if f.exists():
            try:
                parts.append(pd.read_csv(f, usecols=["ts", "venue", "kind", "market", "bid", "ask", "fair", "start", "end"]))
            except Exception:
                pass
    if not parts:
        return pd.DataFrame()
    m = pd.concat(parts)
    m["o"] = [lab_obj.res.get(f"{v}|{k}") for v, k in zip(m["venue"], m["market"])]
    m = m.dropna(subset=["o", "fair", "bid", "ask"])
    m = m[(m["ask"] > m["bid"]) & (m["ask"] - m["bid"] <= 0.10)]
    if m.empty:
        return pd.DataFrame()
    m["mid"] = (m["bid"] + m["ask"]) / 2
    m["b_fair"] = (m["fair"] - m["o"]) ** 2
    m["b_mkt"] = (m["mid"] - m["o"]) ** 2
    rows = []
    for kind, g in m.groupby("kind"):
        per = g.groupby("market")[["b_fair", "b_mkt"]].mean()
        d = per["b_mkt"] - per["b_fair"]
        t = d.mean() / (d.std(ddof=1) / math.sqrt(len(d))) if len(d) > 2 and d.std() > 0 else np.nan
        rows.append({"market kind": kind, "markets settled": len(per), "quotes": len(g),
                     "Brier ours": per["b_fair"].mean(), "Brier market mid": per["b_mkt"].mean(),
                     "market − ours": d.mean(), "t (by market)": t,
                     "better": "ours" if d.mean() > 0 else "market"})
    return pd.DataFrame(rows)


def health(now) -> pd.DataFrame:
    f = day_file(TICKS, now)
    if not f.exists():
        return pd.DataFrame()
    t = pd.read_csv(f)
    t["ts"] = pd.to_datetime(t["ts"], utc=True)
    t = t[t["ts"] > now - pd.Timedelta(hours=1)]
    rows = []
    for pre, name in (("bn_", "Binance spot"), ("bf_", "Binance perp"), ("by_", "Bybit"), ("ok_", "OKX"), ("cb_", "Coinbase"),
                      ("kr_", "Kraken"), ("db_", "Deribit"), ("dvol", "DVOL")):
        cols = [c for c in t.columns if c.startswith(pre)]
        rows.append({"source": name, "minutes in last hour": len(t),
                     "filled %": 100 * float(t[cols].notna().any(axis=1).mean()) if cols and len(t) else 0.0})
    return pd.DataFrame(rows)


def _fig(fig) -> str:
    b = io.BytesIO()
    fig.savefig(b, format="png", dpi=110, bbox_inches="tight")
    return base64.b64encode(b.getvalue()).decode()


def charts(led) -> dict:
    void = void_mask(led)
    led = led[~void]
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out = {}
    done = led[led["event"].isin(["CLOSE", "INSTANT"])].copy()
    done["pnl"] = pd.to_numeric(done["pnl"], errors="coerce")
    done = done.dropna(subset=["pnl"])
    if done.empty:
        return out
    done["t_end"] = done["exit_ts"].fillna(done["ts"])
    for fam, g in done.groupby("family"):
        fig, ax = plt.subplots(figsize=(8, 3))
        for hid, gg in g.groupby("hyp"):
            gg = gg.sort_values("t_end")
            ax.step(gg["t_end"], gg["pnl"].cumsum(), where="post", label=f"{hid} ({len(gg)})", lw=1.2)
        ax.axhline(0, color="grey", lw=0.6)
        ax.set_title(f"{signals.FAMILIES.get(fam, fam)}: cumulative net P&L per trade unit")
        ax.legend(fontsize=7, ncol=4)
        fig.autofmt_xdate()
        out[fam] = _fig(fig)
        plt.close(fig)
    return out


def md(df, d=4):
    if df is None or len(df) == 0:
        return "_(nothing yet)_\n"
    df = df.reset_index() if not isinstance(df.index, pd.RangeIndex) else df

    def f(v):
        if isinstance(v, (float, np.floating)):
            if not np.isfinite(v):
                return "—"
            return f"{v:.{d}f}" if abs(v) >= 10 ** -d or v == 0 else f"{v:.2e}"
        return str(v)
    out = "| " + " | ".join(map(str, df.columns)) + " |\n|" + "---|" * len(df.columns) + "\n"
    for row in df.itertuples(index=False):
        out += "| " + " | ".join(f(v) for v in row) + " |\n"
    return out


def write(lab_obj, now):
    RESULTS.mkdir(parents=True, exist_ok=True)
    led = read_ledger()
    lb, fam = leaderboard(led)
    lb.to_csv(RESULTS / "leaderboard.csv")
    cal = calibration(lab_obj, now)
    cal.to_csv(RESULTS / "calibration.csv", index=False)
    zoo, zdm = VolZoo.score(lab_obj.minutes)
    zoo.to_csv(RESULTS / "volzoo_scores.csv", index=False)
    zdm.to_csv(RESULTS / "volzoo_dm.csv", index=False)
    ic = ML.information_coefficient(lab_obj.minutes)
    hl = health(now)
    try:
        hb = json.loads(HEARTBEAT.read_text(encoding="utf-8"))
    except Exception:
        hb = {}
    figs = charts(led)
    show = lb[["family", "hypothesis", "unit", "trades", "clusters", "open", "mean net", "t", "Holm p", "hit rate", "total net",
               "mean gross", "DSR", "verdict"]]
    up = (now - lab_obj.started).total_seconds() / 3600
    n_closed = int(led["event"].isin(["CLOSE", "INSTANT"]).sum())
    head = (f"Updated {now:%Y-%m-%d %H:%M} UTC · lab running {up:.1f} h · rules frozen {FROZEN} (version {lab_obj.book.code}) · "
            f"{n_closed} settled trades, {len(lab_obj.book.pos)} open · last loop {hb.get('loop_seconds', '—')} s")
    L = ["# Alpha lab — live paper trading\n", f"_{head}_\n",
         "Every rule was written down before its first live trade. Entry and exit at the real bid/ask, real fees, "
         "binaries held to the venue's own settlement. Holm controls the family-wise error over all hypotheses with "
         "≥ 10 trades; DSR is the probability that the per-trade Sharpe beats the best of as many zero-skill rules.\n",
         "## Leaderboard\n", md(show), "\n## Families\n", md(fam),
         "\n## Model vs market: who forecasts short-dated binaries better?\n",
         "Brier score (lower is better) of our fair value and of the market mid over every quoted minute, averaged per "
         "market; t > 2 means our model is significantly better calibrated than the market.\n", md(cal),
         "\n## Volatility zoo (QLIKE, lower is better)\n", md(zoo), "\nDiebold–Mariano vs the lab blend:\n\n", md(zdm),
         "\n## ML forecasts: information coefficient\n", md(ic), "\n## Data health (last hour)\n", md(hl, 1),
         "\n## Rules\n"]
    for h in signals.REGISTRY:
        L.append(f"- **{h.id}** {h.title}. {h.rule}. _{h.source}_")
    REPORT_MD.parent.mkdir(parents=True, exist_ok=True)
    REPORT_MD.write_text("\n".join(L), encoding="utf-8")
    # one self-contained HTML page
    css = ("body{font-family:system-ui,Segoe UI,Arial;margin:16px;max-width:1300px;color:#111;background:#fff}"
           "table{border-collapse:collapse;font-size:12px;margin:8px 0}td,th{border:1px solid #ddd;padding:3px 6px;text-align:right}"
           "th{background:#f3f3f3}td:nth-child(2),td:nth-child(3),td:last-child{text-align:left}h2{margin-top:28px}"
           ".v-ALPHA{background:#d8f5d0}.small{color:#555;font-size:12px}")

    def html_table(df):
        if df is None or len(df) == 0:
            return "<p class=small>(nothing yet)</p>"
        return df.to_html(float_format=lambda v: f"{v:.4g}", na_rep="—", border=0)
    H = [f"<!doctype html><html><head><meta charset=utf-8><meta name=viewport content='width=device-width,initial-scale=1'>"
         f"<title>Alpha lab</title><style>{css}</style><meta http-equiv=refresh content=300></head><body>",
         f"<h1>Alpha lab — live paper trading</h1><p class=small>{head}</p>",
         "<h2>Leaderboard</h2>", html_table(show)]
    for fam_id, img in figs.items():
        H.append(f"<img style='max-width:100%' src='data:image/png;base64,{img}'>")
    for k, title in (("markouts", "Markouts after entry (¢, positive = quote moved our way)"),
                     ("models", "Fair-value models vs market (Brier per market)"),
                     ("capacity", "Capacity at the top of the book"), ("lead_lag", "Do Polymarket 5-min quotes lead Binance?"),
                     ("buckets", "Where the P&L comes from")):
        f = RESULTS / f"analysis_{k}.csv"
        if f.exists():
            try:
                H += [f"<h2>{title}</h2>", html_table(pd.read_csv(f))]
            except Exception:
                pass
    H += ["<h2>Model vs market (Brier)</h2>", html_table(cal), "<h2>Volatility zoo</h2>", html_table(zoo),
          html_table(zdm), "<h2>ML information coefficient</h2>", html_table(ic), "<h2>Data health</h2>", html_table(hl),
          "<h2>Rules</h2><ul>"]
    H += [f"<li><b>{h.id}</b> {h.title}. {h.rule}. <i>{h.source}</i></li>" for h in signals.REGISTRY]
    H.append("</ul></body></html>")
    REPORT_HTML.write_text("\n".join(H), encoding="utf-8")
