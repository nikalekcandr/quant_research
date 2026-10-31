"""W03–W06 live forward test: parse the lab logs, compute every statistic, draw the dashboard.

Data: lab_2026*.log (root of the repo), the Kalshi hourly wing-selling rules frozen on 29.10.2026:
  W03  sell the cheap side at 10–20¢, 5–15 min left, taker, edge ≥ 1¢
  W04  same at 6–10¢, 15–30 min left, taker, edge ≥ 0
  W05  blend fair value, no US-open hours, maker-first, 10–20¢, 5–15 min (fallback 7)
  W06  same, 6–10¢, 15–30 min (fallback 17)
Everything is 1 contract, Kalshi fees included, held to settlement.

Run:  python reports/wings_live.py
Out:  reports/final/figures/fig19_wings_live.png
      reports/final/figures/fig20_wings_history.png
      reports/final/tables/wings_live_{trades,summary,daily,buckets}.csv
      reports/final/tables/wings_live_summary.json
"""
from __future__ import annotations

import glob
import json
import math
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "reports" / "final" / "figures"
TAB = ROOT / "reports" / "final" / "tables"
FIG.mkdir(parents=True, exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)

INK = "#0b1220"
MUTED = "#5b6472"
GRID = "#e8edf5"
LOSS = "#dc2626"
TOTAL = "#0f172a"
C = {"W03": "#4f46e5", "W04": "#0d9488", "W05": "#d97706", "W06": "#db2777"}
LIGHT = {"W03": "#c7d2fe", "W04": "#99f6e4", "W05": "#fde68a", "W06": "#fbcfe8"}
RNG = np.random.default_rng(20261101)
AMBER_B = "#f59e0b"
Z = 1.959963984540054

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10.3,
    "axes.edgecolor": "#cbd5e1", "axes.linewidth": 1.0,
    "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.titleweight": "bold",
})


# ------------------------------------------------------------------ parse logs
pat_open = re.compile(r"^\[(?P<ts>[^\]]+)\] OPEN (?P<hyp>W0[3-6]) (?P<side>yes|no) (?P<market>\S+) "
                      r"@ (?P<price>[\d.]+) edge (?P<edge>-?[\d.]+)")
pat_wing = re.compile(r"^\[(?P<ts>[^\]]+)\] WINGS (?P<hyp>W0[56]) (?P<exec>maker|taker) "
                      r"sold (?P<cheap>yes|no) (?P<market>\S+) @ (?P<cprice>[\d.]+)")
pat_settle = re.compile(r"^\[(?P<ts>[^\]]+)\] SETTLE (?P<hyp>W0[3-6]) (?P<market>\S+) "
                        r"pays (?P<pays>[01]) pnl (?P<pnl>[+-][\d.]+)")

logs = sorted(glob.glob(str(ROOT / "lab_2026*.log")))
opens: dict[tuple[str, str], dict] = {}
trades: list[dict] = []
feed_errors = 0
span = []
for f in logs:
    for line in open(f, encoding="utf-8", errors="replace"):
        line = line.strip()
        m = re.match(r"^\[([^\]]+)\]", line)
        if m:
            span.append(pd.Timestamp(m.group(1)))
        if "markets kalshi failed" in line:
            feed_errors += 1
        m = pat_open.match(line)
        if m:
            d = m.groupdict()
            opens[(d["hyp"], d["market"])] = dict(
                ts=d["ts"], side=d["side"], price=float(d["price"]), edge=float(d["edge"]),
                exec="taker", cheap="no" if d["side"] == "yes" else "yes",
                cprice=float(d["price"]))
            continue
        m = pat_wing.match(line)
        if m:
            d = m.groupdict()
            opens[(d["hyp"], d["market"])] = dict(
                ts=d["ts"], side="no" if d["cheap"] == "yes" else "yes",
                price=1 - float(d["cprice"]), edge=np.nan, exec=d["exec"],
                cheap=d["cheap"], cprice=float(d["cprice"]))
            continue
        m = pat_settle.match(line)
        if m:
            d = m.groupdict()
            o = opens.pop((d["hyp"], d["market"]), None)
            if o is None:
                print("orphan settle", d)
                continue
            trades.append(dict(hyp=d["hyp"], market=d["market"], open_ts=o["ts"], settle_ts=d["ts"],
                               side=o["side"], price=o["price"], edge=o["edge"], exec=o["exec"],
                               cheap=o["cheap"], cprice=o["cprice"], pays=int(d["pays"]),
                               pnl_log=float(d["pnl"])))
assert not opens, f"unmatched opens: {opens}"

t = pd.DataFrame(trades)
t["open_ts"] = pd.to_datetime(t["open_ts"], utc=True)
t["settle_ts"] = pd.to_datetime(t["settle_ts"], utc=True)
t["date"] = t["open_ts"].dt.tz_convert("UTC").dt.date
t["cheap"] = 1 - t["price"]
t["fee"] = np.where(t["exec"] == "maker",
                    np.ceil(np.round(0.0175 * t["price"] * (1 - t["price"]) * 100, 6)) / 100,
                    np.ceil(np.round(0.07 * t["price"] * (1 - t["price"]) * 100, 6)) / 100)
t["pnl"] = t["pays"] - t["price"] - t["fee"]
t["max_loss"] = -(t["price"] + t["fee"])
bad = (t["pnl"] - t["pnl_log"]).abs() > 0.006
print(f"{len(t)} trades, {t.market.nunique()} markets, {len(logs)} log files, "
      f"{feed_errors} Kalshi feed errors, pnl mismatches {bad.sum()}")
assert bad.sum() == 0


# ------------------------------------------------------------------ statistics
def wilson(k: int, n: int) -> tuple[float, float]:
    p = k / n
    den = 1 + Z * Z / n
    c = (p + Z * Z / (2 * n)) / den
    h = Z * math.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)) / den
    return 100 * max(0.0, c - h), 100 * min(1.0, c + h)


def risk_ci(g: pd.DataFrame, reps: int = 20000) -> tuple[float, float]:
    """Mean P&L CI that also prices the risk of losses we have not seen yet.

    Hit probability ~ Beta(wins+½, losses+½); win sizes from the observed wins; if the strategy
    has no loss yet, the loss is the price actually paid plus fee. This is deliberately wider than
    a t-interval, because 5/5 or 8/8 says almost nothing about the loss tail.
    """
    w = g.loc[g.pays == 1, "pnl"].values
    l = g.loc[g.pays == 0, "pnl"].values
    wm = w.mean() if len(w) else 0.0
    lm = l.mean() if len(l) else float(g["max_loss"].mean())
    p = RNG.beta(len(w) + 0.5, len(l) + 0.5, reps)
    k = RNG.binomial(len(g), p)
    means = (k * wm + (len(g) - k) * lm) / len(g)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return 100 * lo, 100 * hi


rows = []
for h in ["W03", "W04", "W05", "W06", "ALL"]:
    g = t if h == "ALL" else t[t.hyp == h]
    n = len(g)
    k = int(g.pays.sum())
    lo_w, hi_w = wilson(k, n)
    lo_c, hi_c = risk_ci(g)
    s = g.sort_values("settle_ts")
    cum = s.pnl.cumsum()
    dd = float((cum - cum.cummax()).min())
    sd = g.pnl.std(ddof=1)
    rows.append(dict(
        hyp=h, n=n, wins=k, losses=n - k, hit=100 * k / n, hit_lo=lo_w, hit_hi=hi_w,
        mean_c=100 * g.pnl.mean(), median_c=100 * g.pnl.median(), min_c=100 * g.pnl.min(),
        max_c=100 * g.pnl.max(), ci_lo_c=lo_c, ci_hi_c=hi_c, total=g.pnl.sum(),
        sd_c=100 * sd, tstat=g.pnl.mean() / (sd / math.sqrt(n)) if sd > 0 else np.nan,
        avg_win_c=100 * g.loc[g.pays == 1, "pnl"].mean() if k else np.nan,
        avg_loss_c=100 * g.loc[g.pays == 0, "pnl"].mean() if n > k else np.nan,
        profit_factor=(g.loc[g.pnl > 0, "pnl"].sum() / abs(g.loc[g.pnl < 0, "pnl"].sum())
                       if (g.pnl < 0).any() else np.nan),
        max_dd=dd, collateral=g.price.sum(), roi=100 * g.pnl.sum() / g.price.sum(),
        avg_cheap_c=100 * g.cheap.mean(), avg_edge_c=100 * g.edge.mean() if g.edge.notna().any() else np.nan,
        maker_share=100 * (g.exec == "maker").mean(),
    ))
S = pd.DataFrame(rows)
S.to_csv(TAB / "wings_live_summary.csv", index=False)

daily = (t.groupby("date").agg(n=("pnl", "size"), wins=("pays", "sum"), pnl=("pnl", "sum"))
         .reset_index())
daily["cum"] = daily.pnl.cumsum()
daily.to_csv(TAB / "wings_live_daily.csv", index=False)

t["cheap_bucket"] = pd.cut(t.cheap, [0.05, 0.10, 0.15, 0.21], labels=["6–10¢", "10–15¢", "15–20¢"])
t["edge_bucket"] = pd.cut(t.edge, [-0.001, 0.005, 0.015, 0.2], labels=["< 0,5¢", "0,5–1,5¢", "> 1,5¢"])
buckets = []
for col, name in [("cheap_bucket", "cheap"), ("edge_bucket", "edge"), ("exec", "exec"), ("side", "side")]:
    g = t.groupby(col, observed=True).agg(n=("pnl", "size"), wins=("pays", "sum"),
                                          mean_c=("pnl", lambda x: 100 * x.mean()),
                                          total=("pnl", "sum")).reset_index()
    g = g.rename(columns={col: "bucket"})
    g.insert(0, "kind", name)
    buckets.append(g)
B = pd.concat(buckets, ignore_index=True)
B.to_csv(TAB / "wings_live_buckets.csv", index=False)

t.sort_values("open_ts").to_csv(TAB / "wings_live_trades.csv", index=False)

summary = dict(
    n=int(len(t)), markets=int(t.market.nunique()), days=int(t.date.nunique()),
    start=str(t.open_ts.min()), end=str(t.settle_ts.max()),
    total=float(t.pnl.sum()), mean_c=float(100 * t.pnl.mean()), hit=float(100 * t.pays.mean()),
    feed_errors=int(feed_errors), log_lines=len(span),
    log_start=str(min(span)), log_end=str(max(span)),
    multi_markets=int((t.groupby("market").size() > 1).sum()),
)
(TAB / "wings_live_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
print(json.dumps(summary, indent=2, ensure_ascii=False))
print(S.round(2).to_string(index=False))


# ------------------------------------------------------------------ figure 19: live dashboard
fig = plt.figure(figsize=(13.8, 11.0))
gs = GridSpec(3, 3, height_ratios=[1.28, 1.0, 1.0], hspace=0.34, wspace=0.33,
              left=0.075, right=0.985, top=0.915, bottom=0.082)

# --- A: cumulative P&L
axA = fig.add_subplot(gs[0, :])
tot = t.sort_values("settle_ts")
axA.plot(tot.settle_ts, tot.pnl.cumsum(), color=TOTAL, lw=2.8,
         drawstyle="steps-post", zorder=5, label="все стратегии")
for h in C:
    g = t[t.hyp == h].sort_values("settle_ts")
    axA.plot(g.settle_ts, g.pnl.cumsum(), color=C[h], lw=2.2, marker="o", ms=4.2,
             drawstyle="steps-post", label=h, zorder=4)
    if len(g):
        axA.annotate(f"{g.pnl.sum():+.2f} $", (g.settle_ts.iloc[-1], g.pnl.sum()),
                     xytext=(6, 0), textcoords="offset points", color=C[h],
                     fontsize=9.3, fontweight="bold", va="center")
axA.axhline(0, color="#94a3b8", lw=1.0)
loss_row = t[t.pays == 0].iloc[0]
axA.scatter([loss_row.settle_ts], [t.sort_values("settle_ts").pnl.cumsum().iloc[-1]],
            s=70, facecolor="white", edgecolor=LOSS, lw=1.8, zorder=8)
axA.annotate("+2,34 $", (tot.settle_ts.iloc[-1], t.pnl.sum()), xytext=(7, 9),
             textcoords="offset points", fontsize=9.6, color=TOTAL, fontweight="bold")
axA.annotate("единственный убыток\n−0,95 $ (W04)", xy=(loss_row.settle_ts, t.pnl.sum()),
             xytext=(-118, -46), textcoords="offset points", fontsize=8.8, color=LOSS,
             arrowprops=dict(arrowstyle="->", color=LOSS, lw=1.0))
axA.set_ylabel("Накопленный P&L, $\n(1 контракт)", fontsize=10.5)
axA.grid(color=GRID, lw=0.8)
axA.set_axisbelow(True)
axA.legend(loc="upper left", frameon=False, ncol=5, fontsize=9.2, handlelength=1.6,
           columnspacing=1.2)
axA.set_xticks([pd.Timestamp("2026-10-29 22:00", tz="UTC"),
                pd.Timestamp("2026-10-30 12:00", tz="UTC"),
                pd.Timestamp("2026-10-31 12:00", tz="UTC"),
                pd.Timestamp("2026-11-01 00:00", tz="UTC")])
axA.set_xticklabels(["29.10", "30.10", "31.10", "01.11"])
axA.set_xlim(t.open_ts.min() - pd.Timedelta(hours=3), t.settle_ts.max() + pd.Timedelta(hours=10))
axA.set_ylim(0, 3.62)

# --- B: per-trade strip
axB = fig.add_subplot(gs[1, :2])
hyps = ["W03", "W04", "W05", "W06"]
for i, h in enumerate(hyps):
    g = t[t.hyp == h]
    jitter = RNG.uniform(-0.14, 0.14, len(g))
    axB.scatter(i + jitter, 100 * g.pnl, s=46, color=C[h], alpha=0.85,
                edgecolor="white", linewidth=0.7, zorder=3)
    axB.scatter([i], [100 * g.pnl.mean()], marker="D", s=95, color=INK,
                edgecolor="white", linewidth=1.0, zorder=5)
axB.axhline(0, color="#94a3b8", lw=1.0)
axB.annotate("−95¢", xy=(1, -95), xytext=(1.32, -80), fontsize=8.8, color=LOSS,
             arrowprops=dict(arrowstyle="->", color=LOSS, lw=0.9))
axB.set_xticks(range(4))
axB.set_xticklabels(hyps)
axB.set_ylim(-108, 26)
axB.set_ylabel("P&L одной сделки, ¢", fontsize=10.5)
axB.set_title("Сделки: почти все +5…+17¢, один убыток −95¢", loc="left", fontsize=11.0, pad=6)
axB.grid(axis="y", color=GRID, lw=0.8)
axB.set_axisbelow(True)
axB.text(0.99, 0.06, "◆ — среднее по стратегии", transform=axB.transAxes, ha="right",
         fontsize=8.4, color=MUTED)

# --- C: hit rate
axC = fig.add_subplot(gs[1, 2])
for i, h in enumerate(hyps):
    r = S[S.hyp == h].iloc[0]
    axC.bar(i, r.hit, width=0.62, color=C[h], alpha=0.9, edgecolor="white")
    axC.errorbar(i, r.hit, yerr=[[r.hit - r.hit_lo], [r.hit_hi - r.hit]],
                 fmt="none", ecolor="#334155", elinewidth=1.2, capsize=4, zorder=4)
    axC.text(i, 7, f"{int(r.wins)}/{int(r.n)}", ha="center", fontsize=8.8,
             color="white", fontweight="bold")
    axC.text(i, r.hit - 7, f"{r.hit:.1f}%", ha="center", va="top", fontsize=9.0,
             color="white", fontweight="bold")
axC.axhspan(89, 95, color="#e2e8f0", alpha=0.8, zorder=0)
axC.text(-0.46, 104, "серая полоса — бэктест 89–95 %", ha="left", fontsize=7.8, color=MUTED)
axC.set_xticks(range(4))
axC.set_xticklabels(hyps)
axC.set_ylim(0, 112)
axC.set_ylabel("Доля выигрышей, %", fontsize=10.0)
axC.set_title("Hit rate и 95 % ДИ", loc="left", fontsize=10.8, pad=6)
axC.grid(axis="y", color=GRID, lw=0.8)
axC.set_axisbelow(True)

# --- D: daily P&L
axD = fig.add_subplot(gs[2, 0])
dates = sorted(t.date.unique())
bottom_pos = np.zeros(len(dates))
bottom_neg = np.zeros(len(dates))
for h in hyps:
    vals = np.array([t[(t.date == d) & (t.hyp == h)].pnl.sum() for d in dates])
    base = np.where(vals >= 0, bottom_pos, bottom_neg)
    axD.bar(range(len(dates)), vals, bottom=base, width=0.62, color=C[h],
            edgecolor="white", linewidth=0.8, label=h)
    bottom_pos = bottom_pos + np.where(vals >= 0, vals, 0)
    bottom_neg = bottom_neg + np.where(vals < 0, vals, 0)
for i, d in enumerate(dates):
    tot_d = t[t.date == d].pnl.sum()
    axD.text(i, tot_d + (0.09 if tot_d >= 0 else -0.12), f"{tot_d:+.2f}",
             ha="center", va="bottom" if tot_d >= 0 else "top", fontsize=9.0, fontweight="bold",
             color=INK)
axD.axhline(0, color="#94a3b8", lw=1.0)
axD.set_xticks(range(len(dates)))
axD.set_xticklabels([d.strftime("%d.%m") for d in dates])
axD.set_ylabel("P&L за день, $", fontsize=10.0)
axD.set_title("По дням (UTC)", loc="left", fontsize=10.8, pad=6)
axD.set_ylim(-1.35, 2.15)
axD.grid(axis="y", color=GRID, lw=0.8)
axD.set_axisbelow(True)
axD.legend(frameon=False, fontsize=8.4, ncol=2, loc="upper right", handlelength=1.0,
           columnspacing=0.9)

# --- E: summary table
axE = fig.add_subplot(gs[2, 1:])
axE.axis("off")
axE.set_title("Сводка: 1 контракт, комиссия Kalshi внутри", loc="left", fontsize=10.8, pad=6)
cols = [0.02, 0.31, 0.46, 0.61, 0.77, 0.99]
headers = ["", "сделок", "hit", "ср. ¢", "итог $", "макс. DD"]
for x, htxt in zip(cols, headers):
    axE.text(x, 0.90, htxt, fontsize=8.6, color=MUTED, ha="right", transform=axE.transAxes)
axE.plot([0.01, 0.99], [0.845, 0.845], color="#cbd5e1", lw=0.9, transform=axE.transAxes)
for j, h in enumerate(hyps + ["ALL"]):
    r = S[S.hyp == h].iloc[0]
    y = 0.74 - j * 0.135
    col = C.get(h, INK)
    axE.text(cols[0], y, h if h != "ALL" else "ВСЕГО", fontsize=10.0, color=col,
             fontweight="bold" if h == "ALL" else "normal", transform=axE.transAxes)
    axE.text(cols[1], y, f"{int(r.n)}", fontsize=10.0, color=INK, ha="right", transform=axE.transAxes)
    axE.text(cols[2], y, f"{r.hit:.1f}%", fontsize=10.0, color=INK, ha="right", transform=axE.transAxes)
    axE.text(cols[3], y, f"{r.mean_c:+.1f}", fontsize=10.0, color=INK, ha="right", transform=axE.transAxes)
    axE.text(cols[4], y, f"{r.total:+.2f}", fontsize=10.0, color=INK, ha="right", transform=axE.transAxes)
    axE.text(cols[5], y, "—" if abs(r.max_dd) < 1e-9 else f"{r.max_dd:+.2f}",
             fontsize=10.0, color=INK, ha="right", transform=axE.transAxes)
axE.text(0.015, 0.02, "DD — максимальная просадка накопленного P&L; «—» = убытков не было",
         fontsize=8.2, color=MUTED, transform=axE.transAxes)

fig.suptitle(f"W03–W06 на Kalshi: {round((t.settle_ts.max()-t.open_ts.min()).total_seconds()/3600)} часа живого форвард-теста", x=0.075, y=0.975,
             ha="left", fontsize=17.5, fontweight="bold", color=INK)
fig.text(0.075, 0.947,
         f"{len(t)} сделок по 1 контракту · {t.open_ts.min():%d.%m %H:%M} – {t.settle_ts.max():%d.%m %H:%M} UTC · "
         f"{t.market.nunique()} рынков · P&L после комиссии · lab_2026*.log",
         fontsize=9.6, color=MUTED)
fig.text(0.075, 0.030,
         "Правила заморожены 29.10.2026: W03/W04 — тейкер, W05/W06 — мейкер-первый с фолбэком; "
         "все сделки удерживались до расчёта.",
         fontsize=8.4, color=MUTED)
fig.text(0.075, 0.013,
         "1 контракт, капитал на сделку ≈ цена покупки. 39 сделок — предварительно: "
         "вердикт по правилам требует ~месяца торговли.",
         fontsize=8.4, color=MUTED)
out = FIG / "fig19_wings_live.png"
fig.savefig(out, dpi=175)
plt.close(fig)
print("saved", out)


# ------------------------------------------------------------------ figure 20: history and expectations
eq = pd.read_csv(ROOT / "results" / "studies" / "wings_equity.csv")
eq["T"] = pd.to_datetime(eq["T"], utc=True)
eq.columns = ["T", "W02", "W03", "W04"]
imp = pd.read_csv(ROOT / "results" / "studies" / "wings_improve.csv")
imp.columns = ["win", "fair", "regime", "n_tr", "c_tr", "t_tr", "n_te", "c_te", "t_te"]
imp["window"] = np.where(imp.win.str.startswith("W03"), "W03", "W04")

fig = plt.figure(figsize=(13.8, 9.9))
gs = GridSpec(2, 2, height_ratios=[1.16, 1.0], hspace=0.24, wspace=0.26,
              left=0.075, right=0.975, top=0.90, bottom=0.075)

# --- historical equity
ax1 = fig.add_subplot(gs[0, :])
ax1.axvspan(pd.Timestamp("2026-09-01", tz="UTC"), eq["T"].max(), color="#eef4fb", zorder=0)
ax1.axhline(0, color="#94a3b8", lw=1.0)
ax1.plot(eq["T"], eq["W02"], color="#94a3b8", lw=1.8, ls=(0, (4, 2)), label="W02 (база, 2–6¢)")
ax1.plot(eq["T"], eq["W03"], color=C["W03"], lw=2.2, label="W03 (10–20¢, 5–15 мин)")
ax1.plot(eq["T"], eq["W04"], color=C["W04"], lw=2.2, label="W04 (6–10¢, 15–30 мин)")
for name, col, dy in [("W04", C["W04"], 0), ("W03", C["W03"], 0), ("W02", "#94a3b8", 0)]:
    ax1.annotate(f"{name} {eq[name].iloc[-1]:+.1f} $", (eq["T"].iloc[-1], eq[name].iloc[-1]),
                 xytext=(8, dy), textcoords="offset points", color=col, fontsize=9.4,
                 fontweight="bold", va="center")
ax1.axvline(pd.Timestamp("2026-10-29 20:00", tz="UTC"), color=INK, lw=1.0, ls=":")
ax1.text(pd.Timestamp("2026-10-29 20:00", tz="UTC"), -31, "  freeze /\n  live", fontsize=8.4,
         color=INK, va="bottom")
ax1.text(pd.Timestamp("2026-09-01 12:00", tz="UTC"), 24.5, "out-of-sample: сен–окт", fontsize=8.8,
         color="#64748b", va="top")
ax1.set_ylabel("Накопленный P&L, $ (1 контракт)", fontsize=10.5)
ax1.set_title("Бэктест июль–октябрь: W02 теряет, W03 и W04 зарабатывают", loc="left",
              fontsize=13.2, pad=8)
ax1.legend(loc="upper left", frameon=False, fontsize=9.2, ncol=3)
ax1.grid(color=GRID, lw=0.8)
ax1.set_axisbelow(True)
ax1.set_ylim(-38, 30)
ax1.set_xlim(eq["T"].min(), eq["T"].max() + pd.Timedelta(days=16))
ax1.xaxis.set_major_locator(mdates.MonthLocator())
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%m.%Y"))

# --- train vs test (W05/W06 study grid)
ax2 = fig.add_subplot(gs[1, 0])
markers = {"EWMA (как W03)": "o", "бустинг": "s", "среднее EWMA·бустинг": "D", "бустинг t4": "^"}
for w, col in [("W03", C["W03"]), ("W04", C["W04"])]:
    d = imp[imp.window == w]
    for fair, mk in markers.items():
        dd = d[d.fair == fair]
        ax2.scatter(dd.c_tr, dd.c_te, s=48, marker=mk, color=col, alpha=0.75,
                    edgecolor="white", linewidth=0.7, zorder=3)
ax2.plot([-2, 12], [-2, 12], color="#94a3b8", ls=(0, (4, 2)), lw=1.1, zorder=1)
chosen = imp[(imp.fair == "среднее EWMA·бустинг") & (imp.regime.str.startswith("no US"))]
for _, r in chosen.iterrows():
    ax2.scatter([r.c_tr], [r.c_te], s=150, facecolor="none", edgecolor=INK, lw=1.6, zorder=6)
ax2.annotate("W05: +4,6¢\n(t 2,9)", (8.29, 4.56), xytext=(-48, 16), textcoords="offset points",
             fontsize=8.4, color=INK,
             arrowprops=dict(arrowstyle="-", color=INK, lw=0.8))
ax2.annotate("W06: +3,5¢\n(t 4,5)", (2.65, 3.54), xytext=(-72, 14), textcoords="offset points",
             fontsize=8.4, color=INK,
             arrowprops=dict(arrowstyle="-", color=INK, lw=0.8))
ax2.set_xlabel("подбор: июль–авг, ¢", fontsize=9.8)
ax2.set_ylabel("проверка: сен–окт, ¢", fontsize=9.8)
ax2.set_title("W05/W06: сетка правил (ниже диагонали — проверка хуже подбора)", loc="left",
              fontsize=10.6, pad=6)
ax2.grid(color=GRID, lw=0.8)
ax2.set_axisbelow(True)
ax2.set_xlim(-1.5, 12.5)
ax2.set_ylim(-1.5, 6.5)
handles = [Line2D([], [], marker=mk, ls="", color="#64748b", markersize=6, label=fair)
           for fair, mk in markers.items()]
ax2.legend(handles=handles, loc="lower right", frameon=False, fontsize=7.6, ncol=2,
           handletextpad=0.3, columnspacing=0.8)


# --- backtest vs live
ax3 = fig.add_subplot(gs[1, 1])
bt = {"W03": 3.60, "W04": 1.80, "W05": 4.56, "W06": 3.54}
xs = np.arange(4)
b1 = ax3.bar(xs - 0.19, [bt[h] for h in ["W03", "W04", "W05", "W06"]],
             width=0.36, color="#cbd5e1", edgecolor="white", label="бэктест сен–окт")
b2 = ax3.bar(xs + 0.19, [S[S.hyp == h].mean_c.iloc[0] for h in ["W03", "W04", "W05", "W06"]],
             width=0.36, color=[C[h] for h in ["W03", "W04", "W05", "W06"]], edgecolor="white",
             label="живой тест")
for i, h in enumerate(["W03", "W04", "W05", "W06"]):
    r = S[S.hyp == h].iloc[0]
    ax3.text(i - 0.19, bt[h] + 0.28, f"{bt[h]:+.1f}", ha="center", fontsize=8.4, color=MUTED)
    ax3.text(i + 0.19, r.mean_c + 0.28, f"{r.mean_c:+.1f}", ha="center", fontsize=8.6,
             color=C[h], fontweight="bold")
ax3.axhline(0, color="#94a3b8", lw=1.0)
ax3.set_xticks(xs)
ax3.set_xticklabels(["W03", "W04", "W05", "W06"])
ax3.set_ylabel("¢ на сделку", fontsize=9.8)
ax3.set_ylim(-0.6, 15.5)
ax3.set_title("Бэктест vs live: ¢ на сделку (n = 8 / 21 / 5 / 5)", loc="left", fontsize=10.6, pad=6)
ax3.grid(axis="y", color=GRID, lw=0.8)
ax3.set_axisbelow(True)
handles3 = [Patch(facecolor="#cbd5e1", edgecolor="white", label="бэктест сен–окт"),
            Patch(facecolor="#334155", edgecolor="white", label="живой тест")]
ax3.legend(handles=handles3, frameon=False, fontsize=8.4, loc="upper right")

fig.suptitle("W03–W06: что обещал бэктест и что случилось вживую", x=0.075, y=0.965,
             ha="left", fontsize=17.0, fontweight="bold", color=INK)
fig.text(0.075, 0.935,
         "Сверху — история на 645–1 665 сделках (июль–октябрь); внизу — сетка W05/W06 и 39 живых сделок 29.10–01.11.2026",
         fontsize=9.6, color=MUTED)
fig.text(0.075, 0.018,
         "Бэктест: studies/wings_ladder.py, studies/wings_improve.csv (1 контракт, комиссия Kalshi). "
         "Живые средние — по 5–21 сделке на стратегию, поэтому доверительные интервалы широкие (см. отчёт).",
         fontsize=8.3, color=MUTED)
out2 = FIG / "fig20_wings_history.png"
fig.savefig(out2, dpi=175)
plt.close(fig)
print("saved", out2)


# ------------------------------------------------------------------ markdown report
def ru(v: float, nd: int = 1) -> str:
    return f"{v:.{nd}f}".replace(".", ",").replace("-", "−")


def pct(v: float, nd: int = 1) -> str:
    return ru(v, nd) + " %"


def ci(lo: float, hi: float, nd: int = 1) -> str:
    return f"{ru(lo, nd)}…{ru(hi, nd)}"


bt = {"W03": (3.60, 427, "сен–окт, тейкер, edge ≥ 1¢"),
      "W04": (1.80, 980, "сен–окт, тейкер, t4"),
      "W05": (4.56, 318, "сен–окт, тейкер, blend + no US open"),
      "W06": (3.54, 496, "сен–окт, тейкер, blend + no US open")}
cmp_rows = []
for h in ["W03", "W04", "W05", "W06"]:
    r = S[S.hyp == h].iloc[0]
    cmp_rows.append(dict(hyp=h, backtest_c=bt[h][0], backtest_n=bt[h][1], live_n=int(r.n),
                         live_mean_c=r.mean_c, live_lo_c=r.ci_lo_c, live_hi_c=r.ci_hi_c))
pd.DataFrame(cmp_rows).to_csv(TAB / "wings_live_vs_backtest.csv", index=False)

L: list[str] = []
L.append("# W03–W06: живой форвард-тест на Kalshi, 29.10–01.11.2026\n")
L.append(f"**Коротко.** {len(t)} сделок по 1 контракту, {int(t.pays.sum())} выигрышей и "
         f"{int((1 - t.pays).sum())} убыток, **+{ru(t.pnl.sum(), 2)} $** после комиссии "
         f"({ru(100 * t.pnl.mean(), 1)}¢ на сделку, {ru(100 * t.pnl.sum() / t.price.sum(), 1)} % на вложенный капитал). "
         "W04 ведёт себя как в бэктесте; W03/W05/W06 выглядят заметно лучше, но на 5–8 сделках это ещё не результат: "
         "одна потеря почти обнуляет запас W03 и уводит W05/W06 в минус.\n")
L.append(f"Период: {t.open_ts.min():%d.%m.%Y %H:%M} — {t.settle_ts.max():%d.%m.%Y %H:%M} UTC "
         f"(52 часа, {t.date.nunique()} календарных дня). Данные: `lab_20261029.log` … `lab_20261101.log`.\n")
L.append("![Живой форвард-тест](final/figures/fig19_wings_live.png)\n")
L.append("![История и ожидания](final/figures/fig20_wings_history.png)\n")
L.append("![Анатомия сделки и риск](final/figures/fig21_wings_risk.png)\n")
L.append("Слева: сделки с edge < 0,5¢ дали в среднем −4,8¢ (включая единственный убыток), с edge > 1,5¢ — +10,5¢. "
         "В центре: чем дешевле крыло, тем меньше премия за сделку. Справа: ещё один убыток по средней цене "
         "превращает W05 и W06 в минус.\n")

L.append("## По стратегиям\n")
L.append("| Стратегия | Сделок | Выигрышей | Hit rate, 95 % ДИ | Ср. P&L, ¢ (ДИ) | Итог, $ | Ср. выигрыш | Убыток | Profit factor | Макс. DD | ROI |")
L.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
for h in ["W03", "W04", "W05", "W06", "ALL"]:
    r = S[S.hyp == h].iloc[0]
    name = "**ВСЕГО**" if h == "ALL" else f"**{h}**"
    pf = "∞" if not np.isfinite(r.profit_factor) else ru(r.profit_factor, 2)
    loss = "—" if not np.isfinite(r.avg_loss_c) else ru(r.avg_loss_c, 1)
    dd = "—" if abs(r.max_dd) < 1e-9 else ru(r.max_dd, 2)
    L.append(f"| {name} | {int(r.n)} | {int(r.wins)} | {pct(r.hit)} ({ci(r.hit_lo, r.hit_hi)}) | "
             f"{ru(r.mean_c)} ({ci(r.ci_lo_c, r.ci_hi_c)}) | {ru(r.total, 2)} | {ru(r.avg_win_c)} | {loss} | "
             f"{pf} | {dd} | {ru(r.roi)} % |")
L.append("")
r_all = S.iloc[-1]
L.append(f"Разброс P&L по всем сделкам: медиана {ru(r_all.median_c)}¢, минимум {ru(r_all.min_c)}¢, "
         f"максимум {ru(r_all.max_c)}¢; среднее {ru(r_all.mean_c)}¢.\n")
L.append("«ДИ» у среднего — не обычный t-интервал, а симуляция с риском ещё не встреченного убытка: "
         "вероятность выигрыша ~ Beta(выигрыши+½, убытки+½), размер убытка — фактическая цена покупки + комиссия. "
         "Для 5/5 и 8/8 это честнее, чем узкий интервал, который «не видит» хвост.\n")

L.append("## По дням (UTC)\n")
L.append("| Дата | Сделок | Выигрышей | P&L, $ | Накопленно, $ |")
L.append("|---|---:|---:|---:|---:|")
for _, r in daily.iterrows():
    L.append(f"| {r.date:%d.%m} | {int(r.n)} | {int(r.wins)} | {ru(r.pnl, 2)} | {ru(r.cum, 2)} |")
L.append("")

L.append("## Разрезы\n")
L.append("| Разрез | Сделок | Hit | Ср. P&L, ¢ | Итог, $ |")
L.append("|---|---:|---:|---:|---:|")
labels = {"cheap": "Цена крыла", "edge": "Edge (оценка − цена)", "exec": "Исполнение", "side": "Наша сторона"}
for _, r in B.iterrows():
    key = r.bucket
    L.append(f"| {labels[r.kind]}: {key} | {int(r.n)} | {pct(100 * r.wins / r.n)} | "
             f"{ru(r.mean_c)} | {ru(r.total, 2)} |")
L.append("")
L.append("Крайняя левая колонка — цена «дешёвой» стороны, которую мы продаём (покупая противоположную). "
         "У W03/W05 это 10–20¢, у W04/W06 — 6–10¢.\n")

one_loss = {}
for h in ["W03", "W05", "W06"]:
    g = t[t.hyp == h]
    one_loss[h] = (g.pnl.sum() - (g.price.mean() + g.fee.mean())) / (len(g) + 1) * 100
L.append("## Что это значит\n")
L.append(f"1. **Хит-рейт {pct(100 * t.pays.mean())}** ({int(t.pays.sum())}/{len(t)}, 95 % ДИ "
         f"{ci(S.iloc[-1].hit_lo, S.iloc[-1].hit_hi)}): "
         "выборка согласуется с бэктестовыми 89–95 %, но интервал накрывает и 90 %, и 99 %.\n")
L.append(f"2. **Весь убыток — одна сделка W04** 01.11 в 00:35 UTC: страйк 77 499,99, NO не исполнился, "
         f"−0,95 $. После неё накопленный P&L упал с 3,29 до 2,34 $.\n")
L.append(f"3. **W04 ближе всех к бэктесту:** {ru(S[S.hyp == 'W04'].mean_c.iloc[0])}¢ против +1,8¢ на 980 сделках "
         f"проверки; хит-рейт 95,2 % против 94 %. Это единственная стратегия с достаточным числом живых сделок, "
         f"чтобы говорить о совпадении, а не о везении.\n")
L.append(f"4. **W03/W05/W06 — идеальные серии 18/18,** средние +7…+13¢. Но если бы в каждой из них случился "
         f"один убыток, средние стали бы {ru(one_loss['W03'])} / {ru(one_loss['W05'])} / {ru(one_loss['W06'])}¢. "
         "Именно поэтому их нельзя считать подтверждением: при бэктестовой частоте выигрыша ожидание W03 ≈ +1,3¢, "
         "а не +12¢.\n")
L.append(f"5. **Edge работает как фильтр:** сделки с edge < 0,5¢ дали {ru(B[(B.kind == 'edge') & (B.bucket == '< 0,5¢')].mean_c.iloc[0])}¢ "
         f"в среднем (включая единственный убыток), с edge > 1,5¢ — "
         f"{ru(B[(B.kind == 'edge') & (B.bucket == '> 1,5¢')].mean_c.iloc[0])}¢. "
         "Знак правильный, но это 39 наблюдений.\n")
L.append(f"6. **Мейкер-первый работает:** 10/10 мейкерских сделок W05/W06 выиграли, "
         f"{ru(100 * t[t.exec == 'maker'].pnl.mean())}¢ на сделку против "
         f"{ru(100 * t[t.exec == 'taker'].pnl.mean())}¢ у тейкера. Экономия на спреде — главное объяснение, "
         "почему live W05/W06 выше тейкеровского бэктеста (+4,6 / +3,5¢).\n")

L.append("## Бэктест vs живой тест\n")
L.append("| Стратегия | Бэктест сен–окт, ¢ | Сделок | Живой тест, ¢ | 95 % ДИ | Сделок |")
L.append("|---|---:|---:|---:|---:|---:|")
for h in ["W03", "W04", "W05", "W06"]:
    r = S[S.hyp == h].iloc[0]
    L.append(f"| {h} | {ru(bt[h][0], 2)} | {bt[h][1]} | {ru(r.mean_c)} | {ci(r.ci_lo_c, r.ci_hi_c)} | {int(r.n)} |")
L.append("")
L.append("W03/W04 — исторические значения из `results/studies/wings_summary.md`; W05/W06 — лучшие строки "
         "`wings_improve.csv` (blend + no US open, тейкер). Живые W05/W06 — мейкерские, поэтому сравнивать нужно "
         "с поправкой на спред.\n")

L.append("## Все сделки\n")
L.append("| # | Открытие (UTC) | Стратегия | Рынок | Сторона | Цена покупки | P&L, ¢ |")
L.append("|---:|---|---|---|---:|---:|---:|")
for i, (_, r) in enumerate(t.sort_values("open_ts").iterrows(), 1):
    L.append(f"| {i} | {r.open_ts:%d.%m %H:%M} | {r.hyp} | {r.market} | {r.side} | "
             f"{ru(100 * r.price)} | {ru(100 * r.pnl)} |")
L.append("")

L.append("## Бэктест по месяцам (для контекста)\n")
L.append("| Стратегия | июл | авг | сен | окт | всего |")
L.append("|---|---:|---:|---:|---:|---:|")
for h, vals in [("W02", [-4.7, -10.9, -18.8, 10.9]), ("W03", [0.7, 4.1, 9.6, 6.0]),
                ("W04", [8.9, -0.6, 9.8, 7.5])]:
    L.append(f"| {h} | " + " | ".join(ru(v, 1) for v in vals) + f" | {ru(sum(vals), 1)} |")
L.append("")
L.append("Источник: `results/studies/wings_summary.md` (1 контракт, комиссия Kalshi, июль–октябрь 2026). "
         "У W03/W04 сентябрь–октябрь — out-of-sample; живой тест начался 29.10.\n")

L.append("## Целостность данных\n")
L.append(f"- 39 строк `OPEN`/`WINGS` и 39 `SETTLE`, без незакрытых позиций; пересчёт P&L по цене, исходу и комиссии "
         f"совпал с логом во всех 39 случаях.\n"
         f"- 28 уникальных рынков; 8 рынков торговались 2–4 стратегиями одновременно (W03∩W05, W04∩W06, W03∩W04, W05∩W06), "
         f"поэтому исходы не независимы.\n"
         f"- За 52 часа в логах {feed_errors} ошибок `markets kalshi failed: HTTP 403` — в эти минуты сигналы "
         f"могли не считаться, так что «отсутствие сделки» не равно «не было сигнала».\n"
         f"- Комиссия: тейкер 7 %·p(1−p), мейкер 1,75 %·p(1−p), округление вверх до цента. На этих ценах — 1¢ "
         f"с обеих сторон.\n"
         f"- Время удержания в логах иногда больше часа (задержка распознавания расчёта), поэтому в статистике "
         f"использовано время открытия, а не длительность.\n")

L.append("## Что дальше\n")
L.append("- Продолжать тест до ~200 сделок на стратегию (примерно месяц): только тогда нижняя граница ДИ среднего перестанет накрывать ноль.\n"
         "- W04 уже ведёт себя как в бэктесте (+1,6¢ против +1,8¢); W03/W05/W06 пока нельзя отличить от «повезло» — 18/18 без единого убытка.\n"
         "- Отдельно проверить мейкерские исполнения W05/W06 на реальной бирже: в бумажном режиме очередь учитывается моделью, но не проверена живыми ордерами.\n")

L.append("## Ограничения\n")
L.append("1. **39 сделок за 52 часа** — это не вердикт. По плану (`wings_summary.md`) для вердикта нужен ~месяц.\n"
         "2. Один убыток на 39 сделок — редкое событие; его оценка хвоста опирается на модель, а не на данные.\n"
         "3. W05/W06 — мейкерские: 10 сделок, все выиграли; реальные очереди и частичные исполнения в бумажном "
         "режиме учтены моделью, но не проверены на бирже.\n"
         "4. Период совпал с несколькими сбоями Kalshi (403), что могло отфильтровать часть сигналов.\n")

L.append("## Файлы\n")
L.append("- `reports/wings_live.py` — парсер логов, статистика и графики.\n"
         "- `reports/final/figures/fig19_wings_live.png` — дашборд живого теста.\n"
         "- `reports/final/figures/fig20_wings_history.png` — бэктест, сетка W05/W06 и сравнение с живыми.\n"
         "- `reports/final/tables/wings_live_trades.csv` — все 39 сделок.\n"
         "- `reports/final/tables/wings_live_summary.csv` — сводка по стратегиям.\n"
         "- `reports/final/tables/wings_live_daily.csv`, `wings_live_buckets.csv`, `wings_live_vs_backtest.csv`.\n"
         "- `reports/final/tables/wings_live_summary.json` — машинная сводка.\n")
(ROOT / "reports" / "wings_live.md").write_text("\n".join(L))
print("saved reports/wings_live.md")


# ------------------------------------------------------------------ figure 21: anatomy and risk
fig = plt.figure(figsize=(14.0, 4.9))
gs = GridSpec(1, 3, wspace=0.30, left=0.065, right=0.985, top=0.80, bottom=0.16)

# A: edge -> pnl (W03/W04 only, they have an edge estimate)
axA = fig.add_subplot(gs[0, 0])
d = t[t.edge.notna()]
for h in ["W03", "W04"]:
    g = d[d.hyp == h]
    axA.scatter(100 * g.edge, 100 * g.pnl, s=52, color=C[h], alpha=0.85,
                edgecolor="white", linewidth=0.7, label=h, zorder=3)
loss_d = d[d.pays == 0]
if len(loss_d):
    axA.annotate("−95¢", (100 * loss_d.edge.iloc[0], 100 * loss_d.pnl.iloc[0]),
                 xytext=(14, 18), textcoords="offset points", fontsize=8.8, color=LOSS,
                 arrowprops=dict(arrowstyle="->", color=LOSS, lw=0.9))
axA.axhline(0, color="#94a3b8", lw=1.0)
axA.set_xlabel("edge = оценка − цена, ¢", fontsize=9.6)
axA.set_ylabel("P&L сделки, ¢", fontsize=9.6)
axA.set_title("Больше edge — больше P&L (W03/W04)", loc="left", fontsize=10.6, pad=6)
axA.grid(color=GRID, lw=0.8)
axA.set_axisbelow(True)
axA.legend(frameon=False, fontsize=8.6, loc="lower right")


# B: pnl by cheap wing bucket
axB = fig.add_subplot(gs[0, 1])
bb = B[B.kind == "cheap"].reset_index(drop=True)
axB.bar(range(len(bb)), bb.mean_c, width=0.6, color=[AMBER_B for _ in bb], edgecolor="white")
for i, r in bb.iterrows():
    axB.text(i, r.mean_c + 0.5, f"{r.mean_c:+.1f}¢", ha="center", fontsize=9.0, fontweight="bold",
             color=INK)
    axB.text(i, -1.4, f"n={int(r.n)}", ha="center", fontsize=8.4, color=MUTED)
axB.axhline(0, color="#94a3b8", lw=1.0)
axB.set_xticks(range(len(bb)))
axB.set_xticklabels(bb.bucket)
axB.set_ylim(-2.2, 19)
axB.set_ylabel("Ср. P&L, ¢", fontsize=9.6)
axB.set_xlabel("цена «дешёвой» стороны", fontsize=9.6)
axB.set_title("Дешёвые крылья дают меньше", loc="left", fontsize=10.6, pad=6)
axB.grid(axis="y", color=GRID, lw=0.8)
axB.set_axisbelow(True)

# C: one more loss
axC = fig.add_subplot(gs[0, 2])
for i, h in enumerate(["W03", "W04", "W05", "W06"]):
    g = t[t.hyp == h]
    now = 100 * g.pnl.mean()
    one = (g.pnl.sum() - (g.price.mean() + g.fee.mean())) / (len(g) + 1) * 100
    axC.plot([one, now], [i, i], color="#cbd5e1", lw=2.2, zorder=1)
    axC.scatter([one], [i], s=60, facecolor="white", edgecolor=LOSS, lw=1.6, zorder=3)
    axC.scatter([now], [i], s=70, color=C[h], zorder=4)
    axC.text(now + 0.9, i, f"{now:+.1f}", va="center", fontsize=8.6, color=C[h], fontweight="bold")
    axC.text(one - 0.9, i, f"{one:+.1f}", va="center", ha="right", fontsize=8.4, color=LOSS)
axC.axvline(0, color=INK, lw=1.0)
axC.set_yticks(range(4))
axC.set_yticklabels(["W03", "W04", "W05", "W06"])
axC.set_ylim(-0.6, 3.6)
axC.set_xlim(-13, 17)
axC.set_xlabel("средний P&L, ¢", fontsize=9.6)
axC.set_title("Одна потеря съедает весь запас", loc="left", fontsize=10.6, pad=6)
axC.grid(axis="x", color=GRID, lw=0.8)
axC.set_axisbelow(True)
axC.text(-12.4, 3.45, "○ — если бы был ещё один убыток", fontsize=7.6, color=MUTED, va="top")

fig.suptitle("Где живёт результат: edge, цена крыла и цена одной ошибки", x=0.065, y=0.95,
             ha="left", fontsize=15.0, fontweight="bold", color=INK)
fig.text(0.065, 0.895,
         "Слева — только W03/W04 (у W05/W06 мейкерский edge не записан в лог). "
         "В центре — все 39 сделок. Справа — гипотетический ещё один убыток по средней цене стратегии.",
         fontsize=8.8, color=MUTED)
out3 = FIG / "fig21_wings_risk.png"
fig.savefig(out3, dpi=175)
plt.close(fig)
print("saved", out3)
