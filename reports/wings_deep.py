"""Wing selling on Kalshi, all six rules: live paper test, history, risk and graphics.

Parses `lab_2026*.log` (root) for W01–W06, recomputes/validates P&L, joins the July–October
backtest (`results/studies/wings_equity.csv`, `wings_improve.csv`) and draws three posters:

  fig22_wings_live.png     live test, all six rules, 29.10–01.11.2026
  fig23_wings_history.png  backtest July–October, monthly, rolling, W05/W06 grid
  fig24_wings_risk.png     bootstrap, P(profit) vs horizon, max-DD, compounding, one-loss

Run:  python reports/wings_deep.py
"""
from __future__ import annotations

import base64
import glob
import html
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
from matplotlib.ticker import FuncFormatter, LogLocator

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "reports" / "final" / "figures"
TAB = ROOT / "reports" / "final" / "tables"
FIG.mkdir(parents=True, exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)

INK = "#0b1220"
MUTED = "#5b6472"
GRID = "#e8edf5"
LOSS = "#dc2626"
C = {"W01": "#94a3b8", "W02": "#475569", "W03": "#4f46e5",
     "W04": "#0d9488", "W05": "#d97706", "W06": "#db2777"}
HYP = list(C)
RNG = np.random.default_rng(20261101)
Z = 1.959963984540054

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10.2,
    "axes.edgecolor": "#cbd5e1", "axes.linewidth": 1.0,
    "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.titleweight": "bold",
})


# ------------------------------------------------------------------ parse logs
pat_open = re.compile(r"^\[(?P<ts>[^\]]+)\] OPEN (?P<hyp>W0[1-6]) (?P<side>yes|no) (?P<market>\S+) "
                      r"@ (?P<price>[\d.]+)(?: edge (?P<edge>-?[\d.]+))?")
pat_wing = re.compile(r"^\[(?P<ts>[^\]]+)\] WINGS (?P<hyp>W0[5-6]) (?P<exec>maker|taker) "
                      r"sold (?P<cheap>yes|no) (?P<market>\S+) @ (?P<cprice>[\d.]+)")
pat_settle = re.compile(r"^\[(?P<ts>[^\]]+)\] SETTLE (?P<hyp>W0[1-6]) (?P<market>\S+) "
                        r"pays (?P<pays>[01]) pnl (?P<pnl>[+-][\d.]+)")

opens: dict[tuple[str, str], dict] = {}
trades: list[dict] = []
orphan_settles: list[dict] = []
feed_errors = 0
for f in sorted(glob.glob(str(ROOT / "lab_2026*.log"))):
    for line in open(f, encoding="utf-8", errors="replace"):
        line = line.strip()
        if "markets kalshi failed" in line:
            feed_errors += 1
        m = pat_open.match(line)
        if m:
            d = m.groupdict()
            opens[(d["hyp"], d["market"])] = dict(
                ts=d["ts"], side=d["side"], price=float(d["price"]),
                edge=float(d["edge"]) if d["edge"] else np.nan,
                exec="taker", cheap=None, cprice=None)
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
            if o is None:                       # Polymarket names are truncated in SETTLE lines
                cands = [k for k in opens if k[0] == d["hyp"] and k[1].startswith(d["market"])]
                if len(cands) == 1:
                    o = opens.pop(cands[0])
            if o is None:
                orphan_settles.append(d)
                continue
            trades.append(dict(hyp=d["hyp"], market=d["market"], open_ts=o["ts"], settle_ts=d["ts"],
                               side=o["side"], price=o["price"], edge=o["edge"], exec=o["exec"],
                               cheap=o["cheap"], cprice=o["cprice"], pays=int(d["pays"]),
                               pnl=float(d["pnl"])))

t = pd.DataFrame(trades)
t["open_ts"] = pd.to_datetime(t["open_ts"], utc=True)
t["settle_ts"] = pd.to_datetime(t["settle_ts"], utc=True)
t["cheap"] = 1 - t["price"]
t["date"] = t["open_ts"].dt.date
t["coverage"] = np.where(t["hyp"].isin(["W01", "W02"]), "29.10", "52 ч")
print(f"{len(t)} live trades: " + ", ".join(f"{h}={int((t.hyp == h).sum())}" for h in HYP))
print(f"orphan settles {len(orphan_settles)}, unmatched opens {len(opens)}, feed 403 {feed_errors}")

# Kalshi-only rules: verify the logged P&L against price, outcome and fee
kal = t[t.hyp.isin(["W03", "W04", "W05", "W06"])].copy()
fee = np.where(kal.exec == "maker",
               np.ceil(np.round(0.0175 * kal.price * (1 - kal.price) * 100, 6)) / 100,
               np.ceil(np.round(0.07 * kal.price * (1 - kal.price) * 100, 6)) / 100)
recalc = kal.pays - kal.price - fee
print("Kalshi pnl max deviation from log:", round(float((recalc - kal.pnl).abs().max()), 6))


# ------------------------------------------------------------------ statistics
def wilson(k: int, n: int) -> tuple[float, float]:
    p = k / n
    den = 1 + Z * Z / n
    c = (p + Z * Z / (2 * n)) / den
    h = Z * math.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)) / den
    return 100 * max(0.0, c - h), 100 * min(1.0, c + h)


def risk_ci(g: pd.DataFrame, reps: int = 30000) -> tuple[float, float]:
    """Mean P&L CI with the loss tail priced in (Beta posterior over the hit rate)."""
    w = g.loc[g.pays == 1, "pnl"].values
    l = g.loc[g.pays == 0, "pnl"].values
    wm = w.mean() if len(w) else 0.0
    lm = l.mean() if len(l) else float(-(g.price.mean() + 0.01))
    p = RNG.beta(len(w) + 0.5, len(l) + 0.5, reps)
    k = RNG.binomial(len(g), p)
    means = (k * wm + (len(g) - k) * lm) / len(g)
    return 100 * np.percentile(means, 2.5), 100 * np.percentile(means, 97.5)


rows = []
for h in HYP:
    g = t[t.hyp == h]
    n, k = len(g), int(g.pays.sum())
    lo, hi = wilson(k, n)
    cl, ch = risk_ci(g)
    s = g.sort_values("settle_ts")
    cum = s.pnl.cumsum()
    dd = float((cum - cum.cummax()).min())
    sd = g.pnl.std(ddof=1)
    rows.append(dict(
        hyp=h, n=n, wins=k, losses=n - k, hit=100 * k / n, hit_lo=lo, hit_hi=hi,
        mean_c=100 * g.pnl.mean(), ci_lo_c=cl, ci_hi_c=ch, total=g.pnl.sum(), sd_c=100 * sd,
        avg_win_c=100 * g.loc[g.pays == 1, "pnl"].mean() if k else np.nan,
        avg_loss_c=100 * g.loc[g.pays == 0, "pnl"].mean() if n > k else np.nan,
        pf=(g.loc[g.pnl > 0, "pnl"].sum() / abs(g.loc[g.pnl < 0, "pnl"].sum())
            if (g.pnl < 0).any() else np.nan),
        max_dd=dd, collateral=g.price.sum(), roi=100 * g.pnl.sum() / g.price.sum(),
        avg_cheap_c=100 * g.cheap.mean(), avg_edge_c=100 * g.edge.mean() if g.edge.notna().any() else np.nan,
        maker=bool((g.exec == "maker").all()), t0=g.open_ts.min(), t1=g.open_ts.max(),
        coverage=g.coverage.iloc[0],
    ))
S = pd.DataFrame(rows)
S.to_csv(TAB / "wings_all_live_summary.csv", index=False)
t.sort_values("open_ts").to_csv(TAB / "wings_all_live_trades.csv", index=False)
print(S[["hyp", "n", "hit", "mean_c", "total", "max_dd"]].round(2).to_string(index=False))


# ------------------------------------------------------------------ history
eq = pd.read_csv(ROOT / "results" / "studies" / "wings_equity.csv")
eq["T"] = pd.to_datetime(eq["T"], utc=True)
eq.columns = ["T", "W02", "W03", "W04"]
eq = eq.set_index("T").sort_index()
monthly = eq.resample("ME").last().diff()
monthly.iloc[0] = eq.resample("ME").last().iloc[0]
hourly = eq.diff().dropna()
hist_rows = []
for h in ["W02", "W03", "W04"]:
    d = hourly[h]
    active = d[d != 0]
    hist_rows.append(dict(
        hyp=h, hours=len(active), wins=int((active > 0).sum()), losses=int((active < 0).sum()),
        hit=100 * (active > 0).mean(), mean_c=100 * active.mean(), total=eq[h].iloc[-1],
        max_dd=float((eq[h] - eq[h].cummax()).min()),
        worst_month=float(monthly[h].min()), best_month=float(monthly[h].max()),
        oos_c=float(eq[h].iloc[-1] - eq[h][eq.index < pd.Timestamp("2026-09-01", tz="UTC")].iloc[-1]),
    ))
H = pd.DataFrame(hist_rows)
H.to_csv(TAB / "wings_all_history_summary.csv", index=False)
monthly.to_csv(TAB / "wings_all_history_monthly.csv")
print(H.round(2).to_string(index=False))

imp = pd.read_csv(ROOT / "results" / "studies" / "wings_improve.csv")
imp.columns = ["win", "fair", "regime", "n_tr", "c_tr", "t_tr", "n_te", "c_te", "t_te"]
imp["window"] = np.where(imp.win.str.startswith("W03"), "W03", "W04")
imp.to_csv(TAB / "wings_all_improve.csv", index=False)

comp = pd.DataFrame([
    dict(rule="W03", f=5, base=222, h1=177, h2=143, dd=-19),
    dict(rule="W03", f=10, base=458, h1=285, h2=172, dd=-37),
    dict(rule="W04", f=5, base=222, h1=137, h2=86, dd=-25),
    dict(rule="W04", f=10, base=468, h1=159, h2=57, dd=-45),
    dict(rule="W03+W04", f=5, base=516, h1=243, h2=110, dd=-34),
])
comp.to_csv(TAB / "wings_all_compounding.csv", index=False)

risk = S[["hyp", "n", "hit", "mean_c", "sd_c", "pf", "max_dd", "roi"]].copy()
risk["sharpe_trade"] = risk.mean_c / risk.sd_c
risk.to_csv(TAB / "wings_all_risk.csv", index=False)
print(risk.round(2).to_string(index=False))


# ================================================================== FIG 22 — live
def money(x, _=None):
    return f"{x:+.0f}" if abs(x) >= 1 else f"{x:+.1f}"


fig = plt.figure(figsize=(14.6, 13.4))
gs = GridSpec(3, 3, figure=fig, hspace=0.55, wspace=0.30,
              left=0.055, right=0.975, top=0.895, bottom=0.055)
fig.text(0.055, 0.958, "Продажа крыльев на Kalshi: все шесть правил в paper-тесте",
         fontsize=21, fontweight="bold", color=INK)
fig.text(0.055, 0.928,
         "171 сделка · 29.10–01.11.2026 · W01/W02 торговали только 29.10, W03–W06 — 52 часа · "
         "P&L на 1 контракт, включая комиссию",
         fontsize=10.6, color=MUTED)

# --- A: W01 / W02 cumulative
ax = fig.add_subplot(gs[0, 0])
for h in ["W01", "W02"]:
    g = t[t.hyp == h].sort_values("settle_ts")
    ax.plot(g.settle_ts, g.pnl.cumsum(), color=C[h], lw=2.2, label=h)
    ax.scatter(g.settle_ts, g.pnl.cumsum(), s=9, color=C[h], zorder=3)
    ax.annotate(f"{h} {g.pnl.sum():+.2f} $", (g.settle_ts.iloc[-1], g.pnl.sum()),
                xytext=(6, 0), textcoords="offset points", va="center",
                fontsize=9.4, fontweight="bold", color=C[h])
ax.axhline(0, color="#94a3b8", lw=0.9)
ax.set_title("W01 и W02: один день, 29.10", loc="left", fontsize=12)

ax.set_ylabel("$ на контракт")
ax.set_ylim(-8.9, 1.7)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.grid(axis="y", color=GRID)

# --- B: W03–W06 cumulative
ax = fig.add_subplot(gs[0, 1])
dy = {"W03": 9, "W04": -13, "W05": 2, "W06": 9}
for h in ["W03", "W04", "W05", "W06"]:
    g = t[t.hyp == h].sort_values("settle_ts")
    ax.step(g.settle_ts, g.pnl.cumsum(), where="post", color=C[h], lw=2.0, label=h)
    ax.annotate(f"{h} {g.pnl.sum():+.2f}", (g.settle_ts.iloc[-1], g.pnl.sum()),
                xytext=(5, dy[h]), textcoords="offset points", fontsize=8.8,
                fontweight="bold", color=C[h])
ax.axhline(0, color="#94a3b8", lw=0.9)
ax.set_title("W03–W06: 52 часа live", loc="left", fontsize=12)
ax.text(0.02, 0.95, "суммарно +2,34 $ · 39 сделок · 1 проигрыш",
        transform=ax.transAxes, fontsize=8.8, color=MUTED, va="top")
ax.set_xlim(pd.Timestamp("2026-10-29 19:00", tz="UTC"), pd.Timestamp("2026-11-01 04:00", tz="UTC"))
ax.set_ylabel("$ на контракт")
ax.set_xticks([pd.Timestamp("2026-10-29 21:00", tz="UTC"), pd.Timestamp("2026-10-30 12:00", tz="UTC"),
               pd.Timestamp("2026-10-31 12:00", tz="UTC"), pd.Timestamp("2026-11-01 00:00", tz="UTC")])
ax.set_xticklabels(["29.10", "30.10", "31.10", "01.11"])
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.grid(axis="y", color=GRID)

# --- C: table
ax = fig.add_subplot(gs[0, 2])
ax.axis("off")
ax.set_title("Сводка live", loc="left", fontsize=12)
cols = [0.00, 0.34, 0.57, 0.79, 1.00]
heads = ["правило", "n", "hit", "ср.¢", "итог $"]
for x, htxt in zip(cols[1:], heads[1:]):
    ax.text(x, 0.94, htxt, ha="right", va="center", fontsize=8.6, color=MUTED, fontweight="bold",
            transform=ax.transAxes)
ax.text(cols[0], 0.94, heads[0], ha="left", va="center", fontsize=8.6, color=MUTED,
        fontweight="bold", transform=ax.transAxes)
ax.plot([0, 1], [0.89, 0.89], color=GRID, lw=1.4, transform=ax.transAxes)
for i, r in S.iterrows():
    y = 0.80 - i * 0.115
    ax.add_patch(plt.Rectangle((0, y - 0.045), 1, 0.09, color=C[r.hyp], alpha=0.07,
                               transform=ax.transAxes, lw=0))
    ax.text(cols[0], y, r.hyp, fontsize=10, fontweight="bold", color=C[r.hyp], va="center",
            transform=ax.transAxes)
    ax.text(cols[1], y, f"{int(r.n)}", ha="right", va="center", fontsize=9.6, transform=ax.transAxes)
    ax.text(cols[2], y, f"{r.hit:.0f}%", ha="right", va="center", fontsize=9.6, transform=ax.transAxes)
    ax.text(cols[3], y, f"{r.mean_c:+.1f}", ha="right", va="center", fontsize=9.6,
            color=LOSS if r.mean_c < 0 else INK, fontweight="bold", transform=ax.transAxes)
    ax.text(cols[4], y, f"{r.total:+.2f}", ha="right", va="center", fontsize=9.6,
            color=LOSS if r.total < 0 else INK, transform=ax.transAxes)
ax.text(0, 0.02, "† W01/W02 — частичное покрытие 29.10;\n   W05/W06 — maker, W01–W04 — taker.",
        fontsize=7.6, color=MUTED, va="bottom", transform=ax.transAxes)

# --- D: mean ± CI
ax = fig.add_subplot(gs[1, 0])
ys = np.arange(len(S))[::-1]
for y, (_, r) in zip(ys, S.iterrows()):
    ax.errorbar(r.mean_c, y, xerr=[[r.mean_c - r.ci_lo_c], [r.ci_hi_c - r.mean_c]],
                fmt="o", color=C[r.hyp], ecolor=C[r.hyp], elinewidth=2.0,
                capsize=0, ms=8, alpha=0.95)
    ax.text(r.ci_hi_c + 0.7, y, f"{r.mean_c:+.1f}", va="center", fontsize=9,
            color=INK, fontweight="bold")
ax.axvline(0, color=LOSS, lw=1.1, ls="--", alpha=0.8)
ax.set_yticks(ys)
ax.set_yticklabels(S.hyp)
ax.set_xlim(-14.5, 21)
ax.set_title("Средний P&L сделки, ¢ (95 % интервал)", loc="left", fontsize=12)
ax.grid(axis="x", color=GRID)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)

# --- E: hit rate
ax = fig.add_subplot(gs[1, 1])
xs = np.arange(len(S))
for x, (_, r) in zip(xs, S.iterrows()):
    ax.bar(x, r.hit, width=0.66, color=C[r.hyp], alpha=0.9)
    ax.errorbar(x, r.hit, yerr=[[r.hit - r.hit_lo], [r.hit_hi - r.hit]],
                fmt="none", ecolor="#334155", elinewidth=1.2, capsize=3)
    ax.text(x, r.hit + 1.0, f"{r.hit:.0f}", ha="center", fontsize=8.6, color=INK)
ax.set_xticks(xs)
ax.set_xticklabels(S.hyp, fontsize=9)
ax.set_ylim(80, 103)
ax.set_ylabel("%")
ax.set_title("Доля выигрышных сделок (Wilson 95 %)", loc="left", fontsize=12)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.grid(axis="y", color=GRID)

# --- F: risk-return
ax = fig.add_subplot(gs[1, 2])
for _, r in S.iterrows():
    ax.scatter(r.avg_win_c, r.hit, s=28 + 2.2 * r.n, color=C[r.hyp], alpha=0.85,
               edgecolor="white", linewidth=1.4, zorder=3)
offs = {"W01": (8, -12), "W02": (-6, 9), "W03": (-24, 7), "W04": (-24, -10),
        "W05": (7, -11), "W06": (-4, 8)}
for _, r in S.iterrows():
    ax.annotate(r.hyp, (r.avg_win_c, r.hit), xytext=offs[r.hyp],
                textcoords="offset points", fontsize=9, fontweight="bold", color=C[r.hyp])
ax.set_xlim(0, 16)
ax.set_ylim(86, 103)
ax.set_xlabel("средний выигрыш, ¢")
ax.set_ylabel("hit, %")
ax.set_title("Хит-рейт против размера выигрыша", loc="left", fontsize=12)
ax.text(0.03, 0.06, "W01: 90 % попаданий, но +2,5¢ против −95¢",
        transform=ax.transAxes, fontsize=8.2, color=LOSS)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.grid(color=GRID)

# --- G: daily P&L
ax = fig.add_subplot(gs[2, 0])
days = sorted(t.date.unique())
xs = np.arange(len(days))
pos = np.zeros(len(days)); neg = np.zeros(len(days))
for h in HYP:
    vals = np.array([t[(t.hyp == h) & (t.date == d)].pnl.sum() for d in days])
    p = np.where(vals > 0, vals, 0); n = np.where(vals < 0, vals, 0)
    ax.bar(xs, p, bottom=pos, color=C[h], width=0.62, label=h, alpha=0.95)
    ax.bar(xs, n, bottom=neg, color=C[h], width=0.62, alpha=0.95)
    pos += p; neg += n
for x, tot in zip(xs, pos + neg):
    ax.text(x, tot + (0.18 if tot >= 0 else -0.35), f"{tot:+.2f}", ha="center",
            fontsize=8.8, fontweight="bold", color=LOSS if tot < 0 else INK)
ax.axhline(0, color="#94a3b8", lw=0.9)
ax.set_xticks(xs)
ax.set_xticklabels([d.strftime("%d.%m") for d in days])
ax.set_ylim(-8.6, 2.8)
ax.set_ylabel("$")
ax.set_title("P&L по дням", loc="left", fontsize=12)
ax.text(0.03, 0.06, "29.10: W01 −7,18 $, W02 +1,06 $",
        transform=ax.transAxes, fontsize=7.8, color=MUTED)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.grid(axis="y", color=GRID)

# --- H: W01 / W02 drawdown
ax = fig.add_subplot(gs[2, 1])
for h in ["W01", "W02"]:
    g = t[t.hyp == h].sort_values("settle_ts")
    cum = g.pnl.cumsum()
    dd = cum - cum.cummax()
    ax.fill_between(g.settle_ts, dd, 0, color=C[h], alpha=0.22)
    ax.plot(g.settle_ts, dd, color=C[h], lw=1.8, label=h)
ax.set_title("Просадка: W01 против W02", loc="left", fontsize=12)
ax.set_ylabel("$ на контракт")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
ax.annotate("−8,33 $", (t[t.hyp == "W01"].settle_ts.iloc[-1], -8.33),
            xytext=(-62, 12), textcoords="offset points", fontsize=9.5,
            color=LOSS, fontweight="bold",
            arrowprops=dict(arrowstyle="-", color=LOSS, lw=0.9))
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.grid(axis="y", color=GRID)
ax.legend(frameon=False, loc="lower left", fontsize=8.8)

# --- I: trade map
ax = fig.add_subplot(gs[2, 2])
ypos = {h: i for i, h in enumerate(HYP[::-1])}
for h in HYP:
    g = t[t.hyp == h]
    w = g[g.pays == 1]
    l = g[g.pays == 0]
    ax.scatter(w.open_ts, [ypos[h]] * len(w), s=20, color=C[h], alpha=0.65,
               edgecolor="white", linewidth=0.4, zorder=3)
    ax.scatter(l.open_ts, [ypos[h]] * len(l), s=58, marker="X", color=LOSS,
               edgecolor="white", linewidth=0.7, zorder=4)
ax.axvline(pd.Timestamp("2026-10-29 20:00", tz="UTC"), color="#94a3b8", ls="--", lw=1.0)
ax.set_yticks(list(ypos.values()))
ax.set_yticklabels(list(ypos.keys()))
ax.set_ylim(-0.7, 5.7)
ax.set_xlim(pd.Timestamp("2026-10-29 14:00", tz="UTC"), pd.Timestamp("2026-11-01 02:00", tz="UTC"))
ax.xaxis.set_major_locator(mdates.DayLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%d.%m"))
ax.set_title("Карта сделок: где были проигрыши", loc="left", fontsize=12)
ax.text(0.99, 0.04, "● выигрыш   ✕ проигрыш", transform=ax.transAxes, ha="right",
        fontsize=8.4, color=MUTED)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.grid(axis="x", color=GRID)

fig.text(0.055, 0.012,
         "Источник: lab_20261029–lab_20261101.log. W01/W02: 8 открытий и 14 расчётов не сопоставлены "
         "из-за обрезки имён Polymarket — итог частичный. W03–W06: P&L сверен с ценой, исходом и комиссией.",
         fontsize=7.8, color=MUTED)
fig.savefig(FIG / "fig22_wings_live_all.png", dpi=200, facecolor="white", bbox_inches="tight", pad_inches=0.25)
plt.close(fig)
print("fig22 ok")


# ================================================================== FIG 23 — history
fig = plt.figure(figsize=(14.6, 14.2))
gs = GridSpec(3, 3, figure=fig, height_ratios=[1.25, 1.0, 1.0], hspace=0.52, wspace=0.30,
              left=0.055, right=0.975, top=0.905, bottom=0.055)
fig.text(0.055, 0.962, "Продажа крыльев: бэктест 1 июля – 28 октября 2026",
         fontsize=21, fontweight="bold", color=INK)
fig.text(0.055, 0.934,
         "W02 / W03 / W04 на часовой лестнице Kalshi · 2 782 часа · 1 контракт на сделку, комиссия включена · "
         "W03 и W04 выбраны на июле–августе, сентябрь–октябрь — out-of-sample",
         fontsize=9.8, color=MUTED)

# --- A: cumulative
ax = fig.add_subplot(gs[0, :])
oos0 = pd.Timestamp("2026-09-01", tz="UTC")
for h in ["W02", "W03", "W04"]:
    ax.plot(eq.index, eq[h], color=C[h], lw=1.7, label=h)
    ax.annotate(f"{h}  {eq[h].iloc[-1]:+.1f} $", (eq.index[-1], eq[h].iloc[-1]),
                xytext=(8, 0), textcoords="offset points", va="center",
                fontsize=10, fontweight="bold", color=C[h])
ax.axvspan(oos0, eq.index[-1], color="#f1f5f9", zorder=0)
ax.text(oos0 + pd.Timedelta(days=3), 25.0, "out-of-sample: сентябрь–октябрь",
        fontsize=9.5, color=MUTED)
ax.axhline(0, color="#94a3b8", lw=0.9)
ax.set_ylabel("$ накопленным итогом")
ax.set_title("Накопленный P&L", loc="left", fontsize=12)
ax.set_xticks([pd.Timestamp(f"2026-{m:02d}-15", tz="UTC") for m in (7, 8, 9, 10)])
ax.set_xticklabels(["07.2026", "08.2026", "09.2026", "10.2026"])
ax.set_xlim(pd.Timestamp("2026-06-28", tz="UTC"), pd.Timestamp("2026-11-01", tz="UTC"))
ax.set_ylim(-38, 30)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.grid(axis="y", color=GRID)

# --- B: monthly
ax = fig.add_subplot(gs[1, 0])
months = monthly.index
xs = np.arange(len(months))
w = 0.26
for j, h in enumerate(["W02", "W03", "W04"]):
    vals = monthly[h].values
    ax.bar(xs + (j - 1) * w, vals, width=w, color=C[h], label=h)
    if h == "W02":
        for x, v in zip(xs + (j - 1) * w, vals):
            ax.text(x, v + (0.55 if v >= 0 else -1.55), f"{v:+.1f}", ha="center", fontsize=7.4,
                    color=INK if v >= 0 else LOSS, fontweight="bold")
ax.axhline(0, color="#94a3b8", lw=0.9)
ax.set_xticks(xs)
ax.set_xticklabels([m.strftime("%m.%Y") for m in months])
ax.set_ylim(-23, 15)
ax.set_ylabel("$")
ax.set_title("P&L по месяцам", loc="left", fontsize=12)
ax.legend(frameon=False, fontsize=8.4, ncol=3, loc="upper left")
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.grid(axis="y", color=GRID)

# --- C: drawdowns
ax = fig.add_subplot(gs[1, 1])
for h in ["W02", "W03", "W04"]:
    dd = eq[h] - eq[h].cummax()
    ax.plot(eq.index, dd, color=C[h], lw=1.5, label=h)
    if h == "W02":
        ax.annotate(f"{dd.min():.1f}", (dd.idxmin(), dd.min()), xytext=(-4, -10),
                    textcoords="offset points", fontsize=8.8, color=C[h], fontweight="bold")
ax.axhline(0, color="#94a3b8", lw=0.9)
ax.set_ylabel("$")
ax.set_title("Просадка", loc="left", fontsize=12)
ax.set_xticks([pd.Timestamp(f"2026-{m:02d}-15", tz="UTC") for m in (7, 8, 9, 10)])
ax.set_xticklabels(["07.2026", "08.2026", "09.2026", "10.2026"])
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.grid(axis="y", color=GRID)
ax.legend(frameon=False, fontsize=8.2, loc="lower left")

# --- D: rolling expectancy
ax = fig.add_subplot(gs[1, 2])
for h, dy in [("W02", -1), ("W03", 8), ("W04", -9)]:
    a = hourly[h][hourly[h] != 0]
    roll = 100 * a.rolling(200).mean()
    ax.plot(a.index[199:], roll.iloc[199:], color=C[h], lw=1.5, label=h)
    ax.annotate(f"{h} {roll.iloc[-1]:+.1f}", (a.index[-1], roll.iloc[-1]), xytext=(6, dy),
                textcoords="offset points", fontsize=8.2, color=C[h], fontweight="bold")
ax.axhline(0, color=LOSS, lw=1.0, ls="--")
ax.set_ylabel("¢ за активный час")
ax.set_title("Скользящее среднее, 200 активных часов", loc="left", fontsize=11.2)
ax.set_xticks([pd.Timestamp(f"2026-{m:02d}-01", tz="UTC") for m in (8, 9, 10)])
ax.set_xticklabels(["08.2026", "09.2026", "10.2026"])
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.grid(axis="y", color=GRID)

# --- E: distribution
ax = fig.add_subplot(gs[2, 0])
for i, h in enumerate(["W02", "W03", "W04"]):
    v = 100 * hourly[h][hourly[h] != 0].values
    ax.scatter(i + RNG.normal(0, 0.085, len(v)), v, s=7, color=C[h], alpha=0.22,
               edgecolor="none")
    ax.scatter([i], [np.median(v)], s=70, color="white", edgecolor=C[h], linewidth=2, zorder=5)
ax.set_yscale("symlog", linthresh=10)
ax.set_xticks([0, 1, 2])
ax.set_xticklabels(["W02", "W03", "W04"])
ax.set_ylim(-110, 30)
ax.set_yticks([-100, -30, -10, 0, 10, 20])
ax.set_yticklabels(["-100", "-30", "-10", "0", "+10", "+20"])
ax.set_ylabel("¢ за активный час")
ax.set_title("Распределение P&L за час", loc="left", fontsize=12)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.grid(axis="y", color=GRID)

# --- F: W05/W06 grid
ax = fig.add_subplot(gs[2, 1])
ax.axhspan(2, 8, color="#f0fdf4", zorder=0)
ax.axhline(2, color="#86efac", lw=1.0, ls="--")
ax.axvline(2, color="#cbd5e1", lw=0.9, ls=":")
for h, win in [("W03", "W03"), ("W04", "W04")]:
    d = imp[imp.window == win]
    ax.scatter(d.t_tr, d.t_te, s=42, color=C[h], alpha=0.75, edgecolor="white", linewidth=0.8,
               label=win, zorder=3)
chosen = imp[(imp.fair == "среднее EWMA·бустинг") & (imp.regime.str.contains("no US"))]
for _, r in chosen.iterrows():
    h = "W05" if r.window == "W03" else "W06"
    ax.scatter(r.t_tr, r.t_te, s=250, marker="*", color=C[h], edgecolor="white",
               linewidth=1.2, zorder=5)
    ax.annotate(h, (r.t_tr, r.t_te), xytext=(8, 6), textcoords="offset points",
                fontsize=10, fontweight="bold", color=C[h])
ax.set_xlabel("t-stat, июль–август")
ax.set_ylabel("t-stat, сентябрь–октябрь")
ax.set_xlim(0, 7.5)
ax.set_ylim(0.8, 5.2)
ax.set_title("W05/W06: 24 варианта, train → test", loc="left", fontsize=12)
ax.text(0.02, 0.95, "435 правил → 2,3 % с t > 2 на тесте", transform=ax.transAxes,
        fontsize=7.6, color=MUTED, va="top")
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.grid(color=GRID)

# --- G: table
ax = fig.add_subplot(gs[2, 2])
ax.axis("off")
ax.set_title("История: сводка", loc="left", fontsize=12)
cols = [0.00, 0.26, 0.44, 0.62, 0.80, 1.00]
heads = ["правило", "часов", "hit", "ср.¢/час", "итог $", "просадка"]
for x, htxt in zip(cols, heads):
    ax.text(x, 0.90, htxt, ha="left" if x == 0 else "right", va="center", fontsize=8.2,
            color=MUTED, fontweight="bold", transform=ax.transAxes)
ax.plot([0, 1], [0.84, 0.84], color=GRID, lw=1.4, transform=ax.transAxes)
for i, (_, r) in enumerate(H.iterrows()):
    y = 0.73 - i * 0.17
    ax.text(cols[0], y, r.hyp, fontsize=11, fontweight="bold", color=C[r.hyp], va="center",
            transform=ax.transAxes)
    ax.text(cols[1], y, f"{int(r.hours)}", ha="right", va="center", fontsize=9.2, transform=ax.transAxes)
    ax.text(cols[2], y, f"{r.hit:.0f}%", ha="right", va="center", fontsize=9.2, transform=ax.transAxes)
    ax.text(cols[3], y, f"{r.mean_c:+.2f}", ha="right", va="center", fontsize=9.2,
            color=INK if r.mean_c >= 0 else LOSS, fontweight="bold", transform=ax.transAxes)
    ax.text(cols[4], y, f"{r.total:+.1f}", ha="right", va="center", fontsize=9.2,
            color=INK if r.total >= 0 else LOSS, transform=ax.transAxes)
    ax.text(cols[5], y, f"{r.max_dd:.1f}", ha="right", va="center", fontsize=9.2,
            color=LOSS, transform=ax.transAxes)
ax.text(0, 0.13, "W03 и W04: +$15.6 и +$17.4 на out-of-sample\n"
                 "W02 теряет в каждом месяце, кроме октября\n"
                 "W01 в истории: 22 908 сделок, −0.94¢, t −4.4",
        fontsize=7.6, color=MUTED, va="top", transform=ax.transAxes, linespacing=1.5)
fig.savefig(FIG / "fig23_wings_history.png", dpi=200, facecolor="white", bbox_inches="tight", pad_inches=0.25)
plt.close(fig)
print("fig23 ok")


# ================================================================== FIG 24 — risk
def sim_means(g: pd.DataFrame, reps: int = 20000) -> np.ndarray:
    w = g.loc[g.pays == 1, "pnl"].values
    l = g.loc[g.pays == 0, "pnl"].values
    wm = w.mean() if len(w) else 0.0
    lm = l.mean() if len(l) else float(-(g.price.mean() + 0.01))
    p = RNG.beta(len(w) + 0.5, len(l) + 0.5, reps)
    k = RNG.binomial(len(g), p)
    return 100 * (k * wm + (len(g) - k) * lm) / len(g)


def p_profit(g: pd.DataFrame, n: int, reps: int = 20000) -> float:
    w = g.loc[g.pays == 1, "pnl"].values
    l = g.loc[g.pays == 0, "pnl"].values
    wm = w.mean() if len(w) else 0.0
    lm = l.mean() if len(l) else float(-(g.price.mean() + 0.01))
    p = RNG.beta(len(w) + 0.5, len(l) + 0.5, reps)
    k = RNG.binomial(n, p)
    return float((k * wm + (n - k) * lm > 0).mean())


sims = {h: sim_means(t[t.hyp == h]) for h in HYP}
Ns = [20, 50, 100, 200, 400]
pp = {h: [100 * p_profit(t[t.hyp == h], n) for n in Ns] for h in HYP}
pd.DataFrame({"hyp": HYP, **{f"p_profit_N{n}": [pp[h][i] for h in HYP] for i, n in enumerate(Ns)}}).to_csv(
    TAB / "wings_all_p_profit.csv", index=False)

fig = plt.figure(figsize=(14.6, 10.6))
gs = GridSpec(2, 3, figure=fig, hspace=0.46, wspace=0.29,
              left=0.055, right=0.975, top=0.885, bottom=0.075)
fig.text(0.055, 0.955, "Риск продажи крыльев: что скрыто за маленькой выборкой",
         fontsize=21, fontweight="bold", color=INK)
fig.text(0.055, 0.925,
         "Бутстрап по фактическим сделкам · loss = цена контракта + комиссия · W01/W02 — 29.10, "
         "W03–W06 — 39 сделок за 52 часа",
         fontsize=9.8, color=MUTED)

# --- A: bootstrap mean
ax = fig.add_subplot(gs[0, 0])
parts = ax.violinplot([sims[h] for h in HYP], positions=np.arange(len(HYP)), widths=0.82,
                      showextrema=False, bw_method=0.25)
for body, h in zip(parts["bodies"], HYP):
    body.set_facecolor(C[h]); body.set_edgecolor(C[h]); body.set_alpha(0.55)
for i, h in enumerate(HYP):
    v = sims[h]
    ax.plot([i, i], [np.percentile(v, 2.5), np.percentile(v, 97.5)], color=C[h], lw=2.2)
    ax.scatter([i], [np.median(v)], s=26, color="white", edgecolor=C[h], linewidth=1.6, zorder=4)
ax.axhline(0, color=LOSS, lw=1.1, ls="--")
ax.set_xticks(np.arange(len(HYP)))
ax.set_xticklabels([f"{h}\n{np.median(sims[h]):+.1f}" for h in HYP])
ax.set_ylabel("¢ на сделку")
ax.set_ylim(-14, 22)
ax.set_title("Бутстрап среднего P&L", loc="left", fontsize=12)
ax.text(0.02, 0.96, "точка — медиана, усы — 95 %", transform=ax.transAxes, fontsize=7.6,
        color=MUTED, va="top")
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.grid(axis="y", color=GRID)

# --- B: P(profit) vs horizon
ax = fig.add_subplot(gs[0, 1])
xs = np.arange(len(Ns))
ldy = {"W01": 0, "W02": -9, "W03": 9, "W04": 0, "W05": 0, "W06": 0}
for h in HYP:
    ax.plot(xs, pp[h], marker="o", ms=5.5, lw=1.8, color=C[h], label=h)
    ax.annotate(h, (xs[-1], pp[h][-1]), xytext=(7, ldy[h]), textcoords="offset points",
                fontsize=8.2, color=C[h], fontweight="bold", va="center")
ax.set_xticks(xs)
ax.set_xticklabels([str(n) for n in Ns])
ax.set_xlabel("число сделок")
ax.set_ylabel("P(итог > 0), %")
ax.set_ylim(-2, 105)
ax.set_xlim(-0.3, 4.75)
ax.set_title("Вероятность выйти в плюс", loc="left", fontsize=12)
ax.text(0.02, 0.97, "пост-оценка hit с учётом неопределённости", transform=ax.transAxes,
        fontsize=7.6, color=MUTED, va="top")
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.grid(axis="y", color=GRID)

# --- C: max drawdown bootstrap
ax = fig.add_subplot(gs[0, 2])
for h in ["W03", "W04"]:
    a = hourly[h][hourly[h] != 0].values
    dds = []
    for _ in range(4000):
        path = RNG.choice(a, size=200, replace=True)
        cum = np.cumsum(path)
        dds.append((cum - np.maximum.accumulate(cum)).min())
    ax.hist(dds, bins=38, color=C[h], alpha=0.55, label=h, edgecolor="white", linewidth=0.4)
    ax.axvline(H.loc[H.hyp == h, "max_dd"].iloc[0], color=C[h], lw=1.8, ls="--")
ax.set_xlabel("макс. просадка за 200 часов, $")
ax.set_ylabel("частота")
ax.set_title("Просадка: бутстрап истории", loc="left", fontsize=12)
ax.legend(frameon=False, fontsize=8.4, loc="upper left")
ax.text(0.02, 0.03, "пунктир — факт; W02 не показан (до −35,7 $)", transform=ax.transAxes,
        fontsize=7.2, color=MUTED, bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.5))
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.grid(axis="y", color=GRID)

# --- D: compounding
ax = fig.add_subplot(gs[1, 0])
labels = [f"{r.rule}\nf={r.f}%" for _, r in comp.iterrows()]
xs = np.arange(len(comp))
w = 0.26
bases = {"W03": ["#4f46e5", "#818cf8", "#c7d2fe"], "W04": ["#0d9488", "#5eead4", "#ccfbf1"]}
for i, r in comp.iterrows():
    shades = bases["W03"] if r.rule == "W03" else bases["W04"] if r.rule == "W04" else ["#7c3aed", "#a78bfa", "#ddd6fe"]
    for j, v in enumerate([r.base, r.h1, r.h2]):
        ax.bar(i + (j - 1) * w, v, width=w, color=shades[j])
        ax.text(i + (j - 1) * w, v + 8, f"{v}", ha="center", fontsize=7.2, color=INK)
ax.axhline(100, color="#94a3b8", lw=1.0, ls="--")
ax.set_xticks(xs)
ax.set_xticklabels(labels, fontsize=8.2)
ax.set_ylabel("$ из $100")
ax.set_ylim(0, 580)
ax.set_title("Компаундинг, сентябрь–октябрь", loc="left", fontsize=12)
ax.text(0.02, 0.94, "бары: без сдвига / −1¢ / −2¢", transform=ax.transAxes,
        fontsize=7.2, color=MUTED, va="top")
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.grid(axis="y", color=GRID)

# --- E: one more loss
ax = fig.add_subplot(gs[1, 1])
rules = ["W02", "W03", "W04", "W05", "W06"]
xs = np.arange(len(rules))
cur, aft = [], []
for h in rules:
    g = t[t.hyp == h]
    loss = g.price.mean() + 0.01
    cur.append(g.pnl.mean() * 100)
    aft.append((g.pnl.sum() - loss) / (len(g) + 1) * 100)
ax.bar(xs - 0.19, cur, width=0.36, color=[C[h] for h in rules], label="сейчас")
ax.bar(xs + 0.19, aft, width=0.36, color=LOSS, alpha=0.75, label="ещё один минус")
for x, a, b in zip(xs, cur, aft):
    ax.text(x - 0.19, a + (0.7 if a >= 0 else -1.8), f"{a:+.1f}", ha="center", fontsize=7.6,
            color=INK if a >= 0 else LOSS)
    ax.text(x + 0.19, b + (0.7 if b >= 0 else -1.8), f"{b:+.1f}", ha="center", fontsize=7.6,
            color=LOSS if b < 0 else INK)
ax.axhline(0, color="#94a3b8", lw=0.9)
ax.set_xticks(xs)
ax.set_xticklabels(rules)
ax.set_ylabel("¢ на сделку")
ax.set_ylim(-13, 18)
ax.set_title("Что сделает ещё один проигрыш", loc="left", fontsize=12)
ax.text(0.02, 0.96, "цветной — сейчас · красный — ещё один минус", transform=ax.transAxes,
        fontsize=7.4, color=MUTED, va="top")
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.grid(axis="y", color=GRID)

# --- F: table
ax = fig.add_subplot(gs[1, 2])
ax.axis("off")
ax.set_title("Риск-метрики", loc="left", fontsize=12)
cols = [0.00, 0.20, 0.38, 0.60, 0.78, 1.00]
heads = ["правило", "n", "hit", "безубыт.", "PF", "просадка"]
for x, htxt in zip(cols, heads):
    ax.text(x, 0.92, htxt, ha="left" if x == 0 else "right", va="center", fontsize=7.8,
            color=MUTED, fontweight="bold", transform=ax.transAxes)
ax.plot([0, 1], [0.87, 0.87], color=GRID, lw=1.3, transform=ax.transAxes)
for i, r in S.iterrows():
    y = 0.78 - i * 0.125
    g = t[t.hyp == r.hyp]
    wm = g.loc[g.pays == 1, "pnl"].mean() if (g.pays == 1).any() else np.nan
    lm = abs(g.loc[g.pays == 0, "pnl"].mean()) if (g.pays == 0).any() else g.price.mean() + 0.01
    be = lm / (wm + lm) * 100
    pf = "∞" if not np.isfinite(r.pf) else f"{r.pf:.2f}"
    ax.text(cols[0], y, r.hyp, fontsize=9.2, fontweight="bold", color=C[r.hyp], va="center",
            transform=ax.transAxes)
    ax.text(cols[1], y, f"{int(r.n)}", ha="right", va="center", fontsize=8.6, transform=ax.transAxes)
    ax.text(cols[2], y, f"{r.hit:.0f}%", ha="right", va="center", fontsize=8.6, transform=ax.transAxes)
    ax.text(cols[3], y, f"{be:.0f}%", ha="right", va="center", fontsize=8.6,
            color=LOSS if be > r.hit else INK, transform=ax.transAxes)
    ax.text(cols[4], y, pf, ha="right", va="center", fontsize=8.6, transform=ax.transAxes)
    ax.text(cols[5], y, f"{r.max_dd:.2f}", ha="right", va="center", fontsize=8.6,
            color=LOSS, transform=ax.transAxes)
ax.text(0, 0.02, "безубыт. = проигрыш / (выигрыш + проигрыш);\n"
                 "PF = ∞ там, где в live ещё не было минусов.",
        fontsize=7.0, color=MUTED, va="bottom", transform=ax.transAxes, linespacing=1.4)
fig.savefig(FIG / "fig24_wings_risk.png", dpi=200, facecolor="white", bbox_inches="tight", pad_inches=0.25)
plt.close(fig)
print("fig24 ok")


# ------------------------------------------------------------------ extra slices for the report
t["cheap_c"] = 100 * t.cheap
t["bucket"] = pd.cut(t.cheap_c, [2, 6, 10, 15, 21], right=False,
                     labels=["2–6¢", "6–10¢", "10–15¢", "15–20¢"])
buckets = (t.groupby(["hyp", "bucket"], observed=True)
           .agg(n=("pnl", "size"), mean_c=("pnl", lambda s: 100 * s.mean()), total=("pnl", "sum"))
           .reset_index())
buckets.to_csv(TAB / "wings_all_buckets.csv", index=False)
execs = (t.groupby("exec").agg(n=("pnl", "size"), hit=("pays", "mean"),
                               mean_c=("pnl", lambda s: 100 * s.mean()), total=("pnl", "sum")).reset_index())
execs.to_csv(TAB / "wings_all_exec.csv", index=False)
edge = t[t.hyp.isin(["W01", "W02", "W03", "W04"])].copy()
edge["bucket"] = pd.cut(100 * edge.edge, [-1, 0.5, 1.5, 10], labels=["<0.5¢", "0.5–1.5¢", ">1.5¢"])
edge = (edge.groupby(["hyp", "bucket"], observed=True)
        .agg(n=("pnl", "size"), mean_c=("pnl", lambda s: 100 * s.mean())).reset_index())
edge.to_csv(TAB / "wings_all_edge.csv", index=False)
w01 = t[(t.hyp == "W01") & (t.pays == 0)].sort_values("open_ts")
w01[["open_ts", "settle_ts", "market", "price", "pnl"]].to_csv(TAB / "wings_all_w01_losses.csv", index=False)
print("extra slices ok")
