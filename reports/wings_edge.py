"""Where does the wing-selling profit come from: edge (fair value vs market) buckets.

W03/W04 have a logged model edge; we rebuild their trade-level backtest from
studies/wings_ladder.py, bucket Sep-Oct trades by edge and sweep the minimum-edge
threshold. W05/W06 do not log edge in live, so they get a separate, explicit treatment.

Outputs:
  reports/final/figures/fig25_wings_edge.png
  reports/final/figures/fig26_wings_w05_w06.png
  reports/final/tables/wings_edge_*.csv

The Russian write-up (reports/wings_edge.md) is maintained alongside this script.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FuncFormatter

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "reports" / "final" / "figures"
TAB = ROOT / "reports" / "final" / "tables"
FIG.mkdir(parents=True, exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT / "studies"))

INK, MUTED, GRID, LOSS = "#0b1220", "#5b6472", "#e8edf5", "#dc2626"
C = {"W03": "#4f46e5", "W04": "#0d9488", "W05": "#d97706", "W06": "#db2777"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10.2,
                     "axes.edgecolor": "#cbd5e1", "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.titleweight": "bold", "figure.facecolor": "white"})


def fee(p):
    return np.ceil(np.round(0.07 * p * (1 - p) * 100, 6)) / 100


def load_cheap_minutes():
    cache = Path("/tmp/wings_c.parquet")
    if cache.exists():
        c = pd.read_parquet(cache)
    else:
        import wings_ladder as wl
        c = wl.load()
        c.to_parquet(cache)
    c["edge_n"] = c.c_bid - c.c_fair_n - fee(1 - c.c_bid)
    c["edge_t"] = c.c_bid - c.c_fair_t - fee(1 - c.c_bid)
    return c


C_ALL = load_cheap_minutes()


def pick(lo, hi, l0, l1, col, margin, mode="price"):
    g = C_ALL[C_ALL.c_bid.between(lo, hi) & C_ALL.left.between(l0, l1) & (C_ALL[col] >= margin)].copy()
    g["edge"] = g[col]
    if mode == "price":
        g = g.sort_values(["T", "ts", "c_bid"], ascending=[True, True, False]).drop_duplicates(["T", "ticker"])
        return g.groupby("T").head(1)
    g = g.sort_values(["T", "ticker", "ts"]).drop_duplicates(["T", "ticker"])
    return g.loc[g.groupby("T")["edge"].idxmax()]


RULES = {
    "W03": dict(lo=.10, hi=.20, l0=5, l1=15, col="edge_n", margin=.01),
    "W04": dict(lo=.06, hi=.10, l0=15, l1=30, col="edge_t", margin=0.0),
}
SETS = {h: pick(**RULES[h]) for h in RULES}
SETS0 = {h: pick(**{**RULES[h], "margin": 0.0}) for h in RULES}
for h in SETS:
    SETS[h]["half"] = np.where(SETS[h]["T"] < pd.Timestamp("2026-09-01", tz="UTC"), "train", "test")
    SETS0[h]["half"] = np.where(SETS0[h]["T"] < pd.Timestamp("2026-09-01", tz="UTC"), "train", "test")

BINS = [0, .5, 1, 1.5, 2, 3, 4, 100]
LABELS = ["0–0.5", "0.5–1", "1–1.5", "1.5–2", "2–3", "3–4", "≥4"]


def bucket_table(g, edge_col="edge"):
    g = g.copy()
    g["b"] = pd.cut(100 * g[edge_col], BINS, labels=LABELS, right=False)
    rows = []
    for lab in LABELS:
        s = g[g.b == lab]
        if len(s) == 0:
            rows.append(dict(bucket=lab, n=0, share_n=np.nan, mean_c=np.nan, ci_lo=np.nan, ci_hi=np.nan,
                             total=0.0, share_total=np.nan, hit=np.nan))
            continue
        m = 100 * s.pnl.mean()
        se = 100 * s.pnl.std(ddof=1) / np.sqrt(len(s)) if len(s) > 1 else np.nan
        rows.append(dict(bucket=lab, n=len(s), share_n=100 * len(s) / len(g), mean_c=m,
                         ci_lo=m - 1.96 * se, ci_hi=m + 1.96 * se, total=s.pnl.sum(),
                         share_total=100 * s.pnl.sum() / g.pnl.sum() if g.pnl.sum() != 0 else np.nan,
                         hit=100 * (s.pnl > 0).mean()))
    return pd.DataFrame(rows)


rows = []
for h in ["W03", "W04"]:
    for half in ["train", "test"]:
        b = bucket_table(SETS0[h][SETS0[h].half == half])
        b.insert(0, "half", half)
        b.insert(0, "hyp", h)
        rows.append(b)
BUCKETS = pd.concat(rows, ignore_index=True)
BUCKETS.to_csv(TAB / "wings_edge_buckets.csv", index=False)

# the rules exactly as they trade today (W03 margin 1c, W04 margin 0)
CUR = {h: pick(**RULES[h]) for h in RULES}
cur_rows = []
for h in ["W03", "W04"]:
    g = CUR[h][CUR[h]["T"] >= pd.Timestamp("2026-09-01", tz="UTC")]
    b = bucket_table(g)
    b.insert(0, "hyp", h)
    cur_rows.append(b)
CURB = pd.concat(cur_rows, ignore_index=True)
CURB.to_csv(TAB / "wings_edge_buckets_current.csv", index=False)

# threshold sweep from the study grid (price-first selection, exactly as W03/W04)
G = pd.read_csv(ROOT / "results" / "studies" / "wings_ladder_grid.csv")
G["strikes/hour"] = G["strikes/hour"].astype(str)
GRIDROWS = {
    "W03": G[(G["band ¢"] == "10–20") & (G["minutes left"] == "5–15") & (G["filter"] == "normal") & (G["strikes/hour"] == "1")],
    "W04": G[(G["band ¢"] == "6–10") & (G["minutes left"] == "15–30") & (G["filter"] == "t4") & (G["strikes/hour"] == "1")],
}
DAYS = (pd.Timestamp("2026-10-29") - pd.Timestamp("2026-09-01")).days
th_rows = []
for h, d in GRIDROWS.items():
    for _, r in d.sort_values("margin ¢").iterrows():
        n_te = int(r["trades test"])
        th_rows.append(dict(hyp=h, margin=float(r["margin ¢"]), n_test=n_te, mean_c_test=float(r["¢/trade test"]),
                            total_test=n_te * float(r["¢/trade test"]) / 100, t_test=float(r["t (hours) test"]),
                            hit_test=100 * float(r["hit test"]), per_day=float(r["¢/trade test"]) * n_te / 100 / DAYS,
                            n_train=int(r["trades train"]), mean_c_train=float(r["¢/trade train"]),
                            t_train=float(r["t (hours) train"])))
THR = pd.DataFrame(th_rows)
THR.to_csv(TAB / "wings_edge_thresholds.csv", index=False)

# concentration: cumulative share of profit as trades are added from the highest edge down
conc = []
for h in ["W03", "W04"]:
    for half in ["train", "test"]:
        g = SETS0[h][SETS0[h].half == half].sort_values("edge", ascending=False)
        cum = g.pnl.cumsum() / g.pnl.sum()
        for i, v in enumerate(cum.values, 1):
            conc.append(dict(hyp=h, half=half, share_trades=i / len(g), cum_share=100 * v))
CONC = pd.DataFrame(conc)
CONC.to_csv(TAB / "wings_edge_concentration.csv", index=False)

# live edges
LT = pd.read_csv(TAB / "wings_all_live_trades.csv", parse_dates=["open_ts", "settle_ts"])
LIVE = LT[LT.hyp.isin(["W03", "W04", "W05", "W06"])].copy()
LIVE["edge_c"] = 100 * LIVE.edge
LIVE["pnl_c"] = 100 * LIVE.pnl
LIVE["price_c"] = 100 * LIVE.price
LIVE["cheap_c"] = 100 * LIVE.cheap
LIVE[["hyp", "open_ts", "market", "price_c", "cheap_c", "edge_c", "pnl_c", "pays"]].to_csv(
    TAB / "wings_edge_live.csv", index=False)

# W05/W06: live prices and history from the improve grid
IMP = pd.read_csv(ROOT / "results" / "studies" / "wings_improve.csv")
IMP.columns = ["win", "fair", "regime", "n_tr", "c_tr", "t_tr", "n_te", "c_te", "t_te"]
imp_rows = []
for win, label in (("W03", "10–20¢ / 5–15 мин"), ("W04", "6–10¢ / 15–30 мин")):
    for fair, regime, name in (("EWMA (как W03)", "все часы", "EWMA, все часы"),
                               ("EWMA (как W03)", "no US open (13–15 UTC)", "EWMA, без 13–15 UTC"),
                               ("среднее EWMA·бустинг", "все часы", "blend, все часы"),
                               ("среднее EWMA·бустинг", "no US open (13–15 UTC)", "blend, без 13–15 UTC")):
        r = IMP[(IMP.win.str.startswith(win)) & (IMP.fair == fair) & (IMP.regime == regime)].iloc[0]
        imp_rows.append(dict(window=label, variant=name, n=int(r.n_te), mean_c=float(r.c_te), t=float(r.t_te)))
IMPW = pd.DataFrame(imp_rows)
IMPW.to_csv(TAB / "wings_edge_w05w06.csv", index=False)

# ------------------------------------------------------------------ figure 25
fig = plt.figure(figsize=(14.6, 10.8))
gs = GridSpec(2, 3, figure=fig, hspace=0.44, wspace=0.30, left=0.055, right=0.975, top=0.855, bottom=0.075)
fig.text(0.055, 0.955, "Edge: где на самом деле лежит прибыль продажи крыльев",
         fontsize=21, fontweight="bold", color=INK)
fig.text(0.055, 0.922, "W03/W04, сентябрь–октябрь 2026 (out-of-sample) · edge = цена крыла − справедливая цена − комиссия · "
                       "1 контракт на сделку", fontsize=9.6, color=MUTED)
fig.text(0.055, 0.896, "Корзины и доли прибыли — все сделки с edge ≥ 0; серая зона — то, что отрезает текущий порог W03 (1¢). "
                       "Усы — 95 % ДИ; в корзинах 40–250 сделок.", fontsize=8.2, color=MUTED)

# A — distribution
ax = fig.add_subplot(gs[0, 0])
bins = np.arange(0, 13.5, 0.5)
for h in ["W03", "W04"]:
    g = SETS0[h][SETS0[h].half == "test"]
    ax.hist(100 * g.edge, bins=bins, weights=np.ones(len(g)) * 100 / len(g), color=C[h],
            alpha=0.40, histtype="stepfilled", label=f"{h} (n={len(g)})")
    ax.axvline(100 * g.edge.median(), color=C[h], lw=1.4, ls="--")
ax.set_xlabel("edge, ¢")
ax.set_ylabel("% сделок")
ax.set_xlim(0, 12)
ax.set_title("Распределение edge у выбранных сделок", loc="left", fontsize=12)
ax.legend(frameon=False, fontsize=8.4)
ax.text(0.97, 0.95, "пунктир — медиана", transform=ax.transAxes, ha="right", va="top",
        fontsize=7.6, color=MUTED)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.grid(axis="y", color=GRID)

# B — mean P&L by bucket
ax = fig.add_subplot(gs[0, 1])
x = np.arange(len(LABELS))
w = 0.36
ax.axvspan(-0.5, 1.5, color="#f1f5f9", zorder=0)
for j, h in enumerate(["W03", "W04"]):
    b = BUCKETS[(BUCKETS.hyp == h) & (BUCKETS.half == "test")].set_index("bucket").reindex(LABELS)
    ax.bar(x + (j - 0.5) * w, b.mean_c, width=w, color=C[h], label=h, alpha=0.92,
           yerr=[b.mean_c - b.ci_lo, b.ci_hi - b.mean_c], capsize=2.5, error_kw=dict(lw=0.9, ecolor="#334155"))
ax.axhline(0, color="#94a3b8", lw=0.9)
ax.set_xticks(x)
ax.set_xticklabels(LABELS, fontsize=8.4)
ax.set_ylabel("¢ на сделку")
ax.set_ylim(-9, 14)
ax.set_title("Средний P&L по корзинам edge", loc="left", fontsize=12)
ax.annotate("0–0,5¢: +9,9¢", xy=(-0.18, 9.9), xytext=(0.55, 12.3), fontsize=7.8, color=C["W03"],
            fontweight="bold", arrowprops=dict(arrowstyle="-", color=C["W03"], lw=0.8))
ax.annotate("≥4¢: −2,3¢", xy=(6 - 0.18, -2.3), xytext=(4.55, -6.8), fontsize=7.8, color=LOSS,
            fontweight="bold", arrowprops=dict(arrowstyle="-", color=LOSS, lw=0.8))

for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.grid(axis="y", color=GRID)

# C — profit share
ax = fig.add_subplot(gs[0, 2])
for j, h in enumerate(["W03", "W04"]):
    b = BUCKETS[(BUCKETS.hyp == h) & (BUCKETS.half == "test")].set_index("bucket").reindex(LABELS)
    ax.bar(x + (j - 0.5) * w, b.share_total, width=w, color=C[h], label=h, alpha=0.92)
ax.axhline(0, color="#94a3b8", lw=0.9)
ax.text(-0.18, 51.5, "+49 %", ha="center", fontsize=7.6, color=C["W03"], fontweight="bold")
ax.text(1.18, 28.0, "+26 %", ha="center", fontsize=7.6, color=C["W04"], fontweight="bold")
ax.text(5.82, -18.8, "−15 %", ha="center", fontsize=7.9, color=LOSS, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(LABELS, fontsize=8.4)
ax.set_ylabel("% итоговой прибыли")
ax.set_ylim(-24, 60)
ax.set_title("Доля прибыли по корзинам edge", loc="left", fontsize=12)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.grid(axis="y", color=GRID)

# D — threshold sweep
ax = fig.add_subplot(gs[1, 0])
pos = {"W03": [0, 1, 2], "W04": [3.6, 4.6]}
for h, ps in pos.items():
    d = THR[(THR.hyp == h)].sort_values("margin")
    for p, (_, r) in zip(ps, d.iterrows()):
        ax.bar(p, r.total_test, width=0.8, color=C[h], alpha=0.55 + 0.2 * (r.margin == 0))
        ax.text(p, r.total_test + 0.7, f"${r.total_test:.1f}\n{int(r.n_test)} · {r.mean_c_test:+.2f}¢",
                ha="center", fontsize=7.2, color=INK)
ax.set_xticks([0, 1, 2, 3.6, 4.6])
ax.set_xticklabels(["0¢", "1¢", "2¢", "0¢", "1¢"])
ax.text(1, -2.6, "W03", ha="center", fontsize=9, fontweight="bold", color=C["W03"])
ax.text(4.1, -2.6, "W04", ha="center", fontsize=9, fontweight="bold", color=C["W04"])
ax.set_ylabel("$ за Sep–Oct")
ax.set_ylim(-5, 24)
ax.set_title("Порог edge: выше — не лучше", loc="left", fontsize=12)
ax.text(0.02, 0.95, "подпись: итог · сделок · ¢/сделку", transform=ax.transAxes, fontsize=7.2,
        color=MUTED, va="top")
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.grid(axis="y", color=GRID)

# E — concentration
ax = fig.add_subplot(gs[1, 1])
for h in ["W03", "W04"]:
    for half, ls, alpha in (("test", "-", 1.0), ("train", "--", 0.45)):
        d = CONC[(CONC.hyp == h) & (CONC.half == half)]
        ax.plot(100 * d.share_trades, d.cum_share, color=C[h], lw=1.7, ls=ls, alpha=alpha,
                label=f"{h} {half}")
ax.axhline(0, color="#94a3b8", lw=0.8)
ax.axvline(10, color="#cbd5e1", lw=0.9, ls=":")
ax.axvline(50, color="#cbd5e1", lw=0.9, ls=":")
ax.set_xlabel("% сделок, отсортированных по edge (сверху вниз)")
ax.set_ylabel("накопленная доля прибыли, %")
ax.set_xlim(0, 100)
ax.set_ylim(-60, 160)
ax.set_title("Прибыль и концентрация по edge", loc="left", fontsize=12)

ax.legend(frameon=False, fontsize=7.2, loc="upper left")
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.grid(color=GRID)

# F — live
ax = fig.add_subplot(gs[1, 2])
for h in ["W03", "W04"]:
    g = LIVE[LIVE.hyp == h]
    win = g[g.pays == 1]
    loss = g[g.pays == 0]
    ax.scatter(win.edge_c, win.pnl_c, s=34, color=C[h], alpha=0.85, edgecolor="white", linewidth=0.6, label=h)
    ax.scatter(loss.edge_c, loss.pnl_c, s=120, marker="X", color=LOSS, edgecolor="white", linewidth=0.8, zorder=5)
ax.axhline(0, color="#94a3b8", lw=0.9)
ax.set_xlabel("edge, ¢")
ax.set_ylabel("P&L, ¢")
ax.set_xlim(0, 8.5)
ax.set_ylim(-104, 25)
ax.set_title("Live: edge и результат", loc="left", fontsize=12)
ax.annotate("W04: edge 0,32¢ → −95¢", (0.32, -95), xytext=(1.6, -78), fontsize=7.8, color=LOSS,
            arrowprops=dict(arrowstyle="-", color=LOSS, lw=0.8))
ax.text(0.56, 0.56, "W03: corr −0,65 (n=8)\nW04: corr +0,17 (n=21)", transform=ax.transAxes,
        ha="left", va="top", fontsize=7.6, color=MUTED)
ax.legend(frameon=False, fontsize=8.4, loc="lower right")
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.grid(color=GRID)

fig.text(0.055, 0.014, "Источник: studies/wings_ladder.py (1-минутные стаканы Kalshi, июль–октябрь), "
                       "реконструкция выбранных сделок W03/W04; live — lab_20261029–lab_20261101.log. "
                       "Edge — модельный, из него вычтена комиссия fee(1−p).", fontsize=7.2, color=MUTED)
fig.savefig(FIG / "fig25_wings_edge.png", dpi=200, facecolor="white", bbox_inches="tight", pad_inches=0.25)
plt.close(fig)
print("fig25 ok")

# ------------------------------------------------------------------ figure 26 — W05/W06
fig = plt.figure(figsize=(14.6, 5.6))
gs = GridSpec(1, 2, figure=fig, wspace=0.25, left=0.06, right=0.975, top=0.80, bottom=0.14)
fig.text(0.06, 0.925, "W05/W06: что известно про их edge", fontsize=19, fontweight="bold", color=INK)
fig.text(0.06, 0.875, "edge в live не логируется · 5 + 5 сделок · история: 24 варианта blend и режимов",
         fontsize=9.6, color=MUTED)

ax = fig.add_subplot(gs[0, 0])
for h in ["W05", "W06"]:
    g = LIVE[LIVE.hyp == h]
    ax.scatter(g.cheap_c, g.pnl_c, s=70, color=C[h], edgecolor="white", linewidth=1.0,
               label=f"{h}: n={len(g)}, {g.pnl_c.mean():+.1f}¢")
ax.set_xlabel("цена проданного крыла, ¢")
ax.set_ylabel("P&L, ¢")
ax.set_xlim(5, 20)
ax.set_ylim(0, 20)
ax.set_title("Live: только цена, edge неизвестен", loc="left", fontsize=12)
ax.text(0.03, 0.06, "все 10 сделок выиграли; 5 сделок на правило — не статистика", transform=ax.transAxes,
        fontsize=7.4, color=MUTED)
ax.legend(frameon=False, fontsize=8.2, loc="lower right")
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.grid(color=GRID)

ax = fig.add_subplot(gs[0, 1])
groups = ["10–20¢\n5–15 мин", "6–10¢\n15–30 мин"]
variants = ["EWMA, все часы", "EWMA, без 13–15 UTC", "blend, все часы", "blend, без 13–15 UTC"]
colors = ["#94a3b8", "#475569", "#4f46e5", "#0d9488"]
hatches = ["", "//", "", ""]
xx = np.arange(len(groups))
w = 0.20
for j, v in enumerate(variants):
    d = IMPW[IMPW.variant == v].reset_index(drop=True)
    ax.bar(xx + (j - 1.5) * w, d.mean_c, width=w, color=colors[j], hatch=hatches[j], edgecolor="white",
           linewidth=0.6)
    for xi, (m, n) in enumerate(zip(d.mean_c, d.n)):
        ax.text(xi + (j - 1.5) * w, m + 0.08, f"{m:.2f}¢\n{n}", ha="center", fontsize=6.4, color=INK)
ax.set_xticks(xx)
ax.set_xticklabels(groups)
ax.set_ylabel("¢ на сделку, Sep–Oct")
ax.set_ylim(0, 6.3)
ax.set_title("История: откуда берётся улучшение", loc="left", fontsize=12)
ax.text(0.0, -0.20, "серые — EWMA, цветные — blend; штриховка — без 13–15 UTC. В этой сетке выбран максимальный edge, "
                    "поэтому база W04 = 1,46¢, а не 1,77¢ из основной сетки.",
        transform=ax.transAxes, fontsize=6.6, color=MUTED)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.grid(axis="y", color=GRID)
fig.savefig(FIG / "fig26_wings_w05_w06.png", dpi=200, facecolor="white", bbox_inches="tight", pad_inches=0.25)
plt.close(fig)
print("fig26 ok")
