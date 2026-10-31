"""Kalshi hourly ladder: how often does a contract priced at p cents resolve YES?

Builds a calibration curve for all 1..99 cent bins, cluster-bootstrap intervals by
snapshot, a 5-cent deviation panel, and the tables behind the figure.

Run:  python reports/kalshi_calibration.py
Out:  reports/final/figures/fig18_kalshi_price_calibration.png
      reports/final/tables/kalshi_calibration_1c.csv
      reports/final/tables/kalshi_calibration_5c.csv
      reports/final/tables/kalshi_calibration_round.csv
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from scipy.special import expit, logit

ROOT = Path(__file__).resolve().parent.parent
TAB = ROOT / "reports" / "final" / "tables"
FIG = ROOT / "reports" / "final" / "figures"
TAB.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

INK = "#0b1220"
MUTED = "#5b6472"
GRID = "#e5e9f2"
BLUE = "#2563eb"
RED = "#e11d48"
AMBER = "#f59e0b"
GRAY = "#94a3b8"
LIGHT = "#cbd5e1"
DIAG = "#64748b"

REPS = 1000
RNG = np.random.default_rng(20261031)
Z = 1.959963984540054


# --------------------------------------------------------------------------- data
d = pd.read_parquet(ROOT / "ladder_eval.parquet")
x = d[d.horizon == "kx1h"].copy()
x["t"] = pd.to_datetime(x["t"])
x["cents"] = x.p * 100.0
x["c"] = np.clip(np.round(x.cents).astype(int), 1, 99)
x = x[(x.p > 0) & (x.p < 1) & x.outcome.notna()].copy()

ev = x.event.astype("category")
x["eid"] = ev.cat.codes.astype(int)
E = len(ev.cat.categories)
N = len(x)
print(f"{N:,} contracts, {E:,} snapshots, {x.t.min()} .. {x.t.max()}")


def wilson(k: int, n: int):
    if n == 0:
        return np.nan, np.nan
    p = k / n
    den = 1 + Z * Z / n
    c = (p + Z * Z / (2 * n)) / den
    h = Z * np.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)) / den
    return max(0.0, c - h), min(1.0, c + h)


def cluster_boot(bin_idx: np.ndarray, nb: int, reps: int = REPS):
    """Resample whole snapshots; return reps x nb matrix of realized frequencies."""
    C = np.zeros((E, nb))
    Y = np.zeros((E, nb))
    np.add.at(C, (x.eid.values, bin_idx), 1.0)
    np.add.at(Y, (x.eid.values, bin_idx), x.outcome.values)
    out = np.empty((reps, nb))
    for r in range(reps):
        w = RNG.multinomial(E, np.full(E, 1.0 / E))
        c = w @ C
        y = w @ Y
        out[r] = np.where(c > 0, y / np.maximum(c, 1.0), np.nan)
    return out


# --------------------------------------------------------------------------- 1-cent bins
cent = x.c.values - 1
S1 = cluster_boot(cent, 99)
rows = []
for c in range(1, 100):
    y = x[x.c == c]
    n = len(y)
    k = int(y.outcome.sum())
    lo_w, hi_w = wilson(k, n)
    lo_c, hi_c = (np.nanpercentile(S1[:, c - 1], [2.5, 97.5]) if n else (np.nan, np.nan))
    rows.append({
        "cent": c,
        "n_contracts": n,
        "n_snapshots": y.event.nunique(),
        "mean_price_c": y.cents.mean() if n else np.nan,
        "realized_pct": 100 * y.outcome.mean() if n else np.nan,
        "cluster_lo_pct": 100 * lo_c,
        "cluster_hi_pct": 100 * hi_c,
        "wilson_lo_pct": 100 * lo_w,
        "wilson_hi_pct": 100 * hi_w,
        "dev_pp": 100 * (y.outcome.mean() - y.p.mean()) if n else np.nan,
    })
t1 = pd.DataFrame(rows)
t1["significant"] = (t1.cluster_lo_pct > t1.mean_price_c) | (t1.cluster_hi_pct < t1.mean_price_c)
t1.to_csv(TAB / "kalshi_calibration_1c.csv", index=False)

# --------------------------------------------------------------------------- 5-cent bins
b5 = (x.c.values - 1) // 5
x["b5"] = b5
S5 = cluster_boot(b5, 20)
rows = []
for b in range(20):
    lo_c = 1 + 5 * b
    hi_c = min(99, 5 + 5 * b)
    y = x[b5 == b]
    lo, hi = np.nanpercentile(S5[:, b], [2.5, 97.5])
    rows.append({
        "lo_c": lo_c, "hi_c": hi_c,
        "n_contracts": len(y),
        "n_snapshots": y.event.nunique(),
        "mean_price_c": y.cents.mean(),
        "realized_pct": 100 * y.outcome.mean(),
        "cluster_lo_pct": 100 * lo,
        "cluster_hi_pct": 100 * hi,
        "dev_pp": 100 * (y.outcome.mean() - y.p.mean()),
    })
t5 = pd.DataFrame(rows)
t5["significant"] = (t5.cluster_lo_pct > t5.mean_price_c) | (t5.cluster_hi_pct < t5.mean_price_c)
t5.to_csv(TAB / "kalshi_calibration_5c.csv", index=False)

# --------------------------------------------------------------------------- round prices
ROUND = list(range(5, 100, 5))
S_round = cluster_boot(x.c.values - 1, 99)[:, [c - 1 for c in ROUND]]
rows = []
for j, c in enumerate(ROUND):
    y = x[x.c == c]
    k = int(y.outcome.sum())
    n = len(y)
    lo_w, hi_w = wilson(k, n)
    lo, hi = np.nanpercentile(S_round[:, j], [2.5, 97.5])
    rows.append({
        "cent": c, "n_contracts": n, "n_snapshots": y.event.nunique(),
        "mean_price_c": y.cents.mean(), "realized_pct": 100 * y.outcome.mean(),
        "cluster_lo_pct": 100 * lo, "cluster_hi_pct": 100 * hi,
        "wilson_lo_pct": 100 * lo_w, "wilson_hi_pct": 100 * hi_w,
        "dev_pp": 100 * (y.outcome.mean() - y.p.mean()),
        "significant": bool((lo > y.p.mean()) or (hi < y.p.mean())),
    })
tr = pd.DataFrame(rows)
tr.to_csv(TAB / "kalshi_calibration_round.csv", index=False)

# --------------------------------------------------------------------------- slope, bias, Brier
x["xy"] = x.cents * x.outcome
g = x.groupby("eid").agg(n=("outcome", "size"), sx=("cents", "sum"), sy=("outcome", "sum"),
                         sxx=("cents", lambda s: (s * s).sum()), sxy=("xy", "sum"))
slope = ((g.sxy.sum() - g.sx.sum() * g.sy.sum() / g.n.sum())
         / (g.sxx.sum() - g.sx.sum() ** 2 / g.n.sum()))
slopes = []
for _ in range(REPS):
    w = RNG.multinomial(E, np.full(E, 1.0 / E))
    n_ = w @ g.n.values
    sx = w @ g.sx.values
    sy = w @ g.sy.values
    sxx = w @ g.sxx.values
    sxy = w @ g.sxy.values
    slopes.append((sxy - sx * sy / n_) / (sxx - sx * sx / n_))
slope_lo, slope_hi = np.percentile(slopes, [2.5, 97.5])
bias_pp = 100 * (x.outcome.mean() - x.p.mean())
brier = float(((x.outcome - x.p) ** 2).mean())
brier_market = brier

# ---- favourite-longshot model: logit(q) = a + b * logit(p), fitted on 5c bins
g5 = x.groupby("b5").agg(n=("outcome", "size"), k=("outcome", "sum"), p=("p", "mean"))
g5 = g5[g5.n >= 100]
bp = g5.p.values
bq = (g5.k / g5.n).values
bn = g5.n.values
Xd = np.column_stack([np.ones(len(g5)), logit(bp)])


def fit_logit(q, n):
    w = n * q * (1 - q)
    ok = (q > 0) & (q < 1)
    return np.linalg.lstsq(Xd[ok] * np.sqrt(w[ok])[:, None],
                           logit(q[ok]) * np.sqrt(w[ok]), rcond=None)[0]


beta_fl = fit_logit(bq, bn)
Cb = np.zeros((E, len(g5)))
Kb = np.zeros((E, len(g5)))
bpos = {b: i for i, b in enumerate(g5.index)}
np.add.at(Cb, (x.eid.values, x.b5.map(bpos).values.astype(int)), 1.0)
np.add.at(Kb, (x.eid.values, x.b5.map(bpos).values.astype(int)), x.outcome.values)
bfl = []
for _ in range(REPS):
    wts = RNG.multinomial(E, np.full(E, 1.0 / E))
    nn = wts @ Cb
    kk = wts @ Kb
    qq = np.divide(kk, nn, out=np.zeros_like(kk), where=nn > 0)
    keep = nn >= 50
    if keep.sum() < 10:
        continue
    w = nn[keep] * qq[keep] * (1 - qq[keep])
    ok = (qq[keep] > 0) & (qq[keep] < 1)
    bfl.append(fit_logit(qq[keep], nn[keep])[1])
bfl = np.array(bfl)
b_lo, b_hi = np.percentile(bfl, [2.5, 97.5])

summary = {
    "b_logit": float(beta_fl[1]), "b_logit_lo": float(b_lo), "b_logit_hi": float(b_hi),
    "n_contracts": int(N), "n_snapshots": int(E),
    "from": str(x.t.min().date()), "to": str(x.t.max().date()),
    "bias_pp": bias_pp, "brier": brier,
    "slope_per_cent": slope, "slope_lo": slope_lo, "slope_hi": slope_hi,
    "slope_per_100c": 100 * slope,
    "realized_5c": float(tr.loc[tr.cent == 5, "realized_pct"].iloc[0]),
    "realized_10c": float(tr.loc[tr.cent == 10, "realized_pct"].iloc[0]),
    "realized_15c": float(tr.loc[tr.cent == 15, "realized_pct"].iloc[0]),
}
(TAB / "kalshi_calibration_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
print(json.dumps(summary, indent=2, ensure_ascii=False))

# --------------------------------------------------------------------------- figure
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10.5,
    "axes.edgecolor": "#cbd5e1", "axes.linewidth": 1.0,
    "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "figure.facecolor": "white", "axes.facecolor": "white",
})
fig = plt.figure(figsize=(12.6, 10.2))
gs = GridSpec(3, 1, height_ratios=[3.6, 2.0, 1.05], hspace=0.12, figure=fig)
ax = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1], sharex=ax)
ax3 = fig.add_subplot(gs[2], sharex=ax)

# --- panel A: reliability
ok = t1.n_contracts >= 30
xx = t1.loc[ok, "mean_price_c"].values
yy = t1.loc[ok, "realized_pct"].values
nn = t1.loc[ok, "n_contracts"].values
lo = t1.loc[ok, "cluster_lo_pct"].values
hi = t1.loc[ok, "cluster_hi_pct"].values
sig = t1.loc[ok, "significant"].values

ax.plot([0, 100], [0, 100], ls=(0, (4, 3)), color=DIAG, lw=1.4, zorder=1,
        label="идеальная калибровка (факт = цена)")
ax.errorbar(xx, yy, yerr=[yy - lo, hi - yy], fmt="none", ecolor="#b8c2d4",
            elinewidth=1.0, capsize=0, alpha=0.75, zorder=2)
ax.scatter(xx, yy, s=np.clip(np.sqrt(nn) * 1.55, 10, 105),
           c=BLUE, edgecolor="white", linewidth=0.7, alpha=0.72, zorder=3)
grid = np.linspace(1, 99, 197) / 100.0
fl = 100 * expit(beta_fl[0] + beta_fl[1] * logit(grid))
ax.plot(100 * grid, fl, color="#0f766e", lw=2.2, zorder=4,
        label="FL-модель: logit(q) = a + b·logit(p)")

for c in (5, 10, 15):
    r = tr[tr.cent == c].iloc[0]
    ax.scatter([r.mean_price_c], [r.realized_pct], marker="D", s=58, color=AMBER,
               edgecolor="white", linewidth=1.2, zorder=6)
    ax.annotate(f"{c}", xy=(r.mean_price_c, r.realized_pct), xytext=(0, -13),
                textcoords="offset points", ha="center", fontsize=8.2,
                color="#92400e", fontweight="bold", zorder=7)

ax.set_ylabel("Доля контрактов, исполнившихся\nв YES, %", fontsize=11)
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.grid(color=GRID, lw=0.8)
ax.set_axisbelow(True)
from matplotlib.lines import Line2D
handles = [
    Line2D([], [], ls=(0, (4, 3)), color=DIAG, lw=1.4, label="идеальная калибровка"),
    Line2D([], [], color="#0f766e", lw=2.2, label="FL-модель (logit–logit)"),
    Line2D([], [], color="#b8c2d4", lw=1.3, label="95% ДИ (кластерный бутстрэп)"),
    Line2D([], [], marker="D", ls="", markerfacecolor=AMBER, markeredgecolor="white",
           markersize=7, label="цены 5 / 10 / 15 ¢"),
]
ax.legend(handles=handles, loc="upper left", frameon=False, fontsize=9.2,
          handletextpad=0.6)
def ru(v, nd=1):
    return f"{v:.{nd}f}".replace(".", ",")


rows_txt = "цена   факт      n\n"
for c in (5, 10, 15):
    r = tr[tr.cent == c].iloc[0]
    rows_txt += f"{c:>3}¢   {ru(r.realized_pct):>4}%   {int(r.n_contracts):>4}\n"
rows_txt += (f"\nнаклон q~p     {ru(100*slope, 3)} [{ru(100*slope_lo, 3)}; {ru(100*slope_hi, 3)}]"
             f"\nнаклон logit  {ru(beta_fl[1], 3)} [{ru(b_lo, 3)}; {ru(b_hi, 3)}]"
             f"\nсмещение {ru(bias_pp, 2).replace('-', '−')} п.п. · Brier {ru(brier, 4)}")
ax.text(0.985, 0.05, rows_txt, transform=ax.transAxes, ha="right", va="bottom",
        fontsize=9.0, color=INK, family="DejaVu Sans Mono",
        bbox=dict(boxstyle="round,pad=0.5", fc="#f8fafc", ec="#e2e8f0", lw=0.9))
ax.set_title("Kalshi, часовые лесенки: цена контракта против частоты исполнения YES",
             loc="left", fontsize=15.5, fontweight="bold", pad=12)

# --- panel B: deviation by 5c bucket
mid = (t5.lo_c + t5.hi_c) / 2
dev = t5.dev_pp.values
colors = np.where(dev < 0, RED, BLUE)
alphas = np.where(t5.significant.values, 0.95, 0.33)
ax2.axhline(0, color=INK, lw=1.0)
ax2.bar(mid, dev, width=3.6, color=colors, alpha=0.9, edgecolor="white", linewidth=0.6, zorder=2)
ax2.errorbar(mid, dev, yerr=[dev - (t5.cluster_lo_pct - t5.mean_price_c).values,
                             (t5.cluster_hi_pct - t5.mean_price_c).values - dev],
             fmt="none", ecolor="#475569", elinewidth=1.0, capsize=2.5, zorder=3)
for m, dv, sg in zip(mid, dev, t5.significant.values):
    if sg:
        ax2.annotate(f"{dv:+.1f}", (m, dv), textcoords="offset points", xytext=(0, 5 if dv > 0 else -12),
                     ha="center", fontsize=8.2, color=INK, fontweight="bold")
ax2.set_ylabel("Факт − цена, п.п.", fontsize=10.5)
ax2.grid(axis="y", color=GRID, lw=0.8)
ax2.set_axisbelow(True)
ax2.text(0.995, 0.92, "подписаны корзины, где 95% ДИ не накрывает цену",
         transform=ax2.transAxes, ha="right", va="top", fontsize=8.6, color=MUTED)

# --- panel C: sample size
ax3.bar(mid, t5.n_contracts, width=3.6, color=LIGHT, edgecolor="white", linewidth=0.6)
ax3.set_yscale("log")
ax3.set_ylabel("Контрактов", fontsize=9.5)
ax3.set_xlabel("Цена контракта (середина корзины), центы", fontsize=11)
ax3.grid(axis="y", color=GRID, lw=0.8)
ax3.set_axisbelow(True)
ax3.set_xlim(0, 100)
ax3.set_ylim(200, 20000)
for m, n_ in zip(mid, t5.n_contracts):
    if n_ >= 3000:
        ax3.annotate(f"{n_:,}".replace(",", " "), (m, n_ * 1.15), ha="center", fontsize=7.8, color=MUTED)
ax3.tick_params(axis="x", labelsize=9.5)

fig.text(0.008, 0.012,
         (f"ladder_eval.parquet · Kalshi KXBTCD, 55 мин · {N:,} контрактов / {E:,} снимков · "
          f"{x.t.min():%m.%Y}–{x.t.max():%m.%Y} · ДИ — кластерный бутстрэп по снимкам, 95%")
         .replace(",", " "),
         fontsize=8.3, color=MUTED)
fig.subplots_adjust(left=0.085, right=0.985, top=0.935, bottom=0.115)
out = FIG / "fig18_kalshi_price_calibration.png"
fig.savefig(out, dpi=170)
plt.close(fig)
print("saved", out)

print(t5.round(2).to_string(index=False))
print(tr.round(2).to_string(index=False))
