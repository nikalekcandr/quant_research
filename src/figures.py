"""Figures for the report. Each one supports a specific claim; nothing decorative.

  fig1_vol_paths.png      realized vs ladder-implied vs Deribit-implied volatility (24h windows)
  fig2_pit.png            PIT histograms: is each density forecast calibrated?
  fig3_ladder_calibration.png  market price vs realized frequency by probability bin (longshot bias)
  fig4_pnl.png            cumulative profit of the trading rule
  fig5_disagreement.png   when the ladder disagrees with HAR, how much of the gap is realized
  fig6_pool_weights.png   real-time weights of the optimal pool on HAR-X, Deribit and the ladder
  fig7_fluctuation.png    rolling DM statistic of HAR + ladder against HAR (Giacomini-Rossi)
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config
from src.utils import ensure, log, read_table

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False, "lines.linewidth": 1.4,
})


def _try(name):
    try:
        return read_table(name)
    except FileNotFoundError:
        return None


def fig_vol_paths():
    p = _try(config.INTERIM / "panel_pm24h")
    if p is None:
        return
    p = p[p["t"] >= pd.Timestamp(config.EVAL_START, tz="UTC") - pd.Timedelta(days=120)]
    p = p[p["l_mkt_mfiv"].notna() | p["l_iv_var"].notna()]
    if p.empty:
        return
    ann = lambda v: np.sqrt(np.asarray(v, float) * 365) * 100
    fig, ax = plt.subplots(figsize=(7.5, 3.2))
    ax.plot(p["T"], ann(p["rv"]), color=MUTED, lw=0.9, alpha=0.8, label="realized (next 24h)")
    ax.plot(p["T"], ann(np.exp(p["l_iv_var"])), color=BLUE, label="Deribit ATM implied")
    ax.plot(p["T"], ann(np.exp(p["l_mkt_mfiv"])), color=ORANGE, label="Polymarket ladder implied (BJN)")
    ax.set_ylabel("annualized volatility, %")
    ax.set_title("24-hour volatility: forecasts made at t vs what happened", loc="left", color=INK)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3)
    fig.tight_layout()
    fig.savefig(config.FIGURES / "fig1_vol_paths.png")
    plt.close(fig)


def fig_pit():
    sc = _try(config.RESULTS / "scores")
    if sc is None:
        return
    models = [("har", "HAR (baseline)"), ("har_m", "HAR + ladder-implied variance"), ("mkt_mz", "Ladder variance, recalibrated")]
    hz = [h for h in ["kx1h", "pm24h", "pm6d"] if h in set(sc["horizon"])]
    fig, axes = plt.subplots(len(hz), 3, figsize=(7.5, 2.0 * len(hz)), sharex=True, squeeze=False)
    for i, h in enumerate(hz):
        for j, (mdl, label) in enumerate(models):
            ax = axes[i, j]
            x = sc[(sc["horizon"] == h) & (sc["model"] == mdl)]["pit"].dropna()
            if len(x):
                ax.hist(x, bins=10, range=(0, 1), color=BLUE, edgecolor="white", linewidth=1.5,
                        weights=np.full(len(x), 10 / len(x)))
                ax.axhline(1, color=MUTED, lw=0.8, ls="--")
            ax.set_ylim(0, 2.2)
            if i == 0:
                ax.set_title(label, fontsize=8.5, color=INK)
            if j == 0:
                ax.set_ylabel(f"{h}\ndensity")
    fig.suptitle("PIT histograms (flat = calibrated; U-shape = intervals too narrow)", x=0.01, ha="left", fontsize=9.5)
    fig.tight_layout()
    fig.savefig(config.FIGURES / "fig2_pit.png")
    plt.close(fig)


def fig_ladder_calibration():
    le = _try(config.RESULTS / "ladder_eval")
    if le is None:
        return
    le = le[le["t"] >= pd.Timestamp(config.EVAL_START, tz="UTC")].dropna(subset=["outcome", "p", "p_harx_d"])
    hz = [h for h in ["kx1h", "pm24h", "pm6d"] if h in set(le["horizon"])]
    if not hz:
        return
    fig, axes = plt.subplots(1, len(hz), figsize=(2.6 * len(hz), 2.8), squeeze=False)
    bins = np.array([0, 0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.65, 0.8, 0.9, 0.95, 0.98, 1.0001])
    for ax, h in zip(axes[0], hz):
        d = le[le["horizon"] == h]
        for col, color, label in [("p", ORANGE, "market price"), ("p_harx_d", BLUE, "model (HAR-X + Deribit)")]:
            b = pd.cut(d[col], bins, right=False)
            g = d.groupby(b, observed=True).agg(x=(col, "mean"), y=("outcome", "mean"), n=("outcome", "size"))
            g = g[g["n"] >= 20]
            ax.plot(g["x"], g["y"], marker="o", ms=4, color=color, label=label)
        ax.plot([0, 1], [0, 1], color=MUTED, lw=0.8, ls="--")
        ax.set_title(h, fontsize=9, color=INK)
        ax.set_xlabel("predicted probability")
        ax.set_aspect("equal")
    axes[0, 0].set_ylabel("realized frequency")
    axes[0, 0].legend(loc="upper left", fontsize=7.5)
    fig.tight_layout()
    fig.savefig(config.FIGURES / "fig3_ladder_calibration.png")
    plt.close(fig)


def fig_pnl():
    tr = _try(config.RESULTS / "trades")
    if tr is None or tr.empty:
        return
    th = max(config.EDGE_THRESHOLDS)
    tr = tr[(tr["model"] == "harx_d") & (tr["threshold"] == th)]
    hz = [h for h in ["kx1h", "pm24h", "pm6d"] if h in set(tr["horizon"])]
    if not hz:
        return
    # one panel per horizon: the number of trades differs by orders of magnitude, so a shared axis would hide two of them
    fig, axes = plt.subplots(1, len(hz), figsize=(2.6 * len(hz), 2.6), squeeze=False)
    for ax, h in zip(axes[0], hz):
        d = tr[tr["horizon"] == h]
        s_ = d.groupby("T")["pnl"].sum().sort_index().cumsum()
        ax.plot(s_.index, s_.values, color=BLUE)
        ax.axhline(0, color=MUTED, lw=0.8)
        ax.set_title(f"{h}: {len(d)} trades, {d['pnl'].mean():+.3f} $/contract", fontsize=8.5, color=INK)
        ax.tick_params(axis="x", labelrotation=45, labelsize=7)
    axes[0, 0].set_ylabel("cumulative profit, $")
    fig.suptitle(f"Trading rule (HAR-X + Deribit), edge threshold {th:.0%}, net of spreads and fees",
                 x=0.01, ha="left", fontsize=9.5)
    fig.tight_layout()
    fig.savefig(config.FIGURES / "fig4_pnl.png")
    plt.close(fig)


def fig_disagreement():
    """When the ladder disagrees with HAR at t, how much of the disagreement is realized?"""
    p = config.TABLES / "disagreement_bins.csv"
    if not p.exists():
        return
    b = pd.read_csv(p)
    hz = [h for h in ["kx1h", "pm24h", "pm6d"] if h in set(b["horizon"])]
    fig, axes = plt.subplots(1, len(hz), figsize=(2.6 * len(hz), 2.7), squeeze=False, sharey=True)
    for ax, h in zip(axes[0], hz):
        d = b[b["horizon"] == h]
        lim = float(np.nanmax(np.abs(d[["gap", "surprise"]].to_numpy()))) * 1.15
        ax.plot([-lim, lim], [-lim, lim], color=MUTED, lw=0.8, ls="--")
        ax.axhline(0, color=GRID, lw=0.8)
        ax.errorbar(d["gap"], d["surprise"], yerr=1.96 * d["se"], fmt="o", ms=4, color=BLUE, ecolor=BLUE, lw=1)
        ax.set_title(h, fontsize=9, color=INK)
        ax.set_xlabel("ladder minus HAR at t, log")
    axes[0, 0].set_ylabel("realized minus HAR, log")
    fig.suptitle("Ladder-HAR disagreement (quintiles) and what was realized; dashed = fully realized",
                 x=0.01, ha="left", fontsize=9.5)
    fig.tight_layout()
    fig.savefig(config.FIGURES / "fig5_disagreement.png")
    plt.close(fig)


def fig_pool_weights():
    w = _try(config.RESULTS / "pool_weights")
    if w is None or w.empty:
        return
    w = w[w["t"] >= pd.Timestamp(config.EVAL_START, tz="UTC")]
    hz = [h for h in ["kx1h", "pm24h", "pm6d"] if h in set(w["horizon"])]
    fig, axes = plt.subplots(1, len(hz), figsize=(2.6 * len(hz), 2.6), squeeze=False, sharey=True)
    labels = {"w_harx": ("HAR-X", MUTED), "w_iv_mz": ("Deribit", ORANGE), "w_mkt_mz": ("ladder", BLUE)}
    for ax, h in zip(axes[0], hz):
        d = w[w["horizon"] == h].sort_values("t")
        for c, (lab, col) in labels.items():
            ax.plot(d["t"], d[c], color=col, label=lab)
        ax.set_ylim(-0.03, 1.03)
        ax.set_title(h, fontsize=9, color=INK)
        ax.tick_params(axis="x", labelrotation=45, labelsize=7)
    axes[0, 0].set_ylabel("weight in the optimal pool")
    axes[0, min(1, len(hz) - 1)].legend(loc="center", fontsize=7.5)
    fig.tight_layout()
    fig.savefig(config.FIGURES / "fig6_pool_weights.png")
    plt.close(fig)


def fig_fluctuation():
    f = _try(config.RESULTS / "fluctuation")
    if f is None or f.empty:
        return
    hz = [h for h in ["kx1h", "pm24h", "pm6d"] if h in set(f["horizon"])]
    fig, axes = plt.subplots(1, len(hz), figsize=(2.6 * len(hz), 2.6), squeeze=False, sharey=True)
    for ax, h in zip(axes[0], hz):
        d = f[f["horizon"] == h]
        cv = float(d["cv"].iloc[0])
        ax.plot(d["t"], d["stat"], color=BLUE)
        for s in (-cv, cv):
            ax.axhline(s, color=MUTED, lw=0.8, ls="--")
        ax.axhline(0, color=GRID, lw=0.8)
        ax.set_title(h, fontsize=9, color=INK)
        ax.tick_params(axis="x", labelrotation=45, labelsize=7)
    axes[0, 0].set_ylabel("rolling DM, HAR + ladder vs HAR")
    fig.suptitle("Fluctuation test: below the lower dashed line the ladder is significantly better",
                 x=0.01, ha="left", fontsize=9.5)
    fig.tight_layout()
    fig.savefig(config.FIGURES / "fig7_fluctuation.png")
    plt.close(fig)


def make_all():
    ensure(config.FIGURES)
    for f in (fig_vol_paths, fig_pit, fig_ladder_calibration, fig_pnl, fig_disagreement, fig_pool_weights,
              fig_fluctuation):
        try:
            f()
        except Exception as e:
            log(f"figure {f.__name__} failed: {e!r}")
    log("figures written to results/figures")
