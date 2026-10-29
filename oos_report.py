"""Out-of-sample report in three samples, with the same models and the same metrics:

  history   EVAL_START .. NEW_DATA_START   pseudo-out-of-sample: rolling forecasts, each made with data before t
  new       NEW_DATA_START .. now          the same rolling forecasts on data that did not exist when the
                                           specification was frozen (15 Oct 2026): no tuning could see it
  live      forecasts written to forecasts_live/ (hash-chained ledger, git) before the outcome was known

    python oos_report.py            reads results/, writes results/tables/oos_*.csv, results/figures/fig8_oos.png,
                                    reports/oos_report.md and results/oos_summary.json

It also checks that the model code is byte-identical to the frozen version and that the live ledger is intact.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import config
from src import evaluate
from src.utils import ensure, log, read_table

MAIN = ["naive", "har", "harx", "harx_d", "harx_all", "combo", "mkt_raw", "mkt_ln_raw", "iv_raw"]
BRIER_MODELS = ["naive", "har", "harx", "harx_d", "harx_all", "combo"]
STEPS = [("har", "naive"), ("harx", "har"), ("harx_d", "harx"), ("harx_all", "harx_d"),
         ("combo", "harx_all"), ("harx_all", "mkt_ln_raw"), ("harx_d", "mkt_ln_raw")]
NAMES = {"naive": "Last window", "har": "Past volatility (HAR)", "harx": "+ Calendar & jumps", "harx_d": "+ Options",
         "harx_all": "+ Ladder", "combo": "Equal-weight blend", "pool": "Learned-weight blend",
         "mkt_raw": "Market: full ladder", "mkt_ln_raw": "Market: ladder centre", "iv_raw": "Market: Deribit",
         "market": "Market price"}
HL = {"kx1h": "1 hour", "pm24h": "24 hours", "pm6d": "6 days"}


# ------------------------------------------------------------------ checks
def code_check() -> pd.DataFrame:
    rows = []
    for rel, want in config.FROZEN_CODE_SHA256.items():
        p = config.ROOT / rel
        got = hashlib.sha256(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest() if p.exists() else "missing"
        rows.append({"file": rel, "frozen_sha256": want[:16], "current_sha256": got[:16], "identical": got == want})
    return pd.DataFrame(rows)


def samples(t: pd.Series) -> pd.Series:
    t = pd.to_datetime(t, utc=True)
    out = pd.Series("history", index=t.index)
    out[t >= pd.Timestamp(config.NEW_DATA_START, tz="UTC")] = "new"
    out[t < pd.Timestamp(config.EVAL_START, tz="UTC")] = None
    return out


# ------------------------------------------------------------------ density forecasts of ln RV
def density_tables(sc: pd.DataFrame):
    rows, dms = [], []
    sc = sc.copy()
    sc["sample"] = samples(sc["t"])
    sc = sc.dropna(subset=["sample"])
    for (h, smp), g in sc.groupby(["horizon", "sample"]):
        mods = list(g["model"].unique())
        cs = evaluate.common_sample(g, mods)
        if cs.empty:
            cs = evaluate.common_sample(g, [m for m in MAIN if m in mods])
        if cs.empty:
            continue
        tab = cs.groupby("model").agg(n=("crps", "size"), crps=("crps", "mean"), logscore=("logscore", "mean"),
                                      qlike=("qlike", "mean"), cov50=("in50", "mean"), cov90=("in90", "mean"))
        for m, r in tab.iterrows():
            rows.append({"horizon": h, "sample": smp, "model": m, **r.to_dict(),
                         "first_t": cs["t"].min(), "last_t": cs["t"].max()})
        for a, b in STEPS:
            if a not in tab.index or b not in tab.index:
                continue
            pa, pb = cs[cs["model"] == a].set_index("t"), cs[cs["model"] == b].set_index("t")
            idx = pa.index.intersection(pb.index)
            if len(idx) < 8:
                continue
            for loss in ("crps", "logscore", "qlike"):
                la, lb = pa.loc[idx, loss].to_numpy(float), pb.loc[idx, loss].to_numpy(float)
                if loss == "logscore":
                    la, lb = -la, -lb
                r = evaluate.dm_test(la, lb, evaluate.overlap(h))
                dms.append({"horizon": h, "sample": smp, "A": a, "B": b, "loss": loss, **r})
    return pd.DataFrame(rows), pd.DataFrame(dms)


# ------------------------------------------------------------------ strike probabilities
def add_model_probs(le: pd.DataFrame, fc: pd.DataFrame, mods: list[str]) -> pd.DataFrame:
    g = le.copy()
    for mdl in mods:
        if f"p_{mdl}" in g and g[f"p_{mdl}"].notna().any():
            continue
        f = fc[fc["model"] == mdl].set_index("T")
        f = f[~f.index.duplicated()]
        if f.empty:
            continue
        sub = f.reindex(pd.DatetimeIndex(g["T"]))
        m = np.column_stack([sub[f"m{j}"].to_numpy(float) for j in (1, 2, 3)])
        s = np.column_stack([sub[f"s{j}"].to_numpy(float) for j in (1, 2, 3)])
        w = np.nan_to_num(np.column_stack([sub[f"w{j}"].to_numpy(float) for j in (1, 2, 3)]))
        m, s = np.where(w > 0, m, 0), np.where(w > 0, s, 1)
        p = evaluate.digital_prob(g["S_ref"].to_numpy(float), g["strike"].to_numpy(float), m, s, w)
        p[~np.isfinite(sub["m1"].to_numpy(float))] = np.nan
        g[f"p_{mdl}"] = p
    return g


def brier_tables(le: pd.DataFrame, sample_col: str = "sample"):
    rows, dms = [], []
    eps = 1e-4
    for (h, smp), g in le.groupby(["horizon", sample_col]):
        mods = [m for m in BRIER_MODELS if f"p_{m}" in g and g[f"p_{m}"].notna().any()]
        d = g.dropna(subset=["outcome", "p"] + [f"p_{m}" for m in mods])
        if d.empty:
            continue
        srcs = {"market": "p", **{m: f"p_{m}" for m in mods}}
        for k, c in srcs.items():
            d[f"b_{k}"] = (np.clip(d[c].to_numpy(float), eps, 1 - eps) - d["outcome"]) ** 2
        snap = d.groupby("T")[[f"b_{k}" for k in srcs]].sum()
        for k in srcs:
            r = {"horizon": h, "sample": smp, "source": k, "n_strikes": len(d), "n_snapshots": len(snap),
                 "brier": d[f"b_{k}"].mean(), "first_T": d["T"].min(), "last_T": d["T"].max()}
            if k != "market" and len(snap) >= 5:
                t = evaluate.dm_test(snap[f"b_{k}"].to_numpy(), snap["b_market"].to_numpy(), evaluate.overlap(h))
                r.update({"dm_vs_market": t["dm"], "p_vs_market": t["p"]})
            rows.append(r)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ live forecasts
def live_tables(fc_hist: dict):
    out = {}
    rows_p = config.TABLES / "live_scores_rows.csv"
    if rows_p.exists():
        sc = pd.read_csv(rows_p, parse_dates=["t"])
        if len(sc):
            tab = sc.groupby(["horizon", "model"]).agg(n=("crps", "size"), crps=("crps", "mean"),
                                                       logscore=("logscore", "mean"), qlike=("qlike", "mean"),
                                                       cov50=("in50", "mean"), cov90=("in90", "mean")).reset_index()
            tab["sample"] = "live"
            out["density"] = tab
    lad_p = config.TABLES / "live_ladder_records.csv"
    if lad_p.exists():
        lr = pd.read_csv(lad_p)
        if len(lr):
            lr["sample"] = "live"
            lr["T"] = pd.to_datetime(lr["T"], utc=True)
            out["brier"] = brier_tables(lr)
    # live vs replay: the same model at the same t, computed live and in the nightly rolling run
    fr = config.TABLES / "live_forecasts_rows.csv"
    if fr.exists():
        lv = pd.read_csv(fr, parse_dates=["t"])
        rows = []
        for h, g in lv.groupby("horizon"):
            hist = fc_hist.get(h)
            if hist is None:
                continue
            hist = hist[["t", "model", "mean_log"]].rename(columns={"mean_log": "replay"})
            hist["t"] = pd.to_datetime(hist["t"], utc=True)
            g = g.copy()
            g["t"] = pd.to_datetime(g["t"], utc=True)
            j = g.merge(hist, on=["t", "model"], how="inner")
            for mdl, gm in j.groupby("model"):
                if len(gm) >= 2:
                    rows.append({"horizon": h, "model": mdl, "n": len(gm),
                                 "corr": float(np.corrcoef(gm["mean_log"], gm["replay"])[0, 1]),
                                 "mean_abs_diff": float((gm["mean_log"] - gm["replay"]).abs().mean())})
        out["replay"] = pd.DataFrame(rows)
    return out


# ------------------------------------------------------------------ live order books (spreads measured by the logger)
def live_spreads() -> pd.DataFrame:
    rows = []
    for src, bid, ask in (("pm_books", "bid", "ask"), ("kalshi", "yes_bid", "yes_ask")):
        # the project's logger, plus the files of the first standalone logger copied to data/live_logger_root
        files = sorted((config.LIVE / src).glob("*.csv.gz")) + sorted((config.DATA / "live_logger_root" / src).glob("*.csv.gz"))
        if not files:
            continue
        parts = []
        for f in files:
            try:
                parts.append(pd.read_csv(f))
            except Exception:
                pass
        if not parts:
            continue
        d = pd.concat(parts, ignore_index=True)
        d["mid"], d["spr"] = (d[bid] + d[ask]) / 2, d[ask] - d[bid]
        d = d[(d["mid"] > 0.02) & (d["mid"] < 0.98) & d["spr"].notna()]
        if src == "pm_books":
            dts = (pd.to_datetime(d["settle_ts"], utc=True, format="mixed") - pd.to_datetime(d["snap_ts"], utc=True, format="mixed")).dt.total_seconds() / 86400
            d["bucket"] = pd.cut(dts, [0, 1.5, 3, 5.5, 8], labels=["Polymarket ≤1.5 d", "Polymarket 1.5–3 d", "Polymarket 3–5.5 d", "Polymarket 5.5–8 d"])
        else:
            d["bucket"] = "Kalshi 1 h"
        for b, g in d.groupby("bucket", observed=True):
            rows.append({"venue_horizon": b, "quotes": len(g), "snapshots": g["snap_ts"].nunique(),
                         "median_spread_c": 100 * g["spr"].median(), "mean_spread_c": 100 * g["spr"].mean(),
                         "first": g["snap_ts"].min(), "last": g["snap_ts"].max()})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ figure
def figure(dens: pd.DataFrame, brier: pd.DataFrame):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    hz = [h for h in ("kx1h", "pm24h", "pm6d") if h in set(dens["horizon"])]
    if not hz:
        return
    fig, axes = plt.subplots(1, len(hz), figsize=(4.2 * len(hz), 3.4), squeeze=False)
    mods = ["har", "harx", "harx_d", "harx_all", "mkt_ln_raw"]
    colors = {"history": "#3333B3", "new": "#1E7B34", "live": "#A12424"}
    for ax, h in zip(axes[0], hz):
        d = dens[(dens["horizon"] == h) & dens["model"].isin(mods)]
        smps = [s for s in ("history", "new", "live") if s in set(d["sample"])]
        x = np.arange(len(mods))
        w = 0.8 / max(1, len(smps))
        for i, smp in enumerate(smps):
            v = d[d["sample"] == smp].set_index("model").reindex(mods)["crps"]
            n = int(d[d["sample"] == smp]["n"].max())
            ax.bar(x + (i - (len(smps) - 1) / 2) * w, v, w, label=f"{smp} (n={n})", color=colors[smp])
        ax.set_xticks(x, [NAMES[m].replace("Market: ", "Mkt: ") for m in mods], rotation=35, ha="right", fontsize=8)
        ax.set_title(f"CRPS of ln RV, {HL[h]}", fontsize=10)
        ax.legend(fontsize=7)
        ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    ensure(config.FIGURES)
    fig.savefig(config.FIGURES / "fig8_oos.png", dpi=160)
    plt.close(fig)


# ------------------------------------------------------------------ markdown
def md(df: pd.DataFrame, floatfmt=4) -> str:
    if df is None or df.empty:
        return "_no data yet_\n"
    d = df.copy()
    for c in d.columns:
        if pd.api.types.is_float_dtype(d[c]):
            d[c] = d[c].map(lambda v: "" if pd.isna(v) else f"{v:.{floatfmt}f}")
    head = "| " + " | ".join(map(str, d.columns)) + " |\n|" + "---|" * len(d.columns) + "\n"
    return head + "".join("| " + " | ".join(map(str, r)) + " |\n" for r in d.itertuples(index=False))


def main(argv=None):
    ensure(config.TABLES)
    code = code_check()
    from src.live_forecast import verify_ledger
    ok, led_msg = verify_ledger()
    try:
        sc = read_table(config.RESULTS / "scores")
    except FileNotFoundError:
        log("oos_report: results/scores missing — run the pipeline first")
        return
    fc = {}
    for h in config.HORIZONS:
        try:
            fc[h] = read_table(config.RESULTS / f"forecasts_{h}")
        except FileNotFoundError:
            pass
    dens, dm = density_tables(sc)
    try:
        le = read_table(config.RESULTS / "ladder_eval")
        le["sample"] = samples(le["t"])
        le = le.dropna(subset=["sample"])
        parts = [add_model_probs(g, fc[h], BRIER_MODELS) for h, g in le.groupby("horizon") if h in fc]
        br = brier_tables(pd.concat(parts, ignore_index=True)) if parts else pd.DataFrame()
    except FileNotFoundError:
        br = pd.DataFrame()
    live = live_tables(fc)
    if "density" in live:
        dens = pd.concat([dens, live["density"]], ignore_index=True)
    if "brier" in live:
        br = pd.concat([br, live["brier"]], ignore_index=True)
    spreads = live_spreads()
    code.to_csv(config.TABLES / "oos_code_check.csv", index=False)
    dens.to_csv(config.TABLES / "oos_density.csv", index=False)
    dm.to_csv(config.TABLES / "oos_dm.csv", index=False)
    br.to_csv(config.TABLES / "oos_brier.csv", index=False)
    spreads.to_csv(config.TABLES / "live_spreads.csv", index=False)
    if "replay" in live:
        live["replay"].to_csv(config.TABLES / "oos_live_vs_replay.csv", index=False)
    figure(dens, br)

    # ---- markdown summary
    L = ["# Out-of-sample report", "",
         f"Specification frozen {config.SPEC_FROZEN}; history = {config.EVAL_START} .. {config.NEW_DATA_START}; "
         f"new = from {config.NEW_DATA_START}; live = forecasts in the hash-chained ledger.", "",
         f"**Model code identical to the frozen version:** {'yes' if code['identical'].all() else 'NO — see table'}  ",
         f"**Live ledger:** {led_msg}", ""]
    for h in ("kx1h", "pm24h", "pm6d"):
        d = dens[dens["horizon"] == h]
        if d.empty:
            continue
        L.append(f"## {HL[h]}\n")
        piv = d[d["model"].isin(MAIN)].pivot_table(index="model", columns="sample", values="crps").reindex(MAIN).dropna(how="all")
        piv.index = [NAMES[m] for m in piv.index]
        ns = d.groupby("sample")["n"].max().to_dict()
        piv.columns = [f"CRPS {c} (n={int(ns.get(c, 0))})" for c in piv.columns]
        L.append(md(piv.reset_index().rename(columns={"index": "forecast"}), 3))
        b = br[br["horizon"] == h] if len(br) else br
        if len(b):
            pb = b.pivot_table(index="source", columns="sample", values="brier")
            pb = pb.reindex([s for s in ["market"] + BRIER_MODELS if s in pb.index])
            pb.index = [NAMES[s] for s in pb.index]
            nsn = b.groupby("sample")["n_snapshots"].max().to_dict()
            pb.columns = [f"Brier {c} ({int(nsn.get(c, 0))} ladders)" for c in pb.columns]
            L.append(md(pb.reset_index().rename(columns={"index": "source"}), 4))
        dd = dm[(dm["horizon"] == h) & (dm["loss"] == "crps")] if len(dm) else dm
        if len(dd):
            x = dd[["sample", "A", "B", "n", "mean_diff", "dm", "p"]].copy()
            x["A"] = x["A"].map(NAMES)
            x["B"] = x["B"].map(NAMES)
            L.append("Diebold–Mariano on CRPS (negative = A better):\n\n" + md(x, 3))
    if len(spreads):
        L.append("## Spreads measured by the live logger\n\n" + md(spreads.drop(columns=["first", "last"]), 2))
    if "replay" in live and len(live["replay"]):
        L.append("## Live forecasts vs the nightly replay of the same t\n\n" + md(live["replay"], 3))
    L.append("## Code check\n\n" + md(code))
    ensure(config.REPORTS)
    (config.REPORTS / "oos_report.md").write_text("\n".join(L), encoding="utf-8")
    summ = {"code_identical": bool(code["identical"].all()), "ledger": led_msg,
            "samples": dens.groupby(["horizon", "sample"])["n"].max().reset_index().to_dict("records") if len(dens) else []}
    (config.RESULTS / "oos_summary.json").write_text(json.dumps(summ, indent=1, default=str), encoding="utf-8")
    log(f"oos report: {config.REPORTS / 'oos_report.md'} ({led_msg}; code identical: {summ['code_identical']})")


if __name__ == "__main__":
    main(sys.argv[1:])
