"""The whole project as a sequence of steps. run_all.py calls these in order.

  download  -> data/raw          (API responses and archives, resumable)
  interim   -> data/interim      (clean tables: minutes, ladders, Deribit implied variance)
  panels    -> data/interim      (one row per forecast occasion with all features and the target)
  models    -> results/forecasts (rolling real-time forecasts of every model)
  evaluate  -> results/tables    (scores, Diebold-Mariano, calibration, trading)
  figures   -> results/figures
  report    -> reports/draft_report.md
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config
from src import build, evaluate, models
from src.measure import deribit_iv
from src.measure.realized import Minutes, seasonal_profiles
from src.utils import ensure, log, read_table, write_table

HORIZONS = list(config.HORIZONS)
LADDER_MODELS = ["harx", "harx_d", "harx_all"]


# ------------------------------------------------------------------ 1. download
def download(skip: tuple = ()):
    """Each source is downloaded independently: if one fails (e.g. blocked), the others still run."""
    from src.download import binance, deribit, kalshi, polymarket
    jobs = [("binance", "Binance 1-minute candles", binance.download),
            ("dvol", "DVOL", deribit.download_dvol),
            ("polymarket", "Polymarket ladders", polymarket.download_history),
            ("deribit", "Deribit option trades", deribit.download_trades),
            ("kalshi", "Kalshi ladders (the longest step)", kalshi.download_history)]
    failed = []
    for key, label, fn in jobs:
        if key in skip:
            continue
        log(f"downloading {label}")
        try:
            fn()
        except Exception as e:
            failed.append(key)
            log(f"!! {label}: download failed: {e}")
            if key == "kalshi":
                log("!! Kalshi is optional: the project runs without it (the 55-minute horizon is then empty). "
                    "See README, section 12.")
    if failed:
        log(f"!! sources that failed: {', '.join(failed)}. Everything else continues; re-run later to resume them.")


# ------------------------------------------------------------------ 2. interim tables
def interim():
    from src.download import binance, deribit, kalshi, polymarket
    ensure(config.INTERIM)
    log("interim: Binance minutes")
    write_table(binance.build_interim(), config.INTERIM / "minutes")
    log("interim: Polymarket ladders")
    lad, px = polymarket.build_interim()
    if len(lad):
        write_table(lad, config.INTERIM / "pm_ladder")
        write_table(px, config.INTERIM / "pm_prices")
    log("interim: Kalshi ladders")
    kx = kalshi.build_interim()
    if len(kx):
        write_table(kx, config.INTERIM / "kalshi")
    log("interim: Deribit implied variance")
    tr = deribit.build_interim_trades()
    if len(tr):
        hz = {h: v["hours"] for h, v in config.HORIZONS.items() if h != "kx1h"}
        write_table(deribit_iv.from_trades(tr, hz), config.INTERIM / "deribit_iv")


def _maybe(name):
    try:
        return read_table(config.INTERIM / name)
    except FileNotFoundError:
        return None


# ------------------------------------------------------------------ 3. panels
def panels():
    from src.download import deribit
    mins = read_table(config.INTERIM / "minutes")
    m = Minutes(mins)
    log(f"minutes: {m.first:%Y-%m-%d} .. {m.last:%Y-%m-%d %H:%M}")
    profiles = seasonal_profiles(m)
    try:
        dvol = deribit.load_dvol()
    except FileNotFoundError:
        dvol = None
    div = _maybe("deribit_iv")
    lad, px, kx = _maybe("pm_ladder"), _maybe("pm_prices"), _maybe("kalshi")
    strikes = []
    for h in ("pm24h", "pm6d"):
        if lad is not None:
            strikes.append(build.polymarket_strikes(lad, px, h))
    if kx is not None:
        strikes.append(build.kalshi_strikes(kx))
    strikes = [s for s in strikes if s is not None and len(s)]
    snaps = build.ladder_snapshots(pd.concat(strikes, ignore_index=True)) if strikes else pd.DataFrame()
    if len(snaps):
        write_table(snaps, config.INTERIM / "ladder_snaps")
    for s in strikes:
        write_table(s, config.INTERIM / f"strikes_{s['horizon'].iloc[0]}")
    if len(snaps):
        ladder_coverage(lad, kx, strikes, snaps).to_csv(ensure(config.TABLES) / "ladder_coverage.csv", index=False)
    if kx is not None and len(kx):
        chk = kalshi_density_check(build.kalshi_strikes(kx, uniform=False))
        if len(chk):
            chk.to_csv(ensure(config.TABLES) / "kalshi_strike_density_check.csv", index=False)
    for h in HORIZONS:
        log(f"panel {h}")
        p = build.build_panel(h, m, profiles, dvol, div, snaps)
        write_table(p, config.INTERIM / f"panel_{h}")
        log(f"panel {h}: {len(p)} occasions, {p['l_mkt_mfiv'].notna().sum()} with a ladder, "
            f"{p['l_iv_var'].notna().sum()} with Deribit ATM IV")
    return m


# ------------------------------------------------------------------ 4. models
def run_models(m=None):
    ensure(config.RESULTS)
    if m is None:
        m = Minutes(read_table(config.INTERIM / "minutes"))
    for h in HORIZONS:
        p = read_table(config.INTERIM / f"panel_{h}")
        log(f"models {h}")
        fc, coefs = models.run_all_models(p, m)
        write_table(fc, config.RESULTS / f"forecasts_{h}")
        for name, cf in coefs.items():
            if len(cf):
                write_table(cf, config.RESULTS / "coefs" / f"{h}_{name}")


# ------------------------------------------------------------------ 5. evaluation
def run_evaluation():
    ensure(config.TABLES)
    eval_start = pd.Timestamp(config.EVAL_START, tz="UTC")
    live_start = pd.Timestamp(config.LIVE_START, tz="UTC")
    all_sc, panels_by_h = [], {}
    lad_summ, lad_dm, lad_cal, trade_rows, trades_all, le_all = [], [], [], [], [], []
    for h in HORIZONS:
        p = read_table(config.INTERIM / f"panel_{h}")
        panels_by_h[h] = p
        fc = read_table(config.RESULTS / f"forecasts_{h}")
        fc = fc[np.isfinite(fc["y"]) & (fc["t"] >= eval_start)]
        sc = evaluate.scores(fc)
        all_sc.append(sc)
        st = _maybe(f"strikes_{h}")
        if st is not None and len(st):
            fc_all = read_table(config.RESULTS / f"forecasts_{h}")
            le = evaluate.ladder_eval(st, fc_all, p, LADDER_MODELS)
            le_all.append(le)
            s1, s2, s3 = evaluate.ladder_scores(le, LADDER_MODELS)
            lad_summ.append(s1), lad_dm.append(s2), lad_cal.append(s3)
            for mdl in ("harx", "harx_d"):
                tr_sum, tr = evaluate.trading(le, mdl)
                trade_rows.append(tr_sum)
                trades_all.append(tr)
    sc = pd.concat(all_sc, ignore_index=True)
    write_table(sc, config.RESULTS / "scores")
    # full evaluation sample per model, and the common sample on which every model has a forecast
    evaluate.summary_table(sc).to_csv(config.TABLES / "scores_all_rows.csv", index=False)
    common = []
    for h, g in sc.groupby("horizon"):
        mods = [x for x in g["model"].unique()]
        common.append(evaluate.common_sample(g, mods))
    cs = pd.concat(common, ignore_index=True)
    evaluate.summary_table(cs).to_csv(config.TABLES / "scores_common_sample.csv", index=False)
    # time-series models only: longer sample (does not need ladders)
    ts_models = ["naive", "garch", "har", "harx", "harx_d", "iv_raw", "iv_mz"]
    ts_common = pd.concat([evaluate.common_sample(g[g["model"].isin(ts_models)], [m for m in ts_models if m in set(g["model"])])
                           for _, g in sc.groupby("horizon")], ignore_index=True)
    evaluate.summary_table(ts_common).to_csv(config.TABLES / "scores_time_series_sample.csv", index=False)
    dm = pd.concat([evaluate.dm_table(cs), evaluate.dm_table(ts_common).assign(sample="time-series")], ignore_index=True)
    dm["sample"] = dm["sample"].fillna("common")
    dm.to_csv(config.TABLES / "diebold_mariano.csv", index=False)
    evaluate.bias_table(panels_by_h).to_csv(config.TABLES / "implied_bias.csv", index=False)
    live = cs[cs["t"] >= live_start]
    if len(live):
        evaluate.summary_table(live).to_csv(config.TABLES / "scores_live_period.csv", index=False)
    if lad_summ:
        pd.concat(lad_summ).to_csv(config.TABLES / "ladder_scores.csv", index=False)
        pd.concat(lad_dm).to_csv(config.TABLES / "ladder_dm.csv", index=False)
        pd.concat(lad_cal).to_csv(config.TABLES / "ladder_calibration.csv", index=False)
        pd.concat(trade_rows).to_csv(config.TABLES / "trading.csv", index=False)
        tr = pd.concat(trades_all, ignore_index=True)
        if len(tr):
            write_table(tr, config.RESULTS / "trades")
        write_table(pd.concat(le_all, ignore_index=True), config.RESULTS / "ladder_eval")
    data_summary(panels_by_h).to_csv(config.TABLES / "data_summary.csv", index=False)
    from src import live_forecast
    live_tab = live_forecast.evaluate_live()
    if len(live_tab):
        log(f"live forecasts scored: {int(live_tab['n'].sum())} forecast-model pairs")
    log("evaluation tables written to results/tables")


def data_summary(panels_by_h: dict) -> pd.DataFrame:
    rows = []
    for h, p in panels_by_h.items():
        has = p["y"].notna()
        rows.append({"horizon": h, "occasions": int(has.sum()), "first": p.loc[has, "t"].min(), "last": p.loc[has, "t"].max(),
                     "with_ladder": int((has & p["l_mkt_mfiv"].notna()).sum()),
                     "first_ladder": p.loc[has & p["l_mkt_mfiv"].notna(), "t"].min(),
                     "with_deribit_atm": int((has & p["l_iv_var"].notna()).sum()),
                     "with_dvol": int((has & p["l_dvol_var"].notna()).sum()),
                     "median_ladder_strikes": p["mkt_n"].median(),
                     "eval_occasions": int((has & (p["t"] >= pd.Timestamp(config.EVAL_START, tz="UTC"))).sum())})
    return pd.DataFrame(rows)


def ladder_coverage(lad, kx, strikes: list, snaps: pd.DataFrame) -> pd.DataFrame:
    """Does the history contain prices for all strikes at the forecast time? (instructor's point 4)"""
    listed = {}
    if lad is not None and len(lad):
        listed["pm"] = lad.groupby("settle_ts")["strike"].nunique()
    if kx is not None and len(kx):
        listed["kx"] = kx.groupby("settle_ts")["ticker"].nunique()
    rows = []
    for st in strikes:
        h = st["horizon"].iloc[0]
        priced = st.groupby("T")["strike"].nunique()
        lst = listed.get("kx" if h == "kx1h" else "pm", pd.Series(dtype=float)).reindex(priced.index)
        sn = snaps[snaps["horizon"] == h]
        rows.append({"horizon": h, "snapshots": len(priced),
                     "median_strikes_listed": float(lst.median()) if len(lst) else np.nan,
                     "median_strikes_priced_at_t": float(priced.median()),
                     "share_snapshots_all_strikes_priced": float((priced >= lst).mean()) if len(lst) else np.nan,
                     "median_informative_strikes": float(sn["mkt_mfiv_n"].median()),
                     "share_with_bjn_variance": float(sn["mkt_mfiv"].notna().mean()),
                     "share_with_lognormal_fit": float(sn["mkt_var_c"].notna().mean()),
                     "share_non_monotone": float((sn["mkt_nonmono"] > 0).mean())})
    return pd.DataFrame(rows)


def kalshi_density_check(st: pd.DataFrame) -> pd.DataFrame:
    """How much precision does the economical archive download lose?

    Archived Kalshi events are downloaded with every second strike ($200) within +-1.5% of the price;
    recent events with every strike ($100) within +-2.5%. On the recent events we compute the BJN variance
    both ways and compare: if the two agree closely, the economical archive is good enough.
    """
    from src.measure import ladder as L
    rows = []
    for T, g in st.groupby("T"):
        k = (g["strike"] + 0.01).round()
        if k.diff().dropna().min() > 150 or len(g) < 8:   # this snapshot already has the economical design
            continue
        ref = float(k.iloc[(g["p"] - 0.5).abs().argmin()])
        red = g[((k / ref - 1).abs() <= 0.015) & (k % 200 == 0)]
        full = L.mfiv(g["strike"], g["p"])
        econ = L.mfiv(red["strike"], red["p"])
        rows.append({"T": T, "n_full": len(g), "n_econ": len(red), "mfiv_full": full["mfiv"], "mfiv_econ": econ["mfiv"]})
    d = pd.DataFrame(rows).dropna()
    if d.empty:
        return d
    lf, le = np.log(d["mfiv_full"]), np.log(d["mfiv_econ"])
    ann = lambda v: np.sqrt(v * 365 * 24 / (55 / 60)) * 100
    return pd.DataFrame([{
        "snapshots": len(d), "median_strikes_full": d["n_full"].median(), "median_strikes_econ": d["n_econ"].median(),
        "corr_log_variance": float(np.corrcoef(lf, le)[0, 1]),
        "median_abs_log_diff": float(np.median(np.abs(lf - le))),
        "mean_log_diff_econ_minus_full": float(np.mean(le - lf)),
        "median_abs_diff_vol_points": float(np.median(np.abs(ann(d["mfiv_econ"]) - ann(d["mfiv_full"]))))}])
