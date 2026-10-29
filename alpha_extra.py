"""Historical follow-ups from the project's list of ideas (section 11.3 of the project document).

  X1  Kalshi fees rounded up to the cent per order: does the +2.4c of the 5c rule on the hour survive?
      (results/trades.parquet, one-contract and ten-contract orders)
  X2  Does the hourly edge grow towards settlement?  Kalshi quotes 55, 40, 25 and 10 minutes before the hour
      (needs the whole-hour candles: python -m src.download.kalshi 2026-07-01 --minutes 60):
      Brier of market / model / hybrid, and the pre-registered hybrid rule at each horizon
  X3  Box arbitrage between Kalshi "above" (KXBTCD) and range buckets (KXBTC) of the same event, minute by
      minute (needs: python -m src.download.kalshi 2026-07-01 --series KXBTC --minutes 60
                  and    python -m src.download.kalshi 2026-07-01 --minutes 60)

    python alpha_extra.py            -> results/tables/extra_*.csv, reports/alpha_extra.md
Every part is skipped with a note when its data are not there yet.
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

import config
from alpha_live import box_arbitrage, kalshi_fee
from src import alpha, evaluate
from src.download import kalshi
from src.measure import ladder
from src.utils import ensure, log, read_table

TAUS = [55, 40, 25, 10]          # minutes before settlement
HYB_THRESHOLD = 0.05             # the rule pre-registered on 15 October (hyb_harx_d, 5c)


# ------------------------------------------------------------------ X1 fee rounding
def x1_fee_rounding() -> pd.DataFrame:
    try:
        tr = read_table(config.RESULTS / "trades")
    except FileNotFoundError:
        return pd.DataFrame()
    tr = tr[tr["horizon"] == "kx1h"].copy()
    if tr.empty:
        return pd.DataFrame()
    price = np.where(tr["side"] == "buy", tr["ask"], tr["bid"])
    gross = tr["pnl"] + config.KALSHI_FEE_RATE * price * (1 - price)    # pnl was computed with the exact fee
    rows = []
    for label, f in (("exact fee (as in the project)", config.KALSHI_FEE_RATE * price * (1 - price)),
                     ("rounded up, 1 contract per order", kalshi_fee(price)),
                     ("rounded up, 10 contracts per order", np.ceil(config.KALSHI_FEE_RATE * 10 * price * (1 - price) * 100 - 1e-9) / 100 / 10)):
        d = tr.assign(pnl2=gross - f)
        for (mdl, th), g in d.groupby(["model", "threshold"]):
            per = g.groupby("T")["pnl2"].sum().sort_index()
            m, se, n = evaluate.hac_mean(per.to_numpy(float), evaluate.nw_lag(len(per), 1))
            rows.append({"fee": label, "model": mdl, "threshold": th, "trades": len(g),
                         "pnl_per_contract": g["pnl2"].mean(), "t": m / se if se else np.nan})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ helpers for the whole-hour candles
def load_candles(series: str, minutes: int = 60) -> pd.DataFrame:
    d = kalshi.raw_dir(series, minutes)
    files = sorted(d.glob("*.csv.gz"))
    if not files:
        return pd.DataFrame()
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df["settle_ts"] = pd.to_datetime(df["close_time"], utc=True)
    df["candle_end"] = pd.to_datetime(df["end_ts"], unit="s", utc=True)
    df["outcome"] = np.where(df["result"] == "yes", 1.0, np.where(df["result"] == "no", 0.0, np.nan))
    return df


def quotes_at(df: pd.DataFrame, tau: int) -> pd.DataFrame:
    """Last candle that ends at or before T - tau and is at most 5 minutes old, per strike."""
    q = df.copy()
    q["t"] = q["settle_ts"] - pd.Timedelta(minutes=tau)
    q = q[(q["candle_end"] <= q["t"]) & (q["candle_end"] > q["t"] - pd.Timedelta(minutes=5))]
    q = q.sort_values("candle_end").groupby("ticker").tail(1)
    q = q[(q["yes_bid"] > 0) & (q["yes_ask"] < 1) & (q["yes_ask"] > q["yes_bid"])]
    return q


# ------------------------------------------------------------------ X2 edge vs time to settlement
def x2_time_to_settlement() -> tuple[pd.DataFrame, str]:
    kx = load_candles("KXBTCD", 60)
    if kx.empty:
        return pd.DataFrame(), "no whole-hour Kalshi candles yet (python -m src.download.kalshi 2026-07-01 --minutes 60)"
    try:
        fc = read_table(config.RESULTS / "forecasts_kx1h")
        mins = read_table(config.INTERIM / "minutes").set_index("open_time")["close"]
    except FileNotFoundError as e:
        return pd.DataFrame(), f"pipeline results missing ({e})"
    fc = fc[fc["model"] == "harx_d"].drop_duplicates("T").set_index("T")
    # BRTI / Binance basis from settled events (median of the previous 48, known before t)
    ev = kx.groupby("settle_ts")["expiration_value"].first().dropna().sort_index()
    pxT = mins.reindex(ev.index - pd.Timedelta(minutes=1))
    ratio = pd.Series(ev.to_numpy(float) / pxT.to_numpy(float), index=ev.index).shift(1).rolling(48, min_periods=5).median()
    rows, trades = [], []
    for tau in TAUS:
        q = quotes_at(kx, tau)
        q = q[q["settle_ts"].isin(fc.index)]
        if q.empty:
            continue
        q["mid"] = (q["yes_bid"] + q["yes_ask"]) / 2
        S = mins.reindex(q["t"] - pd.Timedelta(minutes=1)).to_numpy(float)
        q["S_ref"] = S * q["settle_ts"].map(ratio).fillna(1.0).to_numpy(float)
        f = fc.reindex(pd.DatetimeIndex(q["settle_ts"]))
        frac = np.log(tau / 55.0)        # remaining share of the hour's variance (uniform within the hour)
        m = np.column_stack([f[f"m{j}"].to_numpy(float) + frac for j in (1, 2, 3)])
        s = np.column_stack([f[f"s{j}"].to_numpy(float) for j in (1, 2, 3)])
        w = np.nan_to_num(np.column_stack([f[f"w{j}"].to_numpy(float) for j in (1, 2, 3)]))
        m, s = np.where(w > 0, m, 0), np.where(w > 0, s, 1)
        q["p_model"] = evaluate.digital_prob(q["S_ref"].to_numpy(float), q["strike"].to_numpy(float), m, s, w)
        # hybrid: the market's own median, the model's width
        centers = {}
        for T, g in q.groupby("settle_ts"):
            fit = ladder.fit_lognormal(g["strike"], g["mid"])
            centers[T] = fit["m"]
        cen = q["settle_ts"].map(centers).to_numpy(float)
        q["p_hyb"] = alpha.mix_digital(cen, q["strike"].to_numpy(float), m, s, w, False)
        q = q.dropna(subset=["outcome", "p_model", "p_hyb", "S_ref"])
        q = q[np.isfinite(q["p_hyb"])]
        if q.empty:
            continue
        b = lambda p: ((np.clip(p, 1e-4, 1 - 1e-4) - q["outcome"]) ** 2)   # noqa: E731
        snap = q.assign(bm=b(q["mid"]), bmod=b(q["p_model"]), bh=b(q["p_hyb"])).groupby("settle_ts")[["bm", "bmod", "bh"]].sum()
        dm = evaluate.dm_test(snap["bh"].to_numpy(), snap["bm"].to_numpy(), 1)
        # the pre-registered hybrid rule at this horizon
        buy = q["p_hyb"] - q["yes_ask"] > HYB_THRESHOLD
        sell = (q["yes_bid"] - q["p_hyb"] > HYB_THRESHOLD) & ~buy
        pnl = np.where(buy, q["outcome"] - q["yes_ask"] - kalshi_fee(q["yes_ask"]), 0.0) + \
            np.where(sell, q["yes_bid"] - q["outcome"] - kalshi_fee(q["yes_bid"]), 0.0)
        tr = q.assign(pnl=pnl)[buy | sell]
        per = tr.groupby("settle_ts")["pnl"].sum().sort_index()
        mm, se, n = evaluate.hac_mean(per.to_numpy(float), evaluate.nw_lag(len(per), 1))
        rows.append({"minutes_before": tau, "snapshots": q["settle_ts"].nunique(), "strikes": len(q),
                     "brier_market": b(q["mid"]).mean(), "brier_model": b(q["p_model"]).mean(),
                     "brier_hybrid": b(q["p_hyb"]).mean(), "dm_hybrid_vs_market": dm["dm"], "p": dm["p"],
                     "median_spread_c": 100 * (q["yes_ask"] - q["yes_bid"]).median(),
                     "hyb_trades": len(tr), "hyb_pnl_per_contract": tr["pnl"].mean() if len(tr) else np.nan,
                     "hyb_t": mm / se if se else np.nan})
    return pd.DataFrame(rows), ""


# ------------------------------------------------------------------ X3 box arbitrage, history
def x3_box_history() -> tuple[pd.DataFrame, str]:
    a = load_candles("KXBTCD", 60)
    r = load_candles("KXBTC", 60)
    if a.empty or r.empty:
        return pd.DataFrame(), "needs whole-hour candles of KXBTCD and KXBTC (see the module docstring)"
    r = r.rename(columns={"strike": "floor"})
    r["cap"] = pd.to_numeric(r["cap_strike"], errors="coerce")
    a = a.rename(columns={"yes_bid": "bid", "yes_ask": "ask"})
    r = r.rename(columns={"yes_bid": "bid", "yes_ask": "ask"})
    a["ev_key"] = a["event"].str.replace("KXBTCD", "", regex=False)
    r["ev_key"] = r["event"].str.replace("KXBTC", "", regex=False)
    rows = []
    rg = dict(tuple(r.groupby(["ev_key", "candle_end"])))
    for key, g in a.groupby(["ev_key", "candle_end"]):
        rr = rg.get(key)
        if rr is None:
            continue
        g = g[(g["bid"] > 0) & (g["ask"] < 1)]
        rr = rr[(rr["bid"] >= 0) & (rr["ask"] <= 1)]
        for x in box_arbitrage(g, rr):
            rows.append({"event": key[0], "minute": key[1], **x})
    d = pd.DataFrame(rows)
    note = f"{a.groupby(['ev_key', 'candle_end']).ngroups} event-minutes checked"
    return d, note


def main(argv=None):
    from oos_report import md
    ensure(config.TABLES)
    x1 = x1_fee_rounding()
    x2, n2 = x2_time_to_settlement()
    x3, n3 = x3_box_history()
    x1.to_csv(config.TABLES / "extra_fee_rounding.csv", index=False)
    x2.to_csv(config.TABLES / "extra_time_to_settlement.csv", index=False)
    x3.to_csv(config.TABLES / "extra_box_arbitrage_history.csv", index=False)
    L = ["# Historical follow-ups", "",
         "## X1. Kalshi fee rounded up to the cent", md(x1, 4),
         "## X2. Edge by minutes before settlement (Kalshi, model = + Options, hybrid rule 5c)", md(x2, 4) if len(x2) else f"_{n2}_\n",
         "## X3. Box arbitrage KXBTCD vs KXBTC, minute by minute",
         (md(x3.groupby("how").agg(cases=("edge", "size"), minutes=("minute", "nunique"), median_edge=("edge", "median"),
                                   max_edge=("edge", "max")).reset_index(), 4) + f"\n{n3}\n") if len(x3) else f"_{n3 or 'no arbitrage after fees'}_\n"]
    ensure(config.REPORTS)
    (config.REPORTS / "alpha_extra.md").write_text("\n".join(L), encoding="utf-8")
    log("alpha_extra: reports/alpha_extra.md")


if __name__ == "__main__":
    main(sys.argv[1:])
