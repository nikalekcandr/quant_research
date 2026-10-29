"""Clean study of Kalshi KXBTC15M ("BTC up in 15 minutes?") on the venue's own 1-minute book candles.

Question 1  Is the market well calibrated, and does a random-walk model on Coinbase spot forecast the outcome better
            than the market's book mid (Brier, log loss; paired by market, Diebold–Mariano with market clusters)?
Question 2  Does the model add information on top of the market (logistic encompassing regression, fitted on the
            first half of the sample in time, scored on the second)?
Question 3  Is there money after the spread and the fee (taker at the book's ask / bid, one trade per market,
            thresholds chosen on the first half, P&L on the second)?
Rules fixed before looking at the second half. Everything the model uses is known at the decision minute:
Coinbase close of the minute, a variance forecast from past minutes only, the BRTI–Coinbase basis from past markets.

    python studies/kx15_study.py <raw dir>     (raw dir holds kx15/, coinbase_1m/)
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm, t as student_t

RAW = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/raw")
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("results/studies")
TAUS = (14, 12, 10, 8, 6, 4, 2, 1)       # minutes left at the decision point
MAX_SPREAD = 0.05


def kalshi_fee(p):
    return np.ceil(np.round(0.07 * p * (1 - p) * 100, 6)) / 100


def load():
    mk = pd.read_csv(RAW / "kx15" / "markets.csv")
    mk["close_time"] = pd.to_datetime(mk["close_time"], utc=True)
    mk["open_time"] = pd.to_datetime(mk["open_time"], utc=True)
    mk["y"] = (mk["result"] == "yes").astype(float)
    c = pd.concat([pd.read_csv(f) for f in sorted((RAW / "kx15" / "candles").glob("*.csv")) if f.stat().st_size > 10])
    c["ts"] = pd.to_datetime(c["end_ts"], unit="s", utc=True)
    cb = pd.concat([pd.read_csv(f) for f in sorted((RAW / "coinbase_1m").glob("*.csv"))])
    cb["ts"] = pd.to_datetime(cb["t"], unit="s", utc=True) + pd.Timedelta(minutes=1)    # close time of the candle
    cb = cb.drop_duplicates("ts").set_index("ts").sort_index()
    return mk, c, cb


def variance_model(cb):
    """Per-minute variance forecast known at each minute: EWMA (half-life 60 min) of squared deseasonalised returns
    x hour-of-week seasonal factor (rolling 28-day mean, lagged one day)."""
    full = pd.date_range(cb.index.min(), cb.index.max(), freq="1min")
    px = cb["close"].reindex(full).ffill()
    r2 = np.log(px).diff() ** 2
    how = full.dayofweek * 24 + full.hour
    df = pd.DataFrame({"r2": r2.values, "how": how, "day": full.floor("D")}, index=full)
    daily = df.groupby(["day", "how"])["r2"].mean().unstack()               # day x hour-of-week
    sea = daily.rolling(28, min_periods=7).mean().shift(1)                  # past days only
    sea = sea.div(sea.mean(axis=1), axis=0)                                 # relative shape
    sea_long = sea.stack().rename("sea").reset_index()
    df = df.reset_index().rename(columns={"index": "ts"}).merge(sea_long, on=["day", "how"], how="left").set_index("ts")
    df["sea"] = df["sea"].fillna(1.0)
    z = df["r2"] / df["sea"]
    df["lvl"] = z.ewm(halflife=60, min_periods=120).mean()                  # includes the current minute's return
    return px, df[["sea", "lvl"]]


def seasonal_sum(sea_tbl, ts, minutes):
    """Seasonal factors of the next `minutes` minutes (approximated by the current hour's factor)."""
    return sea_tbl.loc[ts, "sea"] * minutes


def build(mk, c, cb):
    px, vm = variance_model(cb)
    rows = []
    c = c.merge(mk[["ticker", "close_time", "open_time", "floor_strike", "expiration_value", "y"]], on="ticker")
    c["tau"] = ((c["close_time"] - c["ts"]).dt.total_seconds() / 60).round().astype(int)
    c = c[c["tau"].isin(TAUS)]
    # basis: BRTI (the strike = BRTI at the open) minus the Coinbase close at the open, from markets that are
    # already over (shifted one market), rolling median of 96 markets
    st = mk[["ticker", "open_time", "floor_strike"]].copy()
    st["cb_open"] = px.reindex(st["open_time"]).values
    st["basis"] = np.log(st["floor_strike"] / st["cb_open"])
    st = st.sort_values("open_time")
    st["basis_hat"] = st["basis"].rolling(96, min_periods=20).median().shift(1)
    c = c.merge(st[["ticker", "basis", "basis_hat"]], on="ticker", how="left")
    c["S"] = px.reindex(c["ts"]).values
    c["S_lag1"] = px.reindex(c["ts"] - pd.Timedelta(minutes=1)).values
    c["lvl"] = vm["lvl"].reindex(c["ts"]).values
    c["sea"] = vm["sea"].reindex(c["ts"]).values
    # settlement = mean of the 60 one-second BRTI prints of the last minute: variance of tau minutes ending in a
    # one-minute average = (tau - 2/3) minutes of point variance
    eff = np.maximum(c["tau"] - 2 / 3, 1 / 3)
    c["var"] = c["lvl"] * c["sea"] * eff
    for name, s in (("model", c["S"]), ("model_lag1", c["S_lag1"])):
        x = np.log(s / c["floor_strike"]) + c["basis_hat"].fillna(0)
        c[name] = norm.cdf(x / np.sqrt(c["var"]))
        c[name + "_t4"] = student_t.cdf(x / np.sqrt(c["var"] / 2), 4)      # t4 with the same variance
    c["mid"] = (c["bid"] + c["ask"]) / 2
    c["spread"] = c["ask"] - c["bid"]
    ok = c["bid"].gt(0) & c["ask"].lt(1) & c["spread"].le(MAX_SPREAD) & c["model"].notna()
    return c[ok].copy(), st


def clip(p):
    return np.clip(p, 0.001, 0.999)


def scores(d, cols):
    out = []
    per = {}
    for col in cols:
        b = (d[col] - d["y"]) ** 2
        ll = -(d["y"] * np.log(clip(d[col])) + (1 - d["y"]) * np.log(1 - clip(d[col])))
        per[col] = pd.DataFrame({"b": b, "ll": ll, "ticker": d["ticker"]}).groupby("ticker").mean()
    base = per["mid"]
    for col in cols:
        p = per[col]
        rec = {"forecast": col, "markets": len(p), "Brier": p["b"].mean(), "log loss": p["ll"].mean()}
        if col != "mid":
            for k, lab in (("b", "Brier"), ("ll", "log loss")):
                diff = (base[k] - p[k]).dropna()
                rec[f"{lab} gain vs market"] = diff.mean()
                rec[f"t ({lab})"] = diff.mean() / (diff.std() / math.sqrt(len(diff)))
        out.append(rec)
    return pd.DataFrame(out)


def encompassing(d, split):
    import statsmodels.api as sm
    lg = lambda p: np.log(clip(p) / (1 - clip(p)))
    X = pd.DataFrame({"const": 1.0, "logit_mid": lg(d["mid"]), "logit_model": lg(d["model"])})
    tr = d["ts"] < split
    res = sm.GLM(d.loc[tr, "y"], X[tr], family=sm.families.Binomial()).fit(cov_type="cluster",
                                                                            cov_kwds={"groups": d.loc[tr, "ticker"].astype("category").cat.codes})
    p_te = res.predict(X[~tr])
    te = d[~tr].assign(combo=p_te.values)
    return res, te


def trading(d, split, thresholds=(0.0, 0.01, 0.02, 0.03, 0.05, 0.08)):
    rows = []
    for col in ("model", "model_lag1"):
        for th in thresholds:
            for half, g in (("train", d[d["ts"] < split]), ("test", d[d["ts"] >= split])):
                g = g.sort_values("ts")
                e_yes = g[col] - g["ask"] - kalshi_fee(g["ask"])
                e_no = (1 - g[col]) - (1 - g["bid"]) - kalshi_fee(1 - g["bid"])
                sig = g.assign(e_yes=e_yes, e_no=e_no)
                sig = sig[(sig["e_yes"] > th) | (sig["e_no"] > th)]
                first = sig.groupby("ticker").head(1)
                yes = first["e_yes"] > first["e_no"]
                pnl = np.where(yes, first["y"] - first["ask"] - kalshi_fee(first["ask"]),
                               (1 - first["y"]) - (1 - first["bid"]) - kalshi_fee(1 - first["bid"]))
                n = len(pnl)
                rows.append({"fair": col, "min edge": th, "half": half, "trades": n,
                             "mean P&L, ¢": 100 * pnl.mean() if n else np.nan,
                             "t": pnl.mean() / (pnl.std() / math.sqrt(n)) if n > 2 and pnl.std() > 0 else np.nan,
                             "hit": (pnl > 0).mean() if n else np.nan})
    return pd.DataFrame(rows)


def calibration(d, col="mid"):
    b = pd.cut(d[col], np.linspace(0, 1, 11))
    g = d.groupby(b, observed=True).agg(n=("y", "size"), forecast=(col, "mean"), outcome=("y", "mean"))
    g["gap"] = g["outcome"] - g["forecast"]
    return g.reset_index()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    mk, c, cb = load()
    d, st = build(mk, c, cb)
    split = d["ts"].quantile(0.5)
    rep = {"markets": int(d["ticker"].nunique()), "rows": len(d), "from": str(d["ts"].min()), "to": str(d["ts"].max()),
           "split": str(split), "basis bp median": float(10000 * st["basis"].median()),
           "basis bp IQR": float(10000 * (st["basis"].quantile(0.75) - st["basis"].quantile(0.25))),
           "median spread ¢": float(100 * d["spread"].median())}
    print(json.dumps(rep, indent=1))
    cols = ["mid", "model", "model_t4", "model_lag1", "last"]
    d["last"] = d["last"].where(d["last"].between(0.001, 0.999), d["mid"])
    tabs = {"scores_all": scores(d, cols)}
    for tau in TAUS:
        tabs[f"scores_tau{tau}"] = scores(d[d["tau"] == tau], ["mid", "model", "model_lag1"]).assign(tau=tau)
    tabs["scores_by_tau"] = pd.concat([tabs.pop(f"scores_tau{t}") for t in TAUS])
    for half, g in (("first half", d[d["ts"] < split]), ("second half", d[d["ts"] >= split])):
        tabs[f"scores_{half}"] = scores(g, ["mid", "model"])
    res, te = encompassing(d, split)
    tabs["encompassing_coefs"] = pd.DataFrame({"coef": res.params, "t (clustered)": res.tvalues})
    tabs["encompassing_test"] = scores(te, ["mid", "model", "combo"])
    tabs["calibration_market"] = calibration(d, "mid")
    tabs["calibration_model"] = calibration(d, "model")
    tabs["trading"] = trading(d, split)
    for k, v in tabs.items():
        v.to_csv(OUT / f"kx15_{k}.csv", index=k.startswith("enc"))
        print(f"\n== {k}\n{v.round(4).to_string()}")
    (OUT / "kx15_summary.json").write_text(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
