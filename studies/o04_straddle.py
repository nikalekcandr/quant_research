"""O04 backtest: buy (or sell) the ATM straddle of the nearest Deribit daily expiry when the lab's variance forecast
says it is mispriced by more than θ of its price.

Data
  * Deribit option trades of the nearest expiries in the 60 minutes before each entry
      - data/raw/deribit/dailies/*.csv.gz  (jobs/hist_deribit_dailies_now.py: 2023→, entries 00, 09, 16 UTC)
      - or data/raw/deribit/trades/*.json.gz (the project's 16:00 UTC snapshots, 2025-06→)
  * Binance 1-minute closes (data/interim/minutes.parquet [+ optional extra minute csv]) for the forecast and RV
  * settlement: Deribit delivery price (data/raw/deribit/delivery_btc_usd.csv) if present, otherwise the 30-minute
    Binance TWAP before 08:00 UTC × the Deribit-index / Binance basis observed at entry

Pricing at entry t (no look-ahead: only trades with timestamp < t, minutes closed by t)
  * forward F_t = Deribit index of the latest trade × Binance(t) / Binance(time of that trade)
  * mark IV of every trade = Black-76 inversion of its mark price; ATM IV = local quadratic smile in log-moneyness
    evaluated at the straddle strike K* (nearest 500 to F_t); fallback: median IV of trades within ±2%
  * straddle mid = Black-76 call + put at K*; ask / bid = mid × (1 ± h), h = quoted half-spread measured on the
    lab's live Deribit chain (≈ 10% of mid ≤ 8 h, 6.5% 8–24 h, 5.5% beyond) — `--spread` scales it
  * fees: 0.03% of spot per leg, capped at 12.5% of the leg price (as lab.core.option_fee); no delivery fee
Forecast (replica of lab.history.Minutes.forecast_var):
  level = 0.5·EWMA(hl 30 min) + 0.3·mean 24 h + 0.2·mean 8 d of deseasonalised squared 1-min returns,
  shape = hour-of-week profile of the last 90 days; variance to expiry − 15 min.
  fair = E|X − K|, X ~ N(F, F·√var)    (lab.signals._straddle_fair)
Rule O04: long if fair − ask − fee > θ·ask; short if bid − fair − fee > θ·bid; one trade per expiry (first entry
that fires); hold to settlement.
    python studies/o04_straddle.py [--spread 1.0] [--extra-minutes file.csv]
"""
from __future__ import annotations

import argparse
import glob
import gzip
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import config  # noqa: E402

OUT = config.RESULTS / "studies"
FEE_RATE, FEE_CAP = 0.0003, 0.125


# ------------------------------------------------------------------ data
def load_trades(months=None) -> pd.DataFrame:
    files = sorted(glob.glob(str(config.RAW / "deribit" / "dailies" / "*.csv.gz")))
    if months:
        files = [f for f in files if any(Path(f).name.startswith(m) for m in months)]
    if files:
        d = pd.concat([pd.read_csv(f) for f in files if Path(f).stat().st_size > 30], ignore_index=True)
        d["entry"] = pd.to_datetime(d["entry"], utc=True)
        src = "dailies (00/09/16 UTC, 2023→)"
    else:
        rows = []
        for f in sorted(glob.glob(str(config.RAW / "deribit" / "trades" / "*.json.gz"))):
            t = pd.Timestamp(Path(f).name[:12]).tz_localize("UTC")
            for x in json.load(gzip.open(f)):
                rows.append({"entry": t, "ts": x["timestamp"], "instrument": x["instrument_name"], "price": x["price"],
                             "mark_price": x["mark_price"], "iv": x.get("iv"), "index_price": x["index_price"],
                             "direction": x.get("direction"), "amount": x.get("amount")})
        d = pd.DataFrame(rows)
        src = "16:00 UTC snapshots (2025-06→)"
    d["ts"] = pd.to_datetime(d["ts"], unit="ms", utc=True)
    p = d["instrument"].str.split("-", expand=True)
    d["expiry"] = pd.to_datetime(p[1], format="%d%b%y").dt.tz_localize("UTC") + pd.Timedelta(hours=8)
    d["K"] = pd.to_numeric(p[2])
    d["cp"] = p[3]
    d = d[(d["expiry"] > d["entry"] + pd.Timedelta(hours=3)) & (d["expiry"] <= d["entry"] + pd.Timedelta(hours=32))]
    d = d[(d["ts"] < d["entry"]) & (d["mark_price"] > 0)]
    return d, src


def load_minutes(extra: str | None) -> pd.Series:
    m = pd.read_parquet(config.INTERIM / "minutes.parquet", columns=["open_time", "close"])
    s = m.set_index(pd.to_datetime(m["open_time"], utc=True))["close"]
    if extra:
        e = pd.read_csv(extra, parse_dates=["open_time"]).set_index("open_time")["close"]
        e.index = pd.to_datetime(e.index, utc=True)
        s = pd.concat([s, e[e.index > s.index[-1]]])
    return s[~s.index.duplicated()].sort_index()


# ------------------------------------------------------------------ Black-76 in USD per 1 BTC option
def black(F, K, sig, T, cp):
    if sig <= 0 or T <= 0:
        return max(F - K, 0) if cp == "C" else max(K - F, 0)
    v = sig * math.sqrt(T)
    d1 = (math.log(F / K) + v * v / 2) / v
    d2 = d1 - v
    return F * norm.cdf(d1) - K * norm.cdf(d2) if cp == "C" else K * norm.cdf(-d2) - F * norm.cdf(-d1)


def implied(px, F, K, T, cp):
    intr = max(F - K, 0) if cp == "C" else max(K - F, 0)
    if px <= intr + 1e-9 or T <= 0:
        return np.nan
    try:
        return brentq(lambda s: black(F, K, s, T, cp) - px, 1e-3, 10.0)
    except ValueError:
        return np.nan


def half_spread(tau_h):
    return 0.10 if tau_h <= 8 else (0.065 if tau_h <= 24 else 0.055)


def straddle_fair(F, K, sd):
    mu = F - K
    if sd <= 0:
        return abs(mu)
    return sd * math.sqrt(2 / math.pi) * math.exp(-mu * mu / (2 * sd * sd)) + mu * (1 - 2 * norm.cdf(-mu / sd))


# ------------------------------------------------------------------ the lab's forecaster, rolling
class Forecaster:
    def __init__(self, close: pd.Series):
        self.r = np.log(close).diff().dropna()          # index = minute open time (as lab.history)
        self.cache = {}

    def seasonal(self, t):
        day = t.normalize()
        if day not in self.cache:
            r = self.r[(self.r.index < day) & (self.r.index >= day - pd.Timedelta(days=90))]
            r2 = (r ** 2).clip(upper=(r ** 2).quantile(0.999))
            s = r2.groupby(r2.index.dayofweek * 24 + r2.index.hour).mean()
            s = (s / s.mean()).reindex(range(168)).fillna(1.0)
            self.cache[day] = s.rolling(3, center=True, min_periods=1).mean()
        return self.cache[day]

    def var(self, t, end):
        s = self.seasonal(t)
        r = self.r[(self.r.index <= t - pd.Timedelta(minutes=1)) & (self.r.index >= t - pd.Timedelta(days=8))]
        if len(r) < 600:
            return np.nan
        z = (r ** 2) / s.reindex(r.index.dayofweek * 24 + r.index.hour).values
        z = z.clip(upper=z.quantile(0.9995))
        lv = 0.5 * z.ewm(halflife=30).mean().iloc[-1] + 0.3 * z[z.index > z.index[-1] - pd.Timedelta(hours=24)].mean() + 0.2 * z.mean()
        grid = pd.date_range(t, end, freq="1min", inclusive="left")
        return float(s.reindex(grid.dayofweek * 24 + grid.hour).values.sum()) * lv


# ------------------------------------------------------------------ build one row per entry
def build(d, close, fc, delivery):
    rows = []
    c_end = close.index[-1]
    first = d.groupby("entry")["expiry"].transform("min")
    d = d[d["expiry"] == first]          # the nearest expiry only
    for (t, exp), g in d.groupby(["entry", "expiry"]):
        T = (exp - t).total_seconds() / (365 * 86400)
        tau_h = (exp - t).total_seconds() / 3600
        last = g.sort_values("ts").iloc[-1]
        try:
            b_last = close.asof(last["ts"])
            b_t = close.asof(t)
        except KeyError:
            continue
        if not (np.isfinite(b_last) and np.isfinite(b_t)) or t > c_end:
            continue
        basis = last["index_price"] / b_last
        F = b_t * basis
        # mark IV of each trade (forward = its own index price, time to expiry from its timestamp)
        g = g[np.abs(np.log(g["K"] / g["index_price"])) < 0.06]
        if g.empty:
            continue
        ivs = []
        for _, x in g.iterrows():
            Tx = (exp - x["ts"]).total_seconds() / (365 * 86400)
            ivs.append(implied(x["mark_price"] * x["index_price"], x["index_price"], x["K"], Tx, x["cp"]))
        g = g.assign(miv=ivs, m=np.log(g["K"] / g["index_price"])).dropna(subset=["miv"])
        g = g[(g["miv"] > 0.05) & (g["miv"] < 5) & (g["m"].abs() < 0.06)]
        if g.empty:
            continue
        K = round(F / 500) * 500
        mk = math.log(K / F)
        near = g[g["m"].abs() < 0.02]
        if g["K"].nunique() >= 3 and (g["m"] < mk).any() and (g["m"] > mk).any():
            w = np.exp(-np.abs(g["m"] - mk) / 0.02)
            A = np.vstack([np.ones(len(g)), g["m"], g["m"] ** 2]).T
            coef = np.linalg.lstsq(A * np.sqrt(w.values)[:, None], g["miv"].values * np.sqrt(w.values), rcond=None)[0]
            iv = float(coef[0] + coef[1] * mk + coef[2] * mk * mk)
            if not (0.5 * g["miv"].min() < iv < 1.5 * g["miv"].max()):
                iv = float(near["miv"].median()) if len(near) else np.nan
        else:
            iv = float(near["miv"].median()) if len(near) else np.nan
        if not np.isfinite(iv):
            continue
        call, put = black(F, K, iv, T, "C"), black(F, K, iv, T, "P")
        mid = call + put
        var = fc.var(t, exp - pd.Timedelta(minutes=15))
        if not np.isfinite(var):
            continue
        fair = straddle_fair(F, K, F * math.sqrt(var))
        # settlement
        dd = exp.strftime("%Y-%m-%d")
        if delivery is not None and dd in delivery.index:
            S_T = float(delivery[dd])
            src = "deribit"
        else:
            w = close[(close.index > exp - pd.Timedelta(minutes=30)) & (close.index <= exp)]
            if len(w) < 25:
                continue
            S_T = float(w.mean()) * basis
            src = "binance_twap"
        rr = np.log(close[(close.index >= t) & (close.index <= exp)]).diff()
        rows.append({"entry": t, "expiry": exp, "tau_h": tau_h, "F": F, "K": K, "iv": iv, "call": call, "put": put,
                     "mid": mid, "fair": fair, "fc_vol": math.sqrt(var / T) if T > 0 else np.nan,
                     "var_fc": var, "var_iv": iv * iv * T, "rv": float((rr ** 2).sum()), "S_T": S_T, "settle": src,
                     "payoff": abs(S_T - K), "n_trades": len(g)})
    return pd.DataFrame(rows)


def trade(b, theta, side="both", spread=1.0, entry_hours=None):
    x = b if entry_hours is None else b[b["entry"].dt.hour.isin(entry_hours)]
    h = x["tau_h"].map(half_spread) * spread
    ask, bid = x["mid"] * (1 + h), x["mid"] * (1 - h)
    fa = np.minimum(FEE_RATE * x["F"], FEE_CAP * x["call"] * (1 + h)) + np.minimum(FEE_RATE * x["F"], FEE_CAP * x["put"] * (1 + h))
    fb = np.minimum(FEE_RATE * x["F"], FEE_CAP * x["call"] * (1 - h)) + np.minimum(FEE_RATE * x["F"], FEE_CAP * x["put"] * (1 - h))
    if theta is None:          # baselines
        lg = pd.Series(side == "long", index=x.index)
        sh = pd.Series(side == "short", index=x.index)
    else:
        lg = (x["fair"] - ask - fa > theta * ask) & (side in ("both", "long"))
        sh = (bid - x["fair"] - fb > theta * bid) & (side in ("both", "short")) & ~lg
    t = x.assign(side=np.where(lg, "long", np.where(sh, "short", "")), price=np.where(lg, ask, bid), fee=np.where(lg, fa, fb))
    t = t[t["side"] != ""]
    t["pnl"] = np.where(t["side"] == "long", t["payoff"] - t["price"] - t["fee"], t["price"] - t["fee"] - t["payoff"])
    t["ret"] = t["pnl"] / t["price"]                      # per $ of premium paid / received
    return t.sort_values("entry").drop_duplicates("expiry")   # one trade per expiry: the first entry that fires


def hac_t(x, lags=5):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 10:
        return np.nan
    u = x - x.mean()
    s = u @ u / n + 2 * sum((1 - j / (lags + 1)) * (u[j:] @ u[:-j]) / n for j in range(1, lags + 1))
    return x.mean() / math.sqrt(s / n)


def stats(t, label):
    if len(t) == 0:
        return {"rule": label, "trades": 0}
    p = t["pnl"]
    eq = p.cumsum()
    yrs = max((t["expiry"].max() - t["expiry"].min()).days / 365, 1e-9)
    sr = p.mean() / p.std() * math.sqrt(len(t) / yrs) if p.std() > 0 else np.nan
    top = p.sort_values(ascending=False)
    return {"rule": label, "trades": len(t), "long %": 100 * (t["side"] == "long").mean(), "$/trade": p.mean(),
            "median $": p.median(), "ret/premium %": 100 * t["ret"].mean(), "hit %": 100 * (p > 0).mean(),
            "t HAC": hac_t(p), "Sharpe/yr": sr, "total $": p.sum(), "max DD $": (eq - eq.cummax()).min(),
            "total w/o top 5 $": p.sum() - top.head(5).sum()}


def boot_ci(p, n=4000, block=5, seed=1):
    rng = np.random.default_rng(seed)
    p = np.asarray(p, float)
    k = len(p)
    if k < 20:
        return (np.nan, np.nan)
    means = []
    for _ in range(n):
        idx = np.concatenate([np.arange(s, s + block) % k for s in rng.integers(0, k, k // block + 1)])[:k]
        means.append(p[idx].mean())
    return tuple(np.percentile(means, [2.5, 97.5]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spread", type=float, default=1.0)
    ap.add_argument("--extra-minutes", default=None)
    ap.add_argument("--panel", default="o04_panel.csv")
    ap.add_argument("--months", default=None, help="only dailies files whose name starts with one of these, comma-separated")
    a = ap.parse_args()
    d, src = load_trades(a.months.split(",") if a.months else None)
    close = load_minutes(a.extra_minutes)
    close.index = close.index + pd.Timedelta(minutes=1)        # index = close time: price known at that moment
    fc = Forecaster(close.set_axis(close.index - pd.Timedelta(minutes=1)))
    dl = config.RAW / "deribit" / "delivery_btc_usd.csv"
    delivery = pd.read_csv(dl).set_index("date")["delivery_price"] if dl.exists() else None
    print(f"trades: {len(d)} from {src}; entries {d['entry'].nunique()}; minutes to {close.index[-1]}", flush=True)
    b = build(d, close, fc, delivery)
    OUT.mkdir(parents=True, exist_ok=True)
    b.to_csv(OUT / a.panel, index=False)
    print(f"priced entries: {len(b)}, expiries {b['expiry'].nunique()}, {b['expiry'].min():%Y-%m-%d} .. "
          f"{b['expiry'].max():%Y-%m-%d}; settlement: {b['settle'].value_counts().to_dict()}", flush=True)
    return b


if __name__ == "__main__":
    main()
