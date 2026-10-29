"""Quant-research layer: market regimes + a battery of published (and a few unpublished) BTC inefficiencies,
tested on the data the project already downloads. Nothing here touches the frozen forecasting models.

    python research.py                 everything -> reports/research.md, results/research/*.csv, figures
    python research.py --only R7 R14   a subset
    python research.py --quick         skip the slow pieces (monthly Markov refits -> one fit)

Rules fixed before looking at the results (29 October 2026, RESEARCH_FROZEN):
* every hypothesis comes from a paper or a widely repeated market claim, with the sign it predicts;
* ONE primary statistic per hypothesis: the mean of a per-observation series (`obs`), tested with a
  Newey–West t-statistic (lags = overlap of the observations). Regressions are reported as details;
* two-sided p-values; Holm (FWER) and Benjamini–Hochberg (FDR) across the whole family;
* trading rules use only information available at the decision time; costs 5 bp per side (Binance
  USDT-M taker fee), results gross and net; Sharpe ratios are deflated for the number of rules tried
  (Bailey & López de Prado 2014);
* regimes are real-time: Markov-switching volatility state from filtered (not smoothed) probabilities
  with parameters re-estimated every month on past data only; volatility tercile vs the trailing year;
  trend (price above its 200-day average); era (before / after the spot-ETF launch, 11 Jan 2024).
  Every primary result is broken down by regime;
* data after RESEARCH_FROZEN is a clean out-of-sample check: it is reported separately on every run.
"""
from __future__ import annotations

import argparse
import json
import math
import time
import traceback
import warnings
from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd
from scipy import stats

import config
from src.utils import log

warnings.filterwarnings("ignore")

RESEARCH_FROZEN = "2026-10-29"
ETF_LAUNCH = pd.Timestamp("2024-01-11", tz="UTC")
HALVING = pd.Timestamp("2024-04-20", tz="UTC")
COST_BPS = 5.0                  # per side
ET = "America/New_York"
OUT = config.RESULTS / "research"
FIG = OUT / "figures"

# NYSE full-day closures 2023-2026 (weekends excluded separately)
NYSE_HOLIDAYS = {
    "2023-01-02", "2023-01-16", "2023-02-20", "2023-04-07", "2023-05-29", "2023-06-19", "2023-07-04",
    "2023-09-04", "2023-11-23", "2023-12-25",
    "2024-01-01", "2024-01-15", "2024-02-19", "2024-03-29", "2024-05-27", "2024-06-19", "2024-07-04",
    "2024-09-02", "2024-11-28", "2024-12-25",
    "2025-01-01", "2025-01-09", "2025-01-20", "2025-02-17", "2025-04-18", "2025-05-26", "2025-06-19",
    "2025-07-04", "2025-09-01", "2025-11-27", "2025-12-25",
    "2026-01-01", "2026-01-19", "2026-02-16", "2026-04-03", "2026-05-25", "2026-06-19", "2026-07-03",
    "2026-09-07", "2026-11-26", "2026-12-25"}


# =============================================================================================== statistics
def nw_mean(x, lags: int | None = None) -> dict:
    """Mean of x with a Newey–West (Bartlett) standard error."""
    x = np.asarray(pd.Series(x).dropna(), float)
    n = len(x)
    if n < 5:
        return {"mean": np.nan, "se": np.nan, "t": np.nan, "p": np.nan, "n": n}
    if lags is None:
        lags = int(np.floor(4 * (n / 100) ** (2 / 9)))
    lags = max(0, min(lags, n - 2))
    e = x - x.mean()
    s = e @ e / n
    for L in range(1, lags + 1):
        s += 2 * (1 - L / (lags + 1)) * (e[L:] @ e[:-L]) / n
    se = math.sqrt(max(s, 1e-300) / n)
    t = x.mean() / se
    return {"mean": x.mean(), "se": se, "t": t, "p": 2 * stats.norm.sf(abs(t)), "n": n}


def nw_ols(y, X, lags: int | None = None, names=None) -> pd.DataFrame:
    import statsmodels.api as sm
    d = pd.concat([pd.Series(np.asarray(y, float)), pd.DataFrame(np.asarray(X, float))], axis=1).dropna()
    yy, XX = d.iloc[:, 0].values, sm.add_constant(d.iloc[:, 1:].values)
    n = len(yy)
    if lags is None:
        lags = int(np.floor(4 * (n / 100) ** (2 / 9)))
    r = sm.OLS(yy, XX).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    out = pd.DataFrame({"coef": r.params, "t": r.tvalues, "p": r.pvalues},
                       index=["const"] + (names or [f"x{i}" for i in range(1, XX.shape[1])]))
    out.attrs.update(r2=r.rsquared, n=n)
    return out


def holm(p: pd.Series) -> pd.Series:
    p = p.dropna().sort_values()
    m = len(p)
    adj = np.maximum.accumulate([(m - i) * v for i, v in enumerate(p.values)])
    return pd.Series(np.minimum(adj, 1), index=p.index)


def bh(p: pd.Series) -> pd.Series:
    p = p.dropna().sort_values()
    m = len(p)
    q = p.values * m / np.arange(1, m + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    return pd.Series(np.minimum(q, 1), index=p.index)


def zexp(x: pd.Series, min_n: int = 30) -> pd.Series:
    """Real-time z-score: mean and sd from observations strictly before each point."""
    m = x.expanding(min_n).mean().shift(1)
    s = x.expanding(min_n).std().shift(1)
    return (x - m) / s


def sharpe_stats(r: pd.Series, periods: int = 365) -> dict:
    r = r.dropna()
    if len(r) < 20 or r.std() == 0:
        return {"sr": np.nan, "sr_ann": np.nan, "skew": np.nan, "kurt": np.nan, "n": len(r), "ann_ret": np.nan}
    sr = r.mean() / r.std()
    return {"sr": sr, "sr_ann": sr * math.sqrt(periods), "skew": stats.skew(r), "kurt": stats.kurtosis(r, fisher=False),
            "n": len(r), "ann_ret": r.mean() * periods, "ann_vol": r.std() * math.sqrt(periods)}


def deflated_sharpe(sr: float, n: int, skew: float, kurt: float, sr_trials: list[float]) -> tuple[float, float]:
    """(PSR vs 0, DSR) from Bailey & López de Prado (2014); per-period (non-annualised) Sharpe ratios.
    The benchmark SR0 is the expected maximum of len(sr_trials) independent zero-Sharpe rules."""
    if not np.isfinite(sr) or n < 20:
        return np.nan, np.nan
    den = math.sqrt(max(1e-12, 1 - skew * sr + (kurt - 1) / 4 * sr ** 2))
    psr = stats.norm.cdf(sr * math.sqrt(n - 1) / den)
    N = max(2, len(sr_trials))
    v = 1.0 / (n - 1)    # variance of the Sharpe estimator when every rule tried has zero true Sharpe
    g = 0.5772156649
    sr0 = math.sqrt(v) * ((1 - g) * stats.norm.ppf(1 - 1 / N) + g * stats.norm.ppf(1 - 1 / (N * math.e)))
    dsr = stats.norm.cdf((sr - sr0) * math.sqrt(n - 1) / den)
    return psr, dsr


def variance_ratio(r: np.ndarray, q: int) -> tuple[float, float]:
    """Lo–MacKinlay (1988) variance ratio with the heteroskedasticity-robust z* statistic."""
    r = np.asarray(r, float)
    r = r[np.isfinite(r)]
    T = len(r)
    mu = r.mean()
    e = r - mu
    s1 = e @ e / (T - 1)
    rq = np.convolve(r, np.ones(q), "valid")
    m = q * (T - q + 1) * (1 - q / T)
    sq = ((rq - q * mu) ** 2).sum() / m
    vr = sq / s1
    e2 = e ** 2
    den = (e2.sum()) ** 2
    theta = 0.0
    for j in range(1, q):
        d = (e2[j:] @ e2[:-j]) * T / den
        theta += (2 * (q - j) / q) ** 2 * d
    z = (vr - 1) / math.sqrt(theta / T) if theta > 0 else np.nan
    return vr, z


# =============================================================================================== data
class Data:
    def __init__(self):
        t0 = time.time()
        m = pd.read_parquet(config.INTERIM / "minutes.parquet", columns=["open_time", "close"])
        m["open_time"] = pd.to_datetime(m["open_time"], utc=True)
        s = m.set_index("open_time")["close"].astype(float)
        s = s[~s.index.duplicated()].sort_index()
        grid = pd.date_range(s.index[0], s.index[-1], freq="1min")
        self.px = s.reindex(grid).ffill()                     # minute close, forward-filled over rare gaps
        self.lp = np.log(self.px)
        self.r1 = self.lp.diff()
        self.end = self.px.index[-1]
        lp5 = self.lp[self.lp.index.minute % 5 == 4]          # 5-minute closes (minute :04, :09, ...)
        r5 = lp5.diff()
        day = r5.index.floor("D")
        self.rv_d = (r5 ** 2).groupby(day).sum()              # daily realized variance, 5-minute returns
        self.rs_pos = (r5.clip(lower=0) ** 2).groupby(day).sum()
        self.rs_neg = (r5.clip(upper=0) ** 2).groupby(day).sum()
        cnt = r5.groupby(day).count()
        full = cnt[cnt >= 280].index                          # complete UTC days
        self.rv_d, self.rs_pos, self.rs_neg = self.rv_d.loc[full], self.rs_pos.loc[full], self.rs_neg.loc[full]
        d00 = self.lp[self.lp.index.hour.isin([23]) & (self.lp.index.minute == 59)]
        d00.index = d00.index.floor("D")
        self.close_d = d00                                     # log close at 23:59 UTC
        self.r_d = d00.diff().loc[lambda x: x.index.isin(full)]
        h = self.lp[self.lp.index.minute == 59]
        h.index = h.index.floor("h")
        self.r_h = h.diff()
        self.rv_h = (self.r1 ** 2).groupby(self.r1.index.floor("h")).sum()
        self.dvol = None
        f = config.RAW / "deribit" / "dvol.csv.gz"
        if f.exists():
            d = pd.read_csv(f)
            d["ts"] = pd.to_datetime(d["ts"], utc=True)
            self.dvol = d.set_index("ts")["close"].astype(float).sort_index()
            self.dvol = self.dvol[~self.dvol.index.duplicated()]
        cal = pd.read_csv(config.MANUAL / "us_macro_calendar.csv")
        cal["ts"] = [pd.Timestamp(f"{a} {b}").tz_localize(ET).tz_convert("UTC") for a, b in zip(cal["date"], cal["time_et"])]
        self.macro = cal[(cal["ts"] > self.px.index[0] + pd.Timedelta(days=30)) & (cal["ts"] < self.end - pd.Timedelta(hours=6))]
        log(f"research data: minutes {self.px.index[0]:%Y-%m-%d} .. {self.end:%Y-%m-%d %H:%M}, "
            f"{len(self.rv_d)} full days, DVOL {'yes' if self.dvol is not None else 'no'} ({time.time() - t0:.0f}s)")

    def at(self, ts) -> np.ndarray:
        """Log price at the minute close at or before each timestamp (NaN outside the sample)."""
        ts = pd.DatetimeIndex(pd.to_datetime(ts, utc=True))
        out = self.lp.reindex(ts.floor("min") - pd.Timedelta(minutes=1)).to_numpy(copy=True)
        out[(ts < self.px.index[0]) | (ts > self.end)] = np.nan
        return out

    def trading_days(self) -> pd.DatetimeIndex:
        days = pd.date_range(self.px.index[0].date(), self.end.date(), freq="B")
        return pd.DatetimeIndex([d for d in days if d.strftime("%Y-%m-%d") not in NYSE_HOLIDAYS])


def et(day, hhmm: str) -> pd.Timestamp:
    return pd.Timestamp(f"{pd.Timestamp(day):%Y-%m-%d} {hhmm}").tz_localize(ET).tz_convert("UTC")


# =============================================================================================== regimes
# Two-state Gaussian hidden Markov model (Hamilton 1989), Baum–Welch EM; written out here because the
# statsmodels Markov-switching EM is numerically fragile on this series.
def hmm_fit(x, iters=300, tol=1e-8):
    x = np.asarray(x, float)
    q = np.quantile(x, [0.3, 0.7])
    mu = np.array([x[x <= q[0]].mean(), x[x >= q[1]].mean()])
    sd = np.array([x.std() * 0.7, x.std() * 0.7])
    P = np.array([[0.97, 0.03], [0.03, 0.97]])
    pi = np.array([0.5, 0.5])
    ll_old = -np.inf
    T = len(x)
    for _ in range(iters):
        B = stats.norm.pdf(x[:, None], mu, sd) + 1e-300
        a = np.zeros((T, 2)); c = np.zeros(T)
        a[0] = pi * B[0]; c[0] = a[0].sum(); a[0] /= c[0]
        for t in range(1, T):
            a[t] = (a[t - 1] @ P) * B[t]; c[t] = a[t].sum(); a[t] /= c[t]
        b = np.ones((T, 2))
        for t in range(T - 2, -1, -1):
            b[t] = (P @ (B[t + 1] * b[t + 1])) / c[t + 1]
        g = a * b; g /= g.sum(1, keepdims=True)
        xi = np.zeros((2, 2))
        for t in range(T - 1):
            m = a[t][:, None] * P * (B[t + 1] * b[t + 1])[None, :] / c[t + 1]
            xi += m
        P = xi / xi.sum(1, keepdims=True)
        pi = g[0]
        mu = (g * x[:, None]).sum(0) / g.sum(0)
        sd = np.sqrt((g * (x[:, None] - mu) ** 2).sum(0) / g.sum(0)).clip(1e-3)
        ll = np.log(c).sum()
        if ll - ll_old < tol:
            break
        ll_old = ll
    o = np.argsort(mu)
    return {"mu": mu[o], "sd": sd[o], "P": P[np.ix_(o, o)], "pi": pi[o], "ll": ll}

def hmm_filter(x, p):
    """Filtered P(high state | data up to t) — uses no future observation."""
    x = np.asarray(x, float)
    B = stats.norm.pdf(x[:, None], p["mu"], p["sd"]) + 1e-300
    a = np.zeros((len(x), 2)); a[0] = p["pi"] * B[0]; a[0] /= a[0].sum()
    for t in range(1, len(x)):
        a[t] = (a[t - 1] @ p["P"]) * B[t]; a[t] /= a[t].sum()
    return a[:, 1]


def markov_regimes(D: Data, quick: bool = False) -> tuple[pd.Series, dict]:
    """Two-state Markov-switching model of daily log RV. Real-time filtered probability of the high-vol state:
    parameters re-estimated at the start of every month on past data only (first year: burn-in, filtered with
    its own parameters)."""
    y = np.log(D.rv_d.clip(lower=1e-7)).asfreq("D").interpolate(limit=3).dropna()
    probs = pd.Series(np.nan, index=y.index)
    burn = y.index[0] + pd.Timedelta(days=365)
    months = list(pd.date_range(burn.normalize(), y.index[-1], freq="MS"))
    cut0 = months[0] if months else y.index[-1] + pd.Timedelta(days=1)
    p = hmm_fit(y[y.index < cut0].values)
    sub = y[y.index < cut0]
    probs.loc[sub.index] = hmm_filter(sub.values, p)
    if quick:
        p = hmm_fit(y.values)
        f = pd.Series(hmm_filter(y.values, p), index=y.index)
        probs.loc[cut0:] = f.loc[cut0:]
    else:
        for i, m0 in enumerate(months):
            m1 = months[i + 1] if i + 1 < len(months) else y.index[-1] + pd.Timedelta(days=1)
            p = hmm_fit(y[y.index < m0].values)
            sub = y[y.index < m1]
            f = pd.Series(hmm_filter(sub.values, p), index=sub.index)
            probs.loc[f.index[(f.index >= m0)]] = f[f.index >= m0]
    info = {"ann. vol, low state %": float(100 * np.sqrt(np.exp(p["mu"][0]) * 365)),
            "ann. vol, high state %": float(100 * np.sqrt(np.exp(p["mu"][1]) * 365)),
            "sd of log RV low / high": f"{p['sd'][0]:.2f} / {p['sd'][1]:.2f}",
            "P(stay low)": float(p["P"][0, 0]), "P(stay high)": float(p["P"][1, 1]),
            "mean duration low, days": float(1 / (1 - p["P"][0, 0])),
            "mean duration high, days": float(1 / (1 - p["P"][1, 1])),
            "share of days high (real-time)": float((probs > 0.5).mean()),
            "last estimation": f"{y.index[-1]:%Y-%m-%d}"}
    return probs, info


def build_regimes(D: Data, quick: bool = False) -> tuple[pd.DataFrame, dict]:
    """Daily regime labels. The label for day d uses data up to the end of day d-1 only."""
    days = D.rv_d.index
    reg = pd.DataFrame(index=days)
    p, info = markov_regimes(D, quick)
    reg["ms_prob_high"] = p.reindex(days).shift(1)
    reg["ms"] = np.where(reg["ms_prob_high"] > 0.5, "high-vol", "low-vol")
    reg.loc[reg["ms_prob_high"].isna(), "ms"] = np.nan
    lrv = np.log(D.rv_d)
    q1 = lrv.rolling(365, min_periods=90).quantile(1 / 3).shift(1)
    q2 = lrv.rolling(365, min_periods=90).quantile(2 / 3).shift(1)
    lag = lrv.shift(1)
    reg["vol_tercile"] = np.select([lag <= q1, lag >= q2], ["low", "high"], "mid")
    reg.loc[q1.isna(), "vol_tercile"] = np.nan
    c = D.close_d.reindex(days)
    sma = np.exp(D.close_d).rolling(200, min_periods=150).mean().reindex(days)
    reg["trend"] = np.where(np.exp(c.shift(1)) > sma.shift(1), "bull", "bear")
    reg.loc[sma.shift(1).isna(), "trend"] = np.nan
    reg["era"] = np.where(days < ETF_LAUNCH, "pre-ETF", "post-ETF")
    reg["year"] = days.year.astype(str)
    if D.dvol is not None:
        dv = D.dvol.resample("D").last().reindex(days)
        med = dv.rolling(365, min_periods=90).median().shift(1)
        reg["dvol_regime"] = np.where(dv.shift(1) > med, "IV high", "IV low")
        reg.loc[med.isna(), "dvol_regime"] = np.nan
    reg["sample"] = np.where(days >= pd.Timestamp(RESEARCH_FROZEN, tz="UTC"), "post-freeze",
                             np.where(days >= pd.Timestamp(config.NEW_DATA_START, tz="UTC"), "new data", "history"))
    return reg, info


REGIME_COLS = ["ms", "vol_tercile", "trend", "era", "dvol_regime", "year", "sample"]


def by_regime(obs: pd.Series, reg: pd.DataFrame, lags: int | None = None) -> pd.DataFrame:
    """Primary statistic (NW mean of obs) inside every regime, plus a test that the regimes differ."""
    if obs is None or len(obs.dropna()) == 0:
        return pd.DataFrame()
    o = obs.dropna()
    day = pd.DatetimeIndex(o.index).tz_convert("UTC").floor("D") if pd.DatetimeIndex(o.index).tz is not None \
        else pd.DatetimeIndex(o.index).tz_localize("UTC").floor("D")
    rows = []
    for col in REGIME_COLS:
        if col not in reg:
            continue
        lab = reg[col].reindex(day).values
        groups = {}
        for g in pd.Series(lab).dropna().unique():
            x = o.values[lab == g]
            st = nw_mean(x, lags)
            groups[g] = x
            rows.append({"regime": col, "state": g, **{k: st[k] for k in ("mean", "t", "p", "n")}})
        if len(groups) >= 2:   # Kruskal–Wallis: do the regimes differ?
            vals = [v for v in groups.values() if len(v) >= 5]
            if len(vals) >= 2:
                h, pk = stats.kruskal(*vals)
                rows.append({"regime": col, "state": "differ? (Kruskal p)", "mean": np.nan, "t": np.nan, "p": pk,
                             "n": sum(len(v) for v in vals)})
    return pd.DataFrame(rows)


def stable_sign(tab: pd.DataFrame, overall: float) -> str:
    """'yes' if the effect has the same sign in every state of ms/trend/era with n>=20."""
    if tab.empty or not np.isfinite(overall):
        return "n/a"
    sub = tab[tab["regime"].isin(["ms", "trend", "era"]) & tab["n"].ge(20) & tab["mean"].notna()]
    if sub.empty:
        return "n/a"
    same = (np.sign(sub["mean"]) == np.sign(overall)).mean()
    return "yes" if same == 1 else f"{same:.0%} of states"


# =============================================================================================== results
@dataclass
class Result:
    id: str
    title: str
    source: str
    claim: str
    obs: pd.Series | None = None                # per-observation series whose mean is the primary effect
    lags: int | None = None
    unit: str = ""
    scale: float = 1.0                          # display multiplier for the mean (e.g. 1e4 -> bp)
    primary: dict = field(default_factory=dict)
    details: dict = field(default_factory=dict)  # name -> DataFrame
    notes: list = field(default_factory=list)
    strategies: dict = field(default_factory=dict)  # name -> daily net return series (gross kept in details)
    regimes: pd.DataFrame | None = None

    def finish(self, reg: pd.DataFrame):
        if self.obs is not None and not self.primary:
            self.primary = nw_mean(self.obs, self.lags)
        if self.obs is not None:
            self.regimes = by_regime(self.obs, reg, self.lags)
        return self


def add_strat(res, name: str, gross: pd.Series, cost):
    """Store (net, gross) daily P&L of a rule; `gross` holds one return per trade indexed by its exit time."""
    gross = pd.Series(gross).dropna()
    c = cost.reindex(gross.index).fillna(0) if isinstance(cost, pd.Series) else cost
    res.strategies[name] = (daily_pnl(gross - c), daily_pnl(gross))


def daily_pnl(trades: pd.Series) -> pd.Series:
    """Sum trade returns by UTC day of the exit (days without trades = 0 so that Sharpe is per calendar day)."""
    t = trades.dropna()
    idx = pd.DatetimeIndex(t.index)
    d = t.groupby(idx.floor("D")).sum()
    full = pd.date_range(d.index.min(), d.index.max(), freq="D")
    return d.reindex(full, fill_value=0.0)


# =============================================================================================== hypotheses
def r01_seasonality(D: Data, reg) -> Result:
    res = Result("R1", "Intraday/weekly volatility seasonality is stable and predicts out of sample",
                 "Andersen & Bollerslev (1997, 1998); for crypto Eross et al. (2019); Hansen et al. (2024)",
                 "the hour-of-week volatility profile of year y predicts the profile of year y+1 (slope > 0)",
                 unit="slope", lags=24)
    lr = np.log(D.rv_h.clip(lower=1e-12))
    lr = lr[lr > np.log(1e-11)]
    z = lr - lr.rolling(24 * 7, min_periods=24 * 3, center=False).mean().shift(1)   # remove level, real-time
    how = z.index.dayofweek * 24 + z.index.hour
    df = pd.DataFrame({"z": z, "how": how, "year": z.index.year}).dropna()
    obs, rows = [], []
    years = sorted(df["year"].unique())
    for y in years[1:]:
        prof = df[df.year == y - 1].groupby("how")["z"].mean()
        prof = prof - prof.mean()
        cur = df[df.year == y]
        f = prof.reindex(cur["how"]).values
        e0, e1 = ((cur["z"] - cur["z"].mean()) ** 2).sum(), ((cur["z"] - cur["z"].mean() - f) ** 2).sum()
        rows.append({"year": y, "corr(profile y-1, y)": np.corrcoef(prof, cur.groupby("how")["z"].mean().reindex(prof.index))[0, 1],
                     "OOS R2 hourly log-RV": 1 - e1 / e0})
        obs.append(pd.Series((cur["z"].values - cur["z"].mean()) * f / np.var(f), index=cur.index))
    res.obs = pd.concat(obs) if obs else None
    res.details["year-on-year stability"] = pd.DataFrame(rows)
    # share of daily variance in US cash hours / weekend, by year
    et_idx = D.rv_h.index.tz_convert(ET)
    td = set(D.trading_days().strftime("%Y-%m-%d"))
    us = np.array([(d.strftime("%Y-%m-%d") in td) and (9.5 <= d.hour + d.minute / 60 < 16) for d in et_idx])
    wk = D.rv_h.index.dayofweek >= 5
    share = pd.DataFrame({"rv": D.rv_h.values, "us": us, "wk": wk, "year": D.rv_h.index.year}, index=D.rv_h.index)
    srows = []
    for y, g in share.groupby("year"):
        tot = g["rv"].sum()
        srows.append({"year": y, "US cash hours: share of variance": g.loc[g.us, "rv"].sum() / tot,
                      "US cash hours: share of time": g["us"].mean(),
                      "weekend: share of variance": g.loc[g.wk, "rv"].sum() / tot, "weekend: share of time": g["wk"].mean()})
    res.details["variance shares"] = pd.DataFrame(srows)
    prof_all = df.groupby("how")["z"].mean()
    res.details["_profile"] = prof_all
    return res


def r02_sessions(D: Data, reg) -> Result:
    res = Result("R2", "BTC returns accrue outside US trading hours (post-ETF)",
                 "Lou, Polk & Skouras (2019) 'A tug of war: overnight vs intraday'; 2024 market commentary on "
                 "spot-ETF hours", "after the ETF launch, mean return 16:00→9:30 ET minus 9:30→16:00 ET > 0",
                 unit="bp/day", scale=1e4, lags=5)
    days = D.trading_days()
    o = np.array([D.at([et(d, "09:30")])[0] for d in days])
    c = np.array([D.at([et(d, "16:00")])[0] for d in days])
    df = pd.DataFrame({"open": o, "close": c}, index=pd.DatetimeIndex([et(d, "16:00") for d in days]))
    df["r_us"] = df["close"] - df["open"]
    df["r_off"] = df["open"].shift(-1) - df["close"]      # tonight (incl. weekends), known only tomorrow
    df["diff"] = df["r_off"] - df["r_us"]
    df = df.dropna()
    post = df[df.index >= ETF_LAUNCH]
    res.obs = post["diff"]
    rows = []
    for name, g in (("pre-ETF", df[df.index < ETF_LAUNCH]), ("post-ETF", post), ("all", df)):
        for col in ("r_us", "r_off", "diff"):
            st = nw_mean(g[col], 5)
            rows.append({"sample": name, "leg": col, "mean bp": st["mean"] * 1e4, "t": st["t"], "n": st["n"],
                         "ann. return": st["mean"] * 252, "ann. vol": g[col].std() * math.sqrt(252)})
    res.details["session returns"] = pd.DataFrame(rows)
    # strategies: hold only off-hours / only US hours (two trades per day)
    cost = 2 * COST_BPS / 1e4
    off = pd.Series(df["r_off"].values - cost, index=df.index + pd.Timedelta(hours=17, minutes=30))
    us = pd.Series(df["r_us"].values - cost, index=df.index)   # net; add_strat gets gross = net + cost
    add_strat(res, "R2 long off-hours only (post-ETF)", off[off.index >= ETF_LAUNCH] + cost, cost)
    add_strat(res, "R2 long US hours only (post-ETF)", us[us.index >= ETF_LAUNCH] + cost, cost)
    res.details["_cum"] = df[["r_us", "r_off"]].cumsum()
    res.notes.append("A 'hold only off-hours' rule trades twice a day: 10 bp/day of costs ≈ 25%/year — the "
                     "effect would have to be enormous to survive.")
    return res


def r03_intraday_momentum(D: Data, reg) -> Result:
    res = Result("R3", "Intraday momentum: the first half hour predicts the last half hour",
                 "Gao, Han, Li & Zhou (2018, JFE) 'Market intraday momentum'; for crypto Wen, Bouri, Xu & Zhao "
                 "(2022) — momentum and reversal at the UTC day", "sign(r 16:00 prev→10:00 ET) × r(15:30→16:00 ET) > 0",
                 unit="bp/trade", scale=1e4, lags=5)
    days = D.trading_days()
    rows = []
    for i in range(1, len(days)):
        d, p = days[i], days[i - 1]
        a = D.at([et(p, "16:00"), et(d, "10:00"), et(d, "15:00"), et(d, "15:30"), et(d, "16:00")])
        rows.append({"ts": et(d, "16:00"), "x1": a[1] - a[0], "x2": a[3] - a[2], "y": a[4] - a[3]})
    df = pd.DataFrame(rows).set_index("ts").dropna()
    res.obs = np.sign(df["x1"]) * df["y"]
    res.details["GHLZ regression r_last30 on r_first30, r_15:00-15:30"] = nw_ols(df["y"], df[["x1", "x2"]], 5, ["r_first", "r_12th"])
    # crypto variant: UTC day, first half hour -> last half hour (same day)
    days_utc = pd.date_range(D.px.index[0].normalize() + pd.Timedelta(days=1), D.end.normalize() - pd.Timedelta(days=1), freq="D")
    a0, a1 = D.at(days_utc), D.at(days_utc + pd.Timedelta(minutes=30))
    b0, b1 = D.at(days_utc + pd.Timedelta(hours=23, minutes=30)), D.at(days_utc + pd.Timedelta(days=1))
    u = pd.DataFrame({"x": a1 - a0, "y": b1 - b0}, index=days_utc + pd.Timedelta(days=1)).dropna()
    res.details["crypto variant (UTC day) r_23:30-24:00 on r_00:00-00:30"] = nw_ols(u["y"], u[["x"]], 5, ["r_first"])
    cost = 2 * COST_BPS / 1e4
    add_strat(res, "R3 intraday momentum (US close)", res.obs, cost)
    add_strat(res, "R3b intraday momentum (UTC day)", np.sign(u["x"]) * u["y"], cost)
    return res


def r04_tsmom(D: Data, reg) -> Result:
    res = Result("R4", "Time-series momentum at the weekly horizon",
                 "Liu & Tsyvinski (2021, RFS) 'Risks and returns of cryptocurrency'; Moskowitz, Ooi & Pedersen (2012)",
                 "sign(past-week return) × next-week return > 0", unit="bp/week", scale=1e4, lags=2)
    wk = D.close_d.resample("W-SUN").last()
    rw = wk.diff().dropna()
    df = pd.DataFrame({"x": rw.shift(1), "x4": rw.rolling(4).sum().shift(1), "y": rw}).dropna()
    res.obs = np.sign(df["x"]) * df["y"]
    res.details["weekly regression r_{w+1} on r_w, r_{w-3..w}"] = nw_ols(df["y"], df[["x", "x4"]], 2, ["r_w", "r_4w"])
    # variance ratios at several sampling frequencies
    rows = []
    r5 = D.lp[D.lp.index.minute % 5 == 4].diff().dropna()
    for name, r in (("1 min", D.r1.dropna()), ("5 min", r5), ("1 hour", D.r_h.dropna()), ("1 day", D.r_d.dropna())):
        for q in (2, 4, 8, 16):
            vr, z = variance_ratio(r.values, q)
            rows.append({"returns": name, "q": q, "VR": vr, "z* (robust)": z, "p": 2 * stats.norm.sf(abs(z))})
    res.details["Lo-MacKinlay variance ratios"] = pd.DataFrame(rows)
    # strategies
    pos = np.sign(df["x"])
    turn = pos.diff().abs().fillna(1)
    add_strat(res, "R4 weekly TSMOM (long/short)", pos * df["y"], turn * COST_BPS / 1e4)
    lf = (pos > 0).astype(float)
    add_strat(res, "R4b weekly TSMOM (long/flat)", lf * df["y"], lf.diff().abs().fillna(1) * COST_BPS / 1e4)
    h = D.r_h.dropna()
    rev = -np.sign(h.shift(1)) * h
    add_strat(res, "R4c hourly reversal", rev, 2 * COST_BPS / 1e4)
    res.details["hourly reversal (gross)"] = pd.DataFrame([nw_mean(rev * 1e4, 24)])
    return res


def r05_weekday(D: Data, reg) -> Result:
    res = Result("R5", "Day-of-week: Monday returns are higher", "Caporale & Plastun (2019, Finance Research Letters) "
                 "'The day of the week effect in the cryptocurrency market'; Aharon & Qadan (2019)",
                 "Monday return minus the mean of the other six days of the week > 0", unit="bp/day", scale=1e4, lags=1)
    r = D.r_d.dropna()
    df = pd.DataFrame({"r": r, "dow": r.index.dayofweek, "wk": r.index.to_period("W-SUN")})
    g = df.groupby("wk")
    mon = g.apply(lambda x: x.loc[x.dow == 0, "r"].mean() - x.loc[x.dow != 0, "r"].mean() if (x.dow == 0).any() and len(x) >= 5 else np.nan)
    mon.index = mon.index.to_timestamp().tz_localize("UTC")
    res.obs = mon.dropna()
    rows = []
    names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    rv = D.rv_d.reindex(r.index)
    for era, sub in (("pre-ETF", df[df.index < ETF_LAUNCH]), ("post-ETF", df[df.index >= ETF_LAUNCH])):
        for k in range(7):
            x = sub.loc[sub.dow == k, "r"]
            st = nw_mean(x, 0)
            rows.append({"era": era, "day": names[k], "mean bp": st["mean"] * 1e4, "t": st["t"], "n": st["n"],
                         "ann. vol": math.sqrt(rv.reindex(x.index).mean() * 365)})
    res.details["by weekday"] = pd.DataFrame(rows)
    # weekend volatility relative to Tue-Thu, pre vs post ETF (bootstrap by week)
    wv = []
    for era, lo, hi in (("pre-ETF", None, ETF_LAUNCH), ("post-ETF", ETF_LAUNCH, None)):
        x = D.rv_d
        if lo is not None:
            x = x[x.index >= lo]
        if hi is not None:
            x = x[x.index < hi]
        ratio = x[x.index.dayofweek >= 5].mean() / x[x.index.dayofweek.isin([1, 2, 3])].mean()
        weeks = x.index.to_period("W-SUN")
        uw = np.unique(weeks)
        rng = np.random.default_rng(0)
        bs = []
        grp = {w: x[weeks == w] for w in uw}
        for _ in range(300):
            s = pd.concat([grp[w] for w in rng.choice(uw, len(uw))])
            bs.append(s[s.index.dayofweek >= 5].mean() / s[s.index.dayofweek.isin([1, 2, 3])].mean())
        wv.append({"era": era, "weekend RV / Tue-Thu RV": ratio, "90% CI low": np.quantile(bs, .05), "90% CI high": np.quantile(bs, .95)})
    res.details["weekend volatility"] = pd.DataFrame(wv)
    add_strat(res, "R5 long Monday only", df.loc[df.dow == 0, "r"], 2 * COST_BPS / 1e4)
    return res


def r06_macro(D: Data, reg) -> Result:
    res = Result("R6", "Pre-FOMC drift", "Lucca & Moench (2015, JF) 'The pre-FOMC announcement drift'; "
                 "Savor & Wilson (2013) announcement-day premium", "return from 24 h to 15 min before the FOMC "
                 "statement > 0", unit="bp/event", scale=1e4, lags=0)
    cal = D.macro.copy()
    fomc = cal[cal["event"] == "FOMC"]
    a = D.at(fomc["ts"] - pd.Timedelta(hours=24))
    b = D.at(fomc["ts"] - pd.Timedelta(minutes=15))
    res.obs = pd.Series(b - a, index=pd.DatetimeIndex(fomc["ts"])).dropna()
    # unconditional 24-h return at the same clock time for comparison
    allr = pd.Series(D.at(D.rv_d.index + pd.Timedelta(hours=18, minutes=45)) - D.at(D.rv_d.index - pd.Timedelta(hours=5, minutes=15)))
    res.details["benchmark"] = pd.DataFrame([{"mean 24h return, all days, bp": np.nanmean(allr) * 1e4,
                                              "mean pre-FOMC 24h, bp": res.obs.mean() * 1e4,
                                              "share of FOMC with positive drift": (res.obs > 0).mean(), "n FOMC": len(res.obs)}])
    rows, post = [], []
    for ev in sorted(cal["event"].unique()):
        e = cal[cal["event"] == ev]
        t = pd.DatetimeIndex(e["ts"])
        # volatility in the 30 minutes after the release vs the same window on the previous 20 weekdays
        lr = []
        for ts in t:
            w = D.r1.loc[ts: ts + pd.Timedelta(minutes=29)]
            base = []
            for k in range(1, 29):
                b0 = ts - pd.Timedelta(days=k)
                if b0.dayofweek < 5 and not (cal["ts"].sub(b0).abs() < pd.Timedelta(hours=2)).any():
                    base.append((D.r1.loc[b0: b0 + pd.Timedelta(minutes=29)] ** 2).sum())
                if len(base) >= 20:
                    break
            if len(w) >= 25 and len(base) >= 10:
                lr.append(np.log((w ** 2).sum() / np.median(base)))
        st = nw_mean(lr, 0)
        # direction persistence: first 5 minutes after the release -> next 55 minutes
        f5 = D.at(t + pd.Timedelta(minutes=5)) - D.at(t)
        n55 = D.at(t + pd.Timedelta(minutes=60)) - D.at(t + pd.Timedelta(minutes=5))
        pers = nw_mean(np.sign(f5) * n55 * 1e4, 0)
        day_ret = D.r_d.reindex(t.floor("D")).dropna()
        other = D.r_d[~D.r_d.index.isin(pd.DatetimeIndex(cal["ts"]).floor("D"))]
        tw = stats.ttest_ind(day_ret, other.dropna(), equal_var=False)
        rows.append({"event": ev, "n": len(t), "vol 30 min after / normal (x)": math.exp(st["mean"]) if np.isfinite(st["mean"]) else np.nan,
                     "t (log ratio)": st["t"], "announcement-day return bp": day_ret.mean() * 1e4,
                     "other days bp": other.mean() * 1e4, "t (Welch, day return)": tw.statistic, "first 5 min → next 55 min (bp, sign-aligned)": pers["mean"],
                     "t (persistence)": pers["t"]})
        post.append(pd.Series(np.sign(f5) * n55, index=t + pd.Timedelta(minutes=60)))
    res.details["macro releases"] = pd.DataFrame(rows)
    pp = pd.concat(post).dropna()
    add_strat(res, "R6 pre-FOMC long", res.obs, 2 * COST_BPS / 1e4)
    add_strat(res, "R6b post-release continuation", pp, 2 * COST_BPS / 1e4)
    return res


def r07_vol_managed(D: Data, reg) -> Result:
    res = Result("R7", "Volatility-managed exposure beats buy-and-hold on a risk-adjusted basis",
                 "Moreira & Muir (2017, JF) 'Volatility-managed portfolios'; critique: Cederburg, O'Doherty, Wang "
                 "& Yan (2020, JFE)", "alpha of the vol-managed BTC position on buy-and-hold > 0 (net of costs)",
                 unit="bp/day", scale=1e4, lags=5)
    r = np.expm1(D.r_d.dropna())                        # simple daily returns
    rv = D.rv_d.reindex(r.index)
    f = rv.rolling(22, min_periods=15).mean().shift(1)  # forecast of tomorrow's variance from the past month
    target = r.rolling(365, min_periods=180).var().shift(1)   # real-time unconditional variance
    w = (target / f).clip(upper=2.0)                    # leverage capped at 2x
    w = w.dropna()
    rr = r.reindex(w.index)
    cost = w.diff().abs().fillna(w.abs()) * COST_BPS / 1e4
    vm = w * rr - cost
    beta = np.cov(vm, rr)[0, 1] / rr.var()
    res.obs = vm - beta * rr                             # alpha contributions
    res.details["spanning regression r_vm on r_bh"] = nw_ols(vm, rr.to_frame(), 5, ["r_bh"])
    # Moreira–Muir monthly version: weight = c / previous month's realised variance
    mr = (1 + r).resample("ME").prod() - 1
    mv = rv.resample("ME").sum()
    wm = (mv.expanding(6).mean().shift(1) / mv.shift(1)).clip(upper=2.0).dropna()
    vmm = wm * mr.reindex(wm.index) - wm.diff().abs().fillna(wm) * COST_BPS / 1e4
    res.details["monthly Moreira-Muir version: r_vm on r_bh"] = nw_ols(vmm, mr.reindex(wm.index).to_frame(), 2, ["r_bh"])
    stt = []
    for name, x in (("buy & hold", rr), ("vol-managed (daily, net)", vm)):
        s = sharpe_stats(x)
        mdd = (np.log1p(x).cumsum() - np.log1p(x).cumsum().cummax()).min()
        stt.append({"strategy": name, "Sharpe (ann.)": s["sr_ann"], "ann. return": s["ann_ret"], "ann. vol": s["ann_vol"],
                    "max drawdown (log)": mdd, "mean weight": w.mean() if "vol" in name else 1.0})
    res.details["performance"] = pd.DataFrame(stt)
    res.strategies["R7 vol-managed BTC"] = (vm, vm + cost)
    res.strategies["benchmark: buy & hold"] = (rr, rr)
    res.details["_cum"] = pd.DataFrame({"buy & hold": np.log1p(rr).cumsum(), "vol-managed": np.log1p(vm).cumsum()})
    return res


def r08_vrp(D: Data, reg) -> list[Result]:
    if D.dvol is None:
        return []
    a = Result("R8a", "Variance risk premium: implied variance exceeds subsequent realised variance",
               "Carr & Wu (2009, RFS); for BTC Alexander & Imeraj (2021) 'The Bitcoin VIX and its variance risk premium'",
               "DVOL² − realised variance over the next 30 days > 0", unit="var. points (ann., ×100²)", scale=1e4, lags=30)
    b = Result("R8b", "The variance risk premium predicts returns", "Bollerslev, Tauchen & Zhou (2009, RFS)",
               "z(VRP_t) × return over the next 30 days > 0, VRP_t = DVOL²_t − RV(past 30 days)",
               unit="bp/30d (per 1 sd)", scale=1e4, lags=30)
    days = D.rv_d.index
    iv2 = (D.dvol.resample("D").last() / 100) ** 2           # annualised 30-day implied variance at 00:00 UTC
    iv2 = iv2.reindex(days)
    rv = D.rv_d.asfreq("D")
    fwd = rv[::-1].rolling(30, min_periods=28).sum()[::-1].shift(-1) * 365 / 30
    past = rv.rolling(30, min_periods=28).sum().shift(1) * 365 / 30
    vrp_post = (iv2 - fwd.reindex(days)).dropna()
    a.obs = vrp_post
    a.details["levels"] = pd.DataFrame([{"mean DVOL (ann. vol %)": 100 * np.sqrt(iv2.mean()),
                                         "mean realised next 30d (ann. vol %)": 100 * np.sqrt(fwd.mean()),
                                         "share of days IV² > RV": (vrp_post > 0).mean(),
                                         "mean log(IV²/RV)": np.log(iv2 / fwd.reindex(days)).mean()}])
    vrp = (iv2 - past.reindex(days))
    lp = D.close_d.reindex(days)
    rows = []
    for h in (7, 30):
        y = lp.shift(-h) - lp
        rows.append(nw_ols(y, vrp.to_frame(), h, ["VRP"]).assign(horizon_days=h))
        if h == 30:
            b.obs = (zexp(vrp, 90) * y).dropna()
    b.details["predictive regressions"] = pd.concat(rows)
    # Campbell–Thompson out-of-sample R² (expanding window, 30-day horizon)
    y = (lp.shift(-30) - lp)
    df = pd.DataFrame({"y": y, "x": vrp}).dropna()
    pred, hist = [], []
    for i in range(365, len(df)):
        tr = df.iloc[: i - 30]                          # targets that are already known at time i
        bb = np.polyfit(tr["x"], tr["y"], 1)
        pred.append(np.polyval(bb, df["x"].iloc[i]))
        hist.append(tr["y"].mean())
    if pred:
        yy = df["y"].iloc[365:].values
        b.details["out-of-sample R2 (Campbell-Thompson)"] = pd.DataFrame([{"R2_oos": 1 - ((yy - pred) ** 2).sum() / ((yy - hist) ** 2).sum(), "n": len(yy)}])
    a.details["_series"] = pd.DataFrame({"DVOL² (ann.)": iv2, "realised next 30d (ann.)": fwd.reindex(days)})
    return [a, b]


def _panel(h: str) -> pd.DataFrame | None:
    f = config.INTERIM / f"panel_{h}.parquet"
    if not f.exists():
        return None
    p = pd.read_parquet(f)
    for c in ("t", "T", "e"):
        p[c] = pd.to_datetime(p[c], utc=True)
    return p


def r09_ladder_direction(D: Data, reg) -> Result:
    res = Result("R9", "Prediction-market ladders know the direction",
                 "new (this project); related: Wolfers & Zitzewitz (2004) on prediction-market efficiency",
                 "implied drift of the Kalshi hourly ladder (median + s²/2 − log spot, in units of s) × next-hour "
                 "return > 0", unit="bp/occasion (per 1 s)", scale=1e4, lags=1)
    rows, strat = [], {}
    for h, lag in (("kx1h", 1), ("pm24h", 1), ("pm6d", 6)):
        p = _panel(h)
        if p is None:
            continue
        p = p.dropna(subset=["mkt_m", "mkt_s", "S_t", "S_e"]).copy()
        p = p[(p["mkt_s"] > 0) & (p["mkt_s"] < 0.5)]
        p["d"] = (p["mkt_m"] + 0.5 * p["mkt_s"] ** 2 - np.log(p["S_t"])) / p["mkt_s"]
        p["d"] = p["d"].clip(-3, 3)
        p["r"] = np.log(p["S_e"] / p["S_t"])
        p = p.set_index("e").sort_index()
        reg_ = nw_ols(p["r"], p[["d"]], lag, ["drift/s"])
        rows.append(reg_.assign(horizon=h, n=reg_.attrs["n"], R2=reg_.attrs["r2"]))
        if h == "kx1h":
            res.obs = p["d"] * p["r"]
        strat[h] = np.sign(p["d"]) * p["r"] / lag   # overlapping 6-day trades: 1/6 of capital each
    res.details["regression r(t→T) on implied drift"] = pd.concat(rows) if rows else pd.DataFrame()
    for h, s in strat.items():
        add_strat(res, f"R9 trade the ladder drift ({h})", s, 2 * COST_BPS / 1e4 / (6 if h == "pm6d" else 1))
    res.notes.append("Ladder prices are in USD (Kalshi: CF BRTI; Polymarket: Binance BTC/USDT), spot is Binance BTC/USDT: "
                     "the USDT premium and the index basis enter the drift as noise.")
    return res


def _strikes(h: str) -> pd.DataFrame | None:
    f = config.INTERIM / f"strikes_{h}.parquet"
    if not f.exists():
        return None
    s = pd.read_parquet(f)
    for c in ("t", "T"):
        s[c] = pd.to_datetime(s[c], utc=True)
    return s


def kalshi_fee(p):
    """Kalshi taker fee per contract: 0.07 * P * (1 - P), rounded up to the cent."""
    return np.ceil(np.round(config.KALSHI_FEE_RATE * p * (1 - p) * 100, 6)) / 100


def r10_ladder_tails(D: Data, reg) -> Result:
    res = Result("R10", "Crash insurance is overpriced in prediction-market ladders",
                 "options analogue: Bollen & Whaley (2004), Bakshi, Kapadia & Madan (2003); favourite–longshot bias "
                 "(Snowberg & Wolfers 2010)", "Kalshi hourly: implied P(fall > 1 s) − realised frequency > 0",
                 unit="pp", scale=100, lags=1)
    rows, trade_rows = [], []
    for h in ("kx1h", "pm24h", "pm6d"):
        s, p = _strikes(h), _panel(h)
        if s is None or p is None:
            continue
        p = p.dropna(subset=["mkt_s", "S_t", "S_e"]).drop_duplicates(["T", "t"])
        key = p.set_index(["T", "t"])[["mkt_s", "S_t", "S_e"]]
        s = s.merge(key.reset_index(), on=["T", "t"], how="inner")
        out, trades = [], []
        for (T, t), g in s.groupby(["T", "t"]):
            if len(g) < 4:
                continue
            ms_, S, Se = g[["mkt_s", "S_t", "S_e"]].iloc[0].values
            g = g.sort_values("strike")
            x = np.log(g["strike"].values / S)
            pr = np.minimum.accumulate(np.clip(g["p"].values, 0, 1))     # P(above K) must fall with K
            if "expiration_value" in g and g["expiration_value"].notna().any():
                Se = float(g["expiration_value"].dropna().iloc[0])      # the settlement value itself
            row = {"T": T}
            for side, k in (("up", 1.0), ("dn", -1.0)):
                xk = k * ms_
                if x.min() <= xk <= x.max():
                    above = float(np.interp(xk, x, pr))
                    if side == "up":
                        row["p_up"], row["i_up"] = above, float(np.log(Se / S) > xk)
                    else:
                        row["p_dn"], row["i_dn"] = 1 - above, float(np.log(Se / S) < xk)
                # a real contract near the 1-s point, at the real quotes (Kalshi has bid/ask)
                if {"bid", "ask", "outcome"} <= set(g.columns):
                    j = int(np.argmin(np.abs(x - xk)))
                    if abs(x[j] - xk) < 0.5 * ms_ and np.isfinite(g["outcome"].iloc[j]):
                        bid, ask, o = g["bid"].iloc[j], g["ask"].iloc[j], g["outcome"].iloc[j]
                        if side == "up" and bid > 0:      # sell the upside tail: sell YES 'above K' at the bid
                            trades.append({"T": T, "tail": "sell upside tail", "price": bid,
                                           "pnl": bid - o - kalshi_fee(bid)})
                        if side == "dn" and 0 < ask < 1:  # sell the downside tail = buy YES 'above K' at the ask
                            trades.append({"T": T, "tail": "sell downside tail", "price": 1 - ask,
                                           "pnl": o - ask - kalshi_fee(ask)})
            out.append(row)
        o = pd.DataFrame(out).set_index("T").sort_index()
        if o.empty:
            continue
        lag = 6 if h == "pm6d" else 1
        for side in ("up", "dn"):
            if f"p_{side}" not in o:
                continue
            x = (o[f"p_{side}"] - o[f"i_{side}"]).dropna()
            st = nw_mean(x, lag)
            ln = 1 - stats.norm.cdf(1.0)   # lognormal benchmark for a 1-s move (≈15.9%)
            rows.append({"horizon": h, "tail": "fall > 1 s" if side == "dn" else "rise > 1 s",
                         "implied %": 100 * o[f"p_{side}"].mean(), "realised %": 100 * o[f"i_{side}"].mean(),
                         "lognormal %": 100 * ln, "overpricing pp": 100 * st["mean"], "t": st["t"], "n": st["n"]})
            if h == "kx1h" and side == "dn":
                res.obs = x
        if {"p_up", "p_dn"} <= set(o):
            asym = ((o["p_dn"] - o["i_dn"]) - (o["p_up"] - o["i_up"])).dropna()
            st = nw_mean(asym, lag)
            rows.append({"horizon": h, "tail": "down minus up overpricing", "overpricing pp": 100 * st["mean"],
                         "t": st["t"], "n": st["n"]})
        if trades:
            tr = pd.DataFrame(trades).set_index("T").sort_index()
            for name, g in tr.groupby("tail"):
                st = nw_mean(g["pnl"], 1)
                trade_rows.append({"horizon": h, "trade": name, "mean price ¢": 100 * g["price"].mean(),
                                   "mean P&L after fee ¢": 100 * st["mean"], "t": st["t"], "n": st["n"],
                                   "hit rate": (g["pnl"] > 0).mean()})
                # one contract per trade; P&L in cents of a $1 contract, expressed per $1 at risk ~ price
                add_strat(res, f"R10 {name} ({h}, $1 contracts)", g["pnl"], 0.0)
    res.details["tail pricing (mid quotes)"] = pd.DataFrame(rows)
    res.details["tradeable: nearest real strike, real bid/ask, Kalshi fee"] = pd.DataFrame(trade_rows)
    res.notes.append("s is the ladder's own lognormal width, so the test asks whether the ladder's tails are "
                     "consistent with its own centre — no volatility model involved. Tradeable P&L is per $1 "
                     "contract (pays $1), so its Sharpe is not comparable to a BTC position.")
    return res


def r11_funding(D: Data, reg) -> Result | None:
    from src.download import funding
    f = funding.load()
    if f is None or len(f) < 200:
        return None
    res = Result("R11", "Perpetual funding predicts returns (crowded longs lose)",
                 "He, Manela, Ross & von Wachter (2022) 'Fundamentals of perpetual futures'; Christin et al. (2023)",
                 "−z(funding over the past 7 days) × return over the next 7 days > 0", unit="bp/7d (per 1 sd)", scale=1e4, lags=7)
    fd = f.set_index("ts")["rate"].resample("D").sum()
    days = D.rv_d.index
    f7 = fd.rolling(7, min_periods=6).sum().reindex(days)
    lp = D.close_d.reindex(days)
    y = lp.shift(-7) - lp
    res.obs = (-zexp(f7, 90) * y).dropna()
    res.details["regression r(t→t+7d) on funding(t−7d→t)"] = nw_ols(y, f7.to_frame(), 7, ["funding_7d"])
    res.details["funding level"] = pd.DataFrame([{"mean annualised funding %": 100 * fd.mean() * 365,
                                                  "share of positive prints": (f["rate"] > 0).mean(), "n prints": len(f)}])
    # carry: short perp + long spot collects funding (cost: one round trip per month on both legs)
    carry = fd.reindex(pd.date_range(fd.index.min(), fd.index.max(), freq="D"), fill_value=0)
    cc = (carry.index.day == 1) * 4 * COST_BPS / 1e4
    res.strategies["R11 cash-and-carry (collect funding)"] = (carry - cc, carry)
    pos = -np.sign(zexp(f7, 90)).shift(1)
    rr = D.r_d.reindex(days)
    g = (pos * rr).dropna()
    res.strategies["R11b contrarian to funding (daily)"] = (g - pos.diff().abs().fillna(0).reindex(g.index) * COST_BPS / 1e4, g)
    return res


def r12_dvol_dynamics(D: Data, reg) -> Result | None:
    if D.dvol is None:
        return None
    res = Result("R12", "Implied-volatility changes lead BTC returns", "Granger tests as in Bollerslev, Litvinova & "
                 "Tauchen (2006); 'inverted' leverage effect in BTC: Bouri, Azzi & Dyhrberg (2017)",
                 "z(ΔDVOL in hour h) × return in hour h+1 ≠ 0", unit="bp/hour (per 1 sd)", scale=1e4, lags=24)
    dv = np.log(D.dvol).diff()
    r = D.r_h.reindex(dv.index)
    df = pd.DataFrame({"dv": dv, "r": r, "r1": r.shift(-1), "dv1": dv.shift(-1)}).dropna()
    df = df[df["dv"].abs() < 0.3]
    res.obs = (zexp(df["dv"], 500) * df["r1"]).dropna()
    res.details["Granger: r_{h+1} on ΔlogDVOL_h, r_h"] = nw_ols(df["r1"], df[["dv", "r"]], 24, ["dlogDVOL_h", "r_h"])
    res.details["Granger: ΔlogDVOL_{h+1} on r_h, ΔlogDVOL_h"] = nw_ols(df["dv1"], df[["r", "dv"]], 24, ["r_h", "dlogDVOL_h"])
    day = df.index.floor("D")
    rows = []
    for col in ("trend", "ms", "era", "year"):
        lab = reg[col].reindex(day).values
        for g in pd.Series(lab).dropna().unique():
            sub = df[lab == g]
            if len(sub) > 100:
                c = np.corrcoef(sub["r"], sub["dv"])[0, 1]
                rows.append({"regime": col, "state": g, "corr(r_h, ΔlogDVOL_h)": c, "n hours": len(sub),
                             "corr(r_h−, Δ) down hours": np.corrcoef(sub.loc[sub.r < 0, "r"], sub.loc[sub.r < 0, "dv"])[0, 1],
                             "corr(r_h+, Δ) up hours": np.corrcoef(sub.loc[sub.r > 0, "r"], sub.loc[sub.r > 0, "dv"])[0, 1]})
    res.details["leverage effect by regime"] = pd.DataFrame(rows)
    return res


def r13_top_of_hour(D: Data, reg) -> Result:
    res = Result("R13", "Volatility spikes in the first minute of the hour (and at funding times)",
                 "market-microstructure folklore on candle closes / algorithmic rebalancing; funding at 00/08/16 UTC",
                 "log(variance of minute :00 / mean variance of minutes :01–:59), per day, > 0", unit="log ratio", lags=5)
    r2 = D.r1 ** 2
    df = pd.DataFrame({"r2": r2, "min": r2.index.minute, "hour": r2.index.hour, "day": r2.index.floor("D")}).dropna()
    g0 = df[df["min"] == 0].groupby("day")["r2"].mean()
    g1 = df[df["min"] != 0].groupby("day")["r2"].mean()
    res.obs = np.log(g0 / g1).replace([np.inf, -np.inf], np.nan).dropna()
    prof = df.groupby("min")["r2"].mean()
    res.details["_minute_profile"] = prof / prof.drop(0).median()
    fund = df["hour"].isin([0, 8, 16])
    ratio = {}
    for name, sel in (("funding hours", fund), ("other hours", ~fund)):
        a = df[(df["min"] == 0) & sel].groupby("day")["r2"].mean()
        b = df[(df["min"] != 0) & sel].groupby("day")["r2"].mean()
        ratio[name] = np.log(a / b).replace([np.inf, -np.inf], np.nan)
    st = nw_mean((ratio["funding hours"] - ratio["other hours"]).dropna(), 5)
    res.details["minute :00 spike: funding hours (00/08/16 UTC) vs other hours"] = pd.DataFrame(
        [{"spike in funding hours (x)": math.exp(ratio["funding hours"].mean()),
          "spike in other hours (x)": math.exp(ratio["other hours"].mean()),
          "difference t": st["t"], "n days": st["n"]}])
    # robustness: drop US macro-release days (8:30 ET releases put a spike on minute :30)
    rel = set(pd.DatetimeIndex(D.macro["ts"]).floor("D"))
    ex = df[~df["day"].isin(rel)]
    prof_ex = ex.groupby("min")["r2"].mean()
    prof_ex = prof_ex / prof_ex.drop(0).median()
    top = (prof / prof.drop(0).median()).sort_values(ascending=False).head(8)
    res.details["minute-of-hour"] = pd.DataFrame({"minute": top.index, "variance / median minute": top.values,
                                                  "same, macro-release days dropped": prof_ex.reindex(top.index).values})
    opens = pd.DatetimeIndex([et(d, "09:30") for d in D.trading_days()])
    us_open = (D.r1.reindex(opens) ** 2).dropna()
    res.details["US open (minute 9:30 ET) vs average minute"] = pd.DataFrame(
        [{"ratio": us_open.mean() / df["r2"].mean() if len(us_open) else np.nan, "n": len(us_open)}])
    return res


def r14_cme_gap(D: Data, reg) -> Result:
    res = Result("R14", "'CME gaps always fill'", "trader folklore; tested here against the exact reflection-principle "
                 "null for a driftless random walk (Broadie–Glasserman continuity correction)",
                 "share of weekend gaps revisited before the next Friday close − probability under the null > 0",
                 unit="pp", scale=100, lags=1)
    rows = []
    fridays = pd.date_range(D.px.index[0] + pd.Timedelta(days=8), D.end - pd.Timedelta(days=6), freq="W-FRI")
    for fr in fridays:
        tc = pd.Timestamp(f"{fr:%Y-%m-%d} 16:00").tz_localize("America/Chicago").tz_convert("UTC")
        to = (pd.Timestamp(f"{fr:%Y-%m-%d} 17:00") + pd.Timedelta(days=2)).tz_localize("America/Chicago").tz_convert("UTC")
        te = (pd.Timestamp(f"{fr:%Y-%m-%d} 16:00") + pd.Timedelta(days=7)).tz_localize("America/Chicago").tz_convert("UTC")
        if te > D.end:
            break
        lc, lo = D.at([tc])[0], D.at([to])[0]
        g = lo - lc
        if not np.isfinite(g) or abs(g) < 0.0005:   # gaps below 5 bp are noise and fill trivially
            continue
        path = D.lp.loc[to: te]
        filled = (path.min() <= lc) if g > 0 else (path.max() >= lc)
        sig = D.r1.loc[tc - pd.Timedelta(days=7): tc].std()            # per-minute vol, known at the gap
        n = len(path)
        barrier = abs(g) + 0.5826 * sig                                   # discrete-monitoring correction
        p0 = min(1.0, 2 * stats.norm.sf(barrier / (sig * math.sqrt(n))))
        when = path[(path <= lc) if g > 0 else (path >= lc)].index
        rows.append({"ts": to, "gap bp": g * 1e4, "filled": float(filled), "p_null": p0,
                     "hours to fill": (when[0] - to).total_seconds() / 3600 if len(when) else np.nan})
    df = pd.DataFrame(rows).set_index("ts")
    res.obs = df["filled"] - df["p_null"]
    z = (df["filled"].sum() - df["p_null"].sum()) / math.sqrt((df["p_null"] * (1 - df["p_null"])).sum())
    big = df[df["gap bp"].abs() > 100]
    res.details["gap fills"] = pd.DataFrame([{"weeks": len(df), "filled %": 100 * df["filled"].mean(),
                                              "null %": 100 * df["p_null"].mean(), "Poisson-binomial z": z,
                                              "median hours to fill": df["hours to fill"].median(),
                                              "gaps >1%: n": len(big), "gaps >1%: filled %": 100 * big["filled"].mean() if len(big) else np.nan,
                                              "gaps >1%: null %": 100 * big["p_null"].mean() if len(big) else np.nan}])
    res.details["_gaps"] = df
    return res


def r15_round_numbers(D: Data, reg) -> Result:
    res = Result("R15", "Round numbers act as support/resistance", "Osler (2000, 2003) 'Currency orders and exchange-rate "
                 "dynamics'; Mitchell (2001)", "continuation through a $1,000 level − continuation through placebo "
                 "levels ($250/$500/$750 offsets) < 0 (round levels repel)", unit="bp/approach", scale=1e4, lags=1)
    px = D.px.resample("5min").last().dropna()
    lp = np.log(px)
    events = []
    step = 1000.0
    for off in (0.0, 250.0, 500.0, 750.0):
        lvl = np.round((px.values - off) / step) * step + off          # nearest level of this grid
        dist = np.log(px.values / lvl)
        near = np.abs(dist) < 0.0005                                    # within 5 bp
        far = pd.Series(np.abs(dist) > 0.0020, index=px.index)          # > 20 bp away
        far_before = far.rolling(12).min().shift(1).fillna(0).astype(bool).values   # the whole previous hour away
        idx = np.where(near & far_before)[0]
        for i in idx:
            if i < 12 or i + 12 >= len(px):
                continue
            L = lvl[i]
            side = np.sign(lp.iloc[i - 12] - np.log(L))                 # approach from above (+) or below (−)
            cont = -side * (lp.iloc[i + 12] - lp.iloc[i])               # positive = went through the level
            events.append({"ts": px.index[i], "offset": off, "round": off == 0.0, "cont": cont})
    ev = pd.DataFrame(events).set_index("ts").sort_index()
    plc = ev[~ev["round"]].groupby(ev[~ev["round"]].index.to_period("M"))["cont"].mean()
    rd = ev[ev["round"]]
    base = plc.reindex(rd.index.to_period("M")).values
    res.obs = pd.Series(rd["cont"].values - base, index=rd.index).dropna()
    rows = []
    for off, g in ev.groupby("offset"):
        st = nw_mean(g["cont"] * 1e4, 1)
        rows.append({"grid": "$1000 (round)" if off == 0 else f"$1000 + {off:.0f} (placebo)", "approaches": len(g),
                     "mean continuation 1h, bp": st["mean"], "t": st["t"], "share going through": (g["cont"] > 0).mean()})
    res.details["approaches"] = pd.DataFrame(rows)
    return res


def r16_turn_of_month(D: Data, reg) -> Result:
    res = Result("R16", "Turn-of-the-month effect", "Lakonishok & Smidt (1988); Ariel (1987); for BTC flows: ETF "
                 "creations and month-end rebalancing", "mean daily return in [last day, first 3 days] − other days > 0",
                 unit="bp/day", scale=1e4, lags=1)
    r = D.r_d.dropna()
    dom = r.index.day
    last = (r.index + pd.Timedelta(days=1)).day == 1
    tom = last | (dom <= 3)
    df = pd.DataFrame({"r": r, "tom": tom, "m": r.index.to_period("M")})
    by = df.groupby("m").apply(lambda g: g.loc[g.tom, "r"].mean() - g.loc[~g.tom, "r"].mean() if g.tom.any() and (~g.tom).any() else np.nan)
    by.index = by.index.to_timestamp().tz_localize("UTC")
    res.obs = by.dropna()
    rows = []
    for era, sub in (("pre-ETF", df[df.index < ETF_LAUNCH]), ("post-ETF", df[df.index >= ETF_LAUNCH])):
        a, b = nw_mean(sub.loc[sub.tom, "r"] * 1e4, 1), nw_mean(sub.loc[~sub.tom, "r"] * 1e4, 1)
        rows.append({"era": era, "TOM mean bp": a["mean"], "t": a["t"], "other mean bp": b["mean"], "n TOM days": a["n"]})
    res.details["by era"] = pd.DataFrame(rows)
    pos = np.asarray(tom, float)
    g = pd.Series(pos * r.values, index=r.index)
    res.strategies["R16 long turn-of-month only"] = (g - np.abs(np.diff(np.r_[0, pos])) * COST_BPS / 1e4, g)
    return res


def r17_semivariance(D: Data, reg) -> Result:
    res = Result("R17", "Bad volatility predicts future volatility more than good volatility",
                 "Patton & Sheppard (2015, REStat) 'Good volatility, bad volatility'; Barndorff-Nielsen et al. (2010)",
                 "out-of-sample QLIKE(HAR) − QLIKE(HAR with semivariances) > 0 (daily, 1-day ahead)", unit="QLIKE gain", lags=5)
    rv, rp, rn = D.rv_d.asfreq("D"), D.rs_pos.asfreq("D"), D.rs_neg.asfreq("D")
    df = pd.DataFrame({"y": rv.shift(-1), "rv": rv, "rp": rp, "rn": rn,
                       "rw": rv.rolling(7, min_periods=5).mean(), "rm": rv.rolling(30, min_periods=20).mean()}).dropna()
    full = nw_ols(df["y"], df[["rp", "rn", "rw", "rm"]], 5, ["RS+", "RS-", "RV_week", "RV_month"])
    res.details["in-sample HAR-RS (levels)"] = full
    gain, idx = [], []
    for i in range(365, len(df)):
        tr = df.iloc[i - 365: i - 1]
        x0 = np.c_[np.ones(len(tr)), tr[["rv", "rw", "rm"]].values]
        x1 = np.c_[np.ones(len(tr)), tr[["rp", "rn", "rw", "rm"]].values]
        b0 = np.linalg.lstsq(x0, tr["y"].values, rcond=None)[0]
        b1 = np.linalg.lstsq(x1, tr["y"].values, rcond=None)[0]
        row = df.iloc[i]
        f0 = max(np.r_[1, row[["rv", "rw", "rm"]].values] @ b0, 0.2 * row["rm"])
        f1 = max(np.r_[1, row[["rp", "rn", "rw", "rm"]].values] @ b1, 0.2 * row["rm"])
        y = row["y"]
        q = lambda f: y / f - np.log(y / f) - 1
        gain.append(q(f0) - q(f1))
        idx.append(df.index[i] + pd.Timedelta(days=1))
    res.obs = pd.Series(gain, index=pd.DatetimeIndex(idx))
    return res


def r18_models_by_regime(D: Data, reg) -> Result | None:
    f = config.RESULTS / "scores.parquet"
    if not f.exists():
        return None
    s = pd.read_parquet(f)
    s["t"] = pd.to_datetime(s["t"], utc=True)
    res = Result("R18", "Does the ladder's information gain depend on the volatility regime?",
                 "regime-dependent forecast evaluation: Giacomini & Rossi (2010) fluctuation test idea; this project",
                 "Kalshi hourly: CRPS(HAR) − CRPS(HAR-X with ladder) is larger in the high-vol Markov state",
                 unit="CRPS gain", lags=24)
    rows = []
    best = {"kx1h": "harx_all", "pm24h": "harx_all", "pm6d": "harx_all"}
    for h, g in s.groupby("horizon"):
        piv = g.pivot_table(index="t", columns="model", values="crps")
        if "har" not in piv:
            continue
        lab = reg["ms"].reindex(piv.index.floor("D")).values
        keep = ["naive", "harx", "harx_d", "har_m", "harx_m", "harx_all", "mkt_mz", "combo", "pool", "gbm_all"]
        for m in [c for c in keep if c in piv.columns]:
            d = (piv["har"] - piv[m])
            for state in ("low-vol", "high-vol"):
                x = d[lab == state].dropna()
                st = nw_mean(x, 24 if h == "kx1h" else 5)
                rows.append({"horizon": h, "model": m, "state": state, "CRPS gain vs HAR": st["mean"],
                             "t": st["t"], "n": st["n"], "relative gain %": 100 * st["mean"] / piv["har"][lab == state].mean()})
        if h == "kx1h" and best[h] in piv:
            d = (piv["har"] - piv[best[h]]).dropna()
            res.obs = d
            hi = (reg["ms"].reindex(d.index.floor("D")).values == "high-vol").astype(float)
            ok = ~pd.isna(reg["ms"].reindex(d.index.floor("D")).values)
            inter = nw_ols(d.values[ok], hi[ok].reshape(-1, 1), 24, ["high-vol"])
            res.primary = {"mean": inter.loc["high-vol", "coef"], "t": inter.loc["high-vol", "t"],
                           "p": inter.loc["high-vol", "p"], "n": int(ok.sum()), "se": np.nan}
            res.details["interaction: gain on high-vol dummy"] = inter
    res.details["CRPS gain by Markov state"] = pd.DataFrame(rows)
    return res


def r19_regime_har(D: Data, reg) -> Result:
    res = Result("R19", "A regime-aware HAR forecasts daily volatility better",
                 "Markov-switching HAR: Raggi & Bordignon (2012); McAleer & Medeiros (2008); Corsi (2009) baseline",
                 "out-of-sample QLIKE(HAR) − QLIKE(HAR with coefficients interacted with the real-time Markov "
                 "probability) > 0", unit="QLIKE gain", lags=5)
    lrv = np.log(D.rv_d.clip(lower=1e-7)).asfreq("D")
    # probability known at the end of day t (reg is shifted: reg.loc[t+1] uses data up to t)
    pr = reg["ms_prob_high"].shift(-1).reindex(lrv.index)
    df = pd.DataFrame({"y": lrv.shift(-1), "d": lrv, "w": lrv.rolling(7, min_periods=5).mean(),
                       "m": lrv.rolling(30, min_periods=20).mean(), "p": pr}).dropna()
    gains, mse, idx = [], [], []
    for i in range(365, len(df)):
        tr = df.iloc[i - 365: i - 1]
        row = df.iloc[i]
        X0 = np.c_[np.ones(len(tr)), tr[["d", "w", "m"]].values]
        X1 = np.c_[X0, tr["p"].values[:, None] * X0]
        x0 = np.r_[1, row[["d", "w", "m"]].values]
        x1 = np.r_[x0, row["p"] * x0]
        out = []
        for X, x in ((X0, x0), (X1, x1)):
            b = np.linalg.lstsq(X, tr["y"].values, rcond=None)[0]
            e = tr["y"].values - X @ b
            f = math.exp(x @ b) * np.mean(np.exp(e))        # level forecast with the smearing correction
            out.append((f, x @ b))
        y = math.exp(row["y"])
        q = [y / f - math.log(y / f) - 1 for f, _ in out]
        gains.append(q[0] - q[1])
        mse.append((row["y"] - out[0][1]) ** 2 - (row["y"] - out[1][1]) ** 2)
        idx.append(df.index[i] + pd.Timedelta(days=1))
    res.obs = pd.Series(gains, index=pd.DatetimeIndex(idx))
    st = nw_mean(mse, 5)
    res.details["same comparison on squared log errors"] = pd.DataFrame([{"MSE(log) gain": st["mean"], "t": st["t"], "n": st["n"]}])
    return res


def r20_ml_direction(D: Data, reg) -> Result:
    res = Result("R20", "Machine learning predicts the next-hour BTC return",
                 "Gu, Kelly & Xiu (2020, RFS) 'Empirical asset pricing via machine learning'; for crypto Akyildirim et al. (2021)",
                 "walk-forward gradient-boosted trees (20 price/volume/calendar features, retrained monthly on the "
                 "past year): sign(prediction) × next-hour return > 0", unit="bp/hour", scale=1e4, lags=1)
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from lab.ml import features
    m = pd.read_parquet(config.INTERIM / "minutes.parquet", columns=["open_time", "close", "volume"])
    m["open_time"] = pd.to_datetime(m["open_time"], utc=True)
    m = m.set_index("open_time").sort_index()
    m = m[~m.index.duplicated()]
    close, vol = m["close"].astype(float), m["volume"].astype(float)
    lc = np.log(close)
    grid = pd.date_range(close.index[0] + pd.Timedelta(days=5), close.index[-1] - pd.Timedelta(hours=2), freq="5min")
    X = features(close, vol, grid)
    y = lc.reindex(grid + pd.Timedelta(minutes=59), method="ffill").values - lc.reindex(grid - pd.Timedelta(minutes=1), method="ffill").values
    ok = np.isfinite(X.values).all(1) & np.isfinite(y)
    X, y = X[ok], pd.Series(y[ok], index=X.index[ok])
    months = pd.date_range(X.index[0] + pd.Timedelta(days=370), X.index[-1], freq="MS")
    preds = {"gbm": [], "ridge": []}
    for i, m0 in enumerate(months):
        m1 = months[i + 1] if i + 1 < len(months) else X.index[-1] + pd.Timedelta(minutes=5)
        tr = (X.index < m0 - pd.Timedelta(hours=1)) & (X.index >= m0 - pd.Timedelta(days=365))
        te = (X.index >= m0) & (X.index < m1) & (X.index.minute == 0)        # decisions on the hour
        if tr.sum() < 20000 or te.sum() == 0:
            continue
        yt = y[tr].clip(-0.05, 0.05).values
        g = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.03, max_leaf_nodes=15, min_samples_leaf=300,
                                          l2_regularization=1.0, random_state=0).fit(X[tr].values, yt)
        r = make_pipeline(StandardScaler(), Ridge(alpha=30.0)).fit(X[tr].values, yt)
        preds["gbm"].append(pd.Series(g.predict(X[te].values), index=X.index[te]))
        preds["ridge"].append(pd.Series(r.predict(X[te].values), index=X.index[te]))
    rows = []
    for name in ("gbm", "ridge"):
        if not preds[name]:
            continue
        p = pd.concat(preds[name])
        yy = y.reindex(p.index)
        ic = stats.spearmanr(p, yy).correlation
        sig = np.sign(p) * yy
        if name == "gbm":
            res.obs = sig
        thr = (p.abs() > 0.0015)
        tr_ = np.sign(p[thr]) * yy[thr]
        st = nw_mean(sig * 1e4, 1)
        rows.append({"model": name, "hours": len(p), "rank IC": ic, "sign × r, bp": st["mean"], "t": st["t"],
                     "hit rate": (np.sign(p) == np.sign(yy)).mean(), "trades |pred|>15bp": int(thr.sum()),
                     "mean per trade gross bp": tr_.mean() * 1e4 if len(tr_) else np.nan})
        add_strat(res, f"R20 {name} hourly (|pred| > 15 bp)", tr_, 2 * COST_BPS / 1e4)
        add_strat(res, f"R20 {name} hourly (always in)", sig, (np.sign(p).diff().abs().fillna(1) / 2) * 2 * COST_BPS / 1e4)
    res.details["walk-forward results"] = pd.DataFrame(rows)
    res.notes.append("Decisions every hour on the hour, model refitted each month on the previous 365 days only.")
    return res


def _garch_next(fit, kind, rr):
    """One-day-ahead variance from a fitted arch model with FIXED parameters, run through returns up to today
    (the recursion is written out: re-creating arch models every day is far too slow)."""
    if fit is None:
        return np.nan
    p = fit.params
    mu = p.get("mu", 0.0)
    eps = (rr - mu).values
    w = p["omega"]
    a = p.get("alpha[1]", 0.0)
    g = p.get("gamma[1]", 0.0)
    b = p.get("beta[1]", 0.0)
    if kind == "EGARCH":
        lv = math.log(np.var(eps))
        ez = math.sqrt(2 / math.pi)
        for e in eps:
            z = e / math.exp(lv / 2)
            lv = w + a * (abs(z) - ez) + g * z + b * lv
        return math.exp(lv) / 1e4
    v = np.var(eps)
    for e in eps:
        v = w + (a + g * (e < 0)) * e * e + b * v
    return v / 1e4


def r21_vol_zoo(D: Data, reg) -> Result:
    res = Result("R21", "Does anything beat HAR? Daily volatility model zoo", "Hansen & Lunde (2005) 'Does anything beat "
                 "a GARCH(1,1)?'; Andersen, Bollerslev, Diebold & Labys (2003); Corsi (2009)",
                 "1-day-ahead QLIKE(GARCH(1,1)) − QLIKE(HAR) > 0 (realised-measure models beat GARCH)", unit="QLIKE", lags=5)
    from arch import arch_model
    rv = D.rv_d.asfreq("D")
    r = D.r_d.reindex(rv.index)
    lrv = np.log(rv.clip(lower=1e-8))
    dv = (D.dvol.resample("D").last().reindex(rv.index) / 100) ** 2 / 365 if D.dvol is not None else None
    start = rv.index[0] + pd.Timedelta(days=400)
    days = rv.index[rv.index >= start]
    out = {k: [] for k in ("HAR (log)", "HAR (log) + weekday", "HAR (levels)", "EWMA", "GARCH(1,1)", "GJR", "EGARCH", "DVOL", "GBM", "naive")}
    fits = {}
    from sklearn.ensemble import HistGradientBoostingRegressor
    for i, d in enumerate(days):
        hist = rv[rv.index < d].dropna().iloc[-730:]
        if len(hist) < 300:
            continue
        lh = np.log(hist)
        F = pd.DataFrame({"d": lh, "w": lh.rolling(7).mean(), "m": lh.rolling(30).mean()})
        # HAR (log) and HAR (levels): refit every day (cheap)
        dd = pd.concat([lh.shift(-1).rename("y"), F], axis=1).dropna()
        A = np.c_[np.ones(len(dd)), dd[["d", "w", "m"]].values]
        b = np.linalg.lstsq(A, dd["y"].values, rcond=None)[0]
        e = dd["y"].values - A @ b
        x = np.r_[1, F.iloc[-1].values]
        out["HAR (log)"].append(math.exp(x @ b) * np.mean(np.exp(e)))
        # the same with weekday dummies for the target day (BTC weekends are ~0.4x as volatile)
        tgt_dow = (dd.index + pd.Timedelta(days=1)).dayofweek
        Dm = np.stack([(tgt_dow == k).astype(float) for k in range(1, 7)], axis=1)
        b3 = np.linalg.lstsq(np.c_[A, Dm], dd["y"].values, rcond=None)[0]
        e3 = dd["y"].values - np.c_[A, Dm] @ b3
        dm_x = np.array([float(d.dayofweek == k) for k in range(1, 7)])
        out["HAR (log) + weekday"].append(math.exp(np.r_[x, dm_x] @ b3) * np.mean(np.exp(e3)))
        L = pd.DataFrame({"d": hist, "w": hist.rolling(7).mean(), "m": hist.rolling(30).mean()})
        dl = pd.concat([hist.shift(-1).rename("y"), L], axis=1).dropna()
        A2 = np.c_[np.ones(len(dl)), dl[["d", "w", "m"]].values]
        b2 = np.linalg.lstsq(A2, dl["y"].values, rcond=None)[0]
        out["HAR (levels)"].append(max(np.r_[1, L.iloc[-1].values] @ b2, 0.2 * hist.iloc[-30:].mean()))
        out["EWMA"].append(float(hist.ewm(alpha=0.06).mean().iloc[-1]))
        out["naive"].append(float(hist.iloc[-1]))
        out["DVOL"].append(float(dv.get(d - pd.Timedelta(days=1), np.nan)) if dv is not None else np.nan)
        rr = r[r.index < d].dropna().iloc[-1000:] * 100
        if i % 7 == 0 or not fits:      # GARCH family and GBM refitted weekly
            for name, kw in (("GARCH(1,1)", {"vol": "GARCH", "p": 1, "q": 1}), ("GJR", {"vol": "GARCH", "p": 1, "o": 1, "q": 1}),
                             ("EGARCH", {"vol": "EGARCH", "p": 1, "o": 1, "q": 1})):
                try:
                    fits[name] = arch_model(rr, mean="Constant", dist="t", **kw).fit(disp="off")
                except Exception:
                    fits[name] = None
            G = pd.DataFrame({"d": lh, "w": lh.rolling(7).mean(), "m": lh.rolling(30).mean(), "dow": lh.index.dayofweek,
                              "ab": np.log(np.abs(r.reindex(lh.index)) + 1e-4)})
            dg = pd.concat([lh.shift(-1).rename("y"), G], axis=1).dropna()
            gb = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, max_leaf_nodes=8, min_samples_leaf=20,
                                               random_state=0).fit(dg.drop(columns="y").values, dg["y"].values)
            fits["GBM"] = (gb, float(np.mean(np.exp(dg["y"].values - gb.predict(dg.drop(columns="y").values)))))
        for name in ("GARCH(1,1)", "GJR", "EGARCH"):
            out[name].append(_garch_next(fits.get(name), name, rr))
        gb, sm = fits["GBM"]
        xg = np.array([[lh.iloc[-1], lh.iloc[-7:].mean(), lh.iloc[-30:].mean(), (d - pd.Timedelta(days=1)).dayofweek,
                        math.log(abs(r.get(d - pd.Timedelta(days=1), 0) or 0) + 1e-4)]])
        out["GBM"].append(math.exp(gb.predict(xg)[0]) * sm)
    idx = days[-len(out["HAR (log)"]):]
    fc = pd.DataFrame({k: pd.Series(v, index=idx[-len(v):]) for k, v in out.items() if len(v)})
    y = rv.reindex(fc.index)
    q = fc.apply(lambda f: y / f - np.log(y / f) - 1)
    q = q.dropna(how="all")
    rows = []
    for c in q.columns:
        d_ = (q[c] - q["HAR (log)"]).dropna()
        st = nw_mean(d_, 5)
        rows.append({"model": c, "days": int(q[c].notna().sum()), "QLIKE": q[c].mean(),
                     "MSE log": ((np.log(y) - np.log(fc[c])) ** 2).mean(), "QLIKE − HAR(log)": st["mean"], "t": st["t"]})
    res.details["1-day-ahead out-of-sample"] = pd.DataFrame(rows).sort_values("QLIKE")
    res.obs = (q["GARCH(1,1)"] - q["HAR (log)"]).dropna()
    return res


def structural_breaks(D: Data, max_breaks: int = 6, min_weeks: int = 12) -> pd.DataFrame:
    """Optimal partition (dynamic programming, BIC) of weekly mean log RV into segments with their own mean and
    variance — the Bai–Perron idea, used descriptively to date the volatility regimes."""
    w = np.log(D.rv_d.clip(lower=1e-7)).resample("W-SUN").mean().dropna()
    x = w.values
    n = len(x)
    cs, cs2 = np.r_[0, np.cumsum(x)], np.r_[0, np.cumsum(x ** 2)]

    def cost(i, j):   # -2 loglik of segment x[i:j] with its own mean and variance
        m = j - i
        v = max((cs2[j] - cs2[i]) / m - ((cs[j] - cs[i]) / m) ** 2, 1e-6)
        return m * math.log(v)
    best = {}
    INF = float("inf")
    C = np.full((max_breaks + 1, n + 1), INF)
    B = np.zeros((max_breaks + 1, n + 1), int)
    for j in range(min_weeks, n + 1):
        C[0, j] = cost(0, j)
    for k in range(1, max_breaks + 1):
        for j in range((k + 1) * min_weeks, n + 1):
            vals = [(C[k - 1, i] + cost(i, j), i) for i in range(k * min_weeks, j - min_weeks + 1)]
            if vals:
                C[k, j], B[k, j] = min(vals)
    bic = {k: C[k, n] + (2 * (k + 1) + k) * math.log(n) for k in range(max_breaks + 1) if np.isfinite(C[k, n])}
    k = min(bic, key=bic.get)
    cuts, j = [], n
    for kk in range(k, 0, -1):
        i = B[kk, j]
        cuts.append(i)
        j = i
    cuts = sorted(cuts)
    edges = [0] + cuts + [n]
    rows = []
    for a, b in zip(edges[:-1], edges[1:]):
        seg = x[a:b]
        rows.append({"from": f"{w.index[a]:%Y-%m-%d}", "to": f"{w.index[b - 1]:%Y-%m-%d}", "weeks": b - a,
                     "ann. vol %": 100 * math.sqrt(math.exp(seg.mean()) * 365)})
    out = pd.DataFrame(rows)
    out.attrs["bic"] = bic
    return out


ALL = {"R1": r01_seasonality, "R2": r02_sessions, "R3": r03_intraday_momentum, "R4": r04_tsmom, "R5": r05_weekday,
       "R6": r06_macro, "R7": r07_vol_managed, "R8": r08_vrp, "R9": r09_ladder_direction, "R10": r10_ladder_tails,
       "R11": r11_funding, "R12": r12_dvol_dynamics, "R13": r13_top_of_hour, "R14": r14_cme_gap,
       "R15": r15_round_numbers, "R16": r16_turn_of_month, "R17": r17_semivariance, "R18": r18_models_by_regime,
       "R19": r19_regime_har, "R20": r20_ml_direction, "R21": r21_vol_zoo}


# =============================================================================================== report
def fmt(x, d=2):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "—"
    if isinstance(x, (int, np.integer)):
        return f"{x:,}"
    if isinstance(x, (float, np.floating)):
        return f"{x:.{d}f}" if abs(x) >= 10 ** -d or x == 0 else f"{x:.2e}"
    return str(x)


def md_table(df: pd.DataFrame, d=3) -> str:
    if df is None or len(df) == 0:
        return "_(no data)_\n"
    df = df.reset_index() if not isinstance(df.index, pd.RangeIndex) else df
    head = "| " + " | ".join(map(str, df.columns)) + " |\n|" + "---|" * len(df.columns) + "\n"
    body = "".join("| " + " | ".join(fmt(v, d) for v in row) + " |\n" for row in df.itertuples(index=False))
    return head + body


def figures(results: list[Result], reg: pd.DataFrame, D: Data):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    FIG.mkdir(parents=True, exist_ok=True)
    out = {}
    fig, ax = plt.subplots(2, 1, figsize=(11, 5.5), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    ax[0].plot(np.exp(D.close_d), lw=0.8, color="k")
    ax[0].set_yscale("log")
    ax[0].set_ylabel("BTC, $")
    hi = reg["ms"] == "high-vol"
    for a in ax:
        a.fill_between(reg.index, 0, 1, where=hi, transform=a.get_xaxis_transform(), color="tab:red", alpha=0.15, lw=0)
        a.axvline(ETF_LAUNCH, color="tab:blue", ls="--", lw=1)
    ax[1].plot(np.sqrt(D.rv_d.rolling(7).mean() * 365) * 100, lw=0.8)
    ax[1].set_ylabel("RV 7d, ann. %")
    ax[0].set_title("Regimes: red = Markov high-volatility state (real-time filtered), dashed = spot-ETF launch")
    fig.tight_layout()
    fig.savefig(FIG / "regimes.png", dpi=130)
    plt.close(fig)
    out["regimes"] = "figures/regimes.png"
    for r in results:
        try:
            if r.id == "R1" and "_profile" in r.details:
                p = r.details["_profile"].reindex(range(168))
                fig, ax = plt.subplots(figsize=(8, 3.2))
                im = ax.imshow(p.values.reshape(7, 24), aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1)
                ax.set_yticks(range(7), ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
                ax.set_xlabel("hour, UTC")
                ax.set_title("Hourly log-variance relative to the trailing week")
                fig.colorbar(im)
                fig.tight_layout(); fig.savefig(FIG / "R1_seasonality.png", dpi=130); plt.close(fig)
                out["R1"] = "figures/R1_seasonality.png"
            if r.id in ("R2", "R7") and "_cum" in r.details:
                fig, ax = plt.subplots(figsize=(8, 3.2))
                (r.details["_cum"] * 100).plot(ax=ax, lw=1)
                ax.axvline(ETF_LAUNCH, color="grey", ls="--", lw=1)
                ax.set_ylabel("cumulative log return, %")
                ax.set_title(r.title)
                fig.tight_layout(); fig.savefig(FIG / f"{r.id}.png", dpi=130); plt.close(fig)
                out[r.id] = f"figures/{r.id}.png"
            if r.id == "R13" and "_minute_profile" in r.details:
                fig, ax = plt.subplots(figsize=(8, 3))
                r.details["_minute_profile"].plot.bar(ax=ax, width=0.8, color="tab:gray")
                ax.set_xticks(range(0, 60, 5), range(0, 60, 5))
                ax.set_ylabel("variance / median minute")
                ax.set_title("Variance of 1-minute returns by minute of the hour")
                fig.tight_layout(); fig.savefig(FIG / "R13.png", dpi=130); plt.close(fig)
                out["R13"] = "figures/R13.png"
            if r.id == "R14" and "_gaps" in r.details:
                g = r.details["_gaps"]
                bins = pd.cut(g["gap bp"].abs(), [5, 25, 50, 100, 200, 5000])
                t = g.groupby(bins).agg(obs=("filled", "mean"), null=("p_null", "mean"), n=("filled", "size"))
                fig, ax = plt.subplots(figsize=(7, 3))
                x = np.arange(len(t))
                ax.bar(x - 0.2, t["obs"] * 100, 0.4, label="observed")
                ax.bar(x + 0.2, t["null"] * 100, 0.4, label="random walk")
                ax.set_xticks(x, [f"{i}\n(n={n})" for i, n in zip(t.index.astype(str), t["n"])], fontsize=7)
                ax.set_ylabel("% filled within a week")
                ax.set_xlabel("|gap|, bp")
                ax.legend(); ax.set_title("CME weekend gaps: fill rate vs the random-walk null")
                fig.tight_layout(); fig.savefig(FIG / "R14.png", dpi=130); plt.close(fig)
                out["R14"] = "figures/R14.png"
            if r.id == "R8a" and "_series" in r.details:
                fig, ax = plt.subplots(figsize=(8, 3))
                (np.sqrt(r.details["_series"]) * 100).plot(ax=ax, lw=0.8)
                ax.set_ylabel("ann. vol, %")
                ax.set_title("Implied (DVOL) vs subsequently realised 30-day volatility")
                fig.tight_layout(); fig.savefig(FIG / "R8a.png", dpi=130); plt.close(fig)
                out["R8a"] = "figures/R8a.png"
        except Exception:
            traceback.print_exc()
    return out


def write_report(results: list[Result], reg: pd.DataFrame, info: dict, D: Data, figs: dict, elapsed: float,
                 breaks: pd.DataFrame | None = None):
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for r in results:
        p = r.primary or {}
        post = r.obs[pd.DatetimeIndex(r.obs.index) >= pd.Timestamp(RESEARCH_FROZEN, tz="UTC")] if r.obs is not None else pd.Series(dtype=float)
        rows.append({"id": r.id, "hypothesis": r.title, "effect": p.get("mean", np.nan) * r.scale,
                     "post-freeze effect": post.mean() * r.scale if len(post) else np.nan, "post-freeze n": len(post),
                     "unit": r.unit, "t": p.get("t", np.nan), "p": p.get("p", np.nan), "n": p.get("n", 0),
                     "regime-stable sign": stable_sign(r.regimes, p.get("mean", np.nan)) if r.regimes is not None else "n/a"})
    summ = pd.DataFrame(rows).set_index("id")
    summ["Holm p"] = holm(summ["p"]).reindex(summ.index)
    summ["BH q"] = bh(summ["p"]).reindex(summ.index)

    def verdict(row):
        if not np.isfinite(row["p"]):
            return "no data"
        if row["Holm p"] < 0.05:
            return "CONFIRMED (survives Holm)"
        if row["BH q"] < 0.10:
            return "likely (FDR 10%)"
        if row["p"] < 0.05:
            return "weak (p<0.05 only)"
        return "not supported"
    summ["verdict"] = summ.apply(verdict, axis=1)
    summ.to_csv(OUT / "summary.csv")
    # strategies, deflated Sharpe over every rule tried
    strat = {}
    for r in results:
        strat.update(r.strategies)
    st = {k: sharpe_stats(v[0]) for k, v in strat.items()}
    st_g = {k: sharpe_stats(v[1]) for k, v in strat.items()}
    trials = [v["sr"] for k, v in st.items() if np.isfinite(v["sr"]) and not k.startswith("benchmark")]
    srows = []
    for k, v in st.items():
        psr, dsr = deflated_sharpe(v["sr"], v["n"], v["skew"], v["kurt"], trials)
        s = strat[k][0]
        post = s[s.index >= pd.Timestamp(RESEARCH_FROZEN, tz="UTC")]
        srows.append({"strategy": k, "Sharpe gross": st_g[k]["sr_ann"], "Sharpe net": v["sr_ann"], "ann. return net": v.get("ann_ret"),
                      "ann. vol": v.get("ann_vol"), "days": v["n"], "PSR(>0)": psr, "DSR": dsr,
                      "post-freeze days": len(post), "post-freeze return": post.sum() if len(post) else np.nan})
    strat_tab = pd.DataFrame(srows)
    if len(strat_tab):
        strat_tab = strat_tab.sort_values("Sharpe net", ascending=False)
    strat_tab.to_csv(OUT / "strategies.csv", index=False)
    for r in results:
        if r.obs is not None:
            r.obs.rename("obs").to_frame().to_csv(OUT / f"{r.id}_obs.csv.gz")
        if r.regimes is not None and len(r.regimes):
            r.regimes.to_csv(OUT / f"{r.id}_regimes.csv", index=False)
        for name, tab in r.details.items():
            if not name.startswith("_") and isinstance(tab, pd.DataFrame):
                safe = "".join(ch if ch.isalnum() else "_" for ch in name)[:50]
                tab.to_csv(OUT / f"{r.id}_{safe}.csv")
    reg.to_csv(OUT / "regimes_daily.csv")

    L = []
    L.append(f"# Quant research: regimes and BTC inefficiencies\n")
    L.append(f"_Generated {pd.Timestamp.now():%Y-%m-%d %H:%M} in {elapsed / 60:.1f} min. Data: Binance 1-minute "
             f"{D.px.index[0]:%Y-%m-%d} … {D.end:%Y-%m-%d %H:%M} UTC; hypotheses frozen {RESEARCH_FROZEN}. "
             f"Costs {COST_BPS:.0f} bp per side._\n")
    L.append("## 1. Verdicts\n")
    L.append("Primary statistic = mean of the per-observation series, Newey–West t; Holm controls the family-wise "
             "error over all hypotheses, BH the false-discovery rate. 'Regime-stable' = same sign in every Markov, "
             "trend and ETF-era state with ≥20 observations.\n")
    show = summ[["hypothesis", "effect", "unit", "t", "p", "Holm p", "BH q", "n", "regime-stable sign", "verdict",
                 "post-freeze effect", "post-freeze n"]]
    L.append(md_table(show, 3))
    L.append("\n## 2. Strategies (all rules tried, net of costs)\n")
    L.append(f"DSR = probability that the Sharpe ratio is above the best one expected from {len(trials)} unrelated "
             "rules with zero true Sharpe (Bailey & López de Prado 2014). DSR > 0.95 is the bar.\n")
    L.append(md_table(strat_tab, 3))
    L.append("\n## 3. Regimes\n")
    if "regimes" in figs:
        L.append(f"![regimes](../results/research/{figs['regimes']})\n")
    if info:
        L.append(md_table(pd.DataFrame([info]), 3))
    if breaks is not None and len(breaks):
        L.append("\nVolatility regimes dated by an optimal (BIC) segmentation of weekly log RV:\n\n" + md_table(breaks, 1))
        breaks.to_csv(OUT / "structural_breaks.csv", index=False)
    cnt = reg[["ms", "trend", "era", "vol_tercile"]].apply(lambda c: c.value_counts()).T
    L.append("\nDays per state:\n\n" + md_table(cnt.fillna(0).astype(int), 0))
    L.append("\n## 4. Hypotheses in detail\n")
    for r in results:
        L.append(f"### {r.id}. {r.title}\n")
        L.append(f"*Source:* {r.source}  \n*Prediction (fixed in advance):* {r.claim}\n")
        p = r.primary or {}
        if p:
            L.append(f"**Primary:** effect = {fmt(p.get('mean', np.nan) * r.scale, 3)} {r.unit}, t = {fmt(p.get('t'), 2)}, "
                     f"p = {fmt(p.get('p'), 4)}, n = {p.get('n', 0)} → **{summ.loc[r.id, 'verdict']}**\n")
        if r.id in figs:
            L.append(f"![{r.id}](../results/research/{figs[r.id]})\n")
        for name, tab in r.details.items():
            if name.startswith("_") or not isinstance(tab, pd.DataFrame):
                continue
            L.append(f"*{name}*\n\n" + md_table(tab, 3))
        if r.regimes is not None and len(r.regimes):
            t = r.regimes.copy()
            t["mean"] = t["mean"] * r.scale
            L.append(f"*by regime* (effect in {r.unit})\n\n" + md_table(t, 3))
        for n in r.notes:
            L.append(f"> {n}\n")
        L.append("")
    (config.REPORTS / "research.md").write_text("\n".join(L), encoding="utf-8")
    # compact JSON for slides
    js = {"generated": f"{pd.Timestamp.now():%Y-%m-%d %H:%M}", "summary": json.loads(summ.reset_index().to_json(orient="records")),
          "strategies": json.loads(strat_tab.to_json(orient="records")), "markov": info}
    (OUT / "summary.json").write_text(json.dumps(js, indent=1), encoding="utf-8")
    return summ, strat_tab


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--no-download", action="store_true", help="do not fetch funding rates")
    a = ap.parse_args(argv)
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    config.REPORTS.mkdir(parents=True, exist_ok=True)
    try:   # funding history (R11) is small: fetch or extend it here so that the nightly run always has it
        if a.no_download:
            raise RuntimeError("skipped (--no-download)")
        from src.download import funding
        funding.download()
    except Exception as e:
        log(f"research: funding download failed ({str(e)[:120]}); R11 uses what is on disk")
    D = Data()
    log("research: regimes ...")
    reg, info = build_regimes(D, a.quick)
    results = []
    for key, fn in ALL.items():
        if a.only and key not in a.only:
            continue
        t1 = time.time()
        try:
            out = fn(D, reg)
            for r in (out if isinstance(out, list) else [out]):
                if r is not None:
                    results.append(r.finish(reg))
            log(f"research: {key} done ({time.time() - t1:.0f}s)")
        except Exception:
            log(f"!!! research {key} failed")
            traceback.print_exc()
    try:
        breaks = structural_breaks(D)
    except Exception:
        traceback.print_exc()
        breaks = None
    figs = figures(results, reg, D)
    summ, strat = write_report(results, reg, info, D, figs, time.time() - t0, breaks)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(summ[["effect", "unit", "t", "p", "Holm p", "verdict"]].to_string())
        print(strat.head(12).to_string() if len(strat) else "")
    log(f"research: reports/research.md written ({(time.time() - t0) / 60:.1f} min)")


if __name__ == "__main__":
    main()
