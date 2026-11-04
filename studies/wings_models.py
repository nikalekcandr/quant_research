"""Model zoo for the wing fair value (04.11.2026): which variance level × distribution shape prices the cheap side
of the Kalshi hourly ladder best, does any of them add information to the book, and does a better model trade
better? Plus a day-block bootstrap of the frozen W03+W04 book.

Variance (per minute, × hour-of-week profile unless noted; all known before the decision minute):
  ewma60 (frozen rules) · ewma30 · ewma240 · day (24 h mean) · blend (0.5 ewma30 + 0.3 day + 0.2 week, = live lab)
  · har (log-linear fit of the next 30 min on 1 h / 1 d / 1 w, fitted on minutes before July) · dvol (Deribit DVOL²)
Shapes: normal · t3 · t4 · t6 (unit variance) · empirical (z of the Dec–Jun settlements, no look-ahead).
A. Probability accuracy on cheap sides (3–20¢, 5–30 min left, first minute per strike): Brier / log loss vs the
   book mid; encompassing logit(won) ~ logit(mid) + logit(model) fitted Jul–Aug, scored Sep–Oct.
B. Trading: per hour sell, at the first minute with an edge (bid − fair − fee) ≥ margin, the largest-edge cheap side in the W03 and W04
   windows; margin chosen on Jul–Aug; Sep–Oct test.
C. Bootstrap (blocks of days) of the frozen W03+W04 book: CI of $/hour, P(loss) over 30 days.
    python studies/wings_models.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

sys.argv = sys.argv[:1] + ["data/raw/usd_minutes"]
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from studies import brti_proxy as bp  # noqa: E402
from studies import wings_ladder as wl  # noqa: E402

wl.LAD, wl.MIN = Path("data/raw/kalshi_kxbtcd_60m"), Path("data/interim/minutes.parquet")
OUT = Path("results/studies")
SPLIT, JUL = wl.SPLIT, pd.Timestamp("2026-07-01", tz="UTC")


def levels(S: pd.Series) -> pd.DataFrame:
    """per-minute deseasonalised variance levels, each known at the minute (uses r² up to and incl. the minute)."""
    vm = wl.variance_level(S)
    r2 = np.log(S).diff() ** 2
    z = (r2 / vm["sea"]).clip(upper=(r2 / vm["sea"]).quantile(0.9995))
    L = pd.DataFrame(index=S.index)
    L["sea"] = vm["sea"]
    L["ewma60"] = vm["lvl"]
    L["ewma30"] = z.ewm(halflife=30, min_periods=60).mean()
    L["ewma240"] = z.ewm(halflife=240, min_periods=480).mean()
    L["day"] = z.rolling(1440, min_periods=600).mean()
    L["week"] = z.rolling(7 * 1440, min_periods=3000).mean()
    L["blend"] = 0.5 * L.ewma30 + 0.3 * L.day + 0.2 * L.week
    # HAR on log levels, target = mean deseasonalised r² of the next 30 minutes, fitted before July
    h1 = z.rolling(60, min_periods=60).mean()
    fut = z[::-1].rolling(30, min_periods=30).mean()[::-1].shift(-1)
    X = pd.DataFrame({"h": np.log(h1 + 1e-12), "d": np.log(L.day + 1e-12), "w": np.log(L.week + 1e-12)})
    y = np.log(fut + 1e-12)
    tr = (X.index < JUL) & X.notna().all(axis=1) & y.notna() & np.isfinite(y)
    tr = tr & (X.index > JUL - pd.Timedelta(days=150))
    m = sm.OLS(y[tr], sm.add_constant(X[tr])).fit()
    resid_var = m.resid.var()
    L["har"] = np.exp(m.predict(sm.add_constant(X)) + resid_var / 2)       # lognormal mean correction
    dv = pd.read_csv("data/raw/deribit/dvol.csv.gz")
    dv = dv.set_index(pd.to_datetime(dv.ts, utc=True) + pd.Timedelta(hours=1))["close"]  # known at the hour's end
    dv = dv[~dv.index.duplicated()].sort_index().reindex(S.index, method="ffill")
    L["dvol"] = (dv / 100) ** 2 / (365 * 1440)                                # per minute, before the profile
    return L, m.params


SHAPES = ("normal", "t3", "t4", "t6", "empirical")


def cdf(x, shape, emp=None):
    if shape == "normal":
        return stats.norm.cdf(x)
    if shape == "empirical":
        return np.interp(x, emp[0], emp[1], left=0.0, right=1.0)
    nu = int(shape[1:])
    return stats.t.cdf(x * math.sqrt(nu / (nu - 2)), nu)


def empirical_z(E, S, L):
    """z = log(E/(S·b)) / sd_ewma60 of settlements Dec–Jun, τ = 5..30 min; symmetrised CDF grid."""
    b = bp.lagged_basis(E, S)
    zs = []
    for tau in range(5, 31, 5):
        t0 = E.index - pd.Timedelta(minutes=tau)
        var = L["ewma60"].reindex(t0).values * L["sea"].reindex(t0).values * (tau - 2 / 3)
        z = np.log(E.values / (S.reindex(t0).values * b.values)) / np.sqrt(var)
        zs.append(pd.Series(z, index=E.index))
    z = pd.concat(zs)
    z = z[(z.index < JUL) & np.isfinite(z)]
    z = np.concatenate([z.values, -z.values])
    g = np.linspace(-8, 8, 801)
    return (g, np.searchsorted(np.sort(z), g) / len(z)), len(z) // 2


def main():
    E, S = bp.settlements(), bp.binance()
    L, har = levels(S)
    emp, n_emp = empirical_z(E, S, L)
    c = bp.wings_frame(S, E)
    x0 = np.log(c.S * c.basis / c.strike)
    tleft = np.maximum(c.left - 2 / 3, 1 / 3)
    sea = L["sea"].reindex(c.ts).values
    lines = ["# Model zoo for the wing fair value (studies/wings_models.py)", "",
             f"Cheap sides of the Kalshi hourly ladder Jul–Oct 2026; HAR fit before July: {har.round(3).to_dict()}; "
             f"empirical z from {n_emp} settlement × τ points Dec–Jun.", ""]
    models = {}
    for v in ("ewma60", "ewma30", "ewma240", "day", "blend", "har", "dvol"):
        lv = L[v].reindex(c.ts).values
        var = lv * (sea if v != "dvol" else 1.0) * tleft
        for sh in SHAPES:
            if v not in ("ewma60", "blend", "har", "dvol") and sh in ("t3", "t6"):
                continue
            p = cdf(x0 / np.sqrt(var), sh, emp)
            models[f"{v}/{sh}"] = np.where(c.cheap == "yes", p, 1 - p)
    F = pd.DataFrame(models, index=c.index)

    # ---------------------------------------------------------------- A. accuracy
    m = c.c_bid.between(0.03, 0.20) & c.left.between(5, 30)
    u = c[m].sort_values("ts").drop_duplicates(["T", "ticker"]).copy()
    mid = ((u.c_bid + np.where(u.cheap == "yes", u.yes_ask, 1 - u.yes_bid)) / 2).clip(0.005, 0.995)
    u["mid"] = mid
    y = u.c_won.values
    clip = lambda p: np.clip(p, 0.002, 0.998)
    ll = lambda p, yy=None: -((y if yy is None else yy) * np.log(clip(p)) + (1 - (y if yy is None else yy)) * np.log(1 - clip(p)))
    rows = []
    b_mkt, l_mkt = (u.mid - y) ** 2, ll(u.mid.values)
    te = (u["T"] >= SPLIT).values
    rows.append({"forecast": "book mid", "Brier×100": 100 * b_mkt.mean(), "log loss": l_mkt.mean(),
                 "Brier vs mid ×100": 0.0, "t": np.nan, "Sep–Oct log loss": l_mkt[te].mean(),
                 "encompassing β model (t)": "—", "Sep–Oct Δ log loss vs mid": 0.0})
    lm = np.log(clip(u.mid.values) / (1 - clip(u.mid.values)))
    tr = ~te
    base = sm.Logit(y[tr], sm.add_constant(lm[tr])).fit(disp=0)
    l_base = ll(base.predict(sm.add_constant(lm[te])), y[te])
    for k in F.columns:
        p = F.loc[u.index, k].values
        d = (p - y) ** 2 - b_mkt.values
        mm, t, _ = ct(pd.Series(d), u["T"].reset_index(drop=True))
        lp = np.log(clip(p) / (1 - clip(p)))
        X = sm.add_constant(np.column_stack([lm, lp]))
        try:
            fit = sm.Logit(y[tr], X[tr]).fit(disp=0, cov_type="cluster", cov_kwds={"groups": pd.factorize(u["T"])[0][tr]})
            enc = f"{fit.params[2]:.2f} ({fit.tvalues[2]:.1f})"
            dl = (ll(fit.predict(X[te]), y[te]) - l_base).mean()
        except Exception:  # noqa: BLE001
            enc, dl = "—", np.nan
        rows.append({"forecast": k, "Brier×100": 100 * ((p - y) ** 2).mean(), "log loss": ll(p).mean(),
                     "Brier vs mid ×100": 100 * mm, "t": t, "Sep–Oct log loss": ll(p)[te].mean(),
                     "encompassing β model (t)": enc, "Sep–Oct Δ log loss vs mid": dl})
    A = pd.DataFrame(rows)
    A.to_csv(OUT / "wings_models_A.csv", index=False)
    lines += [f"## A. Probability of the cheap side ({len(u)} strikes × hours, 3–20¢, 5–30 min left)",
              "Brier vs mid < 0 = better than the book. Encompassing: logit(won) on logit(mid) + logit(model), fit "
              "Jul–Aug (clustered by hour), Δ log loss on Sep–Oct vs mid-only (< 0 = the model adds information).", "",
              A.round(4).to_markdown(index=False), ""]

    # ---------------------------------------------------------------- B. trading
    rows = []
    for rule, (lo, hi, l0, l1) in {"W03 window": (0.10, 0.20, 5, 15), "W04 window": (0.06, 0.10, 15, 30)}.items():
        w = c.c_bid.between(lo, hi) & c.left.between(l0, l1)
        g0 = c[w]
        for k in F.columns:
            edge = g0.c_bid - F.loc[g0.index, k] - wl.fee(g0.px)
            best = None
            res = {}
            for mg in (0.0, 0.005, 0.01, 0.02, 0.03):
                g = g0[edge >= mg].assign(edge=edge[edge >= mg])
                # no look-ahead: the first minute of the hour with an edge ≥ margin, the largest edge in that minute
                g = g.sort_values(["T", "ts", "edge"], ascending=[True, True, False]).drop_duplicates("T")
                for half, gg in (("tr", g[g["T"] < SPLIT]), ("te", g[g["T"] >= SPLIT])):
                    res[(mg, half)] = gg
                trn = res[(mg, "tr")]
                t_tr = trn.pnl.mean() / (trn.pnl.std(ddof=1) / math.sqrt(len(trn))) if len(trn) > 30 else -np.inf
                if best is None or t_tr > best[1]:
                    best = (mg, t_tr)
            mg = best[0]
            trn, tst = res[(mg, "tr")], res[(mg, "te")]
            rows.append({"window": rule, "model": k, "margin chosen ¢": 100 * mg,
                         "Jul–Aug n": len(trn), "Jul–Aug ¢": 100 * trn.pnl.mean(), "Jul–Aug t": best[1],
                         "Sep–Oct n": len(tst), "Sep–Oct ¢": 100 * tst.pnl.mean(),
                         "Sep–Oct t": tst.pnl.mean() / (tst.pnl.std(ddof=1) / math.sqrt(len(tst))) if len(tst) > 2 else np.nan})
    B = pd.DataFrame(rows)
    B.to_csv(OUT / "wings_models_B.csv", index=False)
    lines += ["## B. Trading: one cheap side per hour, the first minute with edge ≥ margin (margin chosen on Jul–Aug)",
              "Frozen W03 = ewma60/normal margin 1¢; W04 = ewma60/t4 margin 0.", "",
              B.round(2).sort_values(["window", "Sep–Oct t"], ascending=[True, False]).to_markdown(index=False), ""]

    # ---------------------------------------------------------------- C. bootstrap of the frozen book
    book = pd.concat([wl.run(c, *p) for p in RULES.values()])
    te_b = book[book["T"] >= SPLIT]
    daily = te_b.groupby(te_b["T"].dt.floor("D")).pnl.sum()
    hours = te_b.groupby("T").pnl.sum()
    rng = np.random.default_rng(7)
    days = daily.values
    boot = np.array([rng.choice(days, len(days)).mean() for _ in range(20000)])
    m30 = np.array([rng.choice(days, 30).sum() for _ in range(20000)])
    # block of 3 days
    nb = len(days) // 3
    blocks = [days[i:i + 3] for i in range(0, len(days) - 2)]
    bb = np.array([np.concatenate([blocks[j] for j in rng.integers(0, len(blocks), nb)]).mean() for _ in range(20000)])
    lines += ["## C. Frozen W03+W04 book, Sep–Oct, 1 contract per slot: bootstrap", "",
              f"* {len(days)} days, {len(hours)} active hours, mean ${daily.mean():.3f}/day, worst day ${daily.min():.2f}, "
              f"best ${daily.max():.2f}, {100 * (daily < 0).mean():.0f}% losing days.",
              f"* Mean per day, iid days: 95% CI [{np.percentile(boot, 2.5):.3f}; {np.percentile(boot, 97.5):.3f}], "
              f"P(mean ≤ 0) = {100 * (boot <= 0).mean():.2f}%.",
              f"* 3-day blocks: 95% CI [{np.percentile(bb, 2.5):.3f}; {np.percentile(bb, 97.5):.3f}], "
              f"P(mean ≤ 0) = {100 * (bb <= 0).mean():.2f}%.",
              f"* 30 days: P(loss) = {100 * (m30 < 0).mean():.1f}%, 5% quantile ${np.percentile(m30, 5):.2f}, "
              f"median ${np.median(m30):.2f}.", ""]
    (OUT / "wings_models.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


RULES = {"W03": (0.10, 0.20, 5, 15, "normal", 0.01, 1), "W04": (0.06, 0.10, 15, 30, "t4", 0.0, 1)}


def ct(x: pd.Series, cl: pd.Series):
    d = pd.DataFrame({"x": np.asarray(x), "c": np.asarray(cl)}).dropna()
    g = d.groupby("c").x.agg(["sum", "size"])
    mu = d.x.mean()
    se = math.sqrt(((g["sum"] - mu * g["size"]) ** 2).sum()) / len(d)
    return mu, mu / se if se > 0 else np.nan, len(d)


if __name__ == "__main__":
    main()
