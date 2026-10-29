"""Extensions on top of the main evaluation. Every function reads the forecasts and panels written by the
earlier steps, so this step can be rerun on its own: python run_all.py --from extensions

  1. model_confidence_set   Hansen, Lunde & Nason (2011): the set of models that cannot be beaten, with the
                            multiple-testing problem handled by a block bootstrap of all loss series at once.
  2. optimal_pool           Geweke & Amisano (2011): a mixture of the HAR-X, Deribit and ladder densities whose
                            weights maximise the past log score. Weights are re-estimated in real time from
                            forecasts whose outcome was already known, so the pool is a genuine forecast.
  3. encompassing           regression of the HAR surprise on the ladder-HAR disagreement: the slope is the
                            weight the best linear combination puts on the ladder (Fair & Shiller 1990).
  4. conditional_ability    Giacomini & White (2006): does the ladder's advantage depend on things known at t
                            (scheduled macro release, weekend, size of the ladder-HAR disagreement)?
  5. fluctuation            Giacomini & Rossi (2010): the DM statistic on a rolling window with critical values
                            that account for looking at every window.
  6. sell_wings             the variance premium in the tails as a trade: sell cheap contracts on both wings.

Outputs: results/tables/{mcs, pool_scores, pool_dm, pool_weights_summary, encompassing, conditional_ability,
calendar_split, fluctuation_summary, wings}.csv, results/pool_weights.parquet, results/fluctuation.parquet.
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

import config
from src import evaluate
from src.utils import ensure, log, read_table, write_table

HORIZONS = list(config.HORIZONS)
POOL_PARTS = ["harx", "iv_mz", "mkt_mz"]        # time series / options / ladder: the same parts as `combo`


def _forecasts(h):
    fc = read_table(config.RESULTS / f"forecasts_{h}")
    return fc.sort_values(["model", "t"]).reset_index(drop=True)


def _wide(fc: pd.DataFrame, col: str, models: list[str]) -> pd.DataFrame:
    w = fc[fc["model"].isin(models)].pivot_table(index="t", columns="model", values=col, aggfunc="first")
    return w[[m for m in models if m in w]]


# ------------------------------------------------------------------ 1. model confidence set
def _block_indices(n: int, block: int, rng) -> np.ndarray:
    """Circular moving-block bootstrap: n indices made of blocks of consecutive observations."""
    starts = rng.integers(0, n, size=int(np.ceil(n / block)))
    idx = (starts[:, None] + np.arange(block)[None, :]).ravel()[:n]
    return idx % n


def mcs(losses: pd.DataFrame, alpha: float = 0.10, block: int = 10, B: int = 2000, seed: int = 7) -> pd.DataFrame:
    """Model confidence set with the T_max statistic (Hansen, Lunde & Nason 2011, section 3.1.2).

    losses: one column per model, one row per forecast occasion (common sample).
    Returns one row per model: elimination order, MCS p-value, and whether it is in the (1 - alpha) set.
    """
    L = losses.to_numpy(float)
    names = list(losses.columns)
    n, _ = L.shape
    rng = np.random.default_rng(seed)
    boot = np.stack([L[_block_indices(n, block, rng)].mean(axis=0) for _ in range(B)])   # B x m bootstrap means
    mean = L.mean(axis=0)
    alive = list(range(len(names)))
    out, pmax = [], 0.0
    while len(alive) > 1:
        a = np.array(alive)
        d_bar = mean[a] - mean[a].mean()                       # each model's loss relative to the set average
        d_boot = boot[:, a] - boot[:, a].mean(axis=1, keepdims=True)
        var = ((d_boot - d_bar) ** 2).mean(axis=0)
        t = d_bar / np.sqrt(var)
        t_boot = ((d_boot - d_bar) / np.sqrt(var)).max(axis=1)
        p = float((t_boot >= t.max()).mean())
        pmax = max(pmax, p)
        worst = a[int(t.argmax())]
        out.append({"model": names[worst], "eliminated_step": len(out) + 1, "mcs_p": pmax,
                    "mean_loss": mean[worst], "t_vs_set": float(t.max())})
        alive.remove(worst)
    out.append({"model": names[alive[0]], "eliminated_step": None, "mcs_p": 1.0,
                "mean_loss": mean[alive[0]], "t_vs_set": np.nan})
    res = pd.DataFrame(out)
    res["in_mcs"] = res["mcs_p"] >= alpha
    return res


def model_confidence_set(sc: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for h, g in sc.groupby("horizon"):
        mods = list(g["model"].unique())
        cs = evaluate.common_sample(g, mods)
        block = max(2 * evaluate.overlap(h), int(round(len(cs["t"].unique()) ** (1 / 3))))
        for loss in ("qlike", "logscore"):
            w = cs.pivot_table(index="t", columns="model", values=loss)
            if loss == "logscore":
                w = -w
            w = w.dropna()
            if len(w) < 30:
                continue
            r = mcs(w, block=block)
            r.insert(0, "loss", loss)
            r.insert(0, "horizon", h)
            r["n"] = len(w)
            r["block"] = block
            rows.append(r)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


# ------------------------------------------------------------------ 2. optimal prediction pool
def _simplex(k: int, step: float = 0.05) -> np.ndarray:
    m = int(round(1 / step))
    pts = [c for c in itertools.product(range(m + 1), repeat=k) if sum(c) == m]
    return np.array(pts, float) / m


def optimal_pool(fc: pd.DataFrame, horizon: str, parts=POOL_PARTS) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Real-time log-score-optimal weights on the part densities. Returns forecast rows in the same format as
    every other model (model = 'pool') and the weight path."""
    m = _wide(fc, "m1", parts)
    s = _wide(fc, "s1", parts)
    if m.shape[1] < len(parts):
        return pd.DataFrame(), pd.DataFrame()
    meta = fc.drop_duplicates("t").set_index("t")[["horizon", "T", "e", "y", "rv"]]
    idx = m.dropna().index.intersection(s.dropna().index)
    m, s, meta = m.loc[idx], s.loc[idx], meta.loc[idx]
    y = meta["y"].to_numpy(float)
    M, S = m.to_numpy(float), s.to_numpy(float)
    dens = stats.norm.pdf((y[:, None] - M) / S) / S             # part densities at the realized outcome
    have = np.isfinite(y)
    e = meta["e"].dt.tz_localize(None).to_numpy("datetime64[ns]")
    t = idx.tz_localize(None).to_numpy("datetime64[ns]") if idx.tz is not None else idx.to_numpy("datetime64[ns]")
    grid = _simplex(len(parts))
    win = config.ROLL_WINDOW[horizon]
    min_n = config.MIN_TRAIN_MARKET[horizon]
    refit = config.REFIT_EVERY[horizon]
    W = np.full((len(idx), len(parts)), np.nan)
    for c0 in range(0, len(idx), refit):
        k = int(np.searchsorted(e, t[c0], side="right"))      # occasions whose window ended by t: no look-ahead
        tr = np.flatnonzero(have[:k])[-win:]
        if len(tr) < min_n:
            continue
        ll = np.log(np.maximum(dens[tr] @ grid.T, 1e-300)).sum(axis=0)
        W[c0:c0 + refit] = grid[int(ll.argmax())]
    ok = np.isfinite(W).all(axis=1)
    out = meta.reset_index().loc[ok].copy()
    out["model"] = "pool"
    for j in range(3):
        out[f"m{j+1}"] = M[ok, j]
        out[f"s{j+1}"] = S[ok, j]
        out[f"w{j+1}"] = W[ok, j]
    out["mean_log"] = sum(out[f"w{j}"] * out[f"m{j}"] for j in (1, 2, 3))
    out["level"] = sum(out[f"w{j}"] * np.exp(out[f"m{j}"] + out[f"s{j}"] ** 2 / 2) for j in (1, 2, 3))
    weights = pd.DataFrame({"t": idx[ok], "horizon": horizon, **{f"w_{p}": W[ok, j] for j, p in enumerate(parts)}})
    return out, weights


# ------------------------------------------------------------------ 3. encompassing
def encompassing(fc: pd.DataFrame, horizon: str, base: str, rival: str) -> dict:
    """y - f_base = a + lam (f_rival - f_base) + e. lam = 0: base encompasses rival; lam = 1: rival encompasses
    base; in between, lam is the weight of the rival in the best linear combination."""
    f = _wide(fc, "mean_log", [base, rival])
    yv = fc.drop_duplicates("t").set_index("t")["y"]
    d = pd.concat([f, yv], axis=1).dropna()
    d = d[d.index >= pd.Timestamp(config.EVAL_START, tz="UTC")]
    if len(d) < 30:
        return {}
    surprise = d["y"] - d[base]
    gap = d[rival] - d[base]
    X = sm.add_constant(gap.to_numpy())
    lag = evaluate.nw_lag(len(d), evaluate.overlap(horizon))
    res = sm.OLS(surprise.to_numpy(), X).fit(cov_type="HAC", cov_kwds={"maxlags": lag})
    lam, se = res.params[1], res.bse[1]
    return {"horizon": horizon, "base": base, "rival": rival, "n": len(d), "lambda": lam, "se": se,
            "t_lambda_0": lam / se, "t_lambda_1": (lam - 1) / se, "r2": res.rsquared,
            "p_lambda_0": 2 * stats.norm.sf(abs(lam / se)), "p_lambda_1": 2 * stats.norm.sf(abs((lam - 1) / se))}


def disagreement_bins(fc: pd.DataFrame, horizon: str, base="har", rival="mkt_mz", q: int = 5) -> pd.DataFrame:
    """Quintiles of the ladder-minus-HAR disagreement known at t, and how much of it was realized."""
    f = _wide(fc, "mean_log", [base, rival])
    yv = fc.drop_duplicates("t").set_index("t")["y"]
    d = pd.concat([f, yv], axis=1).dropna()
    d = d[d.index >= pd.Timestamp(config.EVAL_START, tz="UTC")]
    if len(d) < 5 * q:
        return pd.DataFrame()
    d["gap"] = d[rival] - d[base]
    d["surprise"] = d["y"] - d[base]
    d["bin"] = pd.qcut(d["gap"], q, labels=False) + 1
    g = d.groupby("bin").agg(n=("gap", "size"), gap=("gap", "mean"), surprise=("surprise", "mean"),
                             se=("surprise", lambda s: s.std() / np.sqrt(len(s)))).reset_index()
    g.insert(0, "horizon", horizon)
    return g


# ------------------------------------------------------------------ 4. conditional predictive ability
def conditional_ability(sc: pd.DataFrame, panel: pd.DataFrame, fc: pd.DataFrame, horizon: str,
                        a="har_m", b="har", loss="qlike") -> pd.DataFrame:
    """d_t = L_a - L_b regressed on instruments known at t (HAC). The joint Wald test of all coefficients is the
    Giacomini-White test of equal conditional predictive ability."""
    la = sc[(sc["horizon"] == horizon) & (sc["model"] == a)].set_index("t")[loss]
    lb = sc[(sc["horizon"] == horizon) & (sc["model"] == b)].set_index("t")[loss]
    d = (la - lb).dropna().rename("d")
    p = panel.set_index("t")
    f = _wide(fc, "mean_log", ["har", "mkt_mz"])
    if "mkt_mz" not in f:
        return pd.DataFrame()
    gap = (f["mkt_mz"] - f["har"]).rename("gap")        # calibrated ladder forecast minus HAR, known at t
    X = pd.concat([d, p[["macro", "weekend"]], gap.abs().rename("abs_gap")], axis=1).dropna()
    X = X.loc[X.index.isin(d.index)]
    if len(X) < 30:
        return pd.DataFrame()
    X["abs_gap"] = (X["abs_gap"] - X["abs_gap"].mean()) / X["abs_gap"].std()
    regs = ["macro", "weekend", "abs_gap"]
    regs = [r for r in regs if X[r].std() > 0]
    Z = sm.add_constant(X[regs])
    lag = evaluate.nw_lag(len(X), evaluate.overlap(horizon))
    res = sm.OLS(X["d"], Z).fit(cov_type="HAC", cov_kwds={"maxlags": lag})
    wald = res.wald_test(np.eye(len(res.params)), scalar=True)
    rows = [{"horizon": horizon, "comparison": f"{a} - {b}", "loss": loss, "term": k, "coef": res.params[k],
             "se": res.bse[k], "t": res.tvalues[k], "n": len(X),
             "share_with_1": float(X[k].mean()) if k in ("macro", "weekend") else np.nan} for k in res.params.index]
    rows.append({"horizon": horizon, "comparison": f"{a} - {b}", "loss": loss, "term": "GW joint test (p)",
                 "coef": float(wald.pvalue), "se": np.nan, "t": float(wald.statistic), "n": len(X), "share_with_1": np.nan})
    return pd.DataFrame(rows)


CALENDAR_PAIRS = [("har_m", "har"), ("mkt_mz", "har"), ("harx_m", "harx"), ("harx_all", "harx_d")]


def calendar_split(sc: pd.DataFrame, panel: pd.DataFrame, horizon: str, loss: str = "qlike") -> pd.DataFrame:
    """The ladder's gain separately on windows entirely on weekdays and on windows that touch a weekend.
    HAR has no calendar term, so against HAR part of the gain may be the calendar, not market information;
    against HAR-X (which has the weekend share) it cannot be."""
    wk = panel.set_index("t")["weekend"]
    groups = {"all": lambda w: np.ones(len(w), bool), "weekday only": lambda w: w == 0,
              "touches weekend": lambda w: w > 0, "weekend only": lambda w: w == 1}
    rows = []
    for a, b in CALENDAR_PAIRS:
        la = sc[(sc["horizon"] == horizon) & (sc["model"] == a)].set_index("t")[loss]
        lb = sc[(sc["horizon"] == horizon) & (sc["model"] == b)].set_index("t")[loss]
        d = (la - lb).dropna()
        if len(d) < 30:
            continue
        w = wk.reindex(d.index).to_numpy(float)
        # shares of the total gain are meaningless when there is no total gain to share
        total = d.sum() if abs(d.sum()) >= 0.05 * d.abs().sum() else 0.0
        for name, f in groups.items():
            m = f(w) & np.isfinite(w) if name != "all" else f(w)
            if m.sum() < 10:
                continue
            # windows of a group are not contiguous: the overlap lag of the horizon still bounds the dependence
            r = evaluate.dm_test(la.reindex(d.index).to_numpy()[m], lb.reindex(d.index).to_numpy()[m],
                                 evaluate.overlap(horizon))
            rows.append({"horizon": horizon, "A": a, "B": b, "loss": loss, "group": name, **r,
                         "share_of_total_gain": float(d[m].sum() / total) if total != 0 else np.nan})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ 5. fluctuation test
def fluctuation_cv(mu: float, alpha: float = 0.05, sims: int = 3000, steps: int = 2000, seed: int = 11) -> float:
    """Critical value of sup_r |B(r) - B(r - mu)| / sqrt(mu) over r in [mu, 1], simulated (two-sided)."""
    rng = np.random.default_rng(seed)
    w = int(round(mu * steps))
    sups = np.empty(sims)
    for i in range(sims):
        B = np.concatenate([[0.0], np.cumsum(rng.standard_normal(steps)) / np.sqrt(steps)])
        sups[i] = np.max(np.abs(B[w:] - B[:-w])) / np.sqrt(mu)
    return float(np.quantile(sups, 1 - alpha))


def fluctuation(sc: pd.DataFrame, horizon: str, a="har_m", b="har", loss="qlike", mu: float = 0.25):
    la = sc[(sc["horizon"] == horizon) & (sc["model"] == a)].set_index("t")[loss]
    lb = sc[(sc["horizon"] == horizon) & (sc["model"] == b)].set_index("t")[loss]
    d = (la - lb).dropna().sort_index()
    n = len(d)
    if n < 60:
        return pd.DataFrame(), {}
    w = int(round(mu * n))
    _, se, _ = evaluate.hac_mean(d.to_numpy(), evaluate.nw_lag(n, evaluate.overlap(horizon)))
    sigma = se * np.sqrt(n)                                    # long-run standard deviation of d
    roll = d.rolling(w).sum() / (np.sqrt(w) * sigma)
    cv = fluctuation_cv(mu)
    path = pd.DataFrame({"t": d.index, "horizon": horizon, "stat": roll.to_numpy(), "cv": cv}).dropna()
    summ = {"horizon": horizon, "comparison": f"{a} - {b}", "loss": loss, "n": n, "window": w, "mu": mu,
            "cv_5pct": cv, "min_stat": path["stat"].min(), "max_stat": path["stat"].max(),
            "share_windows_ladder_better": float((path["stat"] < -cv).mean()),
            "share_windows_ladder_worse": float((path["stat"] > cv).mean()),
            "reject": bool((path["stat"].abs() > cv).any())}
    return path, summ


# ------------------------------------------------------------------ 6. sell the wings
def sell_wings(le: pd.DataFrame, cut: float, fee_rate: float, half_spread: float | None) -> dict:
    """Sell YES on every strike priced at most `cut`, and sell NO (= buy YES at the ask) on every strike priced at
    least 1 - cut. Held to settlement, net of the spread and the taker fee."""
    d = le.dropna(subset=["outcome", "p"]).copy()
    d = d[d["t"] >= pd.Timestamp(config.EVAL_START, tz="UTC")]
    if half_spread is not None or "bid" not in d or d["bid"].isna().all():
        hs = half_spread if half_spread is not None else config.PM_HALF_SPREAD
        d["bid"], d["ask"] = d["p"] - hs, d["p"] + hs
    low = d[(d["p"] <= cut) & (d["bid"] > 0.001)].copy()
    low["pnl"] = low["bid"] - low["outcome"] - fee_rate * low["bid"] * (1 - low["bid"])
    low["wing"] = "low"
    high = d[(d["p"] >= 1 - cut) & (d["ask"] < 0.999)].copy()
    high["pnl"] = high["outcome"] - high["ask"] - fee_rate * high["ask"] * (1 - high["ask"])
    high["wing"] = "high"
    tr = pd.concat([low, high], ignore_index=True)
    if tr.empty:
        return {}
    h = tr["horizon"].iloc[0]
    per_snap = tr.groupby("T")["pnl"].sum()
    mean, se, n = evaluate.hac_mean(per_snap.to_numpy(), evaluate.nw_lag(len(per_snap), evaluate.overlap(h)))
    return {"horizon": h, "cut": cut, "n_trades": len(tr), "n_snapshots": n,
            "share_low_wing": float((tr["wing"] == "low").mean()),
            "avg_price_sold": float(np.where(tr["wing"] == "low", tr["bid"], 1 - tr["ask"]).mean()),
            "win_rate": float((tr["pnl"] > 0).mean()), "pnl_per_contract": float(tr["pnl"].mean()),
            "pnl_per_snapshot": mean, "t_stat": mean / se if se else np.nan,
            "worst_snapshot": float(per_snap.min())}


# ------------------------------------------------------------------ driver
def run():
    ensure(config.TABLES)
    sc = read_table(config.RESULTS / "scores")
    mcs_tab = model_confidence_set(sc)
    if len(mcs_tab):
        mcs_tab.to_csv(config.TABLES / "mcs.csv", index=False)
        log("extensions: model confidence set written")
    pool_sc, pool_dm, weights_all, enc, bins, cpa, cal, flu_paths, flu_summ = [], [], [], [], [], [], [], [], []
    eval_start = pd.Timestamp(config.EVAL_START, tz="UTC")
    for h in HORIZONS:
        try:
            fc = _forecasts(h)
            panel = read_table(config.INTERIM / f"panel_{h}")
        except FileNotFoundError:
            continue
        pool, w = optimal_pool(fc, h)
        if len(pool):
            weights_all.append(w)
            ps = evaluate.scores(pool[np.isfinite(pool["y"]) & (pool["t"] >= eval_start)])
            sc_h = sc[sc["horizon"] == h]
            both = pd.concat([sc_h, ps], ignore_index=True)
            mods = ["pool", "combo", "harx_all", "harx_d", *POOL_PARTS]
            cs = evaluate.common_sample(both[both["model"].isin(mods)], [m for m in dict.fromkeys(mods) if m in set(both["model"])])
            pool_sc.append(evaluate.summary_table(cs))
            for a_, b_ in [("pool", "combo"), ("pool", "harx_all"), ("pool", "harx_d"), ("pool", "harx"),
                           ("pool", "iv_mz"), ("pool", "mkt_mz")]:
                pa, pb = cs[cs["model"] == a_].set_index("t"), cs[cs["model"] == b_].set_index("t")
                ix = pa.index.intersection(pb.index)
                for loss in ("qlike", "logscore"):
                    la, lb = pa.loc[ix, loss].to_numpy(), pb.loc[ix, loss].to_numpy()
                    if loss == "logscore":
                        la, lb = -la, -lb
                    pool_dm.append({"horizon": h, "A": a_, "B": b_, "loss": loss, **evaluate.dm_test(la, lb, evaluate.overlap(h))})
        for base, rival in [("har", "mkt_mz"), ("harx_d", "mkt_mz"), ("har", "iv_mz")]:
            r = encompassing(fc, h, base, rival)
            if r:
                enc.append(r)
        bins.append(disagreement_bins(fc, h))
        cpa.append(conditional_ability(sc, panel, fc, h))
        cal.append(calendar_split(sc, panel, h))
        path, summ = fluctuation(sc, h)
        if summ:
            flu_paths.append(path)
            flu_summ.append(summ)
    if pool_sc:
        pd.concat(pool_sc).to_csv(config.TABLES / "pool_scores.csv", index=False)
        pd.DataFrame(pool_dm).to_csv(config.TABLES / "pool_dm.csv", index=False)
        wa = pd.concat(weights_all, ignore_index=True)
        write_table(wa, config.RESULTS / "pool_weights")
        ws = wa[wa["t"] >= eval_start].groupby("horizon")[[c for c in wa if c.startswith("w_")]].mean().reset_index()
        ws.to_csv(config.TABLES / "pool_weights_summary.csv", index=False)
    if enc:
        pd.DataFrame(enc).to_csv(config.TABLES / "encompassing.csv", index=False)
    bins = [b for b in bins if len(b)]
    if bins:
        pd.concat(bins).to_csv(config.TABLES / "disagreement_bins.csv", index=False)
    cpa = [c for c in cpa if len(c)]
    if cpa:
        pd.concat(cpa).to_csv(config.TABLES / "conditional_ability.csv", index=False)
    cal = [c for c in cal if len(c)]
    if cal:
        pd.concat(cal).to_csv(config.TABLES / "calendar_split.csv", index=False)
    if flu_summ:
        pd.DataFrame(flu_summ).to_csv(config.TABLES / "fluctuation_summary.csv", index=False)
        write_table(pd.concat(flu_paths, ignore_index=True), config.RESULTS / "fluctuation")
    # the variance premium in the tails as a trade
    try:
        le = read_table(config.RESULTS / "ladder_eval")
    except FileNotFoundError:
        le = pd.DataFrame()
    rows = []
    for h, g in le.groupby("horizon") if len(le) else []:
        fee = config.KALSHI_FEE_RATE if h == "kx1h" else 0.0
        for cut in (0.05, 0.10, 0.20):
            r = sell_wings(g, cut, fee, None if h == "kx1h" else config.PM_HALF_SPREAD)
            if r:
                rows.append(r)
    if rows:
        pd.DataFrame(rows).to_csv(config.TABLES / "wings.csv", index=False)
    log("extensions: pool, encompassing, conditional ability, fluctuation and wings tables written")
