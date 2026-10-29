"""Search for trading alpha in the BTC ladders, with protection against fooling ourselves.

Probability sources for every strike (column p_<source>):
  harx_d, harx_all, pool, gbm_all   model densities of the window variance, centred at the current price
  iv_raw                            the options market's view: Deribit implied variance as is (DVOL for kx1h)
  hyb_<model>                       a pure volatility bet: the ladder's own median as the centre and the model's
                                    variance as the width, so the trade disagrees with the market only about width
  recal                             the market price after a real-time logistic recalibration (favourite-longshot)
  xv                                (pm1h only) Kalshi's hourly ladder for the same noon-ET settlement,
                                    interpolated at Polymarket's strikes: the same event priced on another venue

New horizon pm1h: the Polymarket daily ladder 55 minutes before its noon-ET settlement. At that moment Kalshi's
hourly ladder settles at the same instant, so the two venues price (almost) the same event.

Rules against data snooping:
  1. every variant (source x edge threshold) is reported, not only the best one;
  2. chronological hold-out: a variant is chosen on the first half of each horizon's sample, judged on the second;
  3. Hansen's (2005) SPA test over all variants of a horizon: is the best one profitable after accounting for
     how many variants were tried?
  4. Polymarket spreads are assumed (no history of the order book): every Polymarket result is shown for
     half-spreads of 0.5, 1 and 2 cents.

Outputs: results/tables/alpha_{grid,spa,holdout,brier,xvenue_gap,recal}.csv, results/alpha_ladder.parquet.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import special, stats

import config
from src import evaluate, extensions
from src.measure import ladder as ladder_mod
from src.utils import ensure, log, read_table, write_table

THRESHOLDS = (0.02, 0.05, 0.10)
PM_HALF_SPREADS = (0.005, 0.01, 0.02)
MODEL_SOURCES = ["harx_d", "harx_all", "pool", "gbm_all", "iv_raw"]
HYBRID_SOURCES = ["harx_d", "harx_all", "pool"]
PM1H_OFFSET = pd.Timedelta(minutes=55)
PM1H_PRICE_AFTER_MIN = 6      # price points come every 10 minutes (:00, :10, ...); t = HH:05 -> the HH:10 point
PM_PRICE_AFTER_MIN = 2        # pm24h, pm6d: t = HH:00 -> the point a few seconds after HH:00
SPA_B = 2000
RECAL_MIN_TRAIN = {"kx1h": 2000, "pm24h": 300, "pm6d": 300, "pm1h": 300, "pm1h_stale": 300}
RECAL_REFIT = {"kx1h": 24, "pm24h": 5, "pm6d": 5, "pm1h": 5, "pm1h_stale": 5}
BLOCK = {"kx1h": 12, "pm24h": 6, "pm6d": 12, "pm1h": 6, "pm1h_stale": 6}   # mean block length of the stationary bootstrap


# ------------------------------------------------------------------ probabilities
def mix_digital(center, K, m, s, w, minus_half_v: bool) -> np.ndarray:
    """P(S_e > K) when log S_e = center (- v/2 if minus_half_v) + sqrt(v) Z and log v ~ sum_j w_j N(m_j, s_j^2).
    center = log of the current price (minus_half_v=True: zero drift, the mean equals today's price) or
    log of the market's median (minus_half_v=False)."""
    center = np.asarray(center, float)
    lk = np.log(np.asarray(K, float))
    out = np.zeros(len(lk))
    for j in range(m.shape[1]):
        wj = w[:, j]
        if not np.any(wj > 0):
            continue
        v = np.exp(m[:, j][:, None] + np.sqrt(2) * s[:, j][:, None] * evaluate.GH_X[None, :])
        d = (center[:, None] - lk[:, None] - (v / 2 if minus_half_v else 0)) / np.sqrt(v)
        out += wj * (stats.norm.cdf(d) @ evaluate.GH_W) / np.sqrt(np.pi)
    return out


def _components(st: pd.DataFrame, f: pd.DataFrame):
    f = f.drop_duplicates("T").set_index("T")
    sub = f.reindex(pd.DatetimeIndex(st["T"]))
    m = np.column_stack([sub[f"m{j}"].to_numpy(float) for j in evaluate.COMP])
    s = np.column_stack([sub[f"s{j}"].to_numpy(float) for j in evaluate.COMP])
    w = np.nan_to_num(np.column_stack([sub[f"w{j}"].to_numpy(float) for j in evaluate.COMP]))
    ok = np.isfinite(sub["m1"].to_numpy(float))
    return np.where(w > 0, np.nan_to_num(m), 0), np.where(w > 0, np.nan_to_num(s, nan=1.0), 1), w, ok


def add_model_probs(st: pd.DataFrame, fcs: dict, center_log_median: np.ndarray, log_var_shift: float = 0.0):
    for name, f in fcs.items():
        if f is None or f.empty:
            continue
        m, s, w, ok = _components(st, f)
        m = m + log_var_shift
        p = mix_digital(np.log(st["S_ref"].to_numpy(float)), st["strike"], m, s, w, True)
        p[~ok] = np.nan
        st[f"p_{name}"] = p
        if name in HYBRID_SOURCES:
            ph = mix_digital(center_log_median, st["strike"], m, s, w, False)
            ph[~ok | ~np.isfinite(center_log_median)] = np.nan
            st[f"p_hyb_{name}"] = ph
    return st


# ------------------------------------------------------------------ favourite-longshot recalibration
def _logit_fit(x: np.ndarray, y: np.ndarray, ridge: float = 1e-3) -> np.ndarray:
    X = np.column_stack([np.ones_like(x), x])
    b = np.array([0.0, 1.0])
    for _ in range(50):
        p = special.expit(X @ b)
        g = X.T @ (y - p) - ridge * b
        H = (X * (p * (1 - p))[:, None]).T @ X + ridge * np.eye(2)
        step = np.linalg.solve(H, g)
        b = b + step
        if np.abs(step).max() < 1e-8:
            break
    return b


def recalibrate(st: pd.DataFrame, h: str) -> tuple[np.ndarray, list[dict]]:
    """Platt scaling of the market price, re-estimated in real time on strikes that had already settled:
    P(Y = 1) = expit(a + b * logit(price)). b < 1 means extreme prices are too extreme (longshots overpriced)."""
    p = np.clip(st["p"].to_numpy(float), 0.005, 0.995)
    x = special.logit(p)
    y = st["outcome"].to_numpy(float)
    T = st["T"].to_numpy("datetime64[ns]")
    t = st["t"].to_numpy("datetime64[ns]")
    snaps = np.unique(t)
    out = np.full(len(st), np.nan)
    coefs = []
    step = RECAL_REFIT[h]
    for i in range(0, len(snaps), step):
        t0 = snaps[i]
        tr = (T + np.timedelta64(1, "m") <= t0) & np.isfinite(y) & np.isfinite(x)
        if tr.sum() < RECAL_MIN_TRAIN[h]:
            continue
        a, b = _logit_fit(x[tr], y[tr])
        rows = np.isin(t, snaps[i:i + step])
        out[rows] = special.expit(a + b * x[rows])
        coefs.append({"horizon": h, "t": pd.Timestamp(t0, tz="UTC"), "a": a, "b": b, "n_train": int(tr.sum())})
    return out, coefs


# ------------------------------------------------------------------ the pm1h horizon and the Kalshi fair value
def kalshi_fair(kg: pd.DataFrame, K_brti: np.ndarray) -> np.ndarray:
    """Kalshi-implied P(BRTI > K) between Kalshi's informative strikes: monotone prices, probit of the CDF
    interpolated linearly in log strike (the same interpolation as the BJN variance)."""
    out = np.full(len(K_brti), np.nan)
    g = kg.sort_values("strike")
    k = g["strike"].to_numpy(float)
    p = ladder_mod.monotone(g["p"].to_numpy(float))
    sel = (p > 0.002) & (p < 0.998)
    if sel.sum() < 2:
        return out
    x = np.log(k[sel])
    z = stats.norm.ppf(1 - p[sel])
    z = np.maximum.accumulate(z + np.arange(len(z)) * 1e-9)
    xi = np.log(K_brti)
    inside = (xi >= x.min()) & (xi <= x.max())
    out[inside] = 1 - stats.norm.cdf(np.interp(xi[inside], x, z))
    return out


def build_pm1h(snaps: pd.DataFrame, le_kx: pd.DataFrame, stale: bool = False) -> pd.DataFrame:
    """Polymarket daily ladder at t = settlement - 55 min, with the price, the Binance reference, the outcome,
    the Kalshi basis (BRTI/Binance, from earlier events only) and the Kalshi fair value at each strike."""
    try:
        lad = read_table(config.INTERIM / "pm_ladder")
        px = read_table(config.INTERIM / "pm_prices")
        kx_strikes = read_table(config.INTERIM / "strikes_kx1h")
        p_kx = read_table(config.INTERIM / "panel_kx1h")[["t", "S_t"]]
        p_pm = read_table(config.INTERIM / "panel_pm24h")[["T", "S_e"]]
    except FileNotFoundError:
        return pd.DataFrame()
    for df_, cols in ((lad, ["settle_ts", "open_ts"]), (px, ["ts"]), (kx_strikes, ["T", "t"]), (p_kx, ["t"]),
                      (p_pm, ["T"])):
        for c in cols:
            df_[c] = pd.to_datetime(df_[c], utc=True).dt.as_unit("ns")
    s = lad.copy()
    s["T"] = s["settle_ts"]
    s["t"] = s["T"] - PM1H_OFFSET
    s = s[s["t"] >= s["open_ts"] + pd.Timedelta(minutes=15)].sort_values("t")
    # the market price is the first price point AFTER t (the history has one every 10 minutes): the market then
    # knows at least as much as the model, which only uses information up to t
    # (stale=True reproduces the naive design, the last price before t, to show what it does to the results)
    m = pd.merge_asof(s, px.sort_values("ts"), left_on="t", right_on="ts", by="yes_token",
                      direction="backward" if stale else "forward",
                      tolerance=pd.Timedelta(minutes=config.PM_MAX_STALENESS_MIN if stale else PM1H_PRICE_AFTER_MIN)
                      ).dropna(subset=["p"])
    m["price_lag_min"] = (m["ts"] - m["t"]).dt.total_seconds() / 60
    m = m.merge(p_kx.drop_duplicates("t"), on="t", how="left").merge(p_pm.drop_duplicates("T"), on="T", how="left")
    basis = le_kx.groupby("T")["basis"].first()
    m["basis"] = m["T"].map(basis)
    m["horizon"] = "pm1h_stale" if stale else "pm1h"
    m["S_ref"] = m["S_t"]
    m["outcome"] = np.where(m["S_e"].notna(), (m["S_e"] > m["strike"]).astype(float), m["outcome"])
    # Kalshi fair value: the Kalshi event settling at the same instant, quoted at the same t
    m["p_xv"] = np.nan
    kx = kx_strikes[kx_strikes["t"] == kx_strikes["T"] - PM1H_OFFSET]
    for T, g in m.groupby("T"):
        kg = kx[kx["T"] == T]
        b = g["basis"].iloc[0]
        if kg.empty or not np.isfinite(b):
            continue
        m.loc[g.index, "p_xv"] = kalshi_fair(kg, g["strike"].to_numpy(float) * b)
    # centre for the hybrid: the Kalshi ladder's median, converted from BRTI to Binance
    ks = snaps[snaps["horizon"] == "kx1h"].drop_duplicates("T").set_index("T")["mkt_m"]
    m["center"] = m["T"].map(ks) - np.log(m["basis"])
    m = m.dropna(subset=["S_ref"])
    return m[["horizon", "T", "t", "strike", "p", "price_lag_min", "outcome", "S_ref", "S_e", "basis", "p_xv",
              "center", "event_slug"]].reset_index(drop=True)


def fresh_prices(rows: pd.DataFrame, px: pd.DataFrame, after_min: float) -> pd.DataFrame:
    """For Polymarket rows: the first price point at or after t (within after_min minutes). The ladder table
    uses the last price BEFORE t, which can be up to 30 minutes old; a model that knows the price of BTC at t
    would then 'beat' a stale market for free."""
    r = rows[["t", "yes_token"]].reset_index().sort_values("t")
    m = pd.merge_asof(r, px.sort_values("ts")[["ts", "yes_token", "p"]], left_on="t", right_on="ts",
                      by="yes_token", direction="forward", tolerance=pd.Timedelta(minutes=after_min))
    m = m.set_index("index").reindex(rows.index)
    return pd.DataFrame({"p": m["p"], "price_lag_min": (m["ts"] - m["t"]).dt.total_seconds() / 60})


# ------------------------------------------------------------------ trading, SPA and hold-out
def trade(d: pd.DataFrame, col: str, th: float, fee: float, half_spread: float | None):
    """Per-snapshot profit (zeros where nothing was traded) and the trades. Buy YES at the ask when
    source - ask > th, sell YES at the bid when bid - source > th; held to settlement, net of fees."""
    if half_spread is not None:   # Polymarket: no quote history, assume a spread around the price
        bid = np.clip(d["p"].to_numpy(float) - half_spread, 0.001, 0.999)
        ask = np.clip(d["p"].to_numpy(float) + half_spread, 0.001, 0.999)
    else:
        bid, ask = d["bid"].to_numpy(float), d["ask"].to_numpy(float)
    q = d[col].to_numpy(float)
    y = d["outcome"].to_numpy(float)
    ok = np.isfinite(q) & np.isfinite(y) & np.isfinite(bid) & np.isfinite(ask)
    buy = ok & (q - ask > th)
    sell = ok & (bid - q > th) & ~buy
    pnl = np.where(buy, y - ask - fee * ask * (1 - ask), 0.0) + np.where(sell, bid - y - fee * bid * (1 - bid), 0.0)
    traded = buy | sell
    per_snap = pd.Series(pnl, index=d["T"]).groupby(level=0).sum()
    n_snap_tr = pd.Series(traded, index=d["T"]).groupby(level=0).any()
    return per_snap, n_snap_tr, pnl[traded], buy, sell


def _boot_idx(n: int, mean_block: float, rng) -> np.ndarray:
    """Stationary bootstrap (Politis-Romano): blocks of random length with mean `mean_block`, wrapped around."""
    new = rng.random(n) < 1 / mean_block
    new[0] = True
    starts = rng.integers(0, n, n)
    pos = np.arange(n)
    last = np.maximum.accumulate(np.where(new, pos, 0))
    return (starts[last] + pos - last) % n


def spa(P: pd.DataFrame, mean_block: float, B: int = SPA_B, seed: int = 3) -> dict:
    """Hansen's (2005) consistent SPA test. P: per-snapshot profits of k variants (benchmark: not trading).
    H0: no variant has positive expected profit. Studentised max statistic, stationary bootstrap, and the
    recentring that keeps clearly bad variants from inflating the p-value."""
    P = P.loc[:, (P != 0).any()]
    X = P.to_numpy(float)
    n, k = X.shape
    if n < 30 or k == 0:
        return {}
    rng = np.random.default_rng(seed)
    dbar = X.mean(axis=0)
    means = np.empty((B, k))
    for b in range(B):
        means[b] = X[_boot_idx(n, mean_block, rng)].mean(axis=0)
    omega = np.sqrt(n * means.var(axis=0, ddof=1))
    omega = np.where(omega > 0, omega, np.nan)
    tstat = np.sqrt(n) * dbar / omega
    T_obs = max(0.0, float(np.nanmax(tstat)))
    thr = -np.sqrt(2 * np.log(np.log(n)))
    mu_c = np.where(tstat <= thr, dbar, 0.0)
    Z = np.sqrt(n) * (means - dbar + mu_c) / omega
    Tb = np.maximum(0.0, np.nanmax(Z, axis=1))
    best = int(np.nanargmax(tstat))
    return {"n_snapshots": n, "k_variants": k, "best_variant": P.columns[best], "best_mean_per_snapshot": dbar[best],
            "best_t": tstat[best], "naive_p_best": float(1 - stats.norm.cdf(tstat[best])),
            "spa_p": float(np.mean(Tb >= T_obs))}


def holdout(P: pd.DataFrame, n_traded: pd.DataFrame, h: str, min_snaps: int = 15) -> dict:
    """Choose the variant with the highest mean profit per snapshot on the first half of the sample (among
    variants that traded on at least `min_snaps` snapshots there) and report it on the second half only."""
    P = P.sort_index()
    half = len(P) // 2
    dev, test = P.iloc[:half], P.iloc[half:]
    elig = n_traded.sort_index().iloc[:half].sum() >= min_snaps
    if not elig.any():
        return {}
    choice = dev.loc[:, elig].mean().idxmax()
    mean, se, n = evaluate.hac_mean(test[choice].to_numpy(float), evaluate.nw_lag(len(test), evaluate.overlap(h)))
    return {"chosen_on_dev": choice, "dev_from": dev.index.min(), "test_from": test.index.min(),
            "dev_mean_per_snapshot": dev[choice].mean(), "test_mean_per_snapshot": mean,
            "test_t": mean / se if se and np.isfinite(se) else np.nan, "test_snapshots": n,
            "test_snapshots_traded": int(n_traded.sort_index().iloc[half:][choice].sum())}


def run_grid(d: pd.DataFrame, h: str, sources: list[str], fee: float, half_spread: float | None):
    rows, P, NT = [], {}, {}
    for src in sources:
        col = f"p_{src}"
        if col not in d or d[col].notna().sum() == 0:
            continue
        for th in THRESHOLDS:
            per_snap, snap_tr, pnl_tr, buy, sell = trade(d, col, th, fee, half_spread)
            name = f"{src}@{th:.2f}"
            P[name], NT[name] = per_snap, snap_tr
            ts = per_snap[snap_tr]
            mean, se, n = evaluate.hac_mean(ts.to_numpy(float), evaluate.nw_lag(len(ts), evaluate.overlap(h)))
            rows.append({"horizon": h, "half_spread": half_spread, "source": src, "threshold": th,
                         "n_trades": int(len(pnl_tr)), "n_buys": int(buy.sum()), "n_sells": int(sell.sum()),
                         "n_snapshots_traded": int(snap_tr.sum()),
                         "pnl_per_contract": pnl_tr.mean() if len(pnl_tr) else np.nan,
                         "pnl_per_snapshot": mean, "t_stat": mean / se if se and np.isfinite(se) else np.nan,
                         "hit_rate": float((pnl_tr > 0).mean()) if len(pnl_tr) else np.nan})
    return pd.DataFrame(rows), pd.DataFrame(P).fillna(0.0), pd.DataFrame(NT).fillna(False)


def brier_table(d: pd.DataFrame, h: str, sources: list[str]) -> pd.DataFrame:
    """Is a source a better probability than the market itself? Brier score per strike, DM per snapshot."""
    rows = []
    for src in sources:
        col = f"p_{src}"
        if col not in d:
            continue
        x = d.dropna(subset=[col, "p", "outcome"])
        if len(x) < 100:
            continue
        bs = (x[col] - x["outcome"]) ** 2
        bm = (x["p"] - x["outcome"]) ** 2
        snap = pd.DataFrame({"s": bs, "m": bm, "T": x["T"]}).groupby("T").sum()
        r = evaluate.dm_test(snap["s"].to_numpy(), snap["m"].to_numpy(), evaluate.overlap(h))
        rows.append({"horizon": h, "source": src, "n_strikes": len(x), "brier_source": bs.mean(),
                     "brier_market": bm.mean(), **{f"dm_{k}": v for k, v in r.items()}})
    return pd.DataFrame(rows)


def gap_table(d: pd.DataFrame, h: str, ref: str) -> dict:
    """Mean (market - reference) probability gap on strikes where the reference is between 5% and 95%."""
    col = f"p_{ref}"
    if col not in d:
        return {}
    x = d[(d[col] > 0.05) & (d[col] < 0.95)].dropna(subset=["p"])
    if len(x) < 30:
        return {}
    g = (x["p"] - x[col]).groupby(x["T"]).mean()
    mean, se, n = evaluate.hac_mean(g.to_numpy(float), evaluate.nw_lag(len(g), evaluate.overlap(h)))
    above = x[x["p"] > 0.5]
    below = x[x["p"] <= 0.5]
    return {"horizon": h, "reference": ref, "n_strikes": len(x), "n_snapshots": n, "mean_gap": mean,
            "t": mean / se if se else np.nan, "mean_abs_gap": float((x["p"] - x[col]).abs().mean()),
            "gap_when_price_above_half": float((above["p"] - above[col]).mean()) if len(above) else np.nan,
            "gap_when_price_below_half": float((below["p"] - below[col]).mean()) if len(below) else np.nan}


# ------------------------------------------------------------------ driver
def run():
    ensure(config.TABLES)
    try:
        le = read_table(config.RESULTS / "ladder_eval")
        snaps = read_table(config.INTERIM / "ladder_snaps")
    except FileNotFoundError:
        log("alpha: run the evaluate step first")
        return
    for df_ in (le, snaps):
        for c in ("T", "t"):
            df_[c] = pd.to_datetime(df_[c], utc=True).dt.as_unit("ns")
    eval_start = pd.Timestamp(config.EVAL_START, tz="UTC")
    try:
        px = read_table(config.INTERIM / "pm_prices")
        px["ts"] = pd.to_datetime(px["ts"], utc=True).dt.as_unit("ns")
    except FileNotFoundError:
        px = None
    fcs_all, tables = {}, []
    for h in ("kx1h", "pm24h", "pm6d"):
        try:
            fc = read_table(config.RESULTS / f"forecasts_{h}")
        except FileNotFoundError:
            continue
        pool, _ = extensions.optimal_pool(fc, h)
        fcs = {mdl: fc[fc["model"] == mdl] for mdl in MODEL_SOURCES if mdl != "pool"}
        fcs["pool"] = pool
        fcs_all[h] = fcs
        st = le[le["horizon"] == h].copy().reset_index(drop=True)
        if st.empty:
            continue
        if h != "kx1h" and px is not None:
            st["p_before_t"] = st["p"]
            st[["p", "price_lag_min"]] = fresh_prices(st, px, PM_PRICE_AFTER_MIN)
        if h == "kx1h":
            sn = snaps[snaps["horizon"] == h][["T", "t", "mkt_m"]].drop_duplicates(["T", "t"])
            st = st.merge(sn, on=["T", "t"], how="left")
        else:
            # the ladder's median from the same (fresh) prices that are traded, not from the stale ones
            cen = {T: ladder_mod.fit_lognormal(g["strike"], g["p"])["m"]
                   for T, g in st.dropna(subset=["p"]).groupby("T")}
            st["mkt_m"] = st["T"].map(cen)
        st = add_model_probs(st, fcs, st["mkt_m"].to_numpy(float))
        tables.append(st)
    if "kx1h" in fcs_all:
        for stale in (False, True):
            pm1h = build_pm1h(snaps, le[le["horizon"] == "kx1h"], stale=stale)
            if len(pm1h):
                # kx1h densities are for 55 minutes; Polymarket's window to the close of the noon candle is 56
                pm1h = add_model_probs(pm1h, fcs_all["kx1h"], pm1h["center"].to_numpy(float), np.log(56 / 55))
                tables.append(pm1h)
    grids, spas, holds, briers, gaps, recal_coefs, out = [], [], [], [], [], [], []
    for st in tables:
        h = st["horizon"].iloc[0]
        st = st.sort_values(["t", "strike"]).reset_index(drop=True)
        st["p_recal"], coefs = recalibrate(st, h)
        recal_coefs += coefs
        out.append(st)
        d = st[(st["t"] >= eval_start) & st["outcome"].notna() & st["p"].notna()].copy()
        if d.empty:
            continue
        srcs = MODEL_SOURCES + [f"hyb_{m}" for m in HYBRID_SOURCES] + ["recal"] + (["xv"] if h.startswith("pm1h") else [])
        fee = config.KALSHI_FEE_RATE if h == "kx1h" else 0.0
        spreads = [None] if h == "kx1h" else [PM_HALF_SPREADS[0]] if h == "pm1h_stale" else list(PM_HALF_SPREADS)
        for hs in spreads:
            grid, P, NT = run_grid(d, h, srcs, fee, hs)
            grids.append(grid)
            r = spa(P, BLOCK[h])
            if r:
                spas.append({"horizon": h, "half_spread": hs, **r})
            r = holdout(P, NT, h)
            if r:
                holds.append({"horizon": h, "half_spread": hs, **r})
        briers.append(brier_table(d, h, srcs))
        for ref in (["xv", "iv_raw"] if h.startswith("pm1h") else ["iv_raw", "harx_d"]):
            g = gap_table(d, h, ref)
            if g:
                gaps.append(g)
        log(f"alpha: {h} done ({d['T'].nunique()} snapshots, {len(d)} strikes)")
    write_table(pd.concat(out, ignore_index=True), config.RESULTS / "alpha_ladder")
    pd.concat(grids).to_csv(config.TABLES / "alpha_grid.csv", index=False)
    pd.DataFrame(spas).to_csv(config.TABLES / "alpha_spa.csv", index=False)
    pd.DataFrame(holds).to_csv(config.TABLES / "alpha_holdout.csv", index=False)
    pd.concat(briers).to_csv(config.TABLES / "alpha_brier.csv", index=False)
    pd.DataFrame(gaps).to_csv(config.TABLES / "alpha_xvenue_gap.csv", index=False)
    rc = pd.DataFrame(recal_coefs)
    if len(rc):
        rc.groupby("horizon").tail(1).to_csv(config.TABLES / "alpha_recal.csv", index=False)
    log("alpha: grid, SPA, hold-out, Brier and cross-venue tables written")


if __name__ == "__main__":
    run()
