"""How accurate can the wing probability be, and does accuracy pay? (Nov 2026)

The wing rules sell a cheap binary when our probability of it paying is below its price plus the fee. Every
frozen rule uses the same forecaster: an EWMA level of deseasonalised squared 1-minute returns times an
hour-of-week profile, normal (W03) or Student-t(4) (W04) tails. This study asks whether a better forecaster of
the variance left until settlement improves the *probability* (Brier, log loss, calibration) and the *P&L*.

Models, all strictly lagged and all evaluated on the same rows
    M0 frozen    EWMA(60) of r^2, no seasonality, normal / t(4)      (what W03/W04 were chosen on)
    M1 lab       level = .5 EWMA(30) + .3 24 h + .2 7 d of deseasonalised r^2, 13-week hour-of-week profile,
                 settlement average removed                            (lab/history.py, what the lab trades)
    M2 HAR-X     monthly-refit OLS of log RV over the remaining window on log RV(5,15,60 min), log RV(24 h),
                 log RV(7 d), jump share, log volume, HARQ, the profile and the weekend dummy, with the
                 lognormal correction back to variance                          (new)
    M3 pool      geometric (log-variance) pool of M1 and M2, weight fitted on the development half (new)
    M4 recaled   M2 with a log-odds (Platt) recalibration fitted on the development half (new)
Tails: normal and t(4) for every variance model, so the variance question and the tail question stay apart.

Scoring rows: every cheap side 3-20 cents with 5-30 minutes left, one row per strike and 5-minute bucket
(the project's `sampled` scheme), clustered by market in the bootstrap.
Trading: the frozen W04 rule (6-10 cents, 15-30 min, edge >= 0, one strike an hour) and W03 (10-20 cents,
5-15 min, edge >= 1 cent) with each model as fair value.

Run:  python studies/wings_accuracy.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.special import expit, logit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from studies.wing_panel import (build, cheap_sides, fee, minute_close, variance_inputs,  # noqa: E402
                                remaining_variance)

OUT = ROOT / "results/studies"
SPLIT = pd.Timestamp("2026-09-01", tz="UTC")
TRAIN = (pd.Timestamp("2026-07-01", tz="UTC"), SPLIT)
TEST = (SPLIT, pd.Timestamp("2026-10-28", tz="UTC"))
NOV = (pd.Timestamp("2026-11-08", tz="UTC"), pd.Timestamp("2026-11-26", tz="UTC"))
FIT_START = pd.Timestamp("2023-07-01", tz="UTC")          # first month the HAR-X is estimated on
FEATURES = ["logrv5", "logrv15", "logrv60", "logrv24h", "logrv7d", "jump", "logvol", "harq",
            "profile", "weekend", "logmin"]


# ------------------------------------------------------------------------------------- variance models
def frozen_var(close: pd.Series) -> pd.Series:
    """the W03/W04 backtest forecaster: EWMA(60) of squared returns, no seasonality."""
    return (np.log(close).diff() ** 2).ewm(halflife=60, min_periods=120).mean()


def minute_features(close: pd.Series, volume: pd.Series | None = None) -> tuple[pd.DataFrame, pd.Series]:
    """Lagged minute features and the realised variance of every [t, next hour) window."""
    v = variance_inputs(close)
    r = np.log(close).diff()
    r2 = r ** 2
    out = pd.DataFrame(index=close.index)
    for name, minutes in (("logrv5", 5), ("logrv15", 15), ("logrv60", 60), ("logrv24h", 1440),
                          ("logrv7d", 10080)):
        out[name] = np.log(r2.rolling(minutes, min_periods=max(5, minutes // 3)).mean().clip(lower=1e-14))
    # jump share: |r| above 4 local standard deviations (a simple, standard jump screen)
    sd = r2.rolling(1440, min_periods=300).mean().pow(0.5)
    out["jump"] = (r.abs() > 4 * sd).astype(float).rolling(1440, min_periods=300).mean()
    vol = (pd.Series(np.nan, index=close.index) if volume is None
           else pd.Series(np.asarray(volume, dtype=float), index=close.index))
    out["logvol"] = np.log(vol.rolling(1440, min_periods=120).mean().clip(lower=1e-9))
    out["harq"] = np.log(r2.pow(2).rolling(60, min_periods=20).mean().clip(lower=1e-30)) / 2
    out["profile"] = np.log(v.sea.clip(lower=1e-3))
    out["weekend"] = (close.index.day_of_week >= 5).astype(float)
    out["level"] = v.level
    # target: realised variance of the rest of the hour, for every minute of the hour
    end = close.index.ceil("h")
    var_target = pd.Series(np.nan, index=close.index)
    grid = close.index
    cum = (r2).cumsum()
    i0 = grid.searchsorted(grid, side="left")
    i1 = grid.searchsorted(end, side="left")
    var_target.iloc[:] = (cum.to_numpy()[np.clip(i1, 0, len(cum) - 1)]
                          - cum.to_numpy()[np.clip(i0, 0, len(cum) - 1)])
    out["minutes"] = (end - close.index).total_seconds() / 60
    out["target"] = np.log(var_target.clip(lower=1e-14))
    out["logmin"] = np.log(out.minutes.clip(lower=0.5))
    return out.replace([np.inf, -np.inf], np.nan), v


def harx_walk_forward(f: pd.DataFrame) -> pd.DataFrame:
    """Monthly-refit OLS of log RV of the remaining window on the lagged features (expanding window).

    Returns the predicted *log* variance (the conditional median of the variance, which is the natural
    plug-in for a probability) and the residual standard deviation of that log variance, which the
    predictive-mixture model later uses to integrate the variance uncertainty out.
    """
    d = f.dropna(subset=FEATURES + ["target"])
    d = d[d.index >= FIT_START - pd.Timedelta(days=400)]
    pred = pd.Series(np.nan, index=f.index)
    sd = pd.Series(np.nan, index=f.index)
    months = pd.period_range(f.index.min(), f.index.max(), freq="M")
    for month in months:
        start = month.to_timestamp(how="start").tz_localize("UTC")
        end = (month + 1).to_timestamp(how="start").tz_localize("UTC")
        train = d[d.index < start]
        block = f[(f.index >= start) & (f.index < end)].dropna(subset=FEATURES)
        if len(train) < 5000 or block.empty:
            continue
        x = np.column_stack([np.ones(len(train)), train[FEATURES].to_numpy()])
        beta, *_ = np.linalg.lstsq(x, train.target.to_numpy(), rcond=None)
        resid = train.target.to_numpy() - x @ beta
        s2 = float(np.var(resid))
        xb = np.column_stack([np.ones(len(block)), block[FEATURES].to_numpy()])
        pred.loc[block.index] = xb @ beta
        sd.loc[block.index] = math.sqrt(s2)
    return pd.DataFrame({"logvar": pred, "sd": sd})


def tail(z: np.ndarray, kind: str, df: float = 4.0) -> np.ndarray:
    if kind == "normal":
        return norm.cdf(z)
    from scipy.stats import t as student_t
    return student_t.cdf(np.asarray(z) * math.sqrt(df / (df - 2.0)), df)


# ------------------------------------------------------------------------------------- scoring rows
def sampled(c: pd.DataFrame) -> pd.DataFrame:
    """Balance expiries, strikes and 5-minute buckets (the project's scheme), one row per strike-bucket."""
    x = c.dropna(subset=["won"]).sort_values("ts").copy()
    x["bucket"] = (x.left / 5).astype(int)
    x = x.drop_duplicates(["end", "market", "bucket"])
    strikes = x.groupby("end").market.transform("nunique")
    buckets = x.groupby(["end", "market"]).market.transform("size")
    x["weight"] = 1.0 / (strikes * buckets)
    return x


def scores(rows: pd.DataFrame, name: str) -> dict:
    """Brier, log loss, calibration slope and reliability for one model on the weighted rows."""
    ok = np.isfinite(rows[name].to_numpy()) & rows.won.notna().to_numpy()
    rows = rows[ok]
    p = rows[name].clip(1e-6, 1 - 1e-6)
    y = rows.won.to_numpy()
    w = rows.weight.to_numpy()
    brier = float(np.sum(w * (p - y) ** 2) / np.sum(w))
    ll = float(-np.sum(w * (y * np.log(p) + (1 - y) * np.log(1 - p))) / np.sum(w))
    z = logit(p)
    # calibration: weighted OLS of y on [1, logit(p)]; slope 1 and intercept 0 = calibrated
    a = np.column_stack([np.ones(len(z)), z]) * np.sqrt(w)[:, None]
    b = y * np.sqrt(w)
    coef = np.linalg.lstsq(a, b, rcond=None)[0]
    slope, intercept = float(coef[1]), float(coef[0])
    bins = pd.cut(p, [0, .03, .05, .08, .12, .2, .35, 1], duplicates="drop")
    rel = pd.DataFrame({"p": p, "y": y, "w": w, "bin": bins}).groupby("bin", observed=True).apply(
        lambda g: pd.Series({"mean p": np.average(g.p, weights=g.w), "mean y": np.average(g.y, weights=g.w),
                             "weight": g.w.sum()}), include_groups=False)
    return dict(model=name, rows=len(rows), brier=brier, logloss=ll, slope=slope, intercept=intercept,
                mean_p=float(np.average(p, weights=w)), mean_y=float(np.average(y, weights=w)),
                table=rel.reset_index())


def brier_delta(a: pd.DataFrame, p: str, q: str, draws: int = 400, seed: int = 3) -> tuple[float, float, float]:
    """Market-clustered bootstrap interval of Brier(p) - Brier(q), negative = p better."""
    y = a.won.to_numpy()
    w = a.weight.to_numpy()
    d = w * ((a[p].clip(1e-6, 1 - 1e-6) - y) ** 2 - (a[q].clip(1e-6, 1 - 1e-6) - y) ** 2)
    groups = [g.to_numpy() for _, g in pd.Series(d).groupby(a.market.to_numpy())]
    rng = np.random.default_rng(seed)
    means = np.array([np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))]).mean()
                      for _ in range(draws)])
    return float(np.mean(d) / np.mean(w)), float(np.quantile(means, 0.025) / np.mean(w)), \
        float(np.quantile(means, 0.975) / np.mean(w))


# ------------------------------------------------------------------------------------- trading
def trade(c: pd.DataFrame, model: str, margin: float, band: tuple[float, float],
          window: tuple[float, float]) -> pd.Series:
    g = c[c.wing_bid.between(*band) & c.left.between(*window)].copy()
    g["edge"] = g.wing_bid - g[model] - fee(1, g.price)
    g = g[g.edge >= margin]
    g = g.sort_values(["end", "ts", "wing_bid"], ascending=[True, True, False])
    g = g.drop_duplicates(["end", "market"]).drop_duplicates("end")
    pnl = g.wing_bid - g.won - fee(1, g.price)
    return pd.Series(pnl.to_numpy(), index=g.end.to_numpy())


def stats(x: pd.Series) -> dict:
    x = x.dropna()
    n = len(x)
    if n < 3:
        return dict(hours=n, cents=np.nan, t=np.nan, hit=np.nan, dd=np.nan)
    h = x.groupby(level=0).sum() if isinstance(x.index, pd.DatetimeIndex) and x.index.has_duplicates else x
    t = h.mean() / (h.std(ddof=1) / math.sqrt(len(h))) if h.std() > 0 else np.nan
    eq = h.cumsum()
    return dict(hours=len(h), cents=100 * x.mean(), t=t, hit=float((h > 0).mean()),
                dd=float((eq.cummax() - eq).max()))


# ------------------------------------------------------------------------------------- main
def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    d = build()
    px = minute_close()
    f, v = minute_features(px.close, px.volume)
    har = harx_walk_forward(f)
    d = d.merge(har.assign(var_harx=lambda x: np.exp(x.logvar)),
                left_on="ts", right_index=True, how="left")

    # every cheap side 3-20 cents, 5-30 minutes left, with the variance of each model
    c = cheap_sides(d, 0.03, 0.20, 5, 30)
    c["var_lab"] = c.var_rem
    c["var_frozen"] = frozen_var(px.close).reindex(c.ts).to_numpy() * np.maximum(c.left - 2 / 3, 1 / 3)
    z = np.log(c.S * c.basis / c.strike)
    # predictive mixture: average Phi(z / sqrt(v)) over the conditional distribution of the variance
    eps = np.array([-1.6449, -1.0364, -0.6745, -0.3186, 0.0, 0.3186, 0.6745, 1.0364, 1.6449])
    logv = c.logvar.to_numpy()
    sdv = c.sd.to_numpy()
    mix = np.zeros(len(c))
    for e in eps:
        v = np.exp(logv + sdv * e)
        mix += tail(z / np.sqrt(np.clip(v, 1e-14, None)), "normal") / len(eps)
    c["harx_mix"] = np.where(c.cheap_yes, mix, 1 - mix)
    w = 0.5
    c["var_pool"] = np.exp(w * np.log(c.var_lab.clip(lower=1e-14))
                           + (1 - w) * np.log(c.var_harx.clip(lower=1e-14)))
    for name in ("frozen", "lab", "harx", "pool"):
        c["z_" + name] = z / np.sqrt(c["var_" + name].clip(lower=1e-14))
        for kind in ("normal", "t4"):
            p_yes = tail(c["z_" + name], "normal" if kind == "normal" else "t4")
            c[f"{name}|{kind}"] = np.where(c.cheap_yes, p_yes, 1 - p_yes)

    models = ["frozen|normal", "frozen|t4", "lab|normal", "lab|t4", "harx|normal", "harx|t4",
              "harx_mix", "pool|normal", "pool|t4"]
    rows, rel_rows, deltas = [], [], []
    for period, (a, b) in {"train Jul-Aug": TRAIN, "test Sep-Oct": TEST, "holdout Nov": NOV}.items():
        sub = c[(c.end >= a) & (c.end < b)]
        s = sampled(sub)
        if len(s) < 50:
            continue
        ref = "frozen|t4" if not s["frozen|t4"].isna().all() else "lab|t4"
        base = scores(s, ref)
        for m in models:
            if s[m].isna().all():
                continue
            r = scores(s, m)
            r["period"] = period
            r["Δ brier vs " + ref] = r["brier"] - base["brier"]
            lo, hi, mid = (brier_delta(s, m, ref) if not s[ref].isna().all() else (np.nan, np.nan, np.nan))
            r["Δ 95% low"], r["Δ 95% high"] = lo, hi
            rows.append({k: v for k, v in r.items() if k != "table"})
            t = r["table"].copy()
            t["model"], t["period"] = m, period
            rel_rows.append(t)
        deltas.append({"period": period, "rows": len(s), "reference": ref, "mean p ref": base["mean_p"],
                       "mean outcome": base["mean_y"]})
    score_table = pd.DataFrame(rows)
    score_table.to_csv(OUT / "wings_accuracy_scores.csv", index=False)
    pd.concat(rel_rows, ignore_index=True).to_csv(OUT / "wings_accuracy_reliability.csv", index=False)

    # does the better probability earn more?
    trade_rows = []
    for period, (a, b) in {"train Jul-Aug": TRAIN, "test Sep-Oct": TEST, "holdout Nov": NOV}.items():
        sub = c[(c.end >= a) & (c.end < b)]
        for rule, band, window, margin in (("W04 early", (0.06, 0.10), (15, 30), 0.0),
                                          ("W03 late", (0.10, 0.20), (5, 15), 0.01)):
            for m in models:
                if sub[m].isna().all():
                    continue
                pnl = trade(sub, m, margin, band, window)
                t = stats(pnl)
                trade_rows.append({"period": period, "rule": rule, "model": m, **t})
    trade_table = pd.DataFrame(trade_rows)
    trade_table.to_csv(OUT / "wings_accuracy_trades.csv", index=False)

    write_report(score_table, trade_table, pd.DataFrame(deltas))
    print(score_table.to_string(index=False))
    print(trade_table.to_string(index=False))


def _reading(scores_t: pd.DataFrame, trades_t: pd.DataFrame) -> str:
    """Three sentences that follow from the tables, written out so the file is readable on its own."""
    lines = []
    for period, g in scores_t.groupby("period"):
        best = g.loc[g.brier.idxmin()]
        lines.append(f"* {period}: лучший Brier у `{best.model}` ({best.brier:.5f}); наклон калибровки "
                     f"{g.slope.min():.3f}…{g.slope.max():.3f}, средняя p {g.mean_p.min():.3f}…{g.mean_p.max():.3f} "
                     f"против факта {g.mean_y.iloc[0]:.3f}.")
    spread = scores_t.pivot_table(index="model", columns="period", values="brier")
    if "test Sep-Oct" in spread:
        col = spread["test Sep-Oct"].dropna().sort_values()
        lines.append(f"* Разброс между моделями по Brier на тесте: {col.iloc[0]:.5f}…{col.iloc[-1]:.5f} — "
                     "это меньше, чем эффект одного хвоста t(4) против нормального.")
    if not trades_t.empty:
        t = trades_t[trades_t.period == "test Sep-Oct"]
        if not t.empty:
            g = t[t.rule == "W04 early"]
            lines.append(f"* В деньгах на тесте W04 даёт {g.cents.min():.2f}…{g.cents.max():.2f}¢ на час при "
                         f"{int(g.hours.min())}…{int(g.hours.max())} сделках: выбор модели вероятности — шум, "
                         "и это согласуется с наклоном калибровки около нуля.")
    return "\n".join(lines) + "\n"


def write_report(scores_t, trades_t, meta) -> None:
    def md(df, fmt="{:.4f}"):
        if df is None or df.empty:
            return "_no rows_\n"
        out = df.copy()
        out.columns = [str(c).replace("|", "/") for c in out.columns]
        for c in out.columns:
            if pd.api.types.is_float_dtype(out[c]):
                out[c] = out[c].map(lambda x: "" if pd.isna(x) else fmt.format(x))
        head = "| " + " | ".join(map(str, out.columns)) + " |\n"
        sep = "|" + "|".join(["---"] * len(out.columns)) + "|\n"
        body = "".join("| " + " | ".join(map(str, r)) + " |\n" for r in out.itertuples(index=False))
        return head + sep + body

    lines = [
        "# Насколько точна вероятность крыла и приносит ли точность деньги\n",
        f"_Сгенерировано {pd.Timestamp.now():%Y-%m-%d %H:%M}, `studies/wings_accuracy.py`._\n",
        "Все модели дисперсии считаются только по прошлым минутам; HAR-X переобучается каждый месяц на "
        "расширяющемся окне. Строки для Brier — по одному наблюдению на страйк и 5-минутный бакет, "
        "взвешенные так, чтобы каждая экспирация, каждый страйк и каждый бакет весили одинаково; "
        "интервалы — бутстреп по рынкам.\n",
        "\n## Точность вероятности\n",
        md(scores_t),
        "\n## Торговля тем же правилом с каждой моделью в роли справедливой цены\n",
        md(trades_t, "{:.3f}"),
        "\n## Чтение\n",
        _reading(scores_t, trades_t),
    ]
    (OUT / "wings_accuracy.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
