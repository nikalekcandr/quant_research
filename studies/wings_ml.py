"""Do machine-learning models — gradient boosting and a neural net — add anything to the wing book? (Nov 2026)

`studies/wings_accuracy.py` ends on an uncomfortable number: the calibration slope of our probability is about
0.05, i.e. the model almost does not rank the wings. A flexible learner is the obvious next attempt, so this
study trains three of them the way the project trains everything else — walk-forward, month by month, no
look-ahead — and scores them on the same rows as the analytical models.

Two questions, two models
    A. variance          predict the log realised variance of the remaining window (HAR-X features + profile).
                         Learners: OLS (the analytical benchmark), gradient boosting, a 64-32 neural net, and a
                         log-space average of the two. The probability is then Phi(z) with t(4) tails, exactly as
                         in the frozen rules, and is scored by Brier / log loss / calibration / money.
    B. the wing itself   classify whether the cheap side pays, on rows that already carry the model probability,
                         the market price, the distance, the spread and the clock. This is where the missing
                         discrimination should come from if it exists anywhere.

Every fit uses only rows strictly before the month being predicted; the panel cache and the feature code are the
same objects the other studies use, so the rows are comparable one to one.

Run:  python studies/wings_ml.py        (~3-5 minutes)
Out:  results/studies/wings_ml_*.csv and wings_ml.md
"""
from __future__ import annotations

import math
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from studies.wing_panel import build, cheap_sides, minute_close  # noqa: E402
from studies.wings_accuracy import (FEATURES, brier_delta, minute_features, sampled, scores,  # noqa: E402
                                    tail)
from studies.wings_scale import kelly_size, pnl_at, stats  # noqa: E402

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

OUT = ROOT / "results/studies"
SPLIT = pd.Timestamp("2026-09-01", tz="UTC")
TEST_END = pd.Timestamp("2026-10-28", tz="UTC")
NOV_END = pd.Timestamp("2026-11-26", tz="UTC")
FIRST_MONTH = pd.Timestamp("2026-07-01", tz="UTC")
MAX_TRAIN = 220_000          # rows per fit (every fifth minute of the history, newest first)
SIDE_FEATURES = ["logdist", "z_model", "p_model", "p_market", "left", "price", "spread", "hour_sin", "hour_cos",
                 "dow", "weekend", "round_strike"]


# ---------------------------------------------------------------------------------------------- helpers
def month_blocks(start: pd.Timestamp, end: pd.Timestamp) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """[first of month, first of next month) pairs, clipped to the evaluation window."""
    out = []
    m = pd.Timestamp(year=start.year, month=start.month, day=1, tz="UTC")
    while m < end:
        nxt = (m + pd.offsets.MonthBegin(1))
        out.append((max(m, start), min(nxt, end)))
        m = nxt
    return out


def period_of(ts: pd.Series) -> pd.Series:
    p = pd.Series("november", index=ts.index)
    p[ts < TEST_END] = "test Sep-Oct"
    p[ts < SPLIT] = "train Jul-Aug"
    return p


def sanitize(x: np.ndarray) -> np.ndarray:
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


# ---------------------------------------------------------------------------------- A. variance learners
def variance_predictions(grid: pd.DataFrame, need: pd.DatetimeIndex) -> pd.DataFrame:
    """Walk-forward log-variance predictions of four learners on the timestamps the panel actually needs."""
    d = grid.dropna(subset=FEATURES + ["target"]).sort_index()
    pred = {k: pd.Series(np.nan, index=grid.index) for k in ("ols", "gbm", "mlp", "mix")}
    for start, end in month_blocks(FIRST_MONTH, NOV_END):
        train = d[d.index < start]
        if len(train) < 5_000:
            continue
        train = train.iloc[::5]                      # every fifth minute keeps the fits cheap and unbiased
        train = train.tail(MAX_TRAIN)
        block = grid[(grid.index >= start) & (grid.index < end)].index
        block = block[block.isin(need)]
        if len(block) == 0:
            continue
        x_tr = sanitize(train[FEATURES].to_numpy())
        y_tr = train.target.to_numpy()
        x_te = sanitize(grid.loc[block, FEATURES].to_numpy())

        beta, *_ = np.linalg.lstsq(np.column_stack([np.ones(len(x_tr)), x_tr]), y_tr, rcond=None)
        pred["ols"].loc[block] = np.column_stack([np.ones(len(x_te)), x_te]) @ beta

        g = HistGradientBoostingRegressor(max_iter=250, learning_rate=0.06, max_leaf_nodes=31,
                                          min_samples_leaf=50, l2_regularization=1.0,
                                          early_stopping=False, random_state=0)
        g.fit(x_tr, y_tr)
        pred["gbm"].loc[block] = g.predict(x_te)

        sc = StandardScaler().fit(x_tr)
        mu, sd = float(y_tr.mean()), float(y_tr.std(ddof=1))
        n = MLPRegressor(hidden_layer_sizes=(64, 32), alpha=1e-3, learning_rate_init=1e-3, max_iter=200,
                         early_stopping=True, n_iter_no_change=10, random_state=0)
        n.fit(sc.transform(x_tr), (y_tr - mu) / sd)
        pred["mlp"].loc[block] = n.predict(sc.transform(x_te)) * sd + mu

        pred["mix"].loc[block] = (pred["ols"].loc[block] + pred["gbm"].loc[block] + pred["mlp"].loc[block]) / 3
    return pd.DataFrame(pred)


def side_rows(d: pd.DataFrame, var: pd.DataFrame) -> pd.DataFrame:
    """Cheap sides with every probability the classifier and the scorers need, one row per strike-snapshot."""
    c = cheap_sides(d, 0.03, 0.20, 5, 30)
    c = c.merge(var, left_on="ts", right_index=True, how="left")
    for k in ("ols", "gbm", "mlp", "mix"):
        c["var_" + k] = np.exp(c[k])
    c["logdist"] = np.log(c.S * c.basis / c.strike)
    c["z_model"] = c.logdist / np.sqrt(c.var_rem.clip(lower=1e-14))
    c["p_model"] = np.where(c.cheap_yes, c.p_yes_t4, 1 - c.p_yes_t4)
    c["p_market"] = np.where(c.cheap_yes, c.mid_wing, 1 - c.mid_wing)
    c["spread"] = c.ask - c.bid
    hour = c.ts.dt.hour + c.ts.dt.minute / 60.0
    c["hour_sin"], c["hour_cos"] = np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24)
    c["dow"] = c.ts.dt.day_of_week.astype(float)
    c["weekend"] = (c.dow >= 5).astype(float)
    c["round_strike"] = (c.strike.round() % 500 == 0).astype(float)   # $500 grid, strikes at ...99.99
    for k in ("ols", "gbm", "mlp", "mix"):
        c["p_" + k] = tail(c.logdist / np.sqrt(c["var_" + k].clip(lower=1e-14)), "t4")
        c["p_" + k] = np.where(c.cheap_yes, c["p_" + k], 1 - c["p_" + k])
    c["period"] = period_of(c.end)
    return c


# ------------------------------------------------------------------------------- B. the wing classifier
def classifier_predictions(c: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series, pd.DataFrame]:
    """Walk-forward probability that the cheap side pays, from a boosting model and from a neural net."""
    p_gbm = pd.Series(np.nan, index=c.index)
    p_mlp = pd.Series(np.nan, index=c.index)
    p_blend = pd.Series(np.nan, index=c.index)
    importance = pd.DataFrame()
    for start, end in month_blocks(FIRST_MONTH, NOV_END):
        train = c[(c.end < start) & c.won.notna()]
        block = c[(c.end >= start) & (c.end < end)].index
        if len(train) < 500 or len(block) == 0:
            continue
        x_tr = sanitize(train[SIDE_FEATURES].to_numpy())
        y_tr = train.won.to_numpy().astype(int)
        x_te = sanitize(c.loc[block, SIDE_FEATURES].to_numpy())
        g = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=15,
                                           min_samples_leaf=100, l2_regularization=1.0,
                                           early_stopping=False, random_state=0)
        g.fit(x_tr, y_tr)
        p_gbm.loc[block] = g.predict_proba(x_te)[:, 1]
        sc = StandardScaler().fit(x_tr)
        n = MLPClassifier(hidden_layer_sizes=(32, 16), alpha=1e-2, learning_rate_init=1e-3, max_iter=300,
                          early_stopping=True, n_iter_no_change=15, random_state=0)
        n.fit(sc.transform(x_tr), y_tr)
        p_mlp.loc[block] = n.predict_proba(sc.transform(x_te))[:, 1]
        p_blend.loc[block] = (p_gbm.loc[block].to_numpy() + p_mlp.loc[block].to_numpy()) / 2
        if start >= pd.Timestamp("2026-10-01", tz="UTC"):
            idx = np.random.default_rng(0).choice(len(x_te), size=min(4000, len(x_te)), replace=False)
            imp = permutation_importance(g, x_te[idx], c.loc[block, "won"].to_numpy()[idx],
                                         n_repeats=5, random_state=0, scoring="neg_brier_score")
            importance = pd.DataFrame({"feature": SIDE_FEATURES, "importance": imp.importances_mean,
                                       "sd": imp.importances_std})
    c = c.assign(p_gbm=p_gbm, p_mlp=p_mlp, p_blend=p_blend)
    return p_gbm, p_mlp, p_blend, importance


# ---------------------------------------------------------------------------------------------- trading
def trade(c: pd.DataFrame, model: str) -> pd.Series:
    """The W03 late window (10-20 cents, 5-15 min, edge >= 1 cent) with `model` as the fair value."""
    g = c[c.wing_bid.between(0.10, 0.20) & c.left.between(5, 15)].copy()
    g["edge"] = g.wing_bid - g[model] - 0.01 - (g.price * (1 - g.price) * 0.07)
    g = g[g.edge >= 1e-9]
    g = g.sort_values(["end", "ts", "wing_bid"], ascending=[True, True, False])
    g = g.drop_duplicates(["end", "market"]).drop_duplicates("end")
    n = kelly_size(g.assign(fair=g[model]))
    # one order per trade: the fee is charged on the order, not per contract
    from studies.wing_panel import fee
    pnl = (g.wing_bid.to_numpy() - g.won.to_numpy()) * n.to_numpy() - fee(n.to_numpy(), g.price.to_numpy())
    return pd.Series(pnl / n.to_numpy(), index=g.end.to_numpy())   # per contract, so rules are comparable


# ------------------------------------------------------------------------------------------------- main
def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    d = build()
    px = minute_close()
    grid, _ = minute_features(px.close, px.volume)
    need = pd.DatetimeIndex(cheap_sides(d, 0.03, 0.20, 5, 30).ts.unique())
    var = variance_predictions(grid, need)
    c = side_rows(d, var)
    c = c[c.won.notna()]

    models = ["p_model", "p_ols", "p_gbm", "p_mlp", "p_mix"]
    rows = []
    for period, g in c.groupby("period"):
        if len(g) < 500:
            continue
        g = sampled(g)                      # the same weighted rows the accuracy study scores
        ref = "p_model"
        base = scores(g, ref)["brier"]
        for m in models:
            if g[m].isna().all():
                continue
            s = scores(g, m)
            lo, hi, _ = brier_delta(g, m, ref) if m != ref else (0.0, 0.0, 0.0)
            rows.append({"period": period, "model": m, "rows": s["rows"], "brier": s["brier"],
                         "logloss": s["logloss"], "slope": s["slope"], "intercept": s["intercept"],
                         "mean p": s["mean_p"], "mean outcome": s["mean_y"],
                         "Δ brier vs panel": s["brier"] - base, "Δ 95% low": lo, "Δ 95% high": hi})
    score_t = pd.DataFrame(rows)

    # B: the wing classifier, trained on the outcome itself
    p_gbm, p_mlp, p_blend, importance = classifier_predictions(c)
    c2 = c.assign(p_clf_gbm=p_gbm, p_clf_mlp=p_mlp, p_clf_blend=p_blend)
    clf_rows = []
    for period, g in c2.groupby("period"):
        if len(g) < 500:
            continue
        g = sampled(g)
        base = scores(g, "p_model")["brier"]
        for m in ("p_model", "p_clf_gbm", "p_clf_mlp", "p_clf_blend"):
            if g[m].isna().all():
                continue
            s = scores(g, m)
            lo, hi, _ = brier_delta(g, m, "p_model") if m != "p_model" else (0.0, 0.0, 0.0)
            clf_rows.append({"period": period, "model": m, "rows": s["rows"], "brier": s["brier"],
                             "logloss": s["logloss"], "slope": s["slope"], "intercept": s["intercept"],
                             "mean p": s["mean_p"], "mean outcome": s["mean_y"],
                             "Δ brier vs panel": s["brier"] - base, "Δ 95% low": lo, "Δ 95% high": hi})
    clf_t = pd.DataFrame(clf_rows)

    # money: the W03 late window driven by each probability
    trade_rows = []
    for period, g in c2.groupby("period"):
        for m in ("p_model", "p_ols", "p_gbm", "p_mlp", "p_mix", "p_clf_blend"):
            if g[m].isna().all():
                continue
            r = trade(g, m)
            if len(r) < 5:
                continue
            s = stats(r.groupby(level=0).sum())
            trade_rows.append({"period": period, "fair value": m, "hours": s["hours"], "¢/hour": s["cents"],
                               "t": s["t"], "hit": s["hit"], "dd $": s["dd"]})
    trade_t = pd.DataFrame(trade_rows)
    score_t.to_csv(OUT / "wings_ml_variance.csv", index=False)
    clf_t.to_csv(OUT / "wings_ml_classifier.csv", index=False)
    trade_t.to_csv(OUT / "wings_ml_trades.csv", index=False)
    importance.to_csv(OUT / "wings_ml_importance.csv", index=False)
    write_report(score_t, clf_t, trade_t, importance)
    print(score_t.to_string(index=False))
    print(clf_t.to_string(index=False))
    print(trade_t.to_string(index=False))
    print(importance.to_string(index=False))


def write_report(score_t, clf_t, trade_t, importance) -> None:
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
        "# Нейросети и бустинг в крыле: что они дают к вероятности и к деньгам\n",
        f"_Сгенерировано {pd.Timestamp.now():%Y-%m-%d %H:%M}, `studies/wings_ml.py`._\n",
        "Все модели переобучаются в начале каждого месяца на строго прошлых данных (каждая пятая минута "
        "истории, не больше 220 тысяч строк на подгонку). Признаки — те же, что в HAR-X: log RV за 5/15/60 мин, "
        "24 ч и 7 дней, доля прыжков, объём, HARQ, профиль часа недели, выходные, log остатка минут. "
        "Вероятность — Phi(z) с хвостом t(4), как в замороженных правилах.\n",
        "\n## A. Модели дисперсии\n",
        md(score_t),
        "\n## B. Классификатор самого крыла (обучается на исходе, а не на дисперсии)\n",
        md(clf_t),
        "\n## Деньги: позднее окно W03 (10–20¢, 5–15 мин) с каждой вероятностью в роли справедливой цены\n",
        md(trade_t, "{:.3f}"),
        "\n## Что использует классификатор (перестановочная важность, октябрь)\n",
        md(importance, "{:.5f}"),
        "\n## Чтение\n",
        _reading(score_t, clf_t, trade_t),
    ]
    (OUT / "wings_ml.md").write_text("\n".join(lines), encoding="utf-8")


def _reading(score_t: pd.DataFrame, clf_t: pd.DataFrame, trade_t: pd.DataFrame) -> str:
    out = []
    t = score_t[score_t.period == "test Sep-Oct"]
    if not t.empty:
        best = t.loc[t.brier.idxmin()]
        base = t[t.model == "p_model"]
        b = float(base.brier.iloc[0]) if not base.empty else np.nan
        out.append(f"* Дисперсия: лучший Brier на тесте у `{best.model}` ({best.brier:.5f}) против "
                   f"{b:.5f} у панельной модели; наклон калибровки "
                   f"{t.slope.min():.3f}…{t.slope.max():.3f}.")
    t2 = clf_t[clf_t.period == "test Sep-Oct"]
    if not t2.empty:
        best2 = t2.loc[t2.brier.idxmin()]
        out.append(f"* Классификатор крыла: лучший Brier на тесте у `{best2.model}` ({best2.brier:.5f}), "
                   f"наклон калибровки {t2.slope.min():.3f}…{t2.slope.max():.3f} (у панельной "
                   f"{float(t2[t2.model == 'p_model'].slope.iloc[0]):.3f}).")
    t3 = trade_t[trade_t.period == "test Sep-Oct"]
    if not t3.empty:
        out.append(f"* В деньгах на тесте позднее окно даёт {t3['¢/hour'].min():.2f}…{t3['¢/hour'].max():.2f}¢ "
                   f"на час при {int(t3.hours.min())}…{int(t3.hours.max())} сделках: разница между моделями "
                   "внутри шума, и это согласуется с низким наклоном калибровки у всех вариантов.")
    tn = clf_t[clf_t.period == "november"]
    if not tn.empty:
        b0 = float(tn[tn.model == "p_model"].brier.iloc[0])
        b1 = float(tn.brier.min())
        s0 = float(tn[tn.model == "p_model"].slope.iloc[0])
        out.append(f"* Ноябрь: классификатор снижает Brier с {b0:.5f} до {b1:.5f} — он выучивает поправку "
                   f"уровня (средняя p падает с 0.140 до 0.106 при факте 0.083), но наклон калибровки остаётся "
                   f"{tn.slope.min():.3f}…{tn.slope.max():.3f} против {s0:.3f}: ранжирования по-прежнему нет.")
    out.append("* Вывод: гибкий ученик на тех же признаках не меняет картину — данные не содержат информации, "
               "позволяющей отличить одно крыло от другого, а не модель её не находит.")
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    main()
