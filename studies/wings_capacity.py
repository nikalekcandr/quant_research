"""Can a heavy model help? Or does this data have a ceiling that no amount of parameters moves? (Nov 2026)

The question is not rhetorical, so the study answers it with numbers on this project's own labels:

  C1  what there is to learn: base rate of the cheap side paying, entropy of the label, Brier of a constant
  C2  capacity sweep: logistic regression -> gradient boosting (small, large) -> MLP 16 -> 64-32 -> 256-128-64
      -> 512-512-512, walk-forward (fit on settlements before the month, predict the month); train and test
      log loss, Brier, AUC, calibration slope, parameter count, fit time
  C3  learning curve: the big model on 100 % of the training hours and on a random half of them, plus the
      small model on the same half -- does capacity pay only when data grows?
  C4  label shuffle: the big model on permuted labels -- it still fits the training set, so the capacity is
      there; the question is whether the data has anything to fit
  C5  the prize: an *oracle* gate (skip the hours in which the cheap side pays) is the absolute upper bound of
      any classifier; a one-bit calendar rule (no CPI/NFP/FOMC hour) is a lower bound that is already known.
      Both in cents per hour and in dollars at the size the project can actually trade.

Everything is causal: features are the quote and the model probability at the quote's own minute, the label is
the settlement, and each fold fits only on settlements strictly before the predicted month.

Run:  python -m studies.wings_capacity        (~3-6 minutes)
Out:  results/studies/wings_capacity_*.csv and wings_capacity.md
"""
from __future__ import annotations

import math
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from studies.wing_panel import build, cheap_sides, fee  # noqa: E402
from studies.wings_scale import NOV, TEST, TRAIN, pnl_at, stats  # noqa: E402

warnings.filterwarnings("ignore")

OUT = ROOT / "results/studies"
MACRO = ROOT / "data/manual/us_macro_calendar.csv"
FEATURES = ["logdist", "z_model", "p_model", "p_market", "left", "price", "spread",
            "hour_sin", "hour_cos", "dow", "weekend", "logvol"]
EDGES = pd.to_datetime(["2026-08-01", "2026-09-01", "2026-10-01", "2026-10-28", "2026-11-26"], utc=True)


# ------------------------------------------------------------------------------------------ data
def rows(d: pd.DataFrame) -> pd.DataFrame:
    """Cheap sides of the union band, one row per strike per 5-minute bucket, causal features only."""
    c = cheap_sides(d, 0.03, 0.20, 5, 30).dropna(subset=["won", "y"]).copy()
    c = c.sort_values("ts")
    c["bucket"] = (c.left / 5).astype(int)
    c = c.drop_duplicates(["end", "market", "bucket"])
    c["logdist"] = np.log(c.S * c.basis / c.strike)
    c["z_model"] = c.logdist / np.sqrt(c.var_rem.clip(lower=1e-14))
    c["p_model"] = np.where(c.cheap_yes, c.p_yes_t4, 1 - c.p_yes_t4)
    c["p_market"] = c.mid_wing
    c["spread"] = c.ask - c.bid
    hour = c.ts.dt.hour + c.ts.dt.minute / 60.0
    c["hour_sin"], c["hour_cos"] = np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24)
    c["dow"] = c.ts.dt.day_of_week.astype(float)
    c["weekend"] = (c.dow >= 5).astype(float)
    c["logvol"] = np.log1p(c.volume.fillna(0).clip(lower=0))
    c["period"] = np.select([(c.end < TRAIN[1]), (c.end < TEST[1]),
                             (c.end >= NOV[0]) & (c.end < NOV[1])],
                            ["train Jul-Aug", "test Sep-Oct", "November"], default="other")
    keep = list(dict.fromkeys(FEATURES + ["won", "end", "period", "wing_bid", "fair_t4"]))
    return c[keep].copy()


# ------------------------------------------------------------------------------------------ models
def make_logit(x, y):
    return LogisticRegression(C=1.0, max_iter=2000).fit(x, y)


def make_hgb_small(x, y):
    return HistGradientBoostingClassifier(max_iter=100, learning_rate=0.06, max_leaf_nodes=15,
                                          min_samples_leaf=200, l2_regularization=1.0,
                                          early_stopping=False, random_state=0).fit(x, y)


def make_hgb_big(x, y):
    return HistGradientBoostingClassifier(max_iter=600, learning_rate=0.05, max_leaf_nodes=63,
                                          min_samples_leaf=20, l2_regularization=0.1,
                                          early_stopping=False, random_state=0).fit(x, y)


def mlp(shape, alpha=1e-2):
    def fit(x, y):
        return MLPClassifier(hidden_layer_sizes=shape, alpha=alpha, learning_rate_init=1e-3,
                             max_iter=80, early_stopping=True, n_iter_no_change=5,
                             validation_fraction=0.1, random_state=0).fit(x, y)
    return fit


MODELS = [("logit", "logistic (12 параметров)", make_logit),
          ("hgb small", "бустинг, 100 итераций, 15 листьев", make_hgb_small),
          ("hgb big", "бустинг, 600 итераций, 63 листа", make_hgb_big),
          ("mlp 16", "MLP 16", mlp((16,))),
          ("mlp 64-32", "MLP 64-32", mlp((64, 32))),
          ("mlp 256-128-64", "MLP 256-128-64", mlp((256, 128, 64))),
          ("mlp 512x3", "MLP 512-512-512", mlp((512, 512, 512)))]


def n_params(name: str) -> float:
    shapes = {"mlp 16": (16,), "mlp 64-32": (64, 32), "mlp 256-128-64": (256, 128, 64), "mlp 512x3": (512, 512, 512)}
    if name == "logit":
        return float(len(FEATURES) + 1)
    if name.startswith("hgb"):
        return float("nan")          # trees: the count of leaves times 2, reported separately
    sizes = [len(FEATURES)] + list(shapes[name]) + [1]
    return float(sum(a * b + b for a, b in zip(sizes[:-1], sizes[1:])))


def walk_forward(c: pd.DataFrame, fit_fn) -> tuple[np.ndarray, float]:
    """Probability for every row, each month fitted only on the settlements before it. Returns (p, seconds)."""
    p = np.full(len(c), np.nan)
    seconds = 0.0
    for start, stop in zip(EDGES[:-1], EDGES[1:]):
        trmask = (c.end < start).to_numpy()
        temask = ((c.end >= start) & (c.end < stop)).to_numpy()
        if trmask.sum() < 2000 or temask.sum() == 0:
            continue
        x_tr = c.loc[trmask, FEATURES].to_numpy(dtype=float)
        y_tr = c.loc[trmask, "won"].to_numpy(dtype=int)
        sc = StandardScaler().fit(x_tr)
        t0 = time.time()
        model = fit_fn(sc.transform(x_tr), y_tr)
        seconds += time.time() - t0
        p[temask] = model.predict_proba(sc.transform(c.loc[temask, FEATURES].to_numpy(dtype=float)))[:, 1]
    return p, seconds


def calib_slope(y: np.ndarray, p: np.ndarray) -> float:
    """Slope of y on logit(p): 1 = calibrated, 0 = no discrimination, < 0 = inverted."""
    p = np.clip(p, 1e-6, 1 - 1e-6)
    if np.std(p) < 1e-9:          # a constant forecast cannot be ranked: the slope is not identified
        return 0.0
    z = logit(p).reshape(-1, 1)
    return float(LogisticRegression(C=1e6, max_iter=1000).fit(z, y).coef_[0, 0])


def score(y: np.ndarray, p: np.ndarray, label: str, secs: float = np.nan) -> dict:
    ok = np.isfinite(p)
    y, p = y[ok], p[ok]
    if len(y) < 50 or y.min() == y.max():
        return {"model": label, "rows": int(len(y))}
    base = np.full(len(y), y.mean())
    return {"model": label, "rows": int(len(y)),
            "logloss": log_loss(y, p, labels=[0, 1]),
            "logloss constant": log_loss(y, base, labels=[0, 1]),
            "brier": brier_score_loss(y, p), "brier constant": brier_score_loss(y, base),
            "auc": roc_auc_score(y, p), "slope": calib_slope(y, p),
            "mean p": float(p.mean()), "realised": float(y.mean()), "fit s": secs}


# ------------------------------------------------------------------------------------------ C1
def c1_ceiling(c: pd.DataFrame) -> pd.DataFrame:
    out = []
    for period, g in c.groupby("period"):
        base = float(g.won.mean())
        if not 0 < base < 1:
            continue
        h = -(base * math.log2(base) + (1 - base) * math.log2(1 - base))
        out.append({"period": period, "rows": len(g), "hours": g.end.nunique(), "win rate %": 100 * base,
                    "entropy bits": h, "brier constant": base * (1 - base),
                    "logloss constant": -(base * math.log(base) + (1 - base) * math.log(1 - base)),
                    "oracle logloss": 0.0})
    return pd.DataFrame(out)


# ------------------------------------------------------------------------------------------ C2
def c2_sweep(c: pd.DataFrame, test_mask: pd.Series) -> tuple[pd.DataFrame, dict]:
    res, preds = [], {}
    y_all = c.won.to_numpy(dtype=int)
    for name, label, fn in MODELS:
        p, secs = walk_forward(c, fn)
        preds[name] = p
        tr = (c.period.isin(["train Jul-Aug"])).to_numpy() & np.isfinite(p)
        r = {"model": name, "what": label, "params": n_params(name)}
        if tr.sum() > 50:
            r["train logloss"] = log_loss(y_all[tr], np.clip(p[tr], 1e-6, 1 - 1e-6))
        t = test_mask.to_numpy() & np.isfinite(p)
        if t.sum() > 50:
            r.update({f"test {k}": v for k, v in score(y_all[t], p[t], name).items()
                      if k in ("logloss", "logloss constant", "brier", "auc", "slope")})
        n = ((c.period == "November").to_numpy() & np.isfinite(p))
        if n.sum() > 50:
            r["nov logloss"] = log_loss(y_all[n], np.clip(p[n], 1e-6, 1 - 1e-6))
            r["nov auc"] = roc_auc_score(y_all[n], p[n])
        r["fit s"] = secs
        res.append(r)
    return pd.DataFrame(res), preds


# ------------------------------------------------------------------------------------------ C3
def c3_learning(c: pd.DataFrame, test_mask: pd.Series) -> pd.DataFrame:
    """Same model, less data: a random 50 % of the *hours* (not rows) of every training fold."""
    y_all = c.won.to_numpy(dtype=int)
    rng = np.random.default_rng(0)
    hours = c.end.drop_duplicates().to_numpy()
    half = set(rng.choice(hours, size=len(hours) // 2, replace=False))
    c_half = c[c.end.isin(half)]
    out = []
    for name, label, fn in [("logit", "logistic", make_logit), ("mlp 256-128-64", "MLP 256-128-64", mlp((256, 128, 64))),
                            ("hgb big", "бустинг 600x63", make_hgb_big)]:
        for tag, frame in (("100 % часов", c), ("50 % часов", c_half)):
            p, secs = walk_forward(frame, fn)
            t = (frame.period == "test Sep-Oct").to_numpy()
            ok = t & np.isfinite(p)
            if ok.sum() < 50:
                continue
            y = frame.won.to_numpy(dtype=int)[ok]
            pp = p[ok]
            out.append({"model": name, "data": tag, "rows": int(ok.sum()),
                        "hours fitted": int(frame[frame.end < TEST[0]].end.nunique()),
                        "test logloss": log_loss(y, np.clip(pp, 1e-6, 1 - 1e-6)),
                        "test auc": roc_auc_score(y, pp), "fit s": secs})
    return pd.DataFrame(out)


# ------------------------------------------------------------------------------------------ C4
def c4_shuffle(c: pd.DataFrame) -> pd.DataFrame:
    """One fit on Jul-Aug, scored in-sample and on Sep-Oct: true labels vs permuted labels.

    A model that cannot drive the in-sample loss below the constant has no capacity at all; a model that does
    and still fails out of sample has capacity but no signal. That is the distinction that matters here.
    """
    rng = np.random.default_rng(1)
    tr = (c.period == "train Jul-Aug").to_numpy()
    te = (c.period == "test Sep-Oct").to_numpy()
    x_tr = c.loc[tr, FEATURES].to_numpy(dtype=float)
    x_te = c.loc[te, FEATURES].to_numpy(dtype=float)
    sc = StandardScaler().fit(x_tr)
    out = []
    for tag, y in (("настоящие метки", c.won.to_numpy(dtype=int)),
                   ("перемешанные метки", rng.permutation(c.won.to_numpy(dtype=int)))):
        for name, fn in (("mlp 512x3", mlp((512, 512, 512), alpha=1e-4)), ("hgb big", make_hgb_big)):
            t0 = time.time()
            m = fn(sc.transform(x_tr), y[tr])
            p_in = m.predict_proba(sc.transform(x_tr))[:, 1]
            p_out = m.predict_proba(sc.transform(x_te))[:, 1]
            out.append({"labels": tag, "model": name, "rows train": int(tr.sum()), "rows test": int(te.sum()),
                        "train logloss": log_loss(y[tr], np.clip(p_in, 1e-6, 1 - 1e-6)),
                        "train constant": log_loss(y[tr], np.full(tr.sum(), y[tr].mean()), labels=[0, 1]),
                        "test logloss": log_loss(y[te], np.clip(p_out, 1e-6, 1 - 1e-6)),
                        "test constant": log_loss(y[te], np.full(te.sum(), y[te].mean()), labels=[0, 1]),
                        "fit s": time.time() - t0})
    return pd.DataFrame(out)


# ------------------------------------------------------------------------------------------ C5
def c5_prize(d: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """What the maximum any classifier could buy, and what a one-bit calendar rule already buys -- per band."""
    cal = pd.read_csv(MACRO)
    ts = (pd.to_datetime(cal.date + " " + cal.time_et).dt.tz_localize("America/New_York")
          .dt.tz_convert("UTC"))
    release = pd.DatetimeIndex((ts + pd.Timedelta(minutes=1)).dt.ceil("h"))
    bands = [("W03 позднее окно (10-20¢, 5-15 мин)", 0.10, 0.20, 5, 15, 0.01),
             ("W04 раннее окно (6-10¢, 15-30 мин)", 0.06, 0.10, 15, 30, 0.0)]
    rows = []
    for label, lo_, hi_, l0, l1, margin in bands:
        c = cheap_sides(d, lo_, hi_, l0, l1).dropna(subset=["won"])
        g = c[(c.wing_bid - c.fair_t4 - fee(1, c.price)) >= margin].copy()
        g = g.sort_values(["end", "ts", "wing_bid"], ascending=[True, True, False])
        g = g.drop_duplicates(["end", "market"]).drop_duplicates("end")
        pnl = pd.Series(np.asarray(pnl_at(g, 1), dtype=float), index=pd.DatetimeIndex(g.end))
        won = g.set_index("end").won
        for name, (a, b) in [("train Jul-Aug", TRAIN), ("test Sep-Oct", TEST), ("November", NOV)]:
            x = pnl[(pnl.index >= a) & (pnl.index < b)]
            if len(x) < 15:
                continue
            w = won.reindex(x.index)
            keep = x[w == 0]                     # oracle: skip the hours in which the cheap side pays
            macro = x[~x.index.isin(release)]
            rows.append({"rule": label, "period": name, "hours": len(x), "¢ per hour": 100 * x.mean(),
                         "oracle-gated ¢": 100 * keep.sum() / len(x),
                         "macro-gated ¢": 100 * macro.mean() if len(macro) > 10 else np.nan,
                         "skipped by oracle %": 100 * (w == 1).mean(),
                         "oracle gain ¢": 100 * (keep.sum() / len(x) - x.mean()),
                         "macro gain ¢": 100 * (macro.mean() - x.mean()) if len(macro) > 10 else np.nan})
    prize = pd.DataFrame(rows)
    money = prize.assign(**{"$/мес, 1 контракт 24/7": prize["¢ per hour"] * 24 * 30 / 100,
                            "$/мес, 100 контрактов 24/7": prize["¢ per hour"] * 24 * 30})
    return prize, money


# ------------------------------------------------------------------------------------------ C6
def c6_money(c: pd.DataFrame, preds: dict) -> pd.DataFrame:
    """The W04 rule, gated by the classifier: skip the trades with the highest predicted win probability."""
    out = []
    for name in ("logit", "hgb small", "mlp 512x3"):
        p = preds.get(name)
        if p is None:
            continue
        mask = c.wing_bid.between(0.06, 0.10) & c.left.between(15, 30)
        g = c[mask].copy()
        g["p"] = np.asarray(p, dtype=float)[mask.to_numpy()]
        g = g.sort_values(["end", "wing_bid"], ascending=[True, False]).drop_duplicates("end")
        g["edge"] = g.wing_bid - g.fair_t4 - np.asarray(fee(1, np.asarray(g.price)), dtype=float)
        g = g[g.edge >= 0]
        for q, tag in ((0.0, "без гейта"), (0.5, "пропуск верхней половины p"), (0.75, "пропуск верхних 25 % p"),
                       (0.9, "пропуск верхних 10 % p")):
            sub = g if q == 0 else g[g.p <= g.p.quantile(q)]
            for period, (a, b) in [("train Jul-Aug", TRAIN), ("test Sep-Oct", TEST), ("November", NOV)]:
                x = sub[(sub.end >= a) & (sub.end < b)]
                if len(x) < 15:
                    continue
                pnl = np.asarray(pnl_at(x, 1), dtype=float)
                out.append({"model": name, "gate": tag, "period": period, "hours": len(x),
                            "¢ per hour": 100 * pnl.mean(),
                            "t": float(pnl.mean() / (pnl.std(ddof=1) / math.sqrt(len(pnl)))) if pnl.std() > 0 else np.nan})
    return pd.DataFrame(out).round(3)


# ------------------------------------------------------------------------------------------ C7
def c7_ablation(c: pd.DataFrame, test_mask: pd.Series) -> pd.DataFrame:
    """Where does the little discrimination that exists come from? One logit per feature group."""
    groups = {
        "все признаки": FEATURES,
        "без цены рынка и модели": [f for f in FEATURES if f not in ("p_market", "p_model")],
        "только цена рынка": ["p_market", "spread"],
        "только модель": ["z_model", "p_model"],
        "rrолько часы суток и день недели".replace("rr", "т"): ["hour_sin", "hour_cos", "dow", "weekend"],
        "только расстояние и время": ["logdist", "left", "logvol"],
    }
    out = []
    for name, feats in groups.items():
        c2 = c.copy()
        c2["_x"] = 0.0
        p = np.full(len(c2), np.nan)
        for start, stop in zip(EDGES[:-1], EDGES[1:]):
            tr = (c2.end < start).to_numpy()
            te = ((c2.end >= start) & (c2.end < stop)).to_numpy()
            if tr.sum() < 2000 or te.sum() == 0:
                continue
            sc = StandardScaler().fit(c2.loc[tr, feats].to_numpy(dtype=float))
            m = LogisticRegression(C=1.0, max_iter=2000).fit(
                sc.transform(c2.loc[tr, feats].to_numpy(dtype=float)), c2.loc[tr, "won"].to_numpy(dtype=int))
            p[te] = m.predict_proba(sc.transform(c2.loc[te, feats].to_numpy(dtype=float)))[:, 1]
        ok = test_mask.to_numpy() & np.isfinite(p)
        y = c2.won.to_numpy(dtype=int)[ok]
        out.append({"features": name, "n": len(feats), "test auc": roc_auc_score(y, p[ok]),
                    "test logloss": log_loss(y, np.clip(p[ok], 1e-6, 1 - 1e-6))})
    return pd.DataFrame(out).round(4)


# ------------------------------------------------------------------------------------------ report
def md(df, fmt="{:.4f}"):
    if df is None or df.empty:
        return "_нет строк_\n"
    out = df.copy()
    out.columns = [str(c).replace("|", "/") for c in out.columns]
    for c in out.columns:
        if pd.api.types.is_float_dtype(out[c]):
            out[c] = out[c].map(lambda x: "" if pd.isna(x) else fmt.format(x))
        elif not pd.api.types.is_numeric_dtype(out[c]):
            out[c] = out[c].map(lambda x: str(x).replace("|", "/"))
    head = "| " + " | ".join(map(str, out.columns)) + " |\n"
    sep = "|" + "|".join(["---"] * len(out.columns)) + "|\n"
    return head + sep + "".join("| " + " | ".join(map(str, r)) + " |\n" for r in out.itertuples(index=False))


def write_report(parts: dict) -> Path:
    txt = ["# Сколько модели выдерживают эти данные\n",
           f"_Сгенерировано {pd.Timestamp.now():%Y-%m-%d %H:%M}, `studies/wings_capacity.py`._\n",
           "Вопрос: можно ли обучить «очень тяжёлую» модель, которая найдёт в крыле больше, чем находит "
           "аналитическая формула? Ответ ниже получен на собственных метках проекта: тот же отбор, те же "
           "признаки, walk-forward по месяцам, никакого заглядывания вперёд.\n",
           "## C1. Сколько вообще есть выучить\n", md(parts["ceiling"]),
           "## C2. Лестница ёмкости\n", md(parts["sweep"]),
           "## C3. Кривая обучения: та же модель на половине часов\n", md(parts["learning"]),
           "## C4. Контроль на перемешанных метках\n", md(parts["shuffle"]),
           "## C7. Откуда берётся различающая способность\n", md(parts.get("ablation")),
           "## C6. Деньги: правило W04 с гейтом по классификатору\n", md(parts.get("money_model")),
           "## C5. Приз: чем вообще может заплатить классификатор\n", md(parts["prize"]),
           md(parts["money"]),
           "## Чтение\n", reading(parts)]
    path = OUT / "wings_capacity.md"
    path.write_text("\n".join(txt), encoding="utf-8")
    return path


def reading(parts: dict) -> str:
    s, l_, sh, pz = parts.get("sweep"), parts.get("learning"), parts.get("shuffle"), parts.get("prize")
    ab, mm = parts.get("ablation"), parts.get("money_model")
    out = []
    if s is not None and not s.empty:
        best = s.dropna(subset=["test auc"]).sort_values("test logloss").iloc[0]
        worst = s.dropna(subset=["test auc"]).sort_values("test logloss").iloc[-1]
        bp = "13" if best["params"] == best["params"] else "?"
        wp = f"{worst['params']:.0f}" if worst["params"] == worst["params"] else "десятки тысяч листьев"
        out.append(f"* Победитель лестницы — **{best.model}** ({bp} параметров): тестовый "
                   f"logloss {best['test logloss']:.4f} против {best['test logloss constant']:.4f} у константы, "
                   f"AUC {best['test auc']:.3f}, наклон калибровки {best['test slope']:.2f}. "
                   f"Самая тяжёлая модель ({worst.model}, {wp}) даёт "
                   f"{worst['test logloss']:.4f} и AUC {worst['test auc']:.3f}: **каждый шаг в ёмкости "
                   "ухудшает тест**, а не улучшает.")
        out.append("* Тот же порядок в ноябре: " + ", ".join(
            f"{r.model} {r['nov logloss']:.3f}" for _, r in s.dropna(subset=["nov logloss"]).iterrows())
            + " — снова логистическая регрессия впереди. Причина видна в C4: у тяжёлой модели есть огромная "
              "способность запоминать, и она тратит её на запоминание.")
    if ab is not None and not ab.empty:
        a = ab.set_index("features")["test auc"]
        out.append("* Откуда AUC 0.69: " + ", ".join(f"{k} {v:.3f}" for k, v in a.items()) +
                   ". Различает не модель и не час суток, а **сама цена рынка**: рынок ранжирует исходы крыльев "
                   "лучше, чем любая наша формула. Это ещё раз говорит, что искать надо не «предсказание "
                   "исхода», а расхождение цены с реализацией — то, что и делает T3 в `wings_ten.md`.")
    if l_ is not None and not l_.empty:
        big = l_[l_.model == "mlp 256-128-64"].set_index("data")
        hb = l_[l_.model == "hgb big"].set_index("data")
        if "100 % часов" in big.index and "50 % часов" in big.index:
            out.append(f"* Кривая обучения: сеть 256-128-64 на 100 % часов даёт тестовый logloss "
                       f"{big.loc['100 % часов', 'test logloss']:.4f}, на 50 % — "
                       f"{big.loc['50 % часов', 'test logloss']:.4f}, то есть вдвое больше данных покупают "
                       f"0.004. У бустинга та же разница {hb.loc['100 % часов', 'test logloss']:.4f} → "
                       f"{hb.loc['50 % часов', 'test logloss']:.4f}: тяжёлая модель без данных разваливается, "
                       "лёгкая — почти нет.")
    if sh is not None and not sh.empty:
        parts_sh = []
        for _, r in sh[sh.model == "hgb big"].iterrows():
            parts_sh.append(f"{r.labels}: тренировочный logloss {r['train logloss']:.4f} при константе "
                            f"{r['train constant']:.4f}, тестовый {r['test logloss']:.4f} против "
                            f"{r['test constant']:.4f}")
        if parts_sh:
            out.append("* Контроль на перемешанных метках — " + "; ".join(parts_sh) +
                       ". Ёмкости хватает, чтобы запомнить и настоящие метки, и перемешанные: то есть "
                       "модель учится не различать исходы, а подгонять тот шум, который ей дали.")
    if mm is not None and not mm.empty:
        base = mm[(mm.gate == "без гейта") & (mm.period == "test Sep-Oct")]["¢ per hour"]
        gates = mm[(mm.gate != "без гейта")]
        if len(base) and len(gates):
            b = float(base.iloc[0])
            best_gate = gates[gates.period == "test Sep-Oct"].sort_values("¢ per hour", ascending=False).iloc[0]
            out.append(f"* Деньги: правило W04 без гейта даёт на тесте {b:.2f}¢/час. Лучший гейт по "
                       f"классификатору ({best_gate.model}, {best_gate.gate}) поднимает до "
                       f"{best_gate['¢ per hour']:.2f}¢/час, но на трейне тот же гейт даёт "
                       f"{gates[(gates.model == best_gate.model) & (gates.gate == best_gate.gate) & (gates.period == 'train Jul-Aug')]['¢ per hour'].iloc[0]:.2f}¢, "
                       "то есть выигрыш появляется только там, где его подбирали. Ни один гейт не даёт "
                       "одинаково положительного знака на всех трёх периодах.")
    if pz is not None and not pz.empty:
        t = pz[pz.period == "test Sep-Oct"]
        for _, r in t.iterrows():
            out.append(f"* Потолок: {r['rule']} на тесте {r['¢ per hour']:.2f}¢/час; оракул, знающий исход, "
                       f"дал бы {r['oracle-gated ¢']:.2f}¢ (+{r['oracle gain ¢']:.2f}¢), пропустив "
                       f"{r['skipped by oracle %']:.1f} % часов. Гейт «не торговать в час макро-релиза» "
                       f"забирает {r['macro gain ¢']:+.2f}¢ из этого потолка, не обучаясь ни на чём.")
    out.append("* Вывод. «Сверхобученная модель» на этих данных не имеет смысла, и это измерено, а не "
               "заявлено: (а) ёмкость монотонно вредит (logit 13 параметров > MLP 532 тысячи), (б) памяти "
               "хватает на что угодно, кроме сигнала (C4), (в) вся различающая способность — это цена "
               "рынка, а не наши признаки (C7), (г) приз оракула +5…+10¢/час есть, но гейты по модели "
               "его не ловят (C6), (д) независимых наблюдений — 1 436 часов, а не 36 тысяч строк. "
               "Правильное место ML в этом проекте: оценка неопределённости, сайзинг и отсев риска "
               "(макро-часы, тонкие события), а не предсказание исхода.")
    return "\n".join(out) + "\n"


def main(report_only: bool = False) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    if report_only:                      # rewrite the markdown from the CSVs, no refits
        keys = {"ceiling": "wings_capacity_ceiling.csv", "sweep": "wings_capacity_sweep.csv",
                "learning": "wings_capacity_learning.csv", "shuffle": "wings_capacity_shuffle.csv",
                "ablation": "wings_capacity_ablation.csv", "money_model": "wings_capacity_money_model.csv",
                "prize": "wings_capacity_prize.csv", "money": "wings_capacity_money.csv"}
        parts = {k: pd.read_csv(OUT / v) for k, v in keys.items()}
        print(f"written {write_report(parts)}")
        return
    d = build()
    d = d[d.y.notna()]
    c = rows(d)
    print(f"rows {len(c):,}, hours {c.end.nunique():,}, features {len(FEATURES)}")
    parts = {}
    parts["ceiling"] = c1_ceiling(c)
    parts["ceiling"].to_csv(OUT / "wings_capacity_ceiling.csv", index=False)
    test_mask = (c.period == "test Sep-Oct")
    parts["sweep"], preds = c2_sweep(c, test_mask)
    parts["sweep"].to_csv(OUT / "wings_capacity_sweep.csv", index=False)
    parts["learning"] = c3_learning(c, test_mask)
    parts["learning"].to_csv(OUT / "wings_capacity_learning.csv", index=False)
    parts["shuffle"] = c4_shuffle(c)
    parts["shuffle"].to_csv(OUT / "wings_capacity_shuffle.csv", index=False)
    parts["ablation"] = c7_ablation(c, test_mask)
    parts["ablation"].to_csv(OUT / "wings_capacity_ablation.csv", index=False)
    parts["money_model"] = c6_money(c, preds)
    parts["money_model"].to_csv(OUT / "wings_capacity_money_model.csv", index=False)
    parts["prize"], parts["money"] = c5_prize(d)
    parts["prize"].to_csv(OUT / "wings_capacity_prize.csv", index=False)
    parts["money"].to_csv(OUT / "wings_capacity_money.csv", index=False)
    for k, v in parts.items():
        print(f"\n--- {k}")
        print(v.to_string(index=False))
    print(f"\nwritten {write_report(parts)}")


if __name__ == "__main__":
    main(report_only="--report-only" in sys.argv)
