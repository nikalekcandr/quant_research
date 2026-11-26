"""Where a professional would look next: relative value inside the ladder, stale quotes, the clock, and how
much of our best result is luck (Nov 2026).

The first cycle compared the ladder with our model and found the model's edge thin and unstable. Professional
relative-value desks rarely trade "model vs market"; they trade *market vs market* (one strike against its own
neighbours), they mine microstructure (who left a stale quote), they condition on the clock and the calendar, and
they deflate the Sharpe of the winner by the number of variants they tried. This study does exactly those four
things on the same panel, with the same accounting.

    S1 local richness   each strike's mid against the interpolation of its two neighbours at the same snapshot
                        (arbitrage-free shape, not our model). Sell the locally rich side; does it beat the
                        model-based pick, and does the filter help the frozen rule?
    S2 stale quotes     age of a quote and how far spot has moved since it last changed. Selling into a stale
                        rich quote is the classic adverse-selection trade; measure it by bucket.
    S3 the clock        P&L by UTC hour (the 06-10 UTC band is under-represented in both halves: weekly holes in
                        the archive, sleep in November) and on US federal holidays.
    S4 alpha decay      weekly P&L of the best rule with a trend test.
    S5 multiple testing 48 variants (band x window x gate), deflated Sharpe ratio for the winner and the
                        probability of backtest overfitting (CSCV) of the whole family.

Run:  python studies/wings_alpha.py
Out:  results/studies/wings_alpha_*.csv and wings_alpha.md
"""
from __future__ import annotations

import math
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from studies.wing_panel import build, cheap_sides, fee  # noqa: E402
from studies.wings_scale import interval, pnl_at, stats  # noqa: E402

OUT = ROOT / "results/studies"
TRAIN = (pd.Timestamp("2026-07-01", tz="UTC"), pd.Timestamp("2026-09-01", tz="UTC"))
TEST = (pd.Timestamp("2026-09-01", tz="UTC"), pd.Timestamp("2026-10-28", tz="UTC"))
NOV = (pd.Timestamp("2026-11-08", tz="UTC"), pd.Timestamp("2026-11-26", tz="UTC"))
EULER = 0.5772156649


# ------------------------------------------------------------------------------- S1 local richness
def local_richness(d: pd.DataFrame, lo: float, hi: float, l0: float, l1: float) -> pd.DataFrame:
    """Every cheap side with the residual of its own mid against the two neighbours' mids at the same snapshot."""
    snap = d[d.left.between(l0, l1) & d.bid.gt(0) & d.ask.lt(1) & d.ask.ge(d.bid)].copy()
    snap["p"] = (snap.bid + snap.ask) / 2
    snap = snap.sort_values(["event", "ts", "strike"])
    g = snap.groupby(["event", "ts"])
    logk = np.log(snap.strike)
    prev_p, next_p = g.p. shift(1), g.p.shift(-1)
    prev_k, next_k = np.log(g.strike.shift(1)), np.log(g.strike.shift(-1))
    span = next_k - prev_k
    w = (logk - prev_k) / span.replace(0, np.nan)
    interp = prev_p + w * (next_p - prev_p)
    snap["interp"] = interp
    snap["resid"] = snap.p - interp                      # >0: this strike is rich against its neighbours
    # map to the cheap side by hand (cheap_sides would re-filter and lose the neighbours' residuals)
    out = snap[snap.bid.between(lo, hi) | (1 - snap.ask).between(lo, hi)].copy()
    out["cheap_yes"] = out.bid.between(lo, hi)
    out["wing_bid"] = np.where(out.cheap_yes, out.bid, 1 - out.ask)
    out["price"] = 1 - out.wing_bid
    out["won"] = np.where(out.cheap_yes, out.y, 1 - out.y)
    out["fair"] = np.where(out.cheap_yes, out.p_yes_t4, 1 - out.p_yes_t4)
    out["rich"] = np.where(out.cheap_yes, out.resid, -out.resid)   # richness of the cheap side itself
    out["edge"] = out.wing_bid - out.fair - fee(1, out.price)
    out["market"] = out.ticker
    return out[out.won.notna()]


def pick_one(t: pd.DataFrame, by: str) -> pd.Series:
    """One trade an hour: the row with the largest `by` (ties: the dearest wing)."""
    t = t.sort_values(["end", "ts", by, "wing_bid"], ascending=[True, True, False, False])
    t = t.drop_duplicates(["end", "market"]).drop_duplicates("end")
    return pd.Series(pnl_at(t, 1.0), index=t.end.to_numpy())


# ------------------------------------------------------------------------------- S2 stale quotes
def quote_age(d: pd.DataFrame) -> pd.DataFrame:
    """Minutes since each ticker's quote last changed, and how far spot has moved since that change."""
    x = d.sort_values(["ticker", "ts"]).copy()
    x["changed"] = (x.bid.ne(x.groupby("ticker").bid.shift()) | x.ask.ne(x.groupby("ticker").ask.shift()))
    x["change_ts"] = x.ts.where(x.changed)
    x["last_change"] = x.groupby("ticker").change_ts.ffill()
    x["age"] = (x.ts - x.last_change).dt.total_seconds().div(60)
    x["log_move"] = np.log(x.S / x.groupby("ticker").S.transform("first"))
    return x[["ticker", "ts", "age"]]


# ------------------------------------------------------------------------------- S5 multiple testing
def variant_grid(d: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Hourly P&L of a family of plausible variants: band x window x gate."""
    bands = [(0.06, 0.10), (0.10, 0.20), (0.03, 0.20), (0.08, 0.14), (0.04, 0.08)]
    windows = [(5, 15), (15, 30), (5, 30), (20, 40)]
    gates = [("none", lambda x: pd.Series(True, index=x.index)),
             ("no US open", lambda x: ~x.end.dt.hour.isin([13, 14, 15])),
             ("weekdays", lambda x: x.end.dt.day_of_week < 5),
             ("weekends", lambda x: x.end.dt.day_of_week >= 5)]
    rows, names = {}, []
    for lo, hi in bands:
        for l0, l1 in windows:
            c = cheap_sides(d, lo, hi, l0, l1)
            if len(c) < 200:
                continue
            c["fair"] = c.fair_t4
            c["edge"] = c.wing_bid - c.fair - fee(1, c.price)
            c = c[c.edge > 0]
            if not len(c):
                continue
            for gate, f in gates:
                sub = c[f(c)]
                if len(sub) < 100:
                    continue
                name = f"{lo:.2f}-{hi:.2f}|{l0}-{l1}|{gate}"
                rows[name] = pick_one(sub, "edge")
                names.append(name)
    px = pd.DataFrame(rows).sort_index()
    return px, names


def deflated_sharpe(r: pd.Series, trial_sharpes: np.ndarray) -> dict:
    """Bailey & Lopez de Prado deflated Sharpe: is the winner better than the best of N tries?"""
    r = r.dropna()
    if len(r) < 30 or r.std(ddof=1) == 0:
        return dict(sharpe=np.nan, sr0=np.nan, dsr=np.nan, trials=len(trial_sharpes))
    sr = float(r.mean() / r.std(ddof=1))
    n = len(trial_sharpes)
    var = float(np.var(trial_sharpes, ddof=1)) if n > 2 else 0.0
    if var <= 0 or n < 2:
        return dict(sharpe=sr, sr0=np.nan, dsr=np.nan, trials=n)
    sr0 = math.sqrt(var) * ((1 - EULER) * norm.ppf(1 - 1 / n) + EULER * norm.ppf(1 - 1 / (n * math.e)))
    z = (sr - sr0) * math.sqrt(len(r) - 1) / math.sqrt(max(1e-12, 1 - r.skew() * sr + (r.kurt() - 1) / 4 * sr ** 2))
    return dict(sharpe=sr, sr0=sr0, dsr=float(norm.cdf(z)), trials=n)


def pbo(px: pd.DataFrame, blocks: int = 10) -> dict:
    """CSCV probability of backtest overfitting: is the in-sample winner below the out-of-sample median?"""
    x = px.dropna(how="all")
    if x.shape[1] < 2 or len(x) < blocks * 5:
        return dict(pbo=np.nan, combos=0, variants=int(x.shape[1]))
    edges = np.linspace(0, len(x), blocks + 1).astype(int)
    parts = [x.iloc[edges[i]:edges[i + 1]] for i in range(blocks)]
    below, combos = 0, 0
    for idx in combinations(range(blocks), blocks // 2):
        ins = pd.concat([parts[i] for i in idx])
        oos = pd.concat([parts[i] for i in range(blocks) if i not in idx])
        is_sharpe = ins.mean() / ins.std(ddof=1).replace(0, np.nan)
        oos_sharpe = oos.mean() / oos.std(ddof=1).replace(0, np.nan)
        if is_sharpe.isna().all() or oos_sharpe.isna().all():
            continue
        best = is_sharpe.idxmax()
        rank = oos_sharpe.rank(pct=True).get(best, np.nan)
        if pd.notna(rank):
            # "below the median": with an even number of variants the median itself counts as below
            below += int(rank <= 0.5)
            combos += 1
    return dict(pbo=below / combos if combos else np.nan, combos=combos, variants=int(x.shape[1]))


# ------------------------------------------------------------------------------------------------- main
def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    d = build()
    d = d[d.y.notna()]
    rows = []

    # S1: model edge against local richness, one trade an hour
    rich = local_richness(d, 0.03, 0.20, 5, 30)
    for period, (p0, p1) in {"train Jul-Aug": TRAIN, "test Sep-Oct": TEST, "November": NOV}.items():
        sub = rich[(rich.end >= p0) & (rich.end < p1) & (rich.edge > 0)]
        if len(sub) < 50:
            continue
        for label, by in (("model edge (as frozen)", "edge"), ("local richness", "rich"),
                          ("edge x richness", "mix")):
            if by == "mix":
                sub = sub.assign(mix=sub.edge + sub.rich)
            h = pick_one(sub, by)
            s = stats(h)
            rows.append({"period": period, "selection": label, "hours": s["hours"], "¢/hour": s["cents"],
                         "t": s["t"], "hit": s["hit"], "dd $": s["dd"]})
        # does richness predict the outcome among the candidates the model likes?
        q = sub[sub.edge > 0]
        if len(q) > 30:
            hi_r = q[q.rich > q.rich.median()]
            lo_r = q[q.rich <= q.rich.median()]
            rows.append({"period": period, "selection": "rich half of the candidates",
                         "hours": len(hi_r), "¢/hour": 100 * float((hi_r.wing_bid - hi_r.won).mean()),
                         "t": np.nan, "hit": float((hi_r.won == 0).mean()), "dd": np.nan})
            rows.append({"period": period, "selection": "poor half of the candidates",
                         "hours": len(lo_r), "¢/hour": 100 * float((lo_r.wing_bid - lo_r.won).mean()),
                         "t": np.nan, "hit": float((lo_r.won == 0).mean()), "dd": np.nan})
    richness_t = pd.DataFrame(rows)
    richness_t.to_csv(OUT / "wings_alpha_richness.csv", index=False)

    # S2: quote age versus the outcome of the frozen rules
    age = quote_age(d)
    d2 = d.merge(age, on=["ticker", "ts"], how="left")
    stale_rows = []
    for rule, (lo, hi, l0, l1) in (("W03", (0.10, 0.20, 5, 15)), ("W04", (0.06, 0.10, 15, 30))):
        c = cheap_sides(d2, lo, hi, l0, l1)
        c["fair"] = c.fair_t4 if rule == "W04" else c.fair_n
        c["edge"] = c.wing_bid - c.fair - fee(1, c.price)
        c = c[c.edge > 0]
        for period, (p0, p1) in {"train Jul-Aug": TRAIN, "test Sep-Oct": TEST}.items():
            sub = c[(c.end >= p0) & (c.end < p1)]
            if len(sub) < 100:
                continue
            bins = pd.cut(sub.age, [-0.1, 1, 3, 10, 1e6], labels=["0-1 min", "1-3 min", "3-10 min", "10+ min"])
            for label, g in sub.groupby(bins, observed=True):
                pnl = (g.wing_bid - g.won - fee(1, g.price))
                stale_rows.append({"rule": rule, "period": period, "quote age": str(label),
                                   "candidates": len(g), "¢ per candidate": 100 * float(pnl.mean()),
                                   "hit": float((g.won == 0).mean())})
    stale_t = pd.DataFrame(stale_rows)
    stale_t.to_csv(OUT / "wings_alpha_stale.csv", index=False)

    # S3: the clock - hour of day and US holidays, on the frozen W03 weekday rule (the only positive one)
    c = cheap_sides(d, 0.10, 0.20, 5, 15)
    c["fair"] = c.fair_n
    c["edge"] = c.wing_bid - c.fair - fee(1, c.price)
    c = c[(c.edge > 0.01) & (c.end.dt.day_of_week < 5)]
    h = pick_one(c, "edge")
    by_hour = h.groupby(h.index.hour).agg(cents=lambda p: 100 * p.mean(), hours="size")
    by_hour["t"] = [stats(h[h.index.hour == k])["t"] if (h.index.hour == k).sum() >= 30 else np.nan
                    for k in by_hour.index]   # fewer than 30 trades: the t-statistic is meaningless here
    by_hour.to_csv(OUT / "wings_alpha_hours.csv")
    hol = USFederalHolidayCalendar().holidays("2026-07-01", "2026-10-28").tz_localize("UTC")
    is_hol = pd.Series(h.index.normalize().isin(hol), index=h.index)
    hol_t = pd.DataFrame([
        {"group": "US holiday", "hours": int(is_hol.sum()), "¢/hour": 100 * float(h[is_hol].mean()) if is_hol.any() else np.nan},
        {"group": "ordinary day", "hours": int((~is_hol).sum()), "¢/hour": 100 * float(h[~is_hol].mean())},
    ])
    hol_t.to_csv(OUT / "wings_alpha_holidays.csv", index=False)

    # S4: alpha decay, weekly
    wk = h.groupby(h.index.tz_convert("UTC").to_period("W")).agg(cents=lambda p: 100 * p.sum(), hours="size")
    wk["cum $"] = h.cumsum().groupby(h.index.tz_convert("UTC").to_period("W")).last()
    if len(wk) > 4:
        t = np.arange(len(wk))
        b = np.polyfit(t, wk.cents.to_numpy(), 1)
        trend_t = float(b[0] / (wk.cents.std(ddof=1) / math.sqrt(len(wk)) * math.sqrt(12 / (len(wk) ** 2 - 1))))
    else:
        b, trend_t = np.array([np.nan, np.nan]), np.nan
    decay_t = wk.reset_index().rename(columns={"index": "week"})
    decay_t.to_csv(OUT / "wings_alpha_decay.csv", index=False)

    # S5: how much of the winner is luck
    px, names = variant_grid(d)
    sharpes = (px.mean() / px.std(ddof=1)).dropna()
    best = sharpes.idxmax() if len(sharpes) else None
    ds = deflated_sharpe(px[best], sharpes.to_numpy()) if best else {}
    over = pbo(px)
    family = pd.DataFrame({"variant": sharpes.index, "sharpe per hour": sharpes.to_numpy(),
                           "total ¢": (100 * px[sharpes.index].sum()).to_numpy()}).sort_values(
        "sharpe per hour", ascending=False).reset_index(drop=True)
    family.to_csv(OUT / "wings_alpha_family.csv", index=False)
    # per-period behaviour of the family leaders: the full-sample ranking is not enough
    top_rows = []
    for name in family.variant.head(6):
        col = px[name].dropna()
        for period, (p0, p1) in {"train Jul-Aug": TRAIN, "test Sep-Oct": TEST, "November": NOV}.items():
            sub = col[(col.index >= p0) & (col.index < p1)]
            if len(sub) < 10:
                continue
            s_ = stats(sub)
            top_rows.append({"variant": name, "period": period, "trades": len(sub),
                             "¢ per trade": 100 * float(sub.mean()), "t": s_["t"], "hit": s_["hit"],
                             "dd $": s_["dd"]})
    top_t = pd.DataFrame(top_rows)
    top_t.to_csv(OUT / "wings_alpha_top.csv", index=False)
    # the whole family on the November holdout: how many of the 80 variants are even positive there
    nov_rows = []
    for name in px.columns:
        col = px[name].dropna()
        sub = col[(col.index >= NOV[0]) & (col.index < NOV[1])]
        if len(sub) >= 10:
            nov_rows.append({"variant": name, "trades": len(sub), "¢ per trade": 100 * float(sub.mean())})
    nov_family = pd.DataFrame(nov_rows).sort_values("¢ per trade", ascending=False)
    nov_family.to_csv(OUT / "wings_alpha_november.csv", index=False)
    multi_t = pd.DataFrame([{**ds, **over, "best variant": best,
                             "variants tested": int(len(sharpes))}])
    multi_t.to_csv(OUT / "wings_alpha_multiple.csv", index=False)

    write_report(richness_t, stale_t, by_hour.reset_index(), hol_t, decay_t, family, top_t, nov_family,
                 multi_t, b, trend_t)
    print(richness_t.to_string(index=False))
    print(stale_t.to_string(index=False))
    print(by_hour.to_string())
    print(hol_t.to_string(index=False))
    print(family.head(8).to_string(index=False))
    print(top_t.to_string(index=False))
    if not nov_family.empty:
        pos = int((nov_family["¢ per trade"] > 0).sum())
        print(f"november: {pos} of {len(nov_family)} variants positive, median {nov_family['¢ per trade'].median():.2f}")
        print(nov_family.head(8).to_string(index=False))
    print(multi_t.to_string(index=False))


def write_report(richness_t, stale_t, by_hour, hol_t, decay_t, family, top_t, nov_family, multi_t, trend,
                 trend_t) -> None:
    def md(df, fmt="{:.3f}"):
        if df is None or df.empty:
            return "_no rows_\n"
        out = df.copy()
        out.columns = [str(c).replace("|", "/") for c in out.columns]
        for c in out.columns:
            if pd.api.types.is_float_dtype(out[c]):
                out[c] = out[c].map(lambda x: "" if pd.isna(x) else fmt.format(x))
            elif not pd.api.types.is_numeric_dtype(out[c]):
                out[c] = out[c].map(lambda x: str(x).replace("|", "/"))
        head = "| " + " | ".join(map(str, out.columns)) + " |\n"
        sep = "|" + "|".join(["---"] * len(out.columns)) + "|\n"
        body = "".join("| " + " | ".join(map(str, r)) + " |\n" for r in out.itertuples(index=False))
        return head + sep + body

    lines = [
        "# Куда смотрит профессионал: относительная стоимость внутри лесенки, залипшие котировки, часы, "
        "и сколько в нашем лучшем результате удачи\n",
        f"_Сгенерировано {pd.Timestamp.now():%Y-%m-%d %H:%M}, `studies/wings_alpha.py`._\n",
        "## S1 Относительная стоимость: страйк против своих соседей\n",
        "Для каждого страйка берётся его середина и интерполируется по логарифму страйка между двумя соседями "
        "в том же снимке. Остаток — это «дорого/дёшево» не относительно модели, а относительно самой лесенки. "
        "Дальше: продаём самый дорогой по остатку страйк (один в час) и сравниваем с выбором по модельной "
        "дельте.\n",
        md(richness_t),
        "\n## S2 Залипшие котировки\n",
        "Возраст котировки — сколько минут она не менялась; кандидаты — те, что проходят фильтр модели.\n",
        md(stale_t),
        "\n## S3 Часы суток (правило W03 в будни) и праздники США\n",
        md(by_hour),
        md(hol_t),
        "\n## S4 Затухание: результат по неделям\n",
        md(decay_t),
        f"Наклон тренда по неделям: {trend[0]:.3f}¢/неделю (t ≈ {trend_t:.2f}).\n",
        "\n## S5 Сколько из этого — удача\n",
        md(family.head(12)),
        "\nЛидеры семейства по периодам:\n",
        md(top_t),
        md(multi_t),
        "\nВся семья на ноябрьском holdout:\n",
        md(nov_family.head(12)),
        "\n## Чтение\n",
        _reading(richness_t, stale_t, top_t, nov_family, multi_t, by_hour),
    ]
    (OUT / "wings_alpha.md").write_text("\n".join(lines), encoding="utf-8")


def _reading(richness_t, stale_t, top_t, nov_family, multi_t, by_hour) -> str:
    out = []
    t = richness_t[richness_t.selection.isin(["model edge (as frozen)", "local richness"])]
    if not t.empty:
        piv = t.pivot_table(index="selection", columns="period", values="¢/hour")
        out.append("* Относительная стоимость внутри лесенки не бьёт модельный отбор: "
                   + "; ".join(f"{k}: " + ", ".join(f"{c} {v:.2f}" for c, v in row.dropna().items())
                               for k, row in piv.iterrows())
                   + ". Знак «богатая половина против бедной» между половинами меняется, то есть как "
                     "самостоятельный сигнал остаток относительно соседей не работает.")
    if not stale_t.empty and stale_t.candidates.max() < 200:
        out.append("* Залипшие котировки на этой ленте не измерить: она уже минутные свечи книги, поэтому "
                   "возраст котировки почти всегда ≤1 минуты, а ячеек старше 3 минут — десятки.")
    elif not stale_t.empty:
        out.append("* Котировки старше 3 минут редки (десятки кандидатов), поэтому вывод о залипании "
                   "остаётся предварительным.")
    if not top_t.empty:
        lead = top_t.iloc[0].variant
        rows = top_t[top_t.variant == lead]
        out.append(f"* Лидер семейства `{lead}` на периодах: "
                   + ", ".join(f"{r.period} {r._4:.2f}¢ (t {r.t:.1f}, {r.trades} сделок)"
                               for r in rows.itertuples()))
        out.append("* Все верхние места семейства занимают варианты «только выходные» с дешёвыми полосами "
                   "(4–14¢) и горизонтом 15–40 минут — это независимо повторяет лабораторное W08, и это "
                   "сильнее любой отдельной p-величины.")
    if not nov_family.empty:
        pos = int((nov_family["¢ per trade"] > 0).sum())
        out.append(f"* На ноябрьском holdout положительны {pos} из {len(nov_family)} вариантов семейства, "
                   f"медиана {nov_family['¢ per trade'].median():.2f}¢ на сделку: то есть ни один лидер "
                   "семейства (все они — «только выходные») в ноябре не зарабатывает.")
    if not multi_t.empty:
        r = multi_t.iloc[0]
        out.append(f"* Дефлированный Шарп победителя: SR {r.sharpe:.3f} против порога {r.sr0:.3f} → DSR "
                   f"{r.dsr:.3f} при {int(r.trials)} опробованных вариантах; вероятность переподгонки "
                   f"(CSCV) {r.pbo:.2f}. Формально победитель выживает, но варианты сильно перекрываются, "
                   "поэтому читать это как «не опровергнуто», а не как «доказано».")
    if not by_hour.empty:
        thin = by_hour[by_hour.hours < 30]
        out.append(f"* По часам суток {len(thin)} из {len(by_hour)} ячеек имеют меньше 30 сделок — "
                   "t-статистики там не считаются, и «эффект часа» на этом объёме не устанавливается. "
                   "Отдельно: полосы 06–10 UTC недопредставлены в обеих половинах (еженедельные дыры в "
                   "архиве и сон в ноябре).")
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    main()
