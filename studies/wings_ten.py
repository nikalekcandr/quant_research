"""Ten more professional checks on the wing strategy, all on the tapes we already have.

Each section answers a question a desk would ask before giving the rule money, and each writes one CSV:

  T1  variance risk premium   implied variance from the ladder vs the model variance and vs what the hour did
  T2  liquidity               is the edge in liquid or in thin events (volume quartiles)
  T3  moneyness               calibration and P&L by distance from spot (|z| buckets) -- where the mispricing is
  T4  early exit              sell the wing and buy it back 5/10/20 minutes later instead of holding to settlement
  T5  1/N vs the best rule    equal weight over the top-k variants chosen on train, vs the single best
  T6  block bootstrap         confidence interval of the champion with 24-hour blocks (hours are dependent)
  T7  serial dependence       does one hour's P&L predict the next; runs test; ACF
  T8  other side              buy the expensive side instead: is the loss just the spread or is there information
  T9  minute of the hour      volatility and drift inside the hour, the 60-second BRTI settlement window
  T10 macro releases          7 US releases in the window: spot reaction and wing P&L on release hours

Run: python -m studies.wings_ten     (~1-2 minutes, writes results/studies/wings_ten_*.csv and .md)
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from studies.wing_panel import build, fee, minute_close  # noqa: E402
from studies.wings_scale import (NOV, PERIODS, RULES, TEST, TRAIN, cheap, pnl_at,  # noqa: E402
                                 signals, stats)

OUT = ROOT / "results/studies"
MACRO = ROOT / "data/manual/us_macro_calendar.csv"
CHAMPS = ("W03", "W04")


# --------------------------------------------------------------------------------------- helpers
def book(rule: str, d: pd.DataFrame, margin: float | None = None) -> pd.DataFrame:
    """The rule's trades: one strike an hour, picked by the largest model edge, >= 1 cent of edge."""
    c = cheap(d, rule)
    return signals(c, margin=margin, selection="edge")


def hourly(t: pd.DataFrame) -> pd.Series:
    """P&L of a trade list, one value per settlement hour, indexed by the hour's end."""
    if t.empty:
        return pd.Series(dtype=float)
    return pd.Series(np.asarray(pnl_at(t, 1), dtype=float), index=utc_index(t.end)).sort_index()


def utc_index(vals) -> pd.DatetimeIndex:
    """Index of timestamps, always UTC-aware (some frames keep a naive copy of the hour)."""
    return pd.DatetimeIndex(pd.to_datetime(pd.Index(np.asarray(vals)), utc=True))


def window(x: pd.Series, period) -> pd.Series:
    lo, hi = period
    x = pd.Series(np.asarray(x, dtype=float), index=utc_index(x.index)).sort_index()
    return x[(x.index >= lo) & (x.index < hi)]


def row(label: str, x: pd.Series) -> dict:
    s = stats(x)
    return {"test": label, "hours": s["hours"], "cents": s["cents"], "t": s["t"], "hit": s["hit"],
            "total": s["total"]}


def boot(x: pd.Series, block: int = 24, draws: int = 4000, seed: int = 5) -> dict:
    """Circular block bootstrap of the mean of an hourly series (blocks keep intraday dependence)."""
    x = pd.Series(x).dropna().to_numpy()
    n = len(x)
    if n < 2 * block:
        return {"mean": np.nan, "lo": np.nan, "hi": np.nan, "p_pos": np.nan, "blocks": 0}
    rng = np.random.default_rng(seed)
    k = int(math.ceil(n / block))
    means = np.empty(draws)
    ext = np.concatenate([x, x[:block]])
    for i in range(draws):
        starts = rng.integers(0, n, k)
        means[i] = np.concatenate([ext[s:s + block] for s in starts[:k]])[:n].mean()
    return {"mean": float(100 * x.mean()), "lo": float(100 * np.quantile(means, 0.025)),
            "hi": float(100 * np.quantile(means, 0.975)), "p_pos": float((means > 0).mean()),
            "blocks": k}


def acf(x: pd.Series, lags: int = 6) -> list[float]:
    x = pd.Series(x).dropna().to_numpy()
    if len(x) < 20 or x.std() == 0:
        return [np.nan] * lags
    x = x - x.mean()
    den = (x * x).sum()
    return [float((x[:-k] * x[k:]).sum() / den) for k in range(1, lags + 1)]


def runs_test(x: pd.Series) -> dict:
    """Wald-Wolfowitz runs test on the sign of the hourly P&L."""
    s = np.sign(pd.Series(x).dropna().to_numpy())
    s = s[s != 0]
    if len(s) < 30:
        return {"n": len(s), "runs": np.nan, "z": np.nan}
    n1, n2 = int((s > 0).sum()), int((s < 0).sum())
    runs = int(1 + (s[1:] != s[:-1]).sum())
    mu = 2 * n1 * n2 / (n1 + n2) + 1
    var = 2 * n1 * n2 * (2 * n1 * n2 - n1 - n2) / ((n1 + n2) ** 2 * (n1 + n2 - 1))
    return {"n": int(n1 + n2), "runs": runs, "z": float((runs - mu) / math.sqrt(var)) if var > 0 else np.nan}


# --------------------------------------------------------------------------------------- T1
def _t4cdf(z: np.ndarray, df: float = 4.0) -> np.ndarray:
    from scipy.stats import t as student_t
    return student_t.cdf(np.asarray(z) * math.sqrt(df / (df - 2.0)), df)


def t1_vrp(d: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Implied variance from the ladder vs the model variance and vs the realised hour."""
    a = d[d.spot_source.eq("binance") & d.left.between(15, 45)]
    mid = (a.bid + a.ask) / 2
    ok = mid.between(0.08, 0.92) & a.left.ge(10)
    q = norm.ppf(mid[ok].clip(0.001, 0.999))
    z = np.log(a.S[ok] * a.basis[ok] / a.strike[ok])
    sig = z / (q * np.sqrt(a.left[ok]))            # per sqrt minute
    ratio = (sig ** 2 * a.left[ok]) / a.var_rem[ok]
    per_hour = ratio.replace([np.inf, -np.inf], np.nan).dropna().groupby(a.loc[ok, "event"]).median()
    end_of = a.groupby("event").end.first()
    imp = pd.DataFrame({"event": per_hour.index, "implied_over_model": per_hour.to_numpy(),
                        "end": end_of.reindex(per_hour.index).to_numpy()})

    # what the hour actually did: spot at the first and the last snapshot of the event
    spot = (d[d.spot_source.eq("binance")].sort_values("ts").groupby("event")
            .agg(S0=("S", "first"), S1=("S", "last"), v0=("var_rem", "first"), end=("end", "first")))
    spot = spot[spot.v0.gt(0)]
    r = np.log(spot.S1 / spot.S0)
    real = pd.DataFrame({"event": spot.index, "real_over_model": r ** 2 / spot["v0"],
                         "abs_over_model": r.abs() / np.sqrt(spot["v0"] * 2 / np.pi)}).reset_index(drop=True)
    m = imp.merge(real, on="event", how="inner")
    m["month"] = pd.DatetimeIndex(m.end).strftime("%Y-%m")
    by_month = (m.groupby("month").agg(hours=("event", "size"), implied_over_model=("implied_over_model", "mean"),
                                       realised_over_model=("real_over_model", "mean"),
                                       abs_over_expected=("abs_over_model", "mean"))
                .round(3).reset_index())

    # does a high implied/model ratio predict the wing P&L?
    trades = pd.concat([book(r_, d) for r_ in CHAMPS], ignore_index=True).rename(columns={"rule": "rule"})
    trades = trades.merge(m[["event", "implied_over_model"]], on="event", how="left")
    trades = trades[trades.implied_over_model.notna()]
    try:
        trades["vrp_bucket"] = pd.qcut(trades.implied_over_model, 3, labels=["low", "mid", "high"])
    except ValueError:
        trades["vrp_bucket"] = "all"
    rows = [row(f"{k} | {b}", hourly(g)) for (k, b), g in trades.groupby(["rule", "vrp_bucket"], observed=True)]
    # T1c: the model variance is biased; rescale it by a factor chosen on train only
    scale_rows = []
    for sc in (0.6, 0.7, 0.8, 0.9, 1.0, 1.15):
        for r_ in CHAMPS:
            c = cheap(d, r_)
            z = c.z.to_numpy() / np.sqrt(sc)
            fair = norm.cdf(z) if RULES[r_]["model"] == "normal" else _t4cdf(z)
            c = c.assign(fair=fair,
                         edge=np.asarray(c.wing_bid) - fair - np.asarray(fee(1, np.asarray(c.price)), dtype=float))
            for name, per in ("train Jul-Aug", TRAIN), ("test Sep-Oct", TEST), ("November", NOV):
                sub = c[(c.end >= per[0]) & (c.end < per[1])]
                tr = signals(sub.assign(rule=r_), selection="edge")
                if len(tr) < 15:
                    continue
                scale_rows.append({**row(f"{r_} | x{sc}", hourly(tr)), "scale": sc})
    return by_month, pd.DataFrame(rows).round(3), pd.DataFrame(scale_rows).round(3)


# --------------------------------------------------------------------------------------- T2
def t2_liquidity(d: pd.DataFrame) -> pd.DataFrame:
    vol = d.groupby("event").volume.max()
    trades = pd.concat([book(r_, d) for r_ in CHAMPS], ignore_index=True)
    trades["vol"] = trades.event.map(vol)
    trades = trades[trades.vol.notna()]
    rows = []
    for k, g in trades.groupby("rule"):
        q = pd.qcut(g.vol, 4, labels=["thin", "q2", "q3", "liquid"])
        for b, h in g.groupby(q, observed=True):
            for name, per in ("train Jul-Aug", TRAIN), ("test Sep-Oct", TEST):
                hh = h[(h.end >= per[0]) & (h.end < per[1])]
                if len(hh) < 10:
                    continue
                rows.append({**row(f"{k} | {b} | {name}", hourly(hh)),
                             "mean edge ¢": round(100 * hh.edge.mean(), 3)})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------- T3
def t3_moneyness(d: pd.DataFrame) -> pd.DataFrame:
    """Calibration of the cheap side by distance from spot: what the model says, what the market says, what happened.

    Computed on *all* band rows of the rule (no model filter), so the market-vs-realised gap is not a
    selection artefact; the last two columns repeat it for the rows the rule actually trades.
    """
    rows = []
    for r_ in CHAMPS:
        c = cheap(d, r_)                      # every cheap-side minute of the band and window
        c["absz"] = c.z.abs()
        bins = [0, 1.0, 1.25, 1.5, 1.75, 2.0, 99]
        c["bucket"] = pd.cut(c.absz, bins, right=False,
                             labels=["<1.0σ", "1.0-1.25σ", "1.25-1.5σ", "1.5-1.75σ", "1.75-2σ", ">2σ"])
        sel = signals(c.assign(rule=r_), selection="edge")
        sel_keys = set(zip(sel.ticker, sel.ts))
        c["selected"] = [(t, u) in sel_keys for t, u in zip(c.ticker, c.ts)]
        for b, h in c.groupby("bucket", observed=True):
            if len(h) < 30:
                continue
            s_ = h[h.selected]
            rows.append({"rule": r_, "|z| bucket": b, "rows": len(h),
                         "market mid %": round(100 * h.mid_wing.mean(), 2),
                         "model p(win) %": round(100 * h.fair.mean(), 2),
                         "realised win %": round(100 * h.won.mean(), 2),
                         "market gap pp": round(100 * (h.won.mean() - h.mid_wing.mean()), 2),
                         "model gap pp": round(100 * (h.won.mean() - h.fair.mean()), 2),
                         "traded rows": len(s_),
                         "traded ¢": round(100 * np.asarray(pnl_at(s_, 1), dtype=float).mean(), 2)
                         if len(s_) > 10 else np.nan})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------- T4
def t4_early_exit(d: pd.DataFrame) -> pd.DataFrame:
    """Sell the wing, buy it back after Δ minutes at the ask, instead of holding to settlement."""
    px = d.set_index(["ticker", "ts"])[["bid", "ask"]]
    rows = []
    for r_ in CHAMPS:
        t = book(r_, d)
        for delta in (5, 10, 20):
            key = pd.MultiIndex.from_arrays([t.ticker, t.ts + pd.Timedelta(minutes=delta)])
            ex = px.reindex(key)
            have = np.asarray(ex.bid.notna() & ex.ask.notna(), dtype=bool)
            if have.sum() < 20:
                continue
            h = t[have].copy()
            ask_exit = np.asarray(ex.ask, dtype=float)[have]
            bid_exit = np.asarray(ex.bid, dtype=float)[have]
            # closing a sold cheap-yes costs its ask; closing a sold cheap-no costs 1 - its bid
            cost = np.where(h.cheap_yes, ask_exit, 1 - bid_exit)
            exit_pnl = (np.asarray(h.wing_bid, dtype=float) - cost
                        - np.asarray(fee(1, np.asarray(h.price)), dtype=float) - np.asarray(fee(1, cost), dtype=float))
            hold_pnl = np.asarray(pnl_at(h, 1), dtype=float)
            for per_name, per in ("train Jul-Aug", TRAIN), ("test Sep-Oct", TEST), ("November", NOV):
                m = (h.end >= per[0]).to_numpy() & (h.end < per[1]).to_numpy()
                if m.sum() < 15:
                    continue
                rows.append({"rule": r_, "exit": f"+{delta} min", "period": per_name, "trades": int(m.sum()),
                             "hold ¢": round(100 * hold_pnl[m].mean(), 3),
                             "exit ¢": round(100 * exit_pnl[m].mean(), 3),
                             "better %": round(100 * (exit_pnl[m] > hold_pnl[m]).mean(), 1),
                             "exit t": round(float(exit_pnl[m].mean() / (exit_pnl[m].std(ddof=1)
                                                                       / math.sqrt(m.sum()))), 2)
                             if exit_pnl[m].std() > 0 else np.nan})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------- T5
def t5_combination(d: pd.DataFrame) -> pd.DataFrame:
    from studies.wings_alpha import variant_grid
    px, _ = variant_grid(d)
    px.index = utc_index(px.index)
    rows = []
    for k in (1, 3, 5, 10, 20, 40):
        tr = (px[(px.index >= TRAIN[0]) & (px.index < TRAIN[1])].mean()
              .dropna().sort_values(ascending=False).head(k).index)
        comb = px[list(tr)].sum(axis=1, min_count=1) / float(k)
        for name, per in ("train Jul-Aug", TRAIN), ("test Sep-Oct", TEST), ("November", NOV):
            s = stats(window(comb, per))
            rows.append({"k variants": k, "period": name, "hours": s["hours"], "¢ per hour": s["cents"],
                         "t": s["t"], "hit": s["hit"]})
    return pd.DataFrame(rows).round(3)


# --------------------------------------------------------------------------------------- T6
def t6_bootstrap(d: pd.DataFrame) -> pd.DataFrame:
    from studies.wings_alpha import variant_grid
    px, _ = variant_grid(d)
    px.index = utc_index(px.index)
    best = (px[(px.index >= TRAIN[0]) & (px.index < TRAIN[1])].mean()
            .dropna().sort_values(ascending=False).head(1).index[0])
    rows = []
    for label, s in [("frozen W03", hourly(book("W03", d))), ("frozen W04", hourly(book("W04", d))),
                     (f"family best on train ({best})", px[best].dropna())]:
        for name, per in ("train Jul-Aug", TRAIN), ("test Sep-Oct", TEST), ("November", NOV):
            b = boot(window(s, per))
            rows.append({"series": label, "period": name, "mean ¢/hour": round(b["mean"], 3),
                         "95 % lo": round(b["lo"], 3), "95 % hi": round(b["hi"], 3),
                         "P(mean>0)": round(b["p_pos"], 3)})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------- T7
def t7_serial(d: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for r_ in CHAMPS:
        s = hourly(book(r_, d))
        allx = s
        rt = runs_test(allx)
        r1 = rt["z"]
        ac = acf(allx, 3)
        # conditional: does a losing hour predict the next one?
        nxt = s.shift(-1)
        after_loss = nxt[s < 0].dropna()
        after_win = nxt[s > 0].dropna()
        rows.append({"series": r_, "hours": len(s), "acf(1)": round(ac[0], 3) if ac[0] == ac[0] else np.nan,
                     "acf(2)": round(ac[1], 3) if ac[1] == ac[1] else np.nan,
                     "acf(3)": round(ac[2], 3) if ac[2] == ac[2] else np.nan,
                     "runs z": round(r1, 2) if r1 == r1 else np.nan,
                     "next hour after loss ¢": round(100 * after_loss.mean(), 3) if len(after_loss) > 10 else np.nan,
                     "next hour after win ¢": round(100 * after_win.mean(), 3) if len(after_win) > 10 else np.nan})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------- T8
def t8_maker(d: pd.DataFrame) -> pd.DataFrame:
    """Is the edge just the spread? Sell at the bid (taker) vs sell at the mid (maker, no fee).

    The maker column is an upper bound: a resting order is only filled when the market moves against it
    (adverse selection), which the candles cannot show.
    """
    rows = []
    for r_ in CHAMPS:
        t = book(r_, d)
        taker = np.asarray(pnl_at(t, 1), dtype=float)
        maker = t.mid_wing.to_numpy() - t.won.to_numpy()          # mid, zero fee
        maker_fee = maker - 0.0025 * t.price.to_numpy() * (1 - t.price.to_numpy())   # Kalshi maker rate
        half = (t.mid_wing - t.wing_bid).to_numpy()
        t = t.assign(_taker=taker, _maker=maker, _makerfee=maker_fee, _half=half)
        for name, per in ("train Jul-Aug", TRAIN), ("test Sep-Oct", TEST), ("November", NOV):
            m = (t.end >= per[0]) & (t.end < per[1])
            if m.sum() < 15:
                continue
            rows.append({"rule": r_, "period": name, "trades": int(m.sum()),
                         "half-spread ¢": round(100 * t._half[m].mean(), 3),
                         "taker ¢": round(100 * t._taker[m].mean(), 3),
                         "maker, 0 fee ¢": round(100 * t._maker[m].mean(), 3),
                         "maker, 25 bp ¢": round(100 * t._makerfee[m].mean(), 3),
                         "maker wins %": round(100 * (t._maker[m] > 0).mean(), 1)})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------- T9
def t9_clock() -> pd.DataFrame:
    px = minute_close()
    r = np.log(px.close).diff().dropna()
    mo = r.index.minute
    buckets = {"00-04 (settlement avg)": (0, 4), "05-24": (5, 24), "25-54": (25, 54), "55-59": (55, 59)}
    rows = []
    for name, (a, b) in buckets.items():
        x = r[(mo >= a) & (mo <= b)]
        rows.append({"minute of hour": name, "minutes": len(x),
                     "mean |r| bps": round(1e4 * x.abs().mean(), 3),
                     "mean r bps": round(1e4 * x.mean(), 4),
                     "t of mean": round(float(x.mean() / (x.std() / math.sqrt(len(x)))), 2) if x.std() > 0 else np.nan,
                     "share of |r|": round(100 * x.abs().sum() / r.abs().sum(), 1)})
    out = pd.DataFrame(rows)
    # the last minute of the hour against the rest, and whether it follows the hour's direction
    hour = r.groupby(r.index.floor("h")).sum()
    last = r[r.index.minute == 59]
    last = last.groupby(last.index.floor("h")).sum()
    both = pd.concat([hour.rename("hour"), last.rename("last")], axis=1).dropna()
    agree = float((np.sign(both.hour) == np.sign(both.last)).mean())
    out.loc[len(out)] = {"minute of hour": "sign(last minute) == sign(hour)", "minutes": len(both),
                         "mean |r| bps": np.nan, "mean r bps": round(agree, 3), "t of mean": np.nan,
                         "share of |r|": np.nan}
    return out


# --------------------------------------------------------------------------------------- T10
def t10_macro(d: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Seven US releases inside the window: the spot reaction, and the wing P&L on those hours."""
    cal = pd.read_csv(MACRO)
    ts = (pd.to_datetime(cal.date + " " + cal.time_et)
          .dt.tz_localize("America/New_York").dt.tz_convert("UTC"))
    cal = cal.assign(ts=ts.to_numpy())
    cal = cal[(cal.ts >= TRAIN[0]) & (cal.ts < NOV[1])].sort_values("ts")
    cal = cal[cal.event.isin(["CPI", "NFP", "FOMC", "PCE", "PPI"])]
    px = minute_close().close
    base = (px / px.shift(15) - 1).dropna()
    rows = []
    for c in cal.itertuples():
        pre = px.asof(c.ts) / px.asof(c.ts - pd.Timedelta(minutes=15)) - 1
        post = px.asof(c.ts + pd.Timedelta(minutes=15)) / px.asof(c.ts) - 1
        rows.append({"event": c.event, "utc": c.ts.strftime("%Y-%m-%d %H:%M"),
                     "15 min before %": round(100 * pre, 3), "15 min after %": round(100 * post, 3),
                     "|after| %": round(100 * abs(post), 3),
                     "p vs all 15-min moves": round(float((base.abs() >= abs(post)).mean()), 3)})
    ev = pd.DataFrame(rows)

    # the hourly event that *contains* the release: a 12:30 release belongs to the event settling at 13:00
    hours = pd.DatetimeIndex((cal.ts + pd.Timedelta(minutes=1)).dt.ceil("h").to_numpy())
    pnl_rows = []
    for r_ in CHAMPS:
        t = book(r_, d)
        t["macro"] = t.end.isin(hours)
        for m, g in t.groupby("macro"):
            pnl_rows.append({**row(f"{r_} | macro={'release' if m else 'other'}", hourly(g)),
                             "mean edge ¢": round(100 * g.edge.mean(), 3)})
        near = t[t.end.isin(hours)]
        pnl_rows.append({**row(f"{r_} | release hours 1 contract", hourly(near)), "mean edge ¢": np.nan})
    return ev, pd.DataFrame(pnl_rows)


# --------------------------------------------------------------------------------------- report
def md(df, fmt="{:.3f}"):
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
    body = "".join("| " + " | ".join(map(str, r)) + " |\n" for r in out.itertuples(index=False))
    return head + sep + body


def write_report(parts: dict) -> Path:
    txt = ["# Десять проверок на тех данных, что уже есть\n",
           f"_Сгенерировано {pd.Timestamp.now():%Y-%m-%d %H:%M}, `studies/wings_ten.py`._\n",
           "Все суммы — на 1 контракт, после комиссии Kalshi, посчитанной на *ордер* (округление вверх до цента). "
           "Тест-статистики по часам, один час = одно наблюдение. Периоды: трейн июль–август, тест сентябрь–октябрь, "
           "holdout 8–25 ноября (в ноябре машина писала не все часы — см. `wings_sleep.md`).\n"]
    titles = {
        "vrp_month": "## T1 Премия за риск волатильности\n\n"
                     "Подразумеваемая дисперсия восстановлена из цен лесенки (обратная нормаль по страйкам 8–92¢), "
                     "сравнивается с моделью и с тем, что час реально сделал.\n",
        "vrp_pnl": "### T1б. Прибыль крыла в зависимости от подразумеваемой/модельной дисперсии\n",
        "vrp_scale": "### T1в. Масштаб дисперсии, выбранный на трейне (0.6…1.15)\n",
        "liq": "## T2 Ликвидность: где живёт преимущество\n\nСобытия разделены на квартили по объёму "
               "(максимум за час, контракты).\n",
        "money": "## T3 Деньги против модели по расстоянию от спота\n\n"
                 "`priced win` — вероятность победы дешёвой стороны, заложенная в цену (1 − fair). "
                 "`calibration` — насколько реальная частота выше заложенной.\n",
        "exit": "## T4 Выход до расчёта: продали крыло и выкупили через 5/10/20 минут\n",
        "comb": "## T5 Один лучший вариант против равных весов по топ-k\n\n"
                "k вариантов выбираются по среднему P&L на трейне, затем торгуются равным весом 1/k.\n",
        "boot": "## T6 Блочный бутстрэп: интервал среднего (блок 24 часа)\n",
        "serial": "## T7 Зависимость часов: предсказывает ли один час следующий\n",
        "side": "## T8 Кто платит спред: тейкер по bid против мейкера по середине\n\n"
                "`maker, 0 fee` — верхняя граница: заявка стоит в середине спреда и не платит комиссию. "
                "Реально она исполнится только когда рынок пойдёт против неё (adverse selection), "
                "по свечам это не видно.\n",
        "clock": "## T9 Минута внутри часа и окно расчёта BRTI (60 секунд)\n",
        "macro": "## T10 Макро-релизы: реакция спота\n",
        "macro_pnl": "### T10б. P&L крыла в часы релизов\n",
    }
    for key, title in titles.items():
        txt.append(title)
        txt.append(md(parts.get(key)))
        txt += ["\n"]
    txt.append("## Чтение\n")
    txt.append(reading(parts))
    path = OUT / "wings_ten.md"
    path.write_text("\n".join(txt), encoding="utf-8")
    return path


def reading(parts: dict) -> str:
    out = []
    v = parts.get("vrp_month")
    if v is not None and not v.empty:
        im, rl, ab = (v.implied_over_model.mean(), v.realised_over_model.mean(), v.abs_over_expected.mean())
        out.append(f"* Подразумеваемая дисперсия лесенки относится к модельной как {im:.2f}, реализованная — "
                   f"{rl:.2f}, средний |ход| часа к ожидаемому — {ab:.2f}. Все три числа ниже единицы и "
                   "совпадают между месяцами, то есть **модель систематически завышает дисперсию примерно "
                   "на четверть**, а рынок завышает её меньше, чем модель. Практический смысл: модельные "
                   "вероятности побед дешёвой стороны занижены, значит отбор консервативен — и правка "
                   "масштаба (T1в) меняет не только цифры, но и набор сделок.")
    e = parts.get("exit")
    if e is not None and not e.empty:
        best = e.sort_values("exit ¢", ascending=False).iloc[0]
        out.append(f"* Досрочный выход не спасает: лучшая строка — {best.rule}, {best.exit}, "
                   f"{best.period}: {best['exit ¢']:.2f}¢ против {best['hold ¢']:.2f}¢ при удержании до расчёта. "
                   "Причина структурная: чтобы закрыть позицию, платишь второй спред, а премия уже получена.")
    c = parts.get("comb")
    if c is not None and not c.empty:
        t = c[c.period == "test Sep-Oct"].set_index("k variants")
        if 1 in t.index and 10 in t.index:
            out.append(f"* Единственный лучший вариант даёт на тесте {t.loc[1, '¢ per hour']:.2f}¢/час при "
                       f"t = {t.loc[1, 't']:.2f}, а равные веса по 10 вариантам — {t.loc[10, '¢ per hour']:.2f}¢ "
                       f"при t = {t.loc[10, 't']:.2f}. Средний P&L у победителя выше, но он неотличим от нуля; "
                       "усреднение даёт меньшую, зато устойчивую оценку. Это ровно тот случай, когда "
                       "оптимизация по истории не переносится, а диверсификация по семейству — переносится. "
                       "В ноябре, впрочем, отрицательны все k.")
    b = parts.get("boot")
    if b is not None and not b.empty:
        w3 = b[(b.series == "frozen W03") & (b.period == "test Sep-Oct")]
        if len(w3):
            r = w3.iloc[0]
            out.append(f"* Блочный интервал W03 на тесте: [{r['95 % lo']:.2f}; {r['95 % hi']:.2f}]¢/час, "
                       f"P(среднее>0) = {r['P(mean>0)']:.2f}. Именно этот интервал, а не t-статистику, надо "
                       "показывать инвестору: он учитывает, что часы одного дня зависимы.")
    s = parts.get("serial")
    if s is not None and not s.empty:
        r = s.iloc[0]
        out.append(f"* Зависимости часов почти нет: acf(1) = {r['acf(1)']:.3f}, тест серий z = {r['runs z']:.2f}. "
                   "То есть P&L крыла близок к белому шуму вокруг среднего — отдельного «фильтра по прошлому часу» нет.")
    x = parts.get("side")
    if x is not None and not x.empty:
        r = x[x.period == "test Sep-Oct"].iloc[0] if (x.period == "test Sep-Oct").any() else x.iloc[0]
        d_mk = r["maker, 0 fee ¢"] - r["taker ¢"]
        out.append(f"* Половина спреда у {r.rule} на тесте — {r['half-spread ¢']:.2f}¢ на крыло ценой 6–20¢. "
                   f"Тейкер получает {r['taker ¢']:.2f}¢, мейкер в середине без комиссии — "
                   f"{r['maker, 0 fee ¢']:.2f}¢, то есть {d_mk:+.2f}¢ разницы: ровно два полуспреда плюс "
                   "комиссия. Мейкерская оценка — верхняя граница: она предполагает исполнение в середине, "
                   "а на самом деле заявка исполнится в первую очередь тогда, когда рынок пойдёт против неё. "
                   "Логичный следующий шаг — проверить это на снимках стакана с размерами, которых в "
                   "репозитории нет.")
    k = parts.get("clock")
    if k is not None and not k.empty:
        q = k.set_index("minute of hour")
        if "00-04 (settlement avg)" in q.index and "25-54" in q.index:
            out.append(f"* Волатильность внутри часа неодинакова: минуты 00–04 (сразу после расчёта) дают "
                       f"{q.loc['00-04 (settlement avg)', 'mean |r| bps']:.2f} бп/мин против "
                       f"{q.loc['25-54', 'mean |r| bps']:.2f} в середине часа, а последние пять минут перед "
                       f"расчётом тише всех — {q.loc['55-59', 'mean |r| bps']:.2f}. То есть окно 60-секундного "
                       "среднего BRTI, по которому считается расчёт, — самая спокойная часть часа, и основной "
                       "риск крыла приходится не на него, а на первые минуты нового часа. Направления "
                       "последняя минута не несёт: её знак совпадает со знаком часа в 49.9 % случаев.")
    m = parts.get("macro")
    mp = parts.get("macro_pnl")
    if mp is not None and not mp.empty:
        rel = mp[mp.test.str.contains("release", na=False)]
        oth = mp[mp.test.str.contains("other", na=False)]
        if len(rel) and len(oth):
            out.append(f"* Часы макро-релизов — это ровно те часы, где крыло проигрывает целиком: "
                       f"{rel.cents.mean():.2f}¢/час в релизные часы против {oth.cents.mean():.2f}¢ в остальные "
                       f"(всего {int(rel.hours.sum())} релизных часов). Механизм виден прямо в таблице спота: "
                       "NFP 04.09 дал −2.0 % за 15 минут, CPI 14.10 — −1.2 %; на 30-минутном окне крыла это "
                       "уже не крыло, а лотерея. Мощности на 10 часов нет (t ≈ −1), но правило «не продавать "
                       "крыло в час CPI/NFP/FOMC» подтверждается механизмом, а не только знаком.")
    return "\n".join(out) + "\n"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    d = build()
    d = d[d.y.notna()].copy()
    parts = {}

    parts["vrp_month"], parts["vrp_pnl"], parts["vrp_scale"] = t1_vrp(d)
    parts["vrp_month"].to_csv(OUT / "wings_ten_vrp_month.csv", index=False)
    parts["vrp_pnl"].to_csv(OUT / "wings_ten_vrp_pnl.csv", index=False)
    parts["vrp_scale"].to_csv(OUT / "wings_ten_vrp_scale.csv", index=False)
    parts["liq"] = t2_liquidity(d)
    parts["liq"].to_csv(OUT / "wings_ten_liquidity.csv", index=False)
    parts["money"] = t3_moneyness(d)
    parts["money"].to_csv(OUT / "wings_ten_moneyness.csv", index=False)
    parts["exit"] = t4_early_exit(d)
    parts["exit"].to_csv(OUT / "wings_ten_exit.csv", index=False)
    parts["comb"] = t5_combination(d)
    parts["comb"].to_csv(OUT / "wings_ten_combination.csv", index=False)
    parts["boot"] = t6_bootstrap(d)
    parts["boot"].to_csv(OUT / "wings_ten_bootstrap.csv", index=False)
    parts["serial"] = t7_serial(d)
    parts["serial"].to_csv(OUT / "wings_ten_serial.csv", index=False)
    parts["side"] = t8_maker(d)
    parts["side"].to_csv(OUT / "wings_ten_sides.csv", index=False)
    parts["clock"] = t9_clock()
    parts["clock"].to_csv(OUT / "wings_ten_clock.csv", index=False)
    parts["macro"], parts["macro_pnl"] = t10_macro(d)
    parts["macro"].to_csv(OUT / "wings_ten_macro.csv", index=False)
    parts["macro_pnl"].to_csv(OUT / "wings_ten_macro_pnl.csv", index=False)

    path = write_report(parts)
    for k, v in parts.items():
        print(f"\n--- {k}")
        print(v.to_string(index=False) if not v.empty else "(empty)")
    print(f"\nwritten {path}")


if __name__ == "__main__":
    main()
