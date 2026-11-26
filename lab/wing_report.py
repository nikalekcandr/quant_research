"""Offline Alpha Lab dashboard for frozen wing strategies."""
from __future__ import annotations

import json
from html import escape
from pathlib import Path
from string import Template

import pandas as pd
from scipy.stats import t as student_t

import config
from lab import signals
from lab.core import LAB, REPORT_HTML, REPORT_MD, atomic_write
from lab.edge_timing import W12_FIRST_EXPIRY
from lab.wing_calibration import PARAMS, VERSION as CAL_VERSION, depth_hybrid, pooled_probability, sampled as calibration_sample
from lab.report import md, void_mask
from trader.monitor import SLOT, demo_checks
from trader.settings import Settings

# Short Russian labels for navigation. The full rule is taken from the registry.
NAMES = {
    "W01": "Все дешёвые крылья", "W02": "Крылья с фильтром модели",
    "W03": "Позднее окно · нормальная модель", "W04": "Раннее окно · Стьюдент t₄",
    "W05": "Позднее окно · мейкер", "W06": "Раннее окно · мейкер",
    "W07": "Позднее окно · фильтр волатильности", "W08": "Выходные · одно крыло",
    "W09": "Polymarket · позднее окно", "W10": "Polymarket · раннее окно",
    "W11": "Выходные · два крыла",
    "W12": "Раннее окно · edge от 1¢",
}

DESCRIPTIONS = {
    "W12": "Kalshi, один бумажный контракт на экспирацию. За 15–30 минут до часа выбирается крыло по 6–10¢ с максимальным доступным перевесом t₄. Вход на первом снимке с edge после комиссии от 1¢, спред до 3¢, в стакане минимум один контракт.",
    "W01": "Kalshi и Polymarket. Продажа крыльев по 2–6¢ минимум за 15 минут до экспирации, одна позиция на страйк.",
    "W02": "Как W01, с дополнительной проверкой модели. Цена продажи должна превышать справедливую цену минимум на 1¢ после комиссии.",
    "W03": "Kalshi, за 5–15 минут до часа. Среди крыльев по 10–20¢ выбирается самое переоценённое по нормальной модели. Минимальный перевес после комиссии 1¢, одна сделка на час.",
    "W04": "Kalshi, за 15–30 минут до часа. Среди крыльев по 6–10¢ выбирается наибольший перевес по модели Стьюдента с четырьмя степенями свободы. После комиссии перевес должен быть не ниже нуля. Один вход на час.",
    "W05": "Kalshi, крыло по 10–20¢ за 5–15 минут до часа. Оценка сочетает EWMA и бустинг. Сначала лимитная заявка, за 7 минут до часа возможен вход по bid при перевесе от 1¢. Исключены экспирации 13–15 UTC.",
    "W06": "Kalshi, крыло по 6–10¢ за 15–30 минут до часа. Модель и исключённые часы как у W05. За 17 минут до часа возможен вход по bid при неотрицательном перевесе после комиссии.",
    "W07": "Условия W03 плюс фильтр волатильности. За последний час реализованная дисперсия должна быть не выше 1,5 от ожидаемой моделью.",
    "W08": "Kalshi, субботние и воскресные экспирации по UTC. За 15–30 минут до часа продаётся самое дорогое доступное крыло из диапазона 6–10¢. Один вход на час, без фильтра модели.",
    "W09": "Polymarket, условия W03. Нормальная модель, крыло по 10–20¢, до часа 5–15 минут, перевес после комиссии от 1¢.",
    "W10": "Polymarket, условия W04. Модель Стьюдента t₄, крыло по 6–10¢, до часа 15–30 минут, неотрицательный перевес после комиссии.",
    "W11": "Условия W08 с двумя возможными входами. Для каждой стороны отдельно выбирается самое дорогое доступное крыло по 6–10¢. Обе сделки относятся к одному исходу цены BTC за час.",
}


def number(x, digits=2, signed=False):
    return "—" if pd.isna(x) else format(x, ("+" if signed else "") + f".{digits}f").replace("-", "−").replace(".", ",")


def stamp(ts):
    return "—" if pd.isna(ts) else ts.strftime("%d.%m.%Y %H:%M")


def table(frame):
    if frame.empty:
        return "<p class='empty'>Пока нет наблюдений.</p>"
    return "<div class='scroll'>" + frame.to_html(index=False, border=0, na_rep="—") + "</div>"


def prepare(ledger):
    w = ledger.loc[ledger.family.eq("W")].copy()
    invalid = w.loc[void_mask(w), "id"]
    w = w[~w.id.isin(invalid)].drop_duplicates(["id", "event"], keep="last")
    closed = w[w.event.isin(["CLOSE", "INSTANT"])]
    opened = w[w.event.eq("OPEN") & ~w.id.isin(closed.id)].copy()
    closed = closed.dropna(subset=["pnl", "ts", "exit_ts"]).sort_values("exit_ts").copy()
    closed["ts"] = pd.to_datetime(closed.ts, utc=True, format="mixed")
    closed["exit_ts"] = pd.to_datetime(closed.exit_ts, utc=True, format="mixed")
    # Binary ledger P&L and fees are quoted per contract. Cash statistics use size.
    closed["pnl"] = closed.pnl * closed.qty
    info = closed["info"].map(lambda x: json.loads(x) if pd.notna(x) else {})
    closed["left"] = pd.to_numeric(info.map(lambda x: x.get("left_min")), errors="coerce")
    closed["fill"] = info.map(lambda x: "Maker" if x.get("maker") else "Taker")
    yes_mid = pd.to_numeric(info.map(lambda x: x.get("mid")), errors="coerce")
    closed["mid_side"] = yes_mid.where(closed.side.eq("yes"), 1 - yes_mid)
    # The recorded time left identifies the contractual hour even for late settlement processing.
    closed["hour"] = (closed.ts + pd.to_timedelta(closed.left, unit="min")).dt.round("h")
    return closed, opened


def model_stats(g):
    """Paired observations of the purchased side. All means here give one vote per trade."""
    m = g.dropna(subset=["price","fair","fee","edge","exit_price"])
    paired = m.dropna(subset=["mid_side"])
    return {"С моделью":len(m), "P модели, %":number(100*m.fair.mean()),
            "Цена входа, ¢":number(100*m.price.mean()), "Edge до комиссии, ¢":number(100*(m.fair-m.price).mean()),
            "Комиссия, ¢":number(100*m.fee.mean()), "Edge нетто, ¢":number(100*m.edge.mean()),
            "Медиана, ¢":number(100*m.edge.median()), "P10 / P90, ¢":" / ".join(number(100*m.edge.quantile(q)) for q in (.1,.9)) if len(m) else "—",
            "Факт нетто, ¢":number(100*(m.pnl/m.qty).mean()), "Выплата 1, %":number(100*m.exit_price.mean()),
            "Brier модели (все)":number(((m.fair-m.exit_price)**2).mean(),4),
            "Пар с mid":len(paired), "Модель − mid, ¢":number(100*(paired.fair-paired.mid_side).mean()),
            "Brier модели (пары)":number(((paired.fair-paired.exit_price)**2).mean(),4),
            "Brier рынка (пары)":number(((paired.mid_side-paired.exit_price)**2).mean(),4)}


def audit(closed):
    path = LAB / "resolutions.json"
    if not path.exists() or closed.empty:
        return "Для сверки нужны закрытые сделки и сохранённые исходы рынков."
    outcomes = json.loads(path.read_text(encoding="utf-8"))
    result = (closed.venue + "|" + closed.instrument).map(outcomes)
    paid = result.where(closed.side == "yes", 1 - result)
    pnl = (paid - closed.price - closed.fee) * closed.qty
    known = int(pnl.notna().sum())
    mismatch = int(((pnl - closed.pnl).abs() > 1e-8).sum())
    return (f"Сверка по исходам площадок охватила {known} из {len(closed)} закрытых сделок. "
            f"Расхождений с журналом: {mismatch}. Для каждого контракта из выплаты вычтены цена входа и комиссия.")


def metrics(g):
    settled = g.groupby("exit_ts").pnl.sum().sort_index().cumsum()
    dd = (settled.cummax().clip(lower=0) - settled).max()
    return {"Сделки": len(g), "Часы": g.hour.nunique(), "В плюс, %": number(100 * (g.pnl > 0).mean(), 1),
            "Нетто, $": number(g.pnl.sum(), signed=True), "¢/контр.": number(100 * g.pnl.sum() / g.qty.sum(), signed=True) if len(g) else "—",
            "Худшая, $": number(g.pnl.min()), "Просадка, $": number(dd)}


def equity(g):
    if g.empty:
        return "<p class='empty'>Кривая появится после первого расчёта.</p>"
    by_time = g.groupby("exit_ts").pnl.sum().sort_index().cumsum()
    series = pd.concat([pd.Series([0.0], index=[g.ts.min()]), by_time])
    lo, hi = min(series.min(), 0), max(series.max(), 0)
    scale = max(hi - lo, .01)
    def y(v):
        return 122 - 105 * (v - lo) / scale
    duration = max((series.index[-1] - series.index[0]).total_seconds(), 1)
    points = [f"48,{y(0):.2f}"]
    prev = 0
    for t, v in series.items():
        x = 48 + 630 * (t - series.index[0]).total_seconds() / duration
        points.extend([f"{x:.2f},{y(prev):.2f}", f"{x:.2f},{y(v):.2f}"])
        prev = v
    grid = "".join(f"<line x1='48' x2='678' y1='{y(v)}' y2='{y(v)}'/><text x='2' y='{y(v)+4}'>{number(v)}</text>"
                   for v in sorted({lo, hi, 0}))
    return (f"<svg viewBox='0 0 690 153' role='img' aria-label='Накопленный чистый результат в долларах'>{grid}"
            f"<polyline points='{' '.join(points)}'/><text x='48' y='148'>{series.index[0]:%d.%m}</text>"
            f"<text x='678' y='148' text-anchor='end'>{series.index[-1]:%d.%m}</text></svg>")


def details(h, g, op, status):
    s = metrics(g)
    winners = g[g.pnl > 0]
    losers = g[g.pnl < 0]
    outlay = ((g.price + g.fee)*g.qty).sum()
    parts = [f"<article id='{h.id}'><div class='eyebrow'>{h.id} &nbsp; / &nbsp; "
             f"{status}</div><h2>{escape(NAMES.get(h.id,h.title))}</h2>",
             f"<p>{escape(DESCRIPTIONS.get(h.id,h.rule))}</p><div class='chart'><h3>Накопленный результат, $</h3>{equity(g)}</div>"]
    extra = {"Закрыто": len(g), "Открыто": len(op), "Прибыльных": int((g.pnl > 0).sum()),
             "Убыточных": int((g.pnl < 0).sum()), "До комиссий, $": number((g.pnl + g.fee*g.qty).sum()),
             "Комиссии, $": number((g.fee*g.qty).sum()), "После комиссий, $": s["Нетто, $"],
             "Среднее, ¢/контр.": s["¢/контр."], "Худшая сделка, $": s["Худшая, $"],
             "Макс. просадка, $": s["Просадка, $"], "Средний edge входа, ¢": number(100*g.edge.mean()),
             "Среднее до часа, мин": number(g.left.mean(), 1),
             "Затраты на входы, $":number(outlay),
             "Нетто / затраты, %":number(100*g.pnl.sum()/outlay) if outlay else "—",
             "Лучшая сделка, $":number(g.pnl.max()),
             "Средний выигрыш, $":number(winners.pnl.mean()), "Средний убыток, $":number(losers.pnl.mean()),
             "Сумма выигрышей / потерь":number(winners.pnl.sum()/-losers.pnl.sum()) if len(losers) else "—"}
    parts.append("<dl>" + "".join(f"<div><dt>{k}</dt><dd>{v}</dd></div>" for k,v in extra.items()) + "</dl>")
    hours = g.groupby("hour").pnl.sum()
    if len(hours) > 2 and (hours < 0).any() and hours.std() > 0:
        margin = student_t.ppf(.975, len(hours)-1) * hours.sem()
        parts.append(f"<p>Среднее за активный час {number(100*hours.mean(), signed=True)}¢. "
                     f"Ориентировочный 95% интервал по {len(hours)} часам: "
                     f"{number(100*(hours.mean()-margin))}…{number(100*(hours.mean()+margin))}¢. "
                     "Интервал для одного правила, без поправки на перебор.</p>")
    elif len(g) and not (g.pnl < 0).any():
        parts.append(f"<p>За {len(hours)} часов убыточных сделок ещё не было. Наибольшая сумма под риском "
                     f"при одном входе в этой выборке — {number((g.price + g.fee).max())} доллара за контракт.</p>")
    segments = []
    for name, group in [("Будни UTC", g[g.hour.dt.day_of_week < 5]), ("Выходные UTC", g[g.hour.dt.day_of_week >= 5]),
                        *[(f"Куплено {side.upper()}", x) for side,x in g.groupby("side")],
                        *list(g.groupby("fill")), *list(g.groupby("venue"))]:
        if len(group):
            segments.append({"Срез": name, **metrics(group)})
    parts.extend(["<h3>Условия входа и исполнение</h3>", table(pd.DataFrame(segments)),
                  f"<p class='note'>Правило зафиксировано {escape(h.frozen)}. {escape(h.retired or '')}</p>"])
    parts.append("<h3>Прогноз в момент решения</h3>")
    if g.fair.notna().any():
        parts.append(table(pd.DataFrame([{"Входы":name, **model_stats(x)} for name,x in g.groupby("fill")])))
        buckets = pd.cut(100*g.edge, [-float("inf"),1,3,5,float("inf")], right=False,
                         labels=["< 1¢","1–<3¢","3–<5¢","≥ 5¢"])
        parts.append(table(pd.DataFrame([{"Edge нетто":str(band), **metrics(x)}
                                         for band,x in g.groupby(buckets,observed=True)])))
    else:
        parts.append("<p>В журнале этого правила нет модельной вероятности для закрытых входов.</p>")
    if len(op):
        parts.extend(["<h3>Открытые позиции</h3>", table(op.assign(Вход=op.ts.map(stamp)).rename(
            columns={"instrument":"Рынок", "price":"Цена, $", "side":"Куплено", "qty":"Контракты"})
            .assign(**{"Модель, %":(100*op.fair).round(2),"Edge нетто, ¢":(100*op.edge).round(2)})
            [["Вход","Рынок","Куплено","Цена, $","Контракты","Модель, %","Edge нетто, ¢"]])])
    journal = pd.DataFrame({"Вход UTC":g.ts.map(stamp), "Расчёт UTC":g.exit_ts.map(stamp), "Рынок":g.instrument,
                            "Куплено":g.side, "Крыло, ¢":(100*(1-g.price)).round(2), "Комиссия, ¢":(100*g.fee).round(2),
                            "Цена входа, ¢":(100*g.price).round(2),"Модель, %":(100*g.fair).round(2),
                            "Edge нетто, ¢":(100*g.edge).round(2),"Тип":g["fill"], "Контракты":g.qty,
                            "Нетто, $":g.pnl.map(lambda v:number(v,4,True))})
    parts.extend([f"<details><summary>Журнал сделок · {len(g)}</summary>", table(journal.iloc[::-1]), "</details></article>"])
    return "".join(parts)


def journal_events(path):
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            yield json.loads(line)
        except json.JSONDecodeError:
            continue  # A writer may still be appending the last line.


def execution():
    records = []
    for name, mode in [("Теневое исполнение", "shadow"), ("Реальный счёт", "live")]:
        path = config.DATA / "trader" / f"journal_{mode}.jsonl"
        if not path.exists():
            continue
        events = list(journal_events(path))
        wings = {(SLOT.get(r["hyp"],r["hyp"]) + "|" + r["event"], r["ticker"])
                 for r in events if r.get("what") == "position_open" and r.get("hyp") in NAMES}
        done = [r for r in events if r.get("what") == "settled" and (r.get("key"),r.get("ticker")) in wings]
        skips = [str(r.get("reason","")) for r in events if r.get("what") == "skip"
                 and str(r.get("intent","")).split("|")[0] in NAMES]
        contracts = sum(r["count"] for r in done)
        pnl = sum(r["pnl"] for r in done)
        records.append({"Режим":name, "Закрыто":len(done), "Контрактов":contracts, "Нетто, $":number(pnl,signed=True),
                        "¢/контр.":number(100*pnl/contracts) if contracts else "—",
                        "Пропуски сигналов":len(skips), "Из них просрочено":sum(s.startswith("stale intent") for s in skips)})
    return pd.DataFrame(records)


def execution_by_rule():
    rows = []
    for mode in ("shadow","live"):
        path = config.DATA / "trader" / f"journal_{mode}.jsonl"
        if not path.exists():
            continue
        by_rule, positions = {}, {}
        for e in journal_events(path):
            kind = e.get("what")
            rule = e.get("hyp") or str(e.get("intent","")).split("|")[0]
            if kind == "settled":
                rule = positions.pop((e.get("key"),e.get("ticker")), None)
            if rule not in NAMES:
                continue
            values = by_rule.setdefault(rule, {"opened":0,"skip":0,"settled":0,"n":0,"pnl":0,"age":[],"slip":[],"edge":[]})
            if kind == "skip":
                values["skip"] += 1
            elif kind == "position_open":
                positions[(SLOT.get(rule,rule)+"|"+e["event"], e["ticker"])] = rule
                values["opened"] += 1
                price = e.get("price")  # fill price; vwap is the pre-order book estimate
                if price is not None and e.get("lab_price") is not None:
                    values["slip"].append(100*(price-e["lab_price"]))
                if e.get("decision_age_s") is not None:
                    values["age"].append(e["decision_age_s"])
                if price is not None and e.get("fair_now") is not None and e.get("filled",0)>0 and e.get("fee") is not None:
                    values["edge"].append(100*(e["fair_now"]-price-e["fee"]/e["filled"]))
            elif kind == "settled":
                values["settled"] += 1
                values["n"] += e["count"]
                values["pnl"] += e["pnl"]
        for rule, v in sorted(by_rule.items()):
            rows.append({"Правило":rule,"Режим":mode,"Открытия":v["opened"],"Пропуски":v["skip"],"Закрытия":v["settled"],
                         "Контракты":v["n"],"Нетто, $":number(v["pnl"],signed=True),
                         "¢/контр.":number(100*v["pnl"]/v["n"]) if v["n"] else "—",
                         "Задержка, с":number(pd.Series(v["age"],dtype=float).mean()),
                         "Цена сверх бумажной, ¢":number(pd.Series(v["slip"],dtype=float).mean()),
                         "Edge при исполнении, ¢":number(pd.Series(v["edge"],dtype=float).mean())})
    return pd.DataFrame(rows)


def shadow_w04_refresh(events):
    opened, rows = {}, []
    for e in events:
        if e.get("what") == "position_open" and e.get("hyp") == "W04":
            opened[(SLOT["W04"]+"|"+e["event"],e["ticker"])] = e
        elif e.get("what") == "settled":
            order = opened.pop((e.get("key"),e.get("ticker")),None)
            if order and order.get("fair_now") is not None and order.get("count",0)>0:
                edge = order["fair_now"]-order["price"]-order["fee"]/order["count"]
                rows.append((pd.Timestamp(order["t"]),edge,e["pnl"]))
    data = pd.DataFrame(rows,columns=["entry","edge","pnl"])
    result = []
    if not data.empty:
        cutoff = pd.Timestamp("2026-11-21",tz="UTC")
        for label,g in [("До 21 ноября",data[data.entry<cutoff]),("С 21 ноября",data[data.entry>=cutoff])]:
            for cents in (0,1,2):
                kept = g[g.edge.ge(cents/100-1e-9)]
                result.append({"Период UTC":label,"Минимум при открытии, ¢":cents,"Открытий":len(kept),
                               "Убыточных":int(kept.pnl.lt(0).sum()),"Нетто пачки, $":round(kept.pnl.sum(),2)})
    return pd.DataFrame(result)


def shadow_edge(enabled):
    """Trace actual take decisions through the production-book simulator by intent id."""
    signals_path, journal = LAB/"intents.jsonl", config.DATA/"trader/journal_shadow.jsonl"
    if not signals_path.exists() or not journal.exists():
        return pd.DataFrame()
    takes = {r["id"]:r for r in journal_events(signals_path)
             if r.get("hyp") in enabled and r.get("kind")=="take" and r.get("edge") is not None}
    results = {}
    for e in journal_events(journal):
        key = e.get("intent")
        if key in takes and e.get("what") in ("position_open","skip"):
            results[key] = e
    rows = []
    for rule in sorted(enabled):
        for low,band in [(True,"< 1¢"),(False,"≥ 1¢")]:
            bucket = [k for k in results if takes[k]["hyp"]==rule and (takes[k]["edge"]<.01)==low]
            fills = [k for k in bucket if results[k]["what"]=="position_open"]
            paired = [k for k in fills if results[k].get("fair_now") is not None and results[k].get("filled",0)>0
                      and results[k].get("fee") is not None and results[k].get("price") is not None]
            if not bucket:
                continue
            decision = [100*takes[k]["edge"] for k in paired]
            refresh = [100*(results[k]["fair_now"]-results[k]["price"]-results[k]["fee"]/results[k]["filled"]) for k in paired]
            rows.append({"Правило":rule,"Edge решения":band,"Решений с ответом":len(bucket),"Теневых открытий":len(fills),
                         "Просрочено":sum(str(results[k].get("reason","")).startswith("stale intent") for k in bucket),
                         "Пар оценок":len(paired),"При сигнале, ¢":number(pd.Series(decision,dtype=float).mean()),
                         "При открытии, ¢":number(pd.Series(refresh,dtype=float).mean())})
    return pd.DataFrame(rows)


def edge_research():
    folder = config.RESULTS / "lab"
    path, meta_path = folder/"edge_timing.csv", folder/"edge_timing_meta.json"
    if not path.exists() or not meta_path.exists():
        return "Данные пересчёта ещё не собраны.", pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    data = pd.read_csv(path)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    late = data[data["sample"].eq("Поздний")].iloc[0]
    note = (f"В ретроспективе базовый вариант воспроизвёл {meta['matches']} из {meta['paper']} входов W04 "
            f"по секунде, страйку и стороне (всего в симуляции {meta['baseline']}). На позднем отрезке "
            f"было {late.observed} часов с наблюдаемым стаканом. Для {late.hours} известны выплаты всех сравниваемых входов.")
    rows = []
    for r in data.itertuples(index=False):
        rows.append({"Период":r.sample,"Порог":r.policy,"Входы / часы":f"{r.entries} / {r.hours}",
                     "Edge, ¢/вход":number(r.model_c_trade),"Факт, ¢/вход":number(r.net_c_trade),
                     "Edge, ¢/час":number(r.model_c_hour),"Факт, ¢/час":number(r.net_c_hour),
                     "Разница с W04, ¢/час":number(r.delta_c_hour,signed=True),
                     "95% интервал разницы":f"{number(r.lo)} … {number(r.hi)}",
                     "Убытки":r.losses,"Цена и edge сохранились":f"{r.next_ok} / {r.next_obs}",
                     "Нетто, $":number(r.net,signed=True),
                     "Δ на всех часах¹, ¢/час":f"{number(r.full_low)} … {number(r.full_high)}"})
    d = pd.DataFrame(rows)
    return (note, d[d["Период"].eq("Поздний")].drop(columns="Период"), d[d["Период"].eq("Ранний")].drop(columns="Период"),
            pd.DataFrame(meta.get("attribution",[])))


def forward_w12(closed,ledger):
    rules = ["W04","W12"]
    opened = ledger[ledger.hyp.isin(rules) & ledger.event.eq("OPEN")].copy()
    left = opened["info"].map(lambda s:json.loads(s).get("left_min") if pd.notna(s) else None)
    opened["hour"] = (pd.to_datetime(opened.ts,utc=True,format="mixed") + pd.to_timedelta(left,unit="min")).dt.round("h")
    opened = opened[opened.hour.ge(W12_FIRST_EXPIRY)]
    pending = opened.loc[~opened.id.isin(closed.id),"hour"].unique()
    g = closed[closed.hyp.isin(rules) & closed.hour.ge(W12_FIRST_EXPIRY) & ~closed.hour.isin(pending)]
    hours = g.hour.nunique()
    message = (f"Первая экспирация W12 — {W12_FIRST_EXPIRY:%d.%m.%Y %H:%M} UTC. "
               f"Завершено {hours} часов со входом хотя бы одного правила. Ожидают исхода или сверки: {len(pending)}.")
    if hours == 0:
        return message,pd.DataFrame()
    rows = []
    for rule in rules:
        x = g[g.hyp.eq(rule)]
        rows.append({"Правило":rule, "Входы":len(x), "Часы без входа":hours-x.hour.nunique(),
                     "Edge, ¢/вход":number(100*x.edge.mean()),"Факт, ¢/общий час":number(100*x.pnl.sum()/hours),
                     "Нетто, $":number(x.pnl.sum(),signed=True)})
    return message,pd.DataFrame(rows)


def staged_results():
    path = config.RESULTS / "lab/scale_in.csv"
    if not path.exists():
        return "Нет пересчёта.", pd.DataFrame()
    data = pd.read_csv(path)
    matched = data[data.status.eq("ok")].copy()
    if matched.empty:
        return "Подходящих парных наблюдений пока нет.", pd.DataFrame()
    hours = pd.to_datetime(matched.hour, utc=True)
    before = hours < pd.Timestamp("2026-11-09",tz="UTC")
    text = (f"Период {hours.min():%d.%m.%Y}–{hours.max():%d.%m.%Y}. Сопоставлено {len(matched)} из {len(data)} сигналов. "
            f"Последующие подходящие снимки есть у {int(matched.followup.sum())} сигналов. "
            f"Один контракт остался в {(matched.contracts==1).sum()} случаях, два — в {(matched.contracts==2).sum()}, "
            f"все три набраны в {(matched.contracts==3).sum()}.")
    rows = []
    for period,g in (("Весь период",matched),("До 09.11",matched[before]),("С 09.11",matched[~before])):
        if g.empty:
            continue
        paired = g.groupby("hour").delta.sum()
        margin = student_t.ppf(.975,len(paired)-1)*paired.sem() if len(paired)>1 else float("nan")
        rows.append({"Период":period, "Сигналы":len(g),"Сразу / частями, контр.":f"{3*len(g)} / {int(g.contracts.sum())}",
                     "Сразу, $":number(g.base.sum(),signed=True),"Частями, $":number(g.staged.sum(),signed=True),
                     "Только 1 сразу, $":number((g.staged-g.additions_net).sum(),signed=True),
                     "Доборы, $":number(g.additions_net.sum(),signed=True),
                     "Δ на сигнал, ¢":number(100*g.delta.mean(),signed=True),
                     "¢/контр. сразу / частями":f"{number(100*g.base.sum()/(3*len(g)))} / {number(100*g.staged.sum()/g.contracts.sum())}",
                     "95% интервал Δ, ¢/час":f"{number(100*(paired.mean()-margin))} … {number(100*(paired.mean()+margin))}",
                     "Вход сразу / частями, ¢":f"{number(100*g.price.mean())} / {number(100*(g.avg_price*g.contracts).sum()/g.contracts.sum())}",
                     "Худший сигнал сразу / частями, $":f"{number(g.base.min())} / {number(g.staged.min())}"})
    return text, pd.DataFrame(rows)


def screenshot_example():
    path = config.RESULTS / "lab/scale_in_example.csv"
    if not path.exists():
        return ""
    d = pd.read_csv(path)
    d["МСК"] = pd.to_datetime(d.ts,utc=True).dt.tz_convert("Europe/Moscow").dt.strftime("%H:%M")
    shown = d[d["МСК"].isin(["22:37","22:39","22:40","22:42","22:43","22:44","22:45","22:47","22:49"])].copy()
    if shown.empty:
        return ""
    cols = {"bid":"Bid YES, ¢","fair_t4":"Цена YES по модели, ¢","edge":"Продажа YES · edge нетто, ¢"}
    for col in cols:
        shown[col] = (100*shown[col]).round(2)
    trace = d[d["МСК"].between("22:37","22:50")].sort_values("ts")
    times = pd.to_datetime(trace.ts,utc=True)
    start = pd.Timestamp("2026-11-20 19:37Z")
    scale = max(.60,trace[["fair_t4","bid"]].max().max())
    chart = ["<div class='chart'><h3>Стакан и модель, ¢ за YES</h3>"
             "<svg viewBox='0 0 690 160' role='img' aria-label='Bid YES и цена модели на страйке 58 800'>"]
    for value in (0,.2,.4,.6):
        y = 127-114*value/scale
        chart.append(f"<line x1='34' x2='678' y1='{y}' y2='{y}'/><text x='3' y='{y+4}'>{100*value:.0f}</text>")
    for col,style in (("bid","stroke:#00745e"),("fair_t4","stroke:#ad582d;stroke-dasharray:5 3")):
        points = " ".join(f"{34+(ts-start).total_seconds()/840*644:.1f},{127-114*v/scale:.1f}"
                          for ts,v in zip(times,trace[col]))
        chart.append(f"<polyline style='{style}' points='{points}'/>")
    chart.append("<text x='34' y='151'>22:37 МСК</text><text x='678' y='151' text-anchor='end'>22:51</text>"
                 "</svg><p class='note'>Сплошная линия — bid YES, пунктир — модель t₄. Точки соединяют доступные снимки.</p></div>")
    return ("<h3>Контракт со скриншота · BTC от $58 800, 20 ноября в 15:00 Нью-Йорка</h3>"
            "<p>Последние сделки на графике биржи и доступный bid могут различаться. Здесь реальные снимки стакана "
            "и оценка t₄ в те же секунды. При росте вероятности выплаты YES продажа даже по более высокой цене "
            "может иметь отрицательный edge.</p>" + "".join(chart) + table(shown[["МСК",*cols]].rename(columns=cols)))


def slot_results(closed):
    """One paper contract per timing window and expiry, with ties in ledger order."""
    candidates = closed.loc[closed.hyp.isin(SLOT) & closed.hour.notna()].copy()
    candidates["slot"] = candidates.hyp.map(SLOT)
    candidates["per_contract"] = candidates.pnl / candidates.qty
    first = candidates.sort_values(["ts"], kind="stable").drop_duplicates(["slot", "hour"])
    rows = []
    for name, ids in [("Раннее · W04/W06", ["early"]), ("Позднее · W03/W05", ["late"]),
                      ("Оба окна", ["early", "late"])]:
        g = first[first.slot.isin(ids)]
        hourly = g.groupby("hour").per_contract.sum()
        rows.append({"Окно":name, "Входы":len(g), "Повторных входов снято":int(candidates.slot.isin(ids).sum())-len(g),
                     "Часы":len(hourly), "Нетто, $":number(hourly.sum(), signed=True),
                     "¢/активный час":number(100*hourly.mean(), signed=True), "Худший час, $":number(hourly.min())})
    return pd.DataFrame(rows)


def overview(active, allowed_paper):
    means = (100 * active.groupby("hyp").pnl.sum() / active.groupby("hyp").qty.sum()).sort_values(ascending=False)
    scale = max(means.abs().max(), .01)
    bars = []
    for rule, value in means.items():
        width = 49 * abs(value) / scale
        side = "left:50%" if value >= 0 else "right:50%;background:#b95e4f"
        bars.append(f"<a class='barrow' href='#{escape(rule)}'><b>{escape(rule)}</b>"
                    f"<span class='track'><i style='{side};width:{width:.2f}%'></i></span>"
                    f"<span>{number(value,signed=True)}¢</span></a>")
    model_lines = [f"<p><a href='#{escape(r['Правило'])}'>{escape(r['Правило'])}</a> · "
                   f"<b>{r['Средний edge нетто, ¢']}¢</b> ожидаемо · {r['Факт, ¢/контр.']}¢ по исходам "
                   f"({r['Допустимые бумажные входы']} входов)</p>" for _,r in allowed_paper.iterrows()]
    return ("<div class='overview'><div><h3>Чистый результат на контракт</h3>" + "".join(bars) +
            "</div><div><h3>Edge правил с допуском к заявкам</h3>" + "".join(model_lines) +
            "<p class='note'>На контракт после комиссии, по закрытым бумажным входам разрешённого типа.</p>"
            "<a href='#model'>Модель и рынок ↓</a></div></div>")


def history():
    path = config.RESULTS / "studies/wings_weekend2.csv"
    if not path.exists():
        return pd.DataFrame()
    d = pd.read_csv(path)
    d = d[d.rule.str.startswith(("W08", "W11"))].copy()
    d["sample"] = d["sample"].replace({"Jul–Aug (found)":"Июль–август · подбор", "Sep–Oct":"Сентябрь–октябрь · проверка"})
    d["rule"] = d["rule"].str[:3]
    return d.rename(columns={"rule":"Правило", "sample":"Период", "hours":"Часы", "trades":"Сделки", "¢/trade":"¢/сделку",
                             "$/hour":"$/час", "worst hour $":"Худший час, $", "hit %":"В плюс, %"})[
        ["Период","Правило","Часы","Сделки","¢/сделку","$/час","Худший час, $","В плюс, %"]].round(3).reset_index(drop=True)


MODEL_LABELS = {"t4":"Исходная t₄", "normal":"Нормальная", "mid":"Середина стакана",
                "market_cal":"Стакан с поправкой по дням недели", "t4_cal":"t₄ с поправкой",
                "hybrid":"Смесь t₄ и стакана", "depth":"Смесь с глубиной стакана", "w06":"Модель W06",
                "odds_pool":"Смесь шансов t₄ и рынка", "pair":"Среднее W04 и W06",
                "calendar_only":"Только рынок и календарь (обучение до сентября)"}


def forward_calibration(now):
    recent = now.floor("D")-pd.Timedelta(days=29)
    parts = []
    for path in sorted((LAB/"wing_forecasts").glob("*.csv")):
        if pd.Timestamp(path.stem,tz="UTC") < recent:
            continue
        frame = pd.read_csv(path)
        parts.append(frame[frame.version.eq(CAL_VERSION)])
    if not parts or not any(len(g) for g in parts):
        return "Ожидается первый подходящий стакан для параллельной записи прогнозов.",pd.DataFrame()
    q = pd.concat(parts,ignore_index=True).dropna(subset=["market","end","ts","recorded_ts"]).drop_duplicates(["market","ts"])
    for col in ("end","ts","recorded_ts"):
        q[col] = pd.to_datetime(q[col],utc=True,format="mixed")
    q = q[(q.recorded_ts<q.end) & (q.recorded_ts<=now)].copy()
    cache = LAB/"resolutions.json"
    outcomes = json.loads(cache.read_text(encoding="utf-8")) if cache.exists() else {}
    y = ("kalshi|"+q.market).map(outcomes).where(q.end<=now)
    archive = config.RESULTS/"lab/wing_forecast_settlements.json"
    if archive.exists():
        values = json.loads(archive.read_text(encoding="utf-8"))
        ladder = q.market.str.extract(r"^(KXBTCD-[0-9A-Z]+)-T(\d+(?:\.\d+)?)$")
        settle = ladder[0].map({event:r["value"] for event,r in values.items()})
        strike = pd.to_numeric(ladder[1],errors="coerce")
        known = settle.notna() & strike.notna() & q.end.le(now)
        y = y.fillna(settle.gt(strike).where(known).astype(float))
    q["y"] = y.where(q.cheap_yes,1-y)
    q["left"] = (q.end-q.ts).dt.total_seconds()/60
    q["pair"] = (q.t4+q.w06)/2
    models = ["t4","mid","w06","pair","market_cal","t4_cal","hybrid"]
    fit_file = config.RESULTS/"lab/wings_odds_pool.json"
    if fit_file.exists():
        coef = json.loads(fit_file.read_text(encoding="utf-8"))["coefficients"]
        q["odds_pool"] = pooled_probability(q.t4,q.mid,q.end.dt.day_of_week>=5,coef)
        models.append("odds_pool")
    if {"bid_size","ask_size","ask","bid","cheap_yes"}.issubset(q.columns):
        q["depth"] = depth_hybrid(q, PARAMS)
        models.append("depth")
    g = calibration_sample(q.dropna(subset=models))
    note = (f"За последние 30 дней записано {len(q)} снимков по {q.market.nunique()} страйкам. "
            f"В парной проверке уже {g.end.nunique()} экспираций. Последняя запись {stamp(q.recorded_ts.max())} UTC.")
    if g.empty:
        return note,pd.DataFrame()
    scores = [{"Модель":MODEL_LABELS[m],"Прогноз, %":number(100*(g[m]*g.weight).sum()/g.weight.sum()),
               "Выплата, %":number(100*(g.y*g.weight).sum()/g.weight.sum()),
               "Brier":number(((g[m]-g.y)**2*g.weight).sum()/g.weight.sum(),5)} for m in models]
    # Replay on identical pre-settlement snapshots. Unknown selected outcomes
    # must stay unknown; a strategy skipping an expiry earns zero there.
    if {"bid","ask","cheap_yes"}.issubset(q.columns):
        from lab.core import kalshi_fee
        quotes = q[q.end.le(now)].dropna(subset=models).copy()
        wing = quotes.bid.where(quotes.cheap_yes,1-quotes.ask)
        quotes["fee"] = (1-wing).map(kalshi_fee)
        quotes["wing"] = wing
        index = pd.Index(quotes.end.unique())
        returns, selected = {}, {}
        for m in models:
            edge = (wing-quotes[m]-quotes.fee).round(10)
            chosen = quotes.assign(edge=edge)[edge.ge(.01)].sort_values(
                ["end","ts","edge","market"],ascending=[True,True,False,True]).drop_duplicates("end").set_index("end")
            returns[m] = (chosen.wing-chosen.y-chosen.fee).reindex(index,fill_value=0.)
            selected[m] = chosen.index
        common = pd.DataFrame(returns).dropna()
        for row,m in zip(scores,models):
            row.update({"Общие часы для прибыли":len(common),"Входы":len(selected[m].intersection(common.index)),
                        "Нетто, $":number(common[m].sum(),signed=True) if len(common) else "—"})
        note += " Расчётный вход при первом edge от 1¢ по цене сохранённого снимка. Исполнимость после записи прогноза не проверена."
    return note,pd.DataFrame(scores)


def calibration_report(now):
    path = config.RESULTS/"lab/wing_calibration.json"
    html, markdown = [], []
    odds_file = config.RESULTS/"lab/wings_odds_pool.json"
    ml_file = config.RESULTS/"lab/wing_regime_ml.json"
    if ml_file.exists():
        ml = json.loads(ml_file.read_text(encoding="utf-8"))
        labels = {"t4":"Исходная t₄","calendar":"Стакан и календарь","fhs_all":"Исторические хвосты",
                  "fhs":"Исторические хвосты по режиму","plain":"Бустинг без признаков режима","trees":"Бустинг с признаками режима"}
        frame = pd.DataFrame(ml["rows"])
        frame["Прогноз"] = frame["Прогноз"].map(labels)
        note = (f"Обучение до сентября: {ml['training_hours']} экспираций. Для всех строк действует одна пауза "
                "волатильности и порог edge 1¢. Brier рассчитан на допущенных снимках, пропущенные часы дают ноль "
                f"для прибыли. Срез до {stamp(pd.Timestamp(ml['until']))} UTC.")
        html.append(f"<details><summary>ML и исторические хвосты BTC</summary><p>{note}</p>{table(frame)}</details>")
        markdown.extend(["### ML и исторические хвосты BTC",note,md(frame)])
    if odds_file.exists():
        pool = json.loads(odds_file.read_text(encoding="utf-8"))
        a,b,w = pool["coefficients"]
        desc = (f"Смесь обучена на {pool['train_hours']} экспирациях июля–августа с суточным разрывом до сентября. "
                f"Вес t₄ в логарифме шансов {number(100*w,1)}%, сдвиг будней {number(a,3)}, выходных {number(b,3)}. "
                "Правило входа одинаковое для всех прогнозов: один контракт при первом чистом edge от 1¢. "
                "До конца октября — реконструкция по минутным свечам без глубины, затем записанные стаканы. "
                f"Срез до {stamp(pd.Timestamp(pool['until']))} UTC.")
        frame = pd.DataFrame(pool["rows"])
        frame["Прогноз"] = frame["Прогноз"].map(MODEL_LABELS)
        html.append(f"<details><summary>Смесь шансов t₄ и рынка</summary><p>{desc}</p>{table(frame)}</details>")
        markdown.extend(["### Смесь шансов t₄ и рынка",desc,md(frame)])
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        params = data["params"]
        desc = (f"Поправка к цене стакана обучена на {data['history_hours']} часах июль–октябрь, отдельно для будней "
                f"и выходных. Поправка t₄ и вес смеси обучены на {data['train_hours']} часах 30 октября – 7 ноября. "
                f"В смеси {number(100*params['weight'],1)}% скорректированной t₄. Срез до {stamp(pd.Timestamp(data['until']))} UTC.")
        html.append(f"<p>{desc}</p>")
        markdown.append(desc)
        check = pd.DataFrame(data.get("archive_check",[]))
        if not check.empty:
            title = "Проверка поправки к стакану на сентябре–октябре"
            html.append(f"<details><summary>{title}</summary><p>Для этой проверки коэффициенты обучены только на июле–августе.</p>{table(check)}</details>")
            markdown.extend(["### "+title, "Обучение на июле–августе.",md(check)])
        for p in data["periods"]:
            if not p["rows"]:
                continue
            note = (f"{p['contracts']} страйков, {p['scored']} экспираций с известным исходом крыла. "
                    f"Взвешенная частота выплаты {number(100*p['payout'])}%. "
                    f"Сделки сравниваются на {p['trade_hours']} из {p['hours']} часов со стаканом. "
                    f"Страйков без исхода: {p['unresolved_markets']}.")
            rows = []
            for r in p["rows"]:
                ci = r["pnl_ci"]
                rows.append({"Модель":MODEL_LABELS[r["model"]],"Прогноз крыла, %":number(100*r["probability"]),
                             "Brier":number(r["brier"],5),"Log-loss":number(r["logloss"],4),
                             "Входы":r["entries"],"Убытки":r["losses"],"Нетто, $":number(r["net"],signed=True),
                             "¢/общий час":number(r["net_hour"]),
                             "95% интервал Δ прибыли к t₄, ¢/час":("—" if ci[0] is None else f"{number(ci[0])} … {number(ci[1])}")})
            frame = pd.DataFrame(rows)
            html.extend([f"<h3>{escape(p['period'])}</h3><p>{note}</p>",table(frame)])
            markdown.extend(["### "+p["period"],note,md(frame)])
    else:
        html.append("<p>Исторический расчёт ещё не выполнен.</p>")
    note, forward = forward_calibration(now)
    html.append(f"<h3>Прогнозы, записанные до экспирации</h3><p>{note}</p>")
    markdown.extend(["### Прогнозы после фиксации",note])
    if not forward.empty:
        html.append(table(forward))
        markdown.append(md(forward))
    return "".join(html),"\n\n".join(markdown)


def write(ledger, now):
    closed, opened = prepare(ledger)
    configured = Settings.load()
    enabled = set(configured.rules)
    rules = sorted((h for h in signals.REGISTRY if h.family == "W"), key=lambda h:(h.id not in enabled, bool(h.retired), h.id))
    rows, panels, choices, models = [], [], [], []
    for h in rules:
        g, op = closed[closed.hyp == h.id], opened[opened.hyp == h.id]
        status = "Допуск к заявкам" if h.id in enabled else ("Архив" if h.retired else "Только бумага")
        ms = model_stats(g)
        rows.append({"ID":h.id, "Стратегия":NAMES.get(h.id,h.title), "Статус":status,
                     **metrics(g), "Edge нетто, ¢":ms["Edge нетто, ¢"], "Открыто":len(op)})
        if not h.retired:
            models.append({"Правило":h.id, **ms})
        choices.append(f"<option value='{h.id}'>{h.id} · {escape(NAMES.get(h.id,h.title))}</option>")
        panels.append(details(h, g, op, status))
    summary = pd.DataFrame(rows)
    comparable = []
    for rule in sorted(enabled):
        selection = closed[closed.hyp.eq(rule)]
        if rule not in configured.maker_rules:
            selection = selection[selection["fill"].eq("Taker")]
        comparable.append({"Правило":rule, "Допустимые бумажные входы":len(selection),
                           "Средний edge нетто, ¢":model_stats(selection)["Edge нетто, ¢"],
                           "Факт, ¢/контр.":metrics(selection)["¢/контр."]})
    allowed_paper = pd.DataFrame(comparable)
    active = closed[closed.hyp.isin([h.id for h in rules if not h.retired])]
    period = f"{stamp(closed.ts.min())} — {stamp(pd.concat([closed.exit_ts, opened.ts]).max())} UTC"
    hist, exe, slots = history(), execution(), slot_results(closed)
    scale_note, scale = staged_results()
    routes = execution_by_rule()
    traced = shadow_edge(enabled)
    demo_path = config.DATA/"trader/journal_demo.jsonl"
    demo = pd.DataFrame([demo_checks(journal_events(demo_path))]) if demo_path.exists() else pd.DataFrame()
    edge_note, validation, training, attribution = edge_research()
    forward_note, forward = forward_w12(closed,ledger)
    calibrated, calibrated_md = calibration_report(now)
    from lab.wing_regime import read as read_regime
    regime = read_regime(now)
    regime_note = ("ПАУЗА заявок" if regime["blocked"] else "Входы разрешены фильтром волатильности") + " · " + regime["reason"]
    if regime.get("range_pct") is not None:
        regime_note += f" · диапазон 60 мин {number(regime['range_pct'])}% · RV 30 мин {number(regime['rv_pct'])}%"
    if regime.get("minute"):
        regime_note += " · "+stamp(pd.Timestamp(regime["minute"]))+" UTC"
    regime_file = config.RESULTS/"lab/wing_regime_replay.csv"
    regime_rows = pd.read_csv(regime_file) if regime_file.exists() else pd.DataFrame()
    agreement_file = config.RESULTS/"lab/wings_two_sided.csv"
    agreement = pd.read_csv(agreement_file) if agreement_file.exists() else pd.DataFrame()
    archive_file = config.RESULTS/"lab/wing_archive_replay.csv"
    archive_replay = pd.read_csv(archive_file) if archive_file.exists() else pd.DataFrame()
    capacity_tables=[]
    stopping_file = config.RESULTS / "lab/wing_stopping.csv"
    if stopping_file.exists():
        cols={"model":"Момент входа","archive_signals":"Сигналы по свечам","archive_fills":"Исполнения",
              "archive_net":"Архив, $","archive_nov":"Архив с ноября, $","archive_p_holm":"p архива",
              "book_signals":"Сигналы по стаканам","book_fills":"Исполнения по стаканам",
              "book_contracts":"Контракты","book_net":"Нетто по стаканам, $","book_p_holm":"p стаканов"}
        frame=pd.read_csv(stopping_file)
        capacity_tables.append("<h3>Когда забирать edge</h3><p>Первый вход, подтверждение, улучшение "
                               "цены или приближение дедлайна. База — первый подходящий снимок. "
                               "Сигналов может стать меньше, а страйк и экспирация входа могут измениться.</p>"
                               +table(frame[list(cols)].rename(columns=cols)))
    shapes_file = config.RESULTS / "lab/wing_shapes.csv"
    if shapes_file.exists():
        shapes = pd.read_csv(shapes_file)
        cols = {"model":"Кривая","archive_hours":"Часы свечей","archive_fills":"Входы",
                "archive_net":"Архив, $","archive_nov":"Архив с ноября, $","archive_p_holm":"p архива",
                "book_fills":"Входы с объёмом","book_contracts":"Контракты","book_net":"Объём, $",
                "book_p_holm":"p стаканов"}
        capacity_tables.append("<h3>Вероятность крыла по форме лестницы</h3>"
                               "<p>Оцениваем каждый страйк по остальным котировкам того же снимка, "
                               "усредняем прогноз с t₄. В архиве проверяется один контракт минутой позже, "
                               "на локальных стаканах — объём в пределах исходного лимита.</p>"
                               + table(shapes[list(cols)].rename(columns=cols)))
    sizing_file = config.RESULTS / "lab/wing_sizing.csv"
    if sizing_file.exists():
        sizing = pd.read_csv(sizing_file).rename(columns={"rule":"Размер позиции","contracts":"Контракты",
                "cost":"Затраты, $","net":"Нетто, $","drawdown":"Макс. просадка, $","worst":"Худший час, $",
                "late":"С 20 ноября, $","p_holm":"p Холма"})
        capacity_tables.append("<h3>Десять способов выбрать объём позиции</h3>"
                               "<p>Одинаковые сигналы соседних страйков и начальный капитал $10 000. "
                               "Проверка лимита на повторном стакане, комиссия пересчитана для каждого объёма. "
                               "Просадка измеряется по закрытым экспирациям.</p>"+table(sizing))
    filters_file = config.RESULTS / "lab/wing_filters.csv"
    if filters_file.exists():
        filters = pd.read_csv(filters_file).rename(columns={"model":"Правило","sample":"Данные",
                 "hours":"Часы","orders":"Сигналы","fills":"Входы","contracts":"Контракты",
                 "net":"Нетто, $","cost":"Затраты, $","worst":"Худший час, $","late":"С 20 ноября, $",
                 "p_holm":"p Холма"})
        capacity_tables.append("<h3>Десять фильтров входа для соседних страйков</h3>"
                               "<p>Архив включает часы сна и проверяет один контракт минутой позже. "
                               "Объём рассчитан по доступным локальным снимкам с задержкой до 90 секунд. "
                               "Каждая серия сравнивается со своей базой.</p>"+table(filters))
    ten_file = config.RESULTS / "lab/wing_ten.csv"
    if ten_file.exists():
        capacity_tables.append("<h3>Десять гипотез против модели соседних страйков</h3>"
                               "<p>Один контракт, проверка исполнения через минуту, комиссия включена. "
                               "Δ показывает прибавку к среднему прогноза t₄ и соседних страйков. "
                               "p Holm учитывает десять одновременных сравнений. "
                               "Столбец с 20 ноября входит в итог за весь период.</p>"
                               + table(pd.read_csv(ten_file)))
    spline_capacity = config.RESULTS / "lab/wing_ten_capacity.csv"
    if spline_capacity.exists():
        check = pd.read_csv(spline_capacity).rename(columns={"model":"Модель","signals":"Сигналы",
                 "contracts":"Контракты позже","cost":"Затраты, $","net":"Нетто, $",
                 "worst":"Худшая экспирация, $","excluded_hours":"Исключено часов"})
        capacity_tables.append("<h3>Проверка сплайновой модели на доступном объёме</h3><p>"
                               "Только часы с локальной записью стакана. После выбора модели по свечам "
                               "проверяем исходный лимит на следующем снимке в пределах 90 секунд. "
                               "Затраты суммируются по всем экспирациям.</p>"+table(check))
    sleep_file = config.RESULTS / "lab/wing_sleep_summary.csv"
    if sleep_file.exists():
        capacity_tables.append("<h3>Непрерывная работа · архивный сценарий</h3>"
                               "<p>Интервалы сна взяты из журнала Windows. Для каждого часа сравниваем "
                               "один контракт по минутным bid/ask Kalshi и проверяем цену минутой позже. "
                               "Часы полного и частичного сна выделены по окну за 15–30 минут до экспирации. "
                               "Модели обучены на архиве до ноября. Brier оценивает вероятность выплаты "
                               "дешёвого крыла; меньшие значения лучше.</p>" + table(pd.read_csv(sleep_file)))
    sensitivity_file = config.RESULTS / "lab/wing_robustness.csv"
    if sensitivity_file.exists():
        capacity_tables.append("<h3>Что останется от edge, если цена ухудшится на 1¢?</h3>"
                               "<p>Для тех же сигналов сдвигаем цены покупки повторного стакана на 1¢ вверх. "
                               "Сохраняем исходный лимит и порог edge после комиссии. Средний edge взвешен "
                               "по исполненному объёму. Последний столбец показывает, сколько долларов "
                               "осталось бы после исключения самой прибыльной экспирации.</p>"
                               + table(pd.read_csv(sensitivity_file)))
    consensus_file = config.RESULTS / "lab/wing_consensus.csv"
    if consensus_file.exists():
        capacity_tables.append("<h3>Устойчивость сигнала соседних страйков</h3>"
                               "<p>Все четыре модели видят одни и те же котировки. mix усредняет t₄ "
                               "и соседние страйки; conservative даёт 75% веса большей вероятности "
                               "выплаты крыла; persistent берёт большее из двух последовательных "
                               "значений mix за 90 секунд. При продаже крыла большая вероятность "
                               "выплаты уменьшает допустимую цену покупки противоположной стороны. "
                               "Последний интервал начинается после среза 22 ноября, 17:00 UTC. "
                               "Доверительный интервал посчитан по дням и относится к нетто с задержкой.</p>"
                               + table(pd.read_csv(consensus_file)))
    quant_file=config.RESULTS/"lab/wing_quant_replay.json"
    if quant_file.exists():
        quant=json.loads(quant_file.read_text(encoding="utf-8"))
        frame=pd.DataFrame(quant["results"]).rename(columns={"cohort":"Выборка","model":"Модель","trades":"Входы",
              "contracts":"Контракты","net":"Нетто, $","delayed":"Нетто позже, $","excluded":"Исключено часов",
              "cost":"Затраты, $","roi":"ROI, %","delta_ci_dollars_hour":"95% Δ к t₄ с задержкой, $/час"})
        capacity_tables.append("<h3>Количественные модели на общей доступной выборке</h3>"
                               "<p>Сравнивайте каждую строку со строкой t₄ внутри той же выборки. "
                               "Доход на ценах снимка и доход при задержке указаны отдельно.</p>"+table(frame))
    quant_archive = config.RESULTS / "lab/wing_quant_archive.csv"
    if quant_archive.exists():
        capacity_tables.append("<h3>CAViaR до ноября · архивный тест</h3>"
                               "<p>Один контракт на вход. Глубина исторического стакана неизвестна. "
                               "Меньший Brier означает более точный вероятностный прогноз.</p>"
                               + table(pd.read_csv(quant_archive)))
    for file,title in [("wing_ledger_capacity.csv","Закрытые сигналы W01–W12"),("wing_capacity.csv","Модели на отложенных часах")]:
        path=config.RESULTS/"lab"/file
        if path.exists():
            data=pd.read_csv(path).rename(columns={"rule":"Правило","model":"Модель","period":"Период UTC","closed":"Закрыто",
                "matched":"Со стаканом","maker":"Мейкер","contracts":"Контракты","capital":"Затраты, $","fees":"Комиссия, $",
                "pnl":"Нетто, $","roi_pct":"ROI, %","trades":"Сделки","worst":"Худшая, $","delayed_contracts":"Объём позже",
                "delayed_pnl":"Нетто позже, $","excluded_hours":"Часы без исхода","delayed_cost":"Затраты позже, $",
                "delayed_roi":"ROI позже, %","next_books":"Есть повторный снимок","pnl_all_min":"Все часы: минимум, $",
                "pnl_all_max":"Все часы: максимум, $","delayed_pnl_all_min":"Позже: минимум, $","delayed_pnl_all_max":"Позже: максимум, $"})
            capacity_tables.append(f"<h3>{title}</h3>{table(data)}")
    neighbors_file = config.RESULTS/"lab/wing_neighbors.csv"
    neighbors = pd.read_csv(neighbors_file) if neighbors_file.exists() else pd.DataFrame()
    stress_file = config.RESULTS/"lab/w04_stress.csv"
    stress = pd.read_csv(stress_file) if stress_file.exists() else pd.DataFrame()
    shadow_file = config.DATA/"trader/journal_shadow.jsonl"
    refresh = shadow_w04_refresh(journal_events(shadow_file)) if shadow_file.exists() else pd.DataFrame()
    template = Template(Path(__file__).with_name("wing_report.html").read_text(encoding="utf-8"))
    checked = audit(closed)
    page = template.substitute(regime=escape(regime_note),regime_table=table(regime_rows) if len(regime_rows) else "",period=period, built=stamp(now), audit=checked, n=len(active), rules=sum(not h.retired for h in rules),
                               hours=active.hour.nunique(), opened=len(opened), summary=table(summary),
                               choices="".join(choices), panels="".join(panels), history=table(hist), execution=table(exe),
                               slots=table(slots), overview=overview(active, allowed_paper), models=table(pd.DataFrame(models)),
                               admitted=escape(", ".join(sorted(enabled))), allowed_paper=table(allowed_paper),
                               routes=table(routes), calibrated=calibrated, traced=table(traced), demo=table(demo), edge_note=escape(edge_note), validation=table(validation),
                               agreement=table(agreement) if len(agreement) else "Ожидается расчёт.",
                               archive_replay=table(archive_replay) if len(archive_replay) else "Ожидается расчёт.",
                               capacity="".join(capacity_tables),
                               neighbors=table(neighbors) if len(neighbors) else "Ожидается расчёт.",
                               entry_stress=table(stress) if len(stress) else "Ожидается расчёт.",
                               refresh_check=table(refresh) if len(refresh) else "Ожидаются расчёты.",
                               training=table(training), attribution=table(attribution), forward_note=escape(forward_note), forward=table(forward) if len(forward) else "",
                               scale_note=escape(scale_note), scale=table(scale), example=screenshot_example())
    atomic_write(REPORT_HTML, page)
    atomic_write(REPORT_MD, "\n\n".join(["# Alpha Lab · Стратегии крыльев BTC", period,
        "Бумажный журнал. Каждое правило оценивается отдельно. Часы сгруппированы по сроку контракта.",
        checked, md(summary), "## Калибровка вероятности крыла",calibrated_md, "## Модель и рынок", md(pd.DataFrame(models)), "## Допуск к исполнению", md(allowed_paper),
        md(routes), "### Перевес на сигнале и при теневом открытии",md(traced),"## Демо Kalshi · проверка заявок", md(demo),
        "## Выбор момента входа", edge_note, "### 9–19 ноября UTC", md(validation), "### 30 октября – 8 ноября UTC", md(training),
        "### Чем отличались входы W12 от W04 на позднем отрезке",md(attribution),
        "## W12 после фиксации правила", forward_note, md(forward),
        "## Согласие моделей и покупка дешёвого крыла", md(agreement),
        "## Архив и направление BTC за пять минут", md(archive_replay),
        "## Соседние страйки, стресс USD и повторный вход",md(neighbors),
        "## W04 · проверка устойчивости edge перед входом",md(stress),
        "## Постепенный вход W04", scale_note, md(scale), "## Один контракт на окно и час", md(slots),
        "## Выходные на исторических котировках", md(hist), "## Исполнение", md(exe),
        "## Правила", *[f"- **{h.id}** {DESCRIPTIONS.get(h.id,h.rule)}" for h in rules]]))

