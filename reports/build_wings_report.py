"""Build the Russian HTML/Markdown report for the wing-selling deep dive."""
from __future__ import annotations

import base64
import html
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "reports" / "final" / "figures"
TAB = ROOT / "reports" / "final" / "tables"

S = pd.read_csv(TAB / "wings_all_live_summary.csv")
T = pd.read_csv(TAB / "wings_all_live_trades.csv", parse_dates=["open_ts", "settle_ts"])
H = pd.read_csv(TAB / "wings_all_history_summary.csv")
M = pd.read_csv(TAB / "wings_all_history_monthly.csv", index_col=0)
PP = pd.read_csv(TAB / "wings_all_p_profit.csv")
CMP = pd.read_csv(TAB / "wings_all_compounding.csv")
B = pd.read_csv(TAB / "wings_all_buckets.csv")
EX = pd.read_csv(TAB / "wings_all_exec.csv")
ED = pd.read_csv(TAB / "wings_all_edge.csv")
L = pd.read_csv(TAB / "wings_all_w01_losses.csv", parse_dates=["open_ts", "settle_ts"])
# produced by reports/wings_edge.py
EB = pd.read_csv(TAB / "wings_edge_buckets.csv")
ET = pd.read_csv(TAB / "wings_edge_thresholds.csv")

COL = {"W01": "#94a3b8", "W02": "#475569", "W03": "#4f46e5",
       "W04": "#0d9488", "W05": "#d97706", "W06": "#db2777"}


def n(x, nd=1, plus=False):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "—"
    s = f"{x:+,.{nd}f}" if plus else f"{x:,.{nd}f}"
    return s.replace(",", " ").replace(".", ",").replace("-", "−")


def k(x):
    return f"{int(x):,}".replace(",", " ")


def tag(h):
    return f'<span class="tag" style="background:{COL[h]}">{h}</span>'


def table(headers, rows, aligns="l" * 20, cls="tbl"):
    out = [f'<div class="tw"><table class="{cls}"><thead><tr>']
    for i, htxt in enumerate(headers):
        a = aligns[i] if i < len(aligns) else "l"
        out.append(f'<th class="n">{htxt}</th>' if a == "r" else f"<th>{htxt}</th>")
    out.append("</tr></thead><tbody>")
    for row in rows:
        out.append("<tr>")
        for i, cell in enumerate(row):
            txt, klass = cell if isinstance(cell, tuple) else (cell, "")
            a = aligns[i] if i < len(aligns) else "l"
            c = " ".join(dict.fromkeys(x for x in ("n" if a == "r" else "", klass) if x))
            out.append(f'<td class="{c}">{txt}</td>' if c else f"<td>{txt}</td>")
        out.append("</tr>")
    out.append('</tbody></table></div>')
    return "".join(out)


def figure(name, caption):
    data = base64.b64encode((FIG / name).read_bytes()).decode()
    return (f'<figure class="fig"><img src="data:image/png;base64,{data}" alt="{caption}">'
            f'<figcaption class="cap">{caption}</figcaption></figure>')


def short_market(m):
    if m.startswith("bitcoin-above-"):
        parts = m.split("-")
        strike = f"{int(parts[2]):,}".replace(",", " ")
        return f"BTC &gt; {strike}"
    if m.startswith("KXBTCD-"):
        strike = m.split("-")[-1].replace("T", "")
        try:
            strike = f"{float(strike):,.2f}".replace(",", " ")
        except ValueError:
            pass
        return f"KXBTCD · {strike}"
    return m


def breakeven(h):
    g = T[T.hyp == h]
    wm = g.loc[g.pays == 1, "pnl"].mean()
    lm = g.loc[g.pays == 0, "pnl"].mean() if (g.pays == 0).any() else -(g.price.mean() + 0.01)
    return 100 * (-lm) / (wm - lm), wm, lm


# ------------------------------------------------------------------ tables
rows = []
for _, r in S.iterrows():
    be, wm, lm = breakeven(r.hyp)
    pf = "∞" if not np.isfinite(r.pf) else n(r.pf, 2)
    ci = f"[{n(r.ci_lo_c, 1)}; {n(r.ci_hi_c, 1)}]"
    rows.append([
        f"{tag(r.hyp)} <span class='mut'>{'†' if r.coverage == '29.10' else ''}</span>",
        (k(r.n), "n"), (n(r.hit, 1), "n"), (n(r.mean_c, 2, True), "n"),
        (ci, "n mut"), (n(r.total, 2, True), "n " + ("neg" if r.total < 0 else "pos")),
        (f"{n(100 * wm, 1, True)} / {n(100 * lm, 1, True)}", "n"),
        (n(be, 1), "n " + ("neg" if be > r.hit else "")),
        (pf, "n"), (n(r.max_dd, 2), "n neg"),
    ])
live_tbl = table(
    ["Правило", "Сделок", "Hit, %", "Ср. P&amp;L, ¢", "95 % интервал, ¢", "Итог, $",
     "Выигрыш / проигрыш, ¢", "Безубыт. hit, %", "PF", "Макс. просадка, $"],
    rows, aligns="lrrrrrrrrr")
live_note = ("† W01/W02 торговали только 29.10; их итог нельзя сравнивать с 52 часами W03–W06. "
             "У W01 8 открытий и 14 расчётов не сопоставлены — итог частичный. "
             "Безубыточный hit = проигрыш / (выигрыш + проигрыш); для правил без минусов проигрыш оценён "
             "по средней цене контракта + комиссия. У правил без проигрышей интервал асимметричен: верхняя "
             "граница упирается в наблюдаемый выигрыш, нижняя учитывает возможный −95¢.")

loss_rows = []
for _, r in L.iterrows():
    loss_rows.append([
        r.open_ts.strftime("%d.%m %H:%M"), r.settle_ts.strftime("%H:%M"),
        short_market(r.market), (n(r.price, 2), "n"), (n(r.pnl, 3, True), "n neg"),
    ])
loss_tbl = table(["Открыта (UTC)", "Расчёт", "Рынок", "Цена", "P&amp;L, $"], loss_rows,
                 aligns="llnrr")

bt = {
    "W01": ("−0,94", "22 908", "бэктест заранее показывал убыток: t −4,4"),
    "W02": ("−0,41", "5 707", "день не отменяет −$23,5 за четыре месяца"),
    "W03": ("+3,6 (OOS)", "427", "live выше OOS в 3,4×, но n = 8"),
    "W04": ("+1,8 (OOS)", "980", "единственное настоящее совпадение"),
    "W05": ("+4,6 (OOS)", "318", "maker-first; n = 5, статистики нет"),
    "W06": ("+3,5 (OOS)", "496", "maker-first; n = 5, статистики нет"),
}
bt_rows = []
for _, r in S.iterrows():
    b, bn, comment = bt[r.hyp]
    bt_rows.append([tag(r.hyp), (k(r.n), "n"), (n(r.mean_c, 2, True), "n"),
                    (b, "n"), (bn, "n"), f"<span class='mut'>{comment}</span>"])
bt_tbl = table(["Правило", "Live, сделок", "Live, ¢/сделку", "Бэктест, ¢/сделку",
                "Бэктест, сделок", "Комментарий"], bt_rows, aligns="lrrrrl")

hist_rows = []
for _, r in H.iterrows():
    hist_rows.append([
        tag(r.hyp), (k(r.hours), "n"), (n(r.hit, 1), "n"), (n(r.mean_c, 2, True), "n"),
        (n(r.total, 1, True), "n " + ("neg" if r.total < 0 else "pos")),
        (n(r.max_dd, 1), "n neg"), (n(r.worst_month, 1, True), "n"),
        (n(r.oos_c, 1, True), "n " + ("neg" if r.oos_c < 0 else "pos")),
    ])
hist_tbl = table(["Правило", "Активных часов", "Hit, %", "Ср. ¢/час", "Итог, $",
                  "Макс. просадка, $", "Худший месяц, $", "Sep–Oct OOS, $"],
                 hist_rows, aligns="lrrrrrrr")

month_rows = []
for idx, r in M.iterrows():
    month_rows.append([pd.to_datetime(idx).strftime("%m.%Y"), (n(r.W02, 1, True), "n"),
                       (n(r.W03, 1, True), "n"), (n(r.W04, 1, True), "n"),
                       (n(r.W02 + r.W03 + r.W04, 1, True), "n")])
month_tbl = table(["Месяц", "W02, $", "W03, $", "W04, $", "Сумма, $"], month_rows,
                  aligns="lrrrr")

pp_rows = []
for _, r in PP.iterrows():
    pp_rows.append([tag(r.hyp), (n(r.p_profit_N20, 1), "n"), (n(r.p_profit_N100, 1), "n"),
                    (n(r.p_profit_N400, 1), "n")])
pp_tbl = table(["Правило", "20 сделок", "100 сделок", "400 сделок"], pp_rows, aligns="lrrr")

cmp_rows = []
for _, r in CMP.iterrows():
    name = r.rule + (f" · f = {int(r.f)} %" if r.rule != "W03+W04" else " · 5 % each")
    cmp_rows.append([name, (f"${int(r.base)}", "n"), (f"${int(r.h1)}", "n"),
                     (f"${int(r.h2)}", "n"), (n(r.dd, 0) + " %", "n neg")])
cmp_tbl = table(["Стратегия", "Без сдвига", "−1¢", "−2¢", "Макс. просадка"], cmp_rows,
                aligns="lrrrr")

ex_rows = []
for _, r in EX.iterrows():
    ex_rows.append([("maker" if r.exec == "maker" else "taker"), (k(r.n), "n"),
                    (n(100 * r.hit, 1), "n"), (n(r.mean_c, 2, True), "n"),
                    (n(r.total, 2, True), "n " + ("neg" if r.total < 0 else "pos"))])
ex_tbl = table(["Исполнение", "Сделок", "Hit, %", "Ср. ¢", "Итог, $"], ex_rows, aligns="lrrrr")

b_rows = []
for _, r in B.iterrows():
    b_rows.append([tag(r.hyp), r.bucket, (k(r.n), "n"), (n(r.mean_c, 1, True), "n"),
                   (n(r.total, 2, True), "n " + ("neg" if r.total < 0 else "pos"))])
b_tbl = table(["Правило", "Цена крыла", "Сделок", "Ср. ¢", "Итог, $"], b_rows, aligns="llrrr")

ed_rows = []
for _, r in ED.iterrows():
    ed_rows.append([tag(r.hyp), html.escape(str(r.bucket)), (k(r.n), "n"),
                    (n(r.mean_c, 1, True), "n")])
ed_tbl = table(["Правило", "Edge", "Сделок", "Ср. ¢"], ed_rows, aligns="llrr")

# ------------------------------------------------------------------ text
CSS = """
:root{--ink:#0b1220;--muted:#5b6472;--line:#e6ebf3;--bg:#f4f6fb;--indigo:#4f46e5;--teal:#0d9488;--red:#dc2626}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,"Helvetica Neue",Arial,sans-serif;line-height:1.55}
.wrap{max-width:1180px;margin:0 auto;padding:28px 22px 64px}
.hero{background:radial-gradient(1200px 500px at 10% 0%,rgba(255,255,255,.18),transparent 60%),linear-gradient(120deg,#312e81,#4f46e5 52%,#0d9488);color:#fff;border-radius:26px;padding:42px 46px 38px;box-shadow:0 24px 60px -30px rgba(49,46,129,.55)}
.hero h1{font-size:38px;line-height:1.12;margin:0 0 10px;letter-spacing:-.02em}
.hero p{margin:0;font-size:16px;color:#dbe3ff;max-width:920px}
.badge{display:inline-flex;gap:8px;align-items:center;background:rgba(255,255,255,.14);border:1px solid rgba(255,255,255,.25);padding:6px 12px;border-radius:999px;font-size:12.5px;margin-bottom:18px}
.kpis{display:grid;grid-template-columns:repeat(4,1fr);gap:16px;margin:22px 0 8px}
.kpi{background:#fff;border:1px solid var(--line);border-radius:18px;padding:18px 20px;box-shadow:0 10px 24px -20px rgba(15,23,42,.25)}
.kpi .v{font-size:29px;font-weight:800;letter-spacing:-.02em;line-height:1.15}
.kpi .l{font-size:12.5px;color:var(--muted);margin-top:3px}
.kpi .s{font-size:12px;color:var(--muted);margin-top:8px}
section{background:#fff;border:1px solid var(--line);border-radius:22px;padding:28px 32px;margin-top:22px;box-shadow:0 12px 30px -26px rgba(15,23,42,.3)}
h2{font-size:23px;margin:0 0 6px;letter-spacing:-.01em}
h2 .num{display:inline-block;background:#eef2ff;color:#4338ca;border-radius:10px;padding:2px 10px;font-size:15px;margin-right:10px;vertical-align:2px}
.lead{color:var(--muted);margin:0 0 18px;font-size:15px}
h3{font-size:17px;margin:24px 0 8px}
p{margin:0 0 12px;font-size:15px}
ul{margin:0 0 12px 20px;padding:0;font-size:15px}
li{margin:5px 0}
.fig{margin:16px 0 8px;border:1px solid var(--line);border-radius:16px;overflow:hidden;background:#fff}
.fig img{width:100%;display:block}
.cap{font-size:13px;color:var(--muted);padding:10px 14px;border-top:1px solid var(--line);background:#fbfcfe}
.note{border-left:4px solid var(--indigo);background:#f5f7ff;padding:12px 16px;border-radius:10px;margin:14px 0;font-size:14.5px}
.warn{border-left:4px solid var(--red);background:#fff5f5;padding:12px 16px;border-radius:10px;margin:14px 0;font-size:14.5px}
table.tbl{width:100%;border-collapse:collapse;font-size:13.5px;margin:10px 0 4px;font-variant-numeric:tabular-nums}
.tbl th{background:#f3f6fb;color:#475569;font-weight:700;text-align:left;padding:9px 10px;border-bottom:1px solid var(--line);font-size:12.5px;white-space:nowrap}
.tbl td{padding:8px 10px;border-bottom:1px solid #eef2f7;vertical-align:top}
.tbl tr:nth-child(even) td{background:#fbfcfe}
.tbl td.n{text-align:right;white-space:nowrap}
.tbl th.n{text-align:right}
.neg{color:var(--red);font-weight:700}
.pos{color:#047857;font-weight:600}
.mut{color:var(--muted)}
.tag{display:inline-block;padding:1px 7px;border-radius:999px;font-size:11.5px;font-weight:700;color:#fff}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:20px}
.tw{overflow-x:auto;-webkit-overflow-scrolling:touch}
.toc{display:flex;flex-wrap:wrap;gap:8px;margin:14px 0 0;padding:0 2px}
.toc a{font-size:12.5px;color:#4338ca;background:#eef2ff;border:1px solid #e0e7ff;text-decoration:none;padding:5px 11px;border-radius:999px}
.toc a:hover{background:#e0e7ff}
@media print{body{background:#fff}section,.hero{box-shadow:none;break-inside:avoid}}
.small{font-size:12.5px;color:var(--muted)}
footer{color:var(--muted);font-size:12.5px;margin-top:26px;text-align:center}
@media(max-width:900px){.kpis{grid-template-columns:repeat(2,1fr)}.grid2{grid-template-columns:1fr}.hero{padding:30px}.hero h1{font-size:29px}section{padding:22px}}
"""

BUCKET_ORDER = ["0–0.5", "0.5–1", "1–1.5", "1.5–2", "2–3", "3–4", "≥4"]
eb_rows = []
for lab in BUCKET_ORDER:
    row = [lab.replace(".", ",")]
    for h in ("W03", "W04"):
        r = EB[(EB.hyp == h) & (EB.half == "test") & (EB.bucket == lab)].iloc[0]
        if int(r.n) == 0:
            row += ["—", "—", "—"]
        else:
            row += [k(r.n), n(r.mean_c, 1, True), n(r.share_total, 0, True) + " %"]
    eb_rows.append(row)
edge_bucket_tbl = table(["Edge, ¢", "W03 сделок", "W03 ¢", "W03 % приб.",
                         "W04 сделок", "W04 ¢", "W04 % приб."], eb_rows, aligns="lrrrrrr")

et_rows = []
for h in ("W03", "W04"):
    for _, r in ET[ET.hyp == h].sort_values("margin").iterrows():
        et_rows.append([h, f"{int(round(r.margin))}¢", k(r.n_test), n(r.mean_c_test, 2, True),
                        n(r.total_test, 2, True), n(r.t_test, 2), f"{r.hit_test:.0f} %"])
edge_thr_tbl = table(["Правило", "Порог", "Сделок", "¢/сделку", "Всего $", "t", "hit"],
                     et_rows, aligns="llrrrrr")

body = f"""
<div class="hero">
  <div class="badge">quant_research · Kalshi hourly ladder · 01.11.2026</div>
  <h1>Продажа крыльев: полный разбор</h1>
  <p>171 live-сделка по шести правилам вместо 39, бэктест за четыре месяца, бутстрап-риск и компаундинг.
  Главный сюжет — не «+$2,34 за 52 часа», а то, как W01 за один час потерял больше, чем остальные заработали за двое суток.</p>
</div>

<div class="kpis">
  <div class="kpi"><div class="v">171</div><div class="l">live-сделка в логах</div><div class="s">W01 101 · W02 31 · W03 8 · W04 21 · W05 5 · W06 5</div></div>
  <div class="kpi"><div class="v">−$3,78</div><div class="l">итог всех шести правил</div><div class="s">W01 −7,18 · W02–W06 +3,40</div></div>
  <div class="kpi"><div class="v">+$2,34</div><div class="l">W03–W06 за 52 часа</div><div class="s">39 сделок · hit 97,4 % · один минус −$0,95</div></div>
  <div class="kpi"><div class="v">9 из 10</div><div class="l">потерь W01 — за один час</div><div class="s">29.10, 16:43–17:43 UTC · p ≈ 0,7 % при бэктестовой частоте 4 %</div></div>
</div>

<nav class="toc">
  <a href="#live">Live</a><a href="#w01">W01</a><a href="#live36">W03–W06</a>
  <a href="#hist">История</a><a href="#cap">Капитал</a><a href="#cal">Калибровка</a><a href="#edge">Edge</a><a href="#next">Выводы</a>
</nav>

<section id="summary">
  <h2><span class="num">01</span>Коротко</h2>
  <p class="lead">Пять выводов, которые меняют картину прошлого отчёта.</p>
  <ul>
    <li><b>В логах не 39 сделок, а 171.</b> Полный парс добавил W01 (101) и W02 (31) — они торговали только 29 октября, но именно они объясняют, чего стоит продажа крыльев без фильтра.</li>
    <li><b>W01 — не «почти безрисковая» стратегия.</b> 90,1 % попаданий, но десять потерь по −95…−98¢ дают −$7,18; средний выигрыш +2,7¢ не покрывает проигрыш. Бэктест на 22 908 сделках давно показывал −0,94¢ и t −4,4.</li>
    <li><b>W02 с фильтром edge ≥ 1¢ в тот же день дал 31/31 и +$1,06</b> — но история W02 это 5 707 сделок и −$23,5. Один хороший день не перекрывает четыре месяца.</li>
    <li><b>W03–W06 заработали +$2,34 за 52 часа</b> при hit 97,4 %, но средний результат всего +6,0¢ за сделку, а один проигрыш стоит −95¢. После ещё одного минуса W03 даёт +1,3¢, W04 — −2,8¢, W05 — −3,5¢, W06 — −9,5¢.</li>
    <li><b>Точечные оценки live выше истории у W03, W05 и W06, но их интервалы слишком широки;</b> почти совпало только W04: +1,57¢ live против +1,8¢ OOS. На 8 и 5 сделках это шум, а не доказательство.</li>
  </ul>
  <div class="note"><b>Как читать цифры.</b> P&amp;L — на один контракт, комиссия Kalshi включена (maker 0,0175 %, taker 0,07 % от цены контракта, округление вверх до цента).
  «Ср. ¢» — средний результат сделки; «hit» — доля расчётов с выплатой 1; «безубыточный hit» — сколько попаданий нужно, чтобы средний результат был нулевым при наблюдаемых размерах выигрыша и проигрыша.</div>
</section>

<section id="live">
  <h2><span class="num">02</span>Все шесть правил в live</h2>
  <p class="lead">29.10–01.11.2026 · один контракт на сделку · W01/W02 — только 29.10, W03–W06 — 52 часа.</p>
  {figure("fig22_wings_live_all.png", "Рис. 22. Live-тест всех шести правил: накопленный P&amp;L, средний результат с 95 % интервалом, hit-рейт, риск-доходность, дни, просадка W01/W02 и карта сделок.")}
  {live_tbl}
  <p class="small">{live_note}</p>
  <h3>Что видно на рисунке</h3>
  <ul>
    <li><b>Масштаб задаёт W01.</b> Его кривая уходит к −$8,3 (максимальная просадка) и заканчивается на −$7,18; W02–W06 на этом фоне выглядят почти плоскими, поэтому W03–W06 показаны отдельно (панель B).</li>
    <li><b>Высокий hit ничего не гарантирует.</b> У W01 90 % попаданий, но соотношение выигрыша к проигрышу 1:36 делает стратегию убыточной.</li>
    <li><b>Maker-сделки W05/W06 дают +10,2¢ на сделку</b> против +4,6¢ у taker-сделок W03/W04 — но это 10 сделок, а не статистика.</li>
    <li><b>29.10 закрылся −$5,63:</b> W01 −$7,18, остальные +$1,55; 30–31.10 W03–W06 добавили +$2,80. Один день с одной просадкой перевесил два спокойных.</li>
  </ul>
</section>

<section id="w01">
  <h2><span class="num">03</span>W01: анатомия убытка</h2>
  <p class="lead">101 сделка, 91 выигрыш, 10 потерь. Средний выигрыш +2,69¢, средний проигрыш −96,3¢, безубыточный hit 97,3 % при фактических 90,1 %.</p>
  {loss_tbl}
  <p class="small">Все десять потерь W01. Девять из них открыты между 16:43 и 17:43 UTC 29.10 — в один и тот же импульс BTC через страйки 79 600–81 000; в выборке одновременно и Kalshi, и Polymarket.</p>
  <h3>Почему это не «не повезло»</h3>
  <p>Если бы частота потерь соответствовала бэктесту (4 %), вероятность увидеть 10 или больше потерь на 101 сделке была бы <b>0,73 %</b>. Это не случайная флуктуация, а реализация коррелированного хвоста: продажа всех страйков без фильтра ставит на то, что цена не пройдёт сразу через несколько границ. В спокойный час это приносит 2–3¢; в трендовый — забирает по 95¢ с каждого страйка одновременно.</p>
  <div class="warn"><b>Урок.</b> W01 — это проданный «хвостовой» риск с отрицательной премией: 96 % попаданий в бэктесте и −0,94¢ за сделку. Live-день лишь показал, как выглядит плохой час: −$7,18 и просадка −$8,33 при среднем выигрыше $0,027.</div>
  <p>Важно: итог W01 частичный — 8 открытий и 14 расчётов не сопоставлены из-за обрезки имён Polymarket в логах. Но даже без них видно, что девять потерь пришли одной волной, а не размазались по дню.</p>
</section>

<section id="w02">
  <h2><span class="num">04</span>W02: фильтр переворачивает день, но не историю</h2>
  <p class="lead">Те же страйки, тот же час, но продаём только там, где bid выше справедливой цены + комиссия (edge ≥ 1¢).</p>
  <div class="grid2">
    <div>
      <h3>Live, 29.10</h3>
      <ul>
        <li>31 сделка, <b>31 выигрыш</b>, +$1,06, +3,42¢ на сделку.</li>
        <li>Средний выигрыш +3,4¢; безубыточный hit 96,6 % — запас всего 3,4 п.п.</li>
        <li>Ни одного минуса — но 31 сделка в один спокойный день.</li>
      </ul>
    </div>
    <div>
      <h3>История, июль–октябрь</h3>
      <ul>
        <li>5 707 сделок, −$23,5, −0,41¢ за сделку, t −0,9.</li>
        <li>Худший месяц −$18,8 (сентябрь), максимальная просадка −$35,7.</li>
        <li>Правило теряет в трёх месяцах из четырёх; октябрь дал +$10,9.</li>
      </ul>
    </div>
  </div>
  <div class="note"><b>Вывод.</b> Фильтр edge действительно убирает часть худших крыльев — в один день это дало разворот с −$7,18 у W01 до +$1,06 у W02. Но на горизонте четырёх месяцев W02 остаётся убыточным: премия за проданный хвост слишком мала. W02 остаётся выключенным.</div>
</section>

<section id="live36">
  <h2><span class="num">05</span>W03–W06: что согласуется с бэктестом</h2>
  <p class="lead">39 сделок, +$2,34, hit 97,4 % [86,8; 99,5], средний +6,0¢ [−7,3; +8,7]. Один проигрыш — W04, 01.11 00:35 UTC, KXBTCD-26OCT3118-T77499.99, −$0,95.</p>
  {bt_tbl}
  <h3>Три вещи, которые важнее среднего</h3>
  <ul>
    <li><b>W04 — единственное честное совпадение.</b> Live +1,57¢ против +1,8¢ OOS на 980 сделках истории. Это не доказательство, но хотя бы не расхождение в разы.</li>
    <li><b>W03, W05 и W06 в live выглядят в 2–3 раза лучше истории.</b> Причина — maker-first исполнение W05/W06 и маленькая выборка: 8 и 5 сделок не отличимы от удачи. W03 +12,4¢ против +3,6¢ OOS — это 8 сделок и интервал [−25,1; +12,4]: почти вся неопределённость лежит в стороне проигрыша.</li>
    <li><b>Все заработанные +$2,34 — это 2,5 проигрыша.</b> Один уже случился; ещё один минус обнуляет W03 и уводит W04–W06 в минус.</li>
  </ul>
  {figure("fig24_wings_risk.png", "Рис. 24. Риск: бутстрап среднего, вероятность выйти в плюс на 20/50/100/200/400 сделках, просадка истории, компаундинг $100, эффект ещё одного проигрыша и риск-метрики.")}
  <h3>Один минус меняет всё</h3>
  <p>После ещё одного проигрыша по средней цене контракта + комиссия:</p>
  <ul>
    <li>W02: +3,4¢ → <b>+0,3¢</b>;</li>
    <li>W03: +12,4¢ → <b>+1,3¢</b>;</li>
    <li>W04: +1,6¢ → <b>−2,8¢</b>;</li>
    <li>W05: +13,2¢ → <b>−3,5¢</b>;</li>
    <li>W06: +7,2¢ → <b>−9,5¢</b>.</li>
  </ul>
  <p>Это и есть главный риск-вывод: при выигрыше 3–13¢ и проигрыше 87–97¢ любая серия из 5–20 сделок выглядит отлично, пока не придёт минус. P(выйти в плюс) на 400 сделках: W03 86 %, W02 86 %, W05 78 %, W06 62 %, W04 56 %, W01 0,2 %.</p>
  {pp_tbl}
  <p class="small">Оценка по пост-распределению hit (Beta по фактическим выигрышам и проигрышам) с фактическими размерами выигрыша/проигрыша; 20 000 симуляций на точку. W03/W05/W06 без минусов — поэтому распределение hit широкое, а не «100 %».</p>
</section>

<section id="hist">
  <h2><span class="num">06</span>История: четыре месяца и 2 782 часа</h2>
  <p class="lead">W02, W03, W04 на часовой лестнице Kalshi, 1 контракт, комиссия; W03/W04 выбраны на июле–августе, сентябрь–октябрь — out-of-sample.</p>
  {figure("fig23_wings_history.png", "Рис. 23. Бэктест: накопленный P&amp;L и OOS, месяцы, просадка, скользящее среднее по 200 активным часам, распределение часового P&amp;L и сетка улучшений W05/W06 (24 варианта).")}
  {hist_tbl}
  {month_tbl}
  <div class="grid2">
    <div>
      <h3>W03 — ровный, но медленный</h3>
      <p>645 часов, +$20,4, +3,17¢/час, t 2,6, hit 89 %, просадка −$4,7. OOS +$15,6 на 427 сделках. Лучший месяц — сентябрь (+$9,6), худший — июль (+$0,7): правило не теряет, но и не разгоняется.</p>
    </div>
    <div>
      <h3>W04 — быстрее и грязнее</h3>
      <p>1 664 часа, +$25,6, +1,59¢/час, t 2,7, hit 94 %, просадка −$6,9. OOS +$17,4 на 980 сделках. Август — единственный минус (−$0,6); сентябрь и октябрь — по +$9,8 и +$7,5.</p>
    </div>
  </div>
  <p>W05/W06 — не новые правила, а улучшения тех же окон: смесь EWMA и GBM для справедливой цены плюс отказ от часов американского открытия (13–15 UTC). На 24 вариантах улучшений W03-окно даёт +4,56¢ (t 2,87) на 318 сделках теста, W04-окно — +3,54¢ (t 4,46) на 496. Но в основной сетке 435 правил только 2,3 % дотягивают до t &gt; 2 на тесте — selection risk остаётся.</p>
</section>

<section id="cap">
  <h2><span class="num">07</span>Капитал и чувствительность к сдвигу</h2>
  <p class="lead">$100 стартового капитала, доля f от текущего капитала как залог, целые контракты, сентябрь–октябрь OOS.</p>
  {cmp_tbl}
  <ul>
    <li><b>Без сдвига:</b> W03 при f = 5 % → $222, при 10 % → $458; W04 → $222 / $468; вместе → $516.</li>
    <li><b>−1¢ на контракт:</b> W03 $177 / $285, W04 $137 / $159, вместе $243. Это всё ещё прибыль, но в 2–3 раза меньше.</li>
    <li><b>−2¢:</b> W03 $143 / $172, W04 $86 / $57, вместе $110. При таком сдвиге W04 уходит ниже стартовых $100, а W03 едва держится.</li>
    <li><b>Просадка:</b> W04 при f = 10 % — −45 %, W03 — −37 %, вместе — −34 %. Для стратегии с одним минусом −95¢ это неприемлемо при агрессивном размере.</li>
  </ul>
  <div class="note"><b>Практический вывод.</b> Базовый сценарий — не «без сдвига», а −1¢: maker-филлы не бесплатны, а очередь на дешёвых крыльях тонкая. При −2¢ W04 перестаёт быть стратегией, а W03 остаётся на грани.</div>
</section>

<section id="cal">
  <h2><span class="num">08</span>Почему крылья вообще продаются</h2>
  <p class="lead">Калибровка Kalshi: 44 514 расчётов, 7 005 снимков, цена 1–99¢.</p>
  <p>Средняя реализованная вероятность 49,68 % против 49,85 % по midpoint; наклон 1,019 [1,011; 1,026]. Но на краях есть систематика: коэффициент favourite–longshot <b>b = 1,058 [1,036; 1,081]</b> — дешёвые контракты (longshots) в среднем переоценены, дорогие (favourites) недооценены. Продажа крыла — это и есть сбор этой премии: мы продаём переоценённый дешёвый контракт и ждём, что он чаще истекает в 0, чем заложено в цене.</p>
  <p>Проблема в том, что премия мала относительно хвоста. W01 продаёт всё подряд, включая крылья, где переоценка не покрывает комиссию и риск; W02 добавляет фильтр edge ≥ 1¢, W03 — окно 10–20¢ и 5–15 минут, W04 — 6–10¢ и 15–30 минут. Чем точнее фильтр, тем ближе стратегия к «продаём только там, где b &gt; 1 + издержки».</p>
  <p>Калибровка также объясняет, почему 90 % hit не спасает: при цене 3¢ рынок закладывает 3 % вероятности, а мы продаём по 3¢ с комиссией; достаточно 4–5 % реализации, чтобы ожидание стало отрицательным.</p>
</section>

<section id="slices">
  <h2><span class="num">09</span>Разрезы: исполнение, цена крыла, edge</h2>
  <div class="grid2">
    <div>
      <h3>Maker против taker</h3>
      {ex_tbl}
      <p class="small">Все maker-сделки — W05/W06; все taker — W01–W04. Сравнение смешивает правило и исполнение, но направление совпадает с гипотезой: maker-филл добавляет 3–5¢.</p>
    </div>
    <div>
      <h3>Edge (только W02–W04)</h3>
      {ed_tbl}
      <p class="small">На 21 live-сделке W04 градиент выглядит монотонным (−4,8 / +6,1 / +7,0¢), но на 980 OOS-сделках он исчезает: корзины дают 1,1–3,3¢ при перекрывающихся интервалах. Это малая выборка, а не свойство edge — см. раздел 10.</p>
    </div>
  </div>
  <h3>Цена крыла</h3>
  {b_tbl}
  <p class="small">Внутри W01 худший вклад дают крылья 6–10¢: 11 сделок, −$2,44, средний −22,2¢ (два проигрыша). W03 монотонно улучшается с ценой: 6–10¢ +9¢ (n=1), 10–15¢ +11,3¢, 15–20¢ +15¢. Это согласуется с калибровкой: чем дороже крыло, тем больше премия и меньше относительный хвост.</p>
</section>

<section id="edge">
  <h2><span class="num">10</span>Edge: сколько стоит ждать максимального преимущества</h2>
  <p class="lead">Прибыль не сосредоточена в сделках с максимальным edge. У W03 корзина ≥ 4¢ — 24 % сделок и −15 % итоговой прибыли; у W04 — 9 % сделок и 3 % прибыли. Основная прибыль лежит в полосе 0–3¢. Поднимать порог и «ждать максимум» невыгодно: итог на тесте падает в 1,7–2,1 раза.</p>
  {figure("fig25_wings_edge.png", "Рис. 25. Edge W03/W04: распределение, средний P&L и доля прибыли по корзинам, порог edge, концентрация прибыли в топе по edge и live-сделки. Корзины — все сделки с edge ≥ 0; серая зона — срез текущего порога W03 (1¢).")}
  <h3>Доли прибыли по корзинам edge (тест, 516 и 980 сделок)</h3>
  {edge_bucket_tbl}
  <p class="small">У W03 половина прибыли — в корзине 0–0,5¢ (hit 96,7 %), а верхняя корзина убыточна (hit 83,6 %). У W04 прибыль размазана по 0–2¢ (73 %), верхняя корзина даёт 3 %. Широкие интервалы: в корзинах 40–250 сделок, поэтому надёжен не отдельный уровень, а отсутствие монотонности.<br>
  Текущее правило W03 (порог 1¢, 427 сделок): 1–1,5 — 27 %, 1,5–2 — 12 %, 2–3 — 51 %, 3–4 — 20 %, ≥4 — −11 % прибыли.</p>
  <h3>Что если поднять минимальный edge</h3>
  {edge_thr_tbl}
  <p class="small">W03: порог 2¢ был лучшим на train (+4,88¢, t = 2,24), но на тесте дал +2,56¢ (t = 1,51) и вдвое меньший итог. W04: порог 1¢ снижает и ¢/сделку, и итог. Текущий порог W03 1¢ не улучшает качество, но отрезает 17 % сделок и $2,9 прибыли.</p>
  <h3>W05/W06: точного edge нет</h3>
  <p>В live W05/W06 edge не логируется, поэтому их 5 + 5 сделок нельзя разложить по корзинам. История говорит, что их улучшение — прежде всего отказ от 13–15 UTC (+1,56¢ для 10–20-окна, +0,77…+1,11¢ для 6–10-окна), а blend помогает только 6–10-окну; max-edge для 10–20-окна хуже.</p>
  {figure("fig26_wings_w05_w06.png", "Рис. 26. W05/W06: live-сделки по цене (edge неизвестен) и историческая сетка EWMA/blend с фильтром 13–15 UTC и без него.")}
  <p class="small">Полный разбор и оговорки: <code>reports/wings_edge.md</code>; код — <code>reports/wings_edge.py</code>; таблицы — <code>reports/final/tables/wings_edge_*.csv</code>.</p>
</section>

<section id="data">
  <h2><span class="num">11</span>Данные и оговорки</h2>
  <ul>
    <li><b>171 сделка</b> собрана из <code>lab_20261029.log</code>–<code>lab_20261101.log</code>. Парсер сопоставляет OPEN/WINGS с SETTLE по (правило, рынок), а для обрезанных имён Polymarket — по префиксу.</li>
    <li><b>W03–W06 сверены точно:</b> максимальное расхождение логированного P&amp;L с пересчётом по цене, исходу и комиссии — 0,000000.</li>
    <li><b>W01/W02 — частичное покрытие:</b> 8 открытий и 14 расчётов не сопоставлены (позиции до начала лога / обрезка имён). Итог W01 (−$7,18) — частичный: несопоставленные расчёты могут сдвинуть его в любую сторону, хотя девять потерь одной волной видны и в сопоставленной части.</li>
    <li><b>Сравнение периодов:</b> W01/W02 торговали 29.10 (3,7 и 6 часов), W03–W06 — 52 часа. Сравнивать итоги в долларах нельзя; сравнение по сделке корректно, но у W01/W02 день один.</li>
    <li><b>403 от Kalshi:</b> 110 ошибок «markets kalshi failed» за три дня. Это не повлияло на расчёты (сделки берутся из лога), но перед масштабированием фид нужно чинить.</li>
    <li><b>Ledger не использован:</b> <code>data/lab/ledger.csv</code> содержит только 11 OPEN W01 и 2 OPEN W02, без SETTLE/WINGS; источник — логи.</li>
  </ul>
</section>

<section id="next">
  <h2><span class="num">12</span>Что делать дальше</h2>
  <ul>
    <li><b>W01 — выключить.</b> Отрицательное ожидание подтверждено и историей (22 908 сделок, t −4,4), и live-днём (−$7,18, 10 потерь).</li>
    <li><b>W02 — оставить выключенным.</b> Один день 31/31 не перекрывает −$23,5 и просадку −$35,7; для возврата нужна новая гипотеза, а не повторение.</li>
    <li><b>W03/W04 — ядро.</b> Одна сделка в час, размер из бюджета «один минус ≤ 2–3 % капитала». При цене контракта ~$0,85–0,94 и потере ~$0,95 на контракт бюджет $0,02–0,03 на $1 капитала даёт 0,02–0,03 контракта; при $10 000 это 200–300 контрактов, но лучше резать до 100–150 с учётом просадки.</li>
    <li><b>W05/W06 — только paper до 100+ сделок.</b> Maker-first и отказ от 13–15 UTC выглядят разумно, но 5 сделок не дают даже знака.</li>
    <li><b>Edge — фильтр, а не ранг.</b> W03: порог 1¢ не улучшает ¢/сделку, но отрезает $2,9 прибыли на тесте; 2¢ — уже потеря. W04: 0¢ лучше 1¢. Не ждать максимум; использовать edge только против ≤ 0.</li>
    <li><b>Базовая оценка — со сдвигом −1¢.</b> При −2¢ W04 теряет преимущество; если maker-филлы окажутся хуже, стратегия не масштабируется.</li>
    <li><b>Вердикт — после 300–400 сделок W03–W06</b> (примерно месяц при 10–20 сделках в день). До этого любые +$2,34 — это аванс, а не прибыль.</li>
    <li><b>Починить фид:</b> 110 ошибок 403 за три дня — риск пропустить вход или расчёт.</li>
  </ul>
  <div class="note"><b>Итог одной фразой.</b> Продажа крыльев зарабатывает на переоценке дешёвых контрактов, но платит за это редкими −95¢; W01 показывает, что без фильтра хвост съедает премию, а W03/W04 — что с фильтром премия остаётся, но её размер (2–4¢ на сделку) требует сотен сделок и терпимости к просадке.</div>
</section>

<footer>
  Данные: lab_20261029–lab_20261101.log · results/studies/wings_equity.csv · wings_improve.csv · wings_summary.md · reports/kalshi_calibration.md.<br>
  Код и таблицы: reports/wings_deep.py, reports/wings_edge.py, reports/build_wings_report.py, reports/final/tables/wings_all_*.csv, wings_edge_*.csv. Рисунки: fig22–fig26.
</footer>
"""

doc = f"""<!doctype html>
<html lang="ru"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Продажа крыльев: полный разбор — 01.11.2026</title>
<style>{CSS}</style></head>
<body><div class="wrap">{body}</div></body></html>
"""
(ROOT / "reports" / "wings_report.html").write_text(doc, encoding="utf-8")
print("html", len(doc) // 1024, "KB")


# ------------------------------------------------------------------ Markdown
def md_table(headers, rows):
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(x) for x in r) + " |")
    return "\n".join(out)


md_live = []
for _, r in S.iterrows():
    be, wm, lm = breakeven(r.hyp)
    md_live.append([r.hyp, k(r.n), n(r.hit, 1), n(r.mean_c, 2, True),
                    f"[{n(r.ci_lo_c, 1)}; {n(r.ci_hi_c, 1)}]", n(r.total, 2, True),
                    f"{n(100 * wm, 1, True)} / {n(100 * lm, 1, True)}", n(be, 1),
                    "∞" if not np.isfinite(r.pf) else n(r.pf, 2), n(r.max_dd, 2)])

md_hist = [[r.hyp, k(r.hours), n(r.hit, 1), n(r.mean_c, 2, True), n(r.total, 1, True),
            n(r.max_dd, 1), n(r.worst_month, 1, True), n(r.oos_c, 1, True)] for _, r in H.iterrows()]

md_months = [[pd.to_datetime(i).strftime("%m.%Y"), n(r.W02, 1, True), n(r.W03, 1, True),
              n(r.W04, 1, True), n(r.W02 + r.W03 + r.W04, 1, True)] for i, r in M.iterrows()]

md_bt = [[r.hyp, k(r.n), n(r.mean_c, 2, True), bt[r.hyp][0], bt[r.hyp][1]] for _, r in S.iterrows()]

md_pp = [[r.hyp, n(r.p_profit_N20, 1), n(r.p_profit_N100, 1), n(r.p_profit_N400, 1)]
         for _, r in PP.iterrows()]

md_cmp = [[r.rule + (f" · f={int(r.f)}%" if r.rule != "W03+W04" else " · 5% each"),
           f"${int(r.base)}", f"${int(r.h1)}", f"${int(r.h2)}", n(r.dd, 0) + "%"]
          for _, r in CMP.iterrows()]

md_loss = [[r.open_ts.strftime("%d.%m %H:%M"), r.settle_ts.strftime("%H:%M"),
            short_market(r.market).replace("&gt;", ">"), n(r.price, 2), n(r.pnl, 3, True)]
           for _, r in L.iterrows()]

md = f"""# Продажа крыльев на Kalshi: полный разбор

**01.11.2026** · 171 live-сделка (W01–W06) · бэктест 1 июля – 28 октября · риск и компаундинг

Источник: `lab_20261029.log`–`lab_20261101.log`, `results/studies/wings_equity.csv`,
`results/studies/wings_improve.csv`, `results/studies/wings_summary.md`, `reports/kalshi_calibration.md`.
Код: `reports/wings_deep.py`, `reports/build_wings_report.py`. HTML-версия: `reports/wings_report.html`.

![Все шесть правил в live](final/figures/fig22_wings_live_all.png)

## Коротко

- В логах **171 сделка**, а не 39: добавились W01 (101) и W02 (31), торговавшие только 29.10.
- **W01 потерял −$7,18**: 90,1 % попаданий, но 10 потерь по −95…−98¢. Девять из десяти — в один час
  16:43–17:43 UTC 29.10; при бэктестовой частоте 4 % вероятность такого — 0,73 %.
- **W02** (edge ≥ 1¢) в тот же день дал 31/31 и +$1,06, но история — 5 707 сделок и −$23,5.
- **W03–W06**: +$2,34 за 52 часа, 39 сделок, hit 97,4 % [86,8; 99,5]; один проигрыш −$0,95.
- С бэктестом согласуется только **W04**: +1,57¢ live против +1,8¢ OOS. W03/W05/W06 выглядят лучше
  истории, но на 8 и 5 сделках это шум.
- Ещё один проигрыш обнуляет W03 (+1,3¢) и уводит W04/W05/W06 в минус (−2,8 / −3,5 / −9,5¢).

## 1. Live: все шесть правил

{md_table(["Правило", "Сделок", "Hit, %", "Ср. P&L, ¢", "95 % интервал", "Итог, $",
           "Выигрыш / проигрыш, ¢", "Безубыт. hit, %", "PF", "Макс. просадка, $"], md_live)}

W01/W02 — только 29.10 (3,7 и 6 часов), W03–W06 — 52 часа. У W01 8 открытий и 14 расчётов не
сопоставлены (обрезка имён Polymarket и позиции до начала лога), итог частичный. P&L на 1 контракт, комиссия Kalshi включена.

## 2. W01: анатомия убытка

| Открыта (UTC) | Расчёт | Рынок | Цена | P&L, $ |
|---|---|---|---|---|
{chr(10).join("| " + " | ".join(map(str, r)) + " |" for r in md_loss)}

Средний выигрыш +2,69¢, средний проигрыш −96,3¢, безубыточный hit 97,3 % при фактических 90,1 %.
Девять потерь открыты в 16:43–17:43 UTC — один импульс BTC через страйки 79 600–81 000, одновременно
Kalshi и Polymarket. Бэктест W01 (22 908 сделок) давно показывал −0,94¢ и t −4,4.

## 3. Live против бэктеста

{md_table(["Правило", "Live, сделок", "Live, ¢/сделку", "Бэктест, ¢/сделку", "Бэктест, сделок"], md_bt)}

W03/W05/W06 в live выглядят в 2–3,5 раза лучше истории, но их интервалы широки; W04 — единственное совпадение точечных оценок.
W05/W06 — maker-first, поэтому их плюс к taker-бэктесту ожидаем, но 5 сделок ничего не доказывают.

## 4. История: июль–октябрь

{md_table(["Правило", "Активных часов", "Hit, %", "Ср. ¢/час", "Итог, $", "Макс. просадка, $",
           "Худший месяц, $", "Sep–Oct OOS, $"], md_hist)}

{md_table(["Месяц", "W02, $", "W03, $", "W04, $", "Сумма, $"], md_months)}

W03: 645 часов, +3,17¢/час, t 2,6, OOS +$15,6. W04: 1 664 часа, +1,59¢/час, t 2,7, OOS +$17,4.
W02 теряет в трёх месяцах из четырёх; октябрь (+$10,9) не перекрывает сентябрь (−$18,8).

![Бэктест](final/figures/fig23_wings_history.png)

## 5. Риск и капитал

Вероятность выйти в плюс (пост-оценка hit, 20 000 симуляций):

{md_table(["Правило", "20 сделок", "100 сделок", "400 сделок"], md_pp)}

$100 капитала, компаундинг, сентябрь–октябрь OOS:

{md_table(["Стратегия", "Без сдвига", "−1¢", "−2¢", "Макс. просадка"], md_cmp)}

Базовый сценарий — не «без сдвига», а −1¢: maker-филлы не бесплатны. При −2¢ W04 уходит ниже $100,
W03 едва держится.

![Риск](final/figures/fig24_wings_risk.png)

## 6. Разрезы

- **Исполнение:** maker 10 сделок, 100 %, +10,2¢, +$1,02; taker 161, 93,2 %, −2,98¢, −$4,80.
  Maker — это W05/W06, taker — W01–W04; сравнение смешивает правило и исполнение.
- **Edge (W02–W04):** W04 при edge < 0,5¢ даёт −4,8¢ (9 сделок), 0,5–1,5¢ — +6,1¢, > 1,5¢ — +7,0¢.
- **Цена крыла:** W01 в 6–10¢ — 11 сделок, −$2,44; W03 монотонно улучшается: 10–15¢ +11,3¢ (4),
  15–20¢ +15,0¢ (3).

## 7. Данные и оговорки

- W03–W06 сверены точно: максимальное расхождение логированного P&L с пересчётом — 0,000000.
- W01/W02: 8 открытий и 14 расчётов не сопоставлены; итог W01 — частичный (несопоставленные расчёты могут сдвинуть его в любую сторону).
- 110 ошибок `markets kalshi failed` (HTTP 403) за три дня — на расчёты не повлияло, но фид нужно чинить.
- `data/lab/ledger.csv` не использован: там 11 OPEN W01 и 2 OPEN W02, без SETTLE/WINGS.

## 8. Что дальше

1. W01 — выключить; W02 — оставить выключенным.
2. W03/W04 — ядро, 1 сделка в час, размер из бюджета «один минус ≤ 2–3 % капитала».
3. W05/W06 — paper до 100+ сделок.
4. Базовая оценка — со сдвигом −1¢; при −2¢ W04 не масштабируется.
5. Вердикт — после 300–400 сделок W03–W06 (примерно месяц).
6. Починить фид Kalshi (110 ошибок 403 за три дня).

**Итог одной фразой.** Продажа крыльев зарабатывает на переоценке дешёвых контрактов, но платит
редкими −95¢; W01 показывает, что без фильтра хвост съедает премию, а W03/W04 — что с фильтром премия
остаётся, но её размер (2–4¢ на сделку) требует сотен сделок и терпимости к просадке.
"""
(ROOT / "reports" / "wings_deep.md").write_text(md, encoding="utf-8")
print("md", len(md) // 1024, "KB")
