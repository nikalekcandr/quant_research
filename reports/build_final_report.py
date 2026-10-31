"""Assemble reports/final_report.html and reports/final_report.md from the derived tables.

Run after reports/final_report_analysis.py:
    python reports/build_final_report.py
"""
from __future__ import annotations

import base64
import html
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
TAB = HERE / "final" / "tables"
FIG = HERE / "final" / "figures"
sys.path.insert(0, str(HERE))
from final_report_analysis import load_forecasts, _model_series, _actual_series  # noqa: E402

# --------------------------------------------------------------------------- helpers
def read(name: str) -> pd.DataFrame:
    return pd.read_csv(TAB / name)


def num(x, nd=3, dash="—"):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return dash
    s = f"{x:,.{nd}f}".replace(",", " ")
    return s.replace(".", ",").replace("-", "−")


def pct(x, nd=1):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "—"
    return f"{x:+.{nd}f}%".replace(".", ",").replace("-", "−")


def pval(p):
    if p is None or (isinstance(p, float) and not np.isfinite(p)):
        return "—"
    if p < 0.001:
        return "&lt; 0,001"
    return f"{p:.3f}".replace(".", ",")


def esc(x):
    return html.escape(str(x))


def img(name: str, caption: str, source: str = "", cls: str = ""):
    data = base64.b64encode((FIG / name).read_bytes()).decode()
    return (f'<figure class="fig {cls}"><img src="data:image/png;base64,{data}" alt="{esc(caption)}">'
            f'<figcaption><b>{caption}</b>{f"<span class=src>{source}</span>" if source else ""}</figcaption></figure>')


def table(headers, rows, cls="", aligns=None, note=None, foot=None):
    aligns = aligns or ["left"] + ["right"] * (len(headers) - 1)
    out = [f'<div class="tablewrap"><table class="tbl {cls}"><thead><tr>']
    out += [f'<th class="{a}">{h}</th>' for h, a in zip(headers, aligns)]
    out.append("</tr></thead><tbody>")
    for r in rows:
        out.append("<tr>" + "".join(f'<td class="{a}">{c}</td>' for c, a in zip(r, aligns)) + "</tr>")
    out.append("</tbody></table>")
    if note:
        out.append(f'<div class="tnote">{note}</div>')
    out.append("</div>")
    return "".join(out)


def badge(text, kind="muted"):
    return f'<span class="badge {kind}">{text}</span>'


def verdict(v: str) -> str:
    v = str(v)
    if "CONFIRMED" in v:
        return badge("подтверждено", "ok")
    if "likely" in v:
        return badge("вероятно (BH)", "warn")
    if "not supported" in v:
        return badge("не подтверждено", "muted")
    return badge(v, "muted")


def live_verdict(v: str) -> str:
    v = str(v).lower()
    if "loses" in v:
        return badge("убыточно", "bad")
    if v == "nothing":
        return badge("ничего", "muted")
    if "collecting" in v:
        return badge("собирает данные", "warn")
    if "no signal" in v:
        return badge("нет сигнала", "muted")
    return badge(v, "muted")


def max_by(df, col):
    return df.loc[df[col].idxmax()] if len(df) else None


def min_by(df, col):
    return df.loc[df[col].idxmin()] if len(df) else None


# --------------------------------------------------------------------------- data
sc = read("scoreboard.csv")
rec = read("recent_oos.csv")
dm = pd.read_csv(ROOT / "results" / "tables" / "oos_dm.csv")
dmq = pd.read_csv(ROOT / "results" / "tables" / "diebold_mariano.csv")
lad = read("brier.csv")
brier_dm = read("brier_dm.csv")
horse = read("horserace.csv")
research = read("research.csv")
lab = read("lab.csv")
implied = read("implied_bias.csv")
regimes = read("regimes.csv")
inventory = read("data_inventory.csv")

LBL = dict(zip(sc.model, sc.model_label))
LBL.update({"har_m": "HAR + лесенка", "mkt_ln_raw": "Лесенка: центр (логнорм.)",
            "mkt_mz": "Лесенка BJN (калибр.)", "market": "Рынок: центр лесенки"})

SHORT = {"R1": "Сезонность волатильности", "R2": "Доходность вне часов США", "R3": "Внутридневной моментум",
         "R4": "Недельный TSMOM", "R5": "Понедельник", "R6": "Дрейф перед FOMC", "R7": "Vol-managed экспозиция",
         "R8a": "Премия за риск дисперсии", "R8b": "VRP предсказывает доходность", "R9": "Лесенка знает направление",
         "R10": "Страховка от краха переоценена", "R11": "Фандинг предсказывает доходность",
         "R12": "IV ведёт доходность", "R13": "Всплеск в первую минуту часа", "R14": "CME-гэпы закрываются",
         "R15": "Круглые уровни", "R16": "Начало месяца", "R17": "Плохая волатильность",
         "R18": "Выигрыш лесенки по режимам", "R19": "HAR с режимами", "R20": "ML предсказывает доходность",
         "R21": "HAR бьёт GARCH"}
research["hypothesis_ru"] = research["id"].map(SHORT).fillna(research["hypothesis"])

# year split from the raw score rows
_sc = pd.read_parquet(ROOT / "scores.parquet", columns=["horizon", "t", "model", "crps"])
_sc["year"] = pd.to_datetime(_sc.t, utc=True).dt.year
_yg = _sc.groupby(["horizon", "year", "model"]).crps.mean().unstack("model")


def podium_html():
    out = ['<div class="podium">']
    for h, title in [("kx1h", "55 минут · Kalshi"), ("pm24h", "24 часа · Polymarket"), ("pm6d", "6 дней · Polymarket")]:
        x = sc[sc.horizon == h].sort_values("crps").head(3)
        out.append(f'<div class="pod"><div class="ph">{title}</div>')
        for i, (_, r) in enumerate(x.iterrows(), 1):
            out.append(f'<div class="pl p{i}"><div class="rank">{i}</div><div><div class="nm">{esc(r.model_label)}</div>'
                       f'<div class="sc">CRPS {num(r.crps)} · QLIKE {num(r.qlike)}</div></div></div>')
        out.append('</div>')
    out.append('</div>')
    return ''.join(out)


def year_table_html():
    rows = []
    for h, hn in [("kx1h", "55 мин"), ("pm24h", "24 ч"), ("pm6d", "6 дн")]:
        for y in (2025, 2026):
            if (h, y) not in _yg.index:
                continue
            r = _yg.loc[(h, y)]
            vals = {m: (float(r[m]) if m in r and pd.notna(r[m]) else np.nan)
                    for m in ("har", "mkt_ln_raw", "harx_all", "iv_mz")}
            if all(np.isnan(v) for v in vals.values()):
                continue
            best = min(v for v in vals.values() if not np.isnan(v))
            cells = []
            for m in ("har", "mkt_ln_raw", "harx_all", "iv_mz"):
                v = vals[m]
                cells.append('<span class="best">—</span>' if np.isnan(v)
                             else (f'<span class="best">{num(v)}</span>' if abs(v - best) < 1e-12 else num(v)))
            rows.append([hn, str(y)] + cells)
    return table(["Горизонт", "Год", "HAR", "Центр лесенки", "HAR-X + Deribit + лесенка", "Deribit IV"],
                 rows, aligns=["left", "right", "right", "right", "right", "right"],
                 note="Средний CRPS по всем прогнозам модели в году, сырые строки scores.parquet. "
                      "Прочерк — метода в этом году ещё не было. Сравнивать можно только внутри строки.")


def V(h, m, col, df=sc):
    r = df[(df.horizon == h) & (df.model == m)]
    return float(r[col].iloc[0]) if len(r) else np.nan


# key numbers
kx = sc[sc.horizon == "kx1h"]
pm24 = sc[sc.horizon == "pm24h"]
pm6 = sc[sc.horizon == "pm6d"]
kx_best_crps = min_by(kx, "crps")
kx_best_q = min_by(kx, "qlike")
pm24_best_crps = min_by(pm24, "crps")
pm24_best_q = min_by(pm24, "qlike")
pm6_best_crps = min_by(pm6, "crps")
pm6_best_q = min_by(pm6, "qlike")

# DM lookups
def dm_row(h, A, B, loss="crps", sample="history"):
    x = dm[(dm.horizon == h) & (dm.A == A) & (dm.B == B) & (dm.loss == loss) & (dm["sample"] == sample)]
    return x.iloc[0] if len(x) else None


ladder_key = dm_row("kx1h", "harx_all", "harx_d")
ladder_key_q = dmq[(dmq.horizon == "kx1h") & (dmq.A == "harx_all") & (dmq.B == "harx_d") & (dmq.loss == "qlike")]
ladder_key_q = ladder_key_q.iloc[0] if len(ladder_key_q) else None
pm24_ladder = dm_row("pm24h", "harx_all", "harx_d")
pm6_ladder = dm_row("pm6d", "harx_all", "harx_d")

# recent OOS
rec_kx = rec[rec.horizon == "kx1h"].sort_values("crps")
rec_pm24 = rec[rec.horizon == "pm24h"].sort_values("crps")
rec_pm6 = rec[rec.horizon == "pm6d"].sort_values("crps")

# interactive weekly series
_d = load_forecasts("kx1h")
_d = _d[_d["T"] >= pd.Timestamp("2026-01-01", tz="UTC")]
_weeks = pd.date_range("2026-01-05", "2026-10-26", freq="W-MON")
_keys = [("actual", "Реализованная", "#0b1220", 3.0),
         ("har", "HAR", "#0d9488", 1.9),
         ("gbm_all", "Бустинг", "#7c3aed", 1.9),
         ("harx_all", "HAR-X + Deribit + лесенка", "#2563eb", 2.0),
         ("mkt_ln_raw", "Лесенка: центр", "#e11d48", 2.0),
         ("iv_mz", "Deribit / DVOL", "#f59e0b", 1.9)]
_series = {}
_act = _actual_series(_d, "W")
_act = _act.set_index("bucket")["act"] * 100
_series["actual"] = [None if pd.isna(_act.get(w, np.nan)) else round(float(_act.get(w)), 1) for w in _weeks]
for key, _, _, _ in _keys[1:]:
    s = _model_series(_d, key, "W").set_index("bucket")["pred"] * 100
    _series[key] = [None if pd.isna(s.get(w, np.nan)) else round(float(s.get(w)), 1) for w in _weeks]
VOL_DATA = {"weeks": [w.strftime("%d.%m") for w in _weeks], "series": _series}
pd.DataFrame({"week": _weeks, **{k: v for k, v in _series.items()}}).to_csv(TAB / "weekly_vol_kx1h.csv", index=False)

# --------------------------------------------------------------------------- scoreboard table
def score_rows(h, n=9, full=False):
    x = sc[sc.horizon == h].sort_values("crps")
    if not full:
        x = x.head(n)
    bc, bq, bl = x.crps.min(), x.qlike.min(), x.logscore.max()
    rows = []
    for _, r in x.iterrows():
        cls_c = "best" if r.crps == bc else ""
        cls_q = "best" if r.qlike == bq else ""
        cls_l = "best" if r.logscore == bl else ""
        if pd.isna(r.mz_p):
            cal = badge("н/д", "muted")
        else:
            cal = badge("ок", "ok") if r.mz_p > 0.05 else badge("сдвиг", "warn")
        rows.append([esc(r.model_label), num(r.n, 0), f'<span class="{cls_c}">{num(r.crps)}</span>',
                     f'<span class="{cls_q}">{num(r.qlike)}</span>', f'<span class="{cls_l}">{num(r.logscore)}</span>',
                     num(r.cov50), num(r.cov90), num(r.mz_b, 2),
                     "—" if pd.isna(r.mz_p) else pval(r.mz_p), cal])
    return rows


SC_HEAD = ["Модель", "n", "CRPS ↓", "QLIKE ↓", "Log score ↑", "cov 50", "cov 90", "MZ b", "p(MZ)", "Калибр."]
SC_ALIGN = ["left"] + ["right"] * 8 + ["center"]

# --------------------------------------------------------------------------- DM table
def dm_table():
    labels = [
        ("har", "naive", "HAR против наивного"),
        ("harx", "har", "HAR-X против HAR"),
        ("harx_d", "harx", "+ Deribit к HAR-X"),
        ("harx_all", "harx_d", "+ лесенка к HAR-X + Deribit"),
        ("combo", "harx_all", "смесь против лучшей модели"),
        ("harx_all", "mkt_ln_raw", "модель против центра лесенки"),
    ]
    rows = []
    for h, hn in [("kx1h", "55 мин"), ("pm24h", "24 ч"), ("pm6d", "6 дн")]:
        for A, B, lab in labels:
            r = dm_row(h, A, B)
            if r is None:
                continue
            q = dmq[(dmq.horizon == h) & (dmq.A == A) & (dmq.B == B) & (dmq.loss == "qlike")]
            q = q.iloc[0] if len(q) else None
            sig_c = "sig" if r.p < 0.05 else ""
            sig_q = "sig" if q is not None and q.p < 0.05 else ""
            dcol = "good" if r.mean_diff < 0 else "bad"
            rows.append([f"<b>{hn}</b>", esc(lab), num(r.mean_diff, 4),
                         f'<span class="{dcol} {sig_c}">{pct(r.dm * 1.0, 2).replace("%", "")}</span>', pval(r.p),
                         num(q.mean_diff, 4) if q is not None else "—",
                         f'<span class="{sig_q}">{pct(q.dm * 1.0, 2).replace("%", "")}</span>' if q is not None else "—",
                         pval(q.p) if q is not None else "—"])
    return table(["Горизонт", "Сравнение", "Δ CRPS", "DM (CRPS)", "p", "Δ QLIKE", "DM (QLIKE)", "p"], rows,
                 note="Отрицательное Δ и DM — первая модель лучше. DM с поправкой Харви–Лейбурна–Ньюболда, "
                      "HAC Newey–West. Подчёркнуты значимые на 5% эффекты; зелёный — улучшение, красный — ухудшение.")

# --------------------------------------------------------------------------- markdown tables
def md_table(df: pd.DataFrame, cols=None, headers=None, floatfmt="{:.3f}") -> str:
    if cols:
        df = df[cols]
    if headers:
        df = df.copy()
        df.columns = headers
    out = ["| " + " | ".join(map(str, df.columns)) + " |", "|" + "|".join(["---"] * len(df.columns)) + "|"]
    for _, r in df.iterrows():
        cells = []
        for v in r:
            if isinstance(v, (float, np.floating)):
                cells.append("" if pd.isna(v) else floatfmt.format(v).replace(".", ",").replace("-", "−"))
            else:
                cells.append(str(v))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


DESC = {
    "scoreboard.csv": "CRPS, QLIKE, log score, покрытие и калибровка: 47 строк (модель × горизонт)",
    "recent_oos.csv": "та же сводка на новой выборке 13–30.10.2026 и Δ к HAR",
    "dm_key.csv": "ключевые тесты Диболда–Мариано: что добавляет каждый блок",
    "horserace.csv": "гонка HAR / GBM / GARCH / EWMA на минутах, QLIKE вне выборки",
    "implied_bias.csv": "подразумеваемая дисперсия против реализованной: DVOL, BJN, логнорм., Student-t",
    "brier.csv": "Brier и log loss на страйк по каждому источнику",
    "brier_dm.csv": "DM «модель против рынка» по Brier и log loss",
    "research.csv": "22 исследовательские гипотезы: эффект, t, Холм, BH, устойчивость",
    "lab.csv": "46 торговых правил лаборатории: сделки, P&L, t, Холм, вердикт",
    "regimes.csv": "шесть режимов волатильности по BIC",
    "seasonality.csv": "средняя волатильность по часам UTC: факт, лесенка, DVOL",
    "data_inventory.csv": "инвентарь источников: период, объём, что даёт, где лежит",
    "weekly_vol_kx1h.csv": "данные интерактивного графика: недельные средние 2026",
}


def table_index():
    rows = []
    for f in sorted(TAB.glob("*.csv")):
        n = sum(1 for _ in f.open("rb")) - 1
        kb = f.stat().st_size / 1024
        rows.append([f'<a href="final/tables/{esc(f.name)}">{esc(f.name)}</a>',
                     num(n, 0), f"{num(kb, 1)} КБ", esc(DESC.get(f.name, "—"))])
    return table(["Файл", "Строк", "Размер", "Что внутри"], rows,
                 aligns=["left", "right", "right", "left"],
                 note="Все таблицы — CSV в UTF-8 с заголовком; открываются в Excel и pandas.read_csv.")


# --------------------------------------------------------------------------- CSS
CSS = """
:root{
  --ink:#0b1220; --ink2:#1e293b; --muted:#5b6472; --faint:#8a94a6;
  --bg:#f3f5f9; --card:#ffffff; --line:#e4e8f0; --line2:#eef1f6;
  --blue:#2563eb; --blue2:#1d4ed8; --red:#e11d48; --amber:#f59e0b; --teal:#0d9488;
  --violet:#7c3aed; --green:#15803d; --greenbg:#dcfce7; --redbg:#fee2e2;
  --shadow:0 1px 2px rgba(15,23,42,.04),0 12px 32px rgba(15,23,42,.06);
}
*{box-sizing:border-box}
html{scroll-behavior:smooth}
body{margin:0;background:var(--bg);color:var(--ink);font-family:'Inter','Segoe UI',system-ui,-apple-system,'Helvetica Neue',Arial,sans-serif;
  font-size:16px;line-height:1.65;-webkit-font-smoothing:antialiased;font-feature-settings:'tnum' 0}
a{color:var(--blue);text-decoration:none}
a:hover{text-decoration:underline}
.wrap{max-width:1180px;margin:0 auto;padding:0 26px}
.hero{position:relative;overflow:hidden;background:
  radial-gradient(900px 420px at 82% -8%,rgba(37,99,235,.45),transparent 60%),
  radial-gradient(700px 380px at 8% 108%,rgba(225,29,72,.20),transparent 60%),
  linear-gradient(135deg,#070d19 0%,#0d1a2f 52%,#132a52 100%);
  color:#fff;padding:64px 0 50px}
.hero:before{content:'';position:absolute;inset:0;background-image:linear-gradient(rgba(255,255,255,.035) 1px,transparent 1px),
  linear-gradient(90deg,rgba(255,255,255,.035) 1px,transparent 1px);background-size:48px 48px;
  mask-image:linear-gradient(to bottom,rgba(0,0,0,.7),transparent 92%);pointer-events:none}
.hero .wrap{position:relative;z-index:2}
.eyebrow{font-size:12px;letter-spacing:.16em;text-transform:uppercase;color:#93c5fd;font-weight:700}
.hero h1{font-size:46px;line-height:1.08;letter-spacing:-.025em;margin:16px 0 16px;max-width:900px;font-weight:800}
.hero h1 em{font-style:normal;color:#93c5fd}
.hero .lead{font-size:18px;line-height:1.55;color:#cbd5e1;max-width:820px;margin:0 0 22px}
.meta{display:flex;flex-wrap:wrap;gap:10px 22px;font-size:13px;color:#94a3b8;align-items:center}
.meta b{color:#e2e8f0;font-weight:600}
.dotsep{width:4px;height:4px;border-radius:50%;background:#475569;display:inline-block}
.kpis{display:grid;grid-template-columns:repeat(5,1fr);gap:12px;margin-top:34px}
.kpi{background:rgba(255,255,255,.055);border:1px solid rgba(255,255,255,.10);border-radius:14px;padding:16px 16px 14px;
  backdrop-filter:blur(8px)}
.kpi .v{font-size:27px;font-weight:800;letter-spacing:-.02em;line-height:1.15;color:#fff}
.kpi .v small{font-size:15px;font-weight:600;color:#93c5fd}
.kpi .l{font-size:11.5px;line-height:1.4;color:#94a3b8;margin-top:7px}
.kpi.accent{background:linear-gradient(140deg,rgba(37,99,235,.34),rgba(37,99,235,.12));border-color:rgba(147,197,253,.34)}
.kpi.accent .v{color:#dbeafe}
.kpi.red{background:linear-gradient(140deg,rgba(225,29,72,.30),rgba(225,29,72,.10));border-color:rgba(251,113,133,.30)}
.kpi.red .v{color:#ffe4e6}
nav.toc{position:sticky;top:0;z-index:50;background:rgba(255,255,255,.88);backdrop-filter:blur(14px);border-bottom:1px solid var(--line)}
nav.toc .wrap{display:flex;gap:6px;overflow-x:auto;padding-top:10px;padding-bottom:10px;scrollbar-width:none}
nav.toc a{white-space:nowrap;font-size:12.5px;font-weight:600;color:var(--muted);padding:6px 10px;border-radius:8px}
nav.toc a:hover{background:#eef2ff;color:var(--blue);text-decoration:none}
section{padding:56px 0 8px;scroll-margin-top:64px}
.sechead{display:grid;grid-template-columns:54px 1fr;gap:16px;align-items:start;margin-bottom:26px}
.sechead .num{font-size:13px;font-weight:800;color:var(--blue);background:#e8efff;border-radius:10px;padding:7px 0;text-align:center;letter-spacing:.04em}
.sechead h2{font-size:30px;line-height:1.18;letter-spacing:-.02em;margin:0 0 6px;font-weight:800}
.sechead .dek{margin:0;color:var(--muted);font-size:15.5px;max-width:900px}
.card{background:var(--card);border:1px solid var(--line);border-radius:16px;padding:26px 28px;box-shadow:var(--shadow);margin-bottom:22px}
.card h3{font-size:20px;line-height:1.3;margin:0 0 10px;letter-spacing:-.01em}
.card h4{font-size:15px;margin:22px 0 8px;color:var(--ink2)}
.card p{margin:0 0 12px}
.card p:last-child{margin-bottom:0}
.muted{color:var(--muted)}
.small{font-size:13.5px}
.tiny{font-size:12px}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:22px}
.grid3{display:grid;grid-template-columns:repeat(3,1fr);gap:16px}
.grid4{display:grid;grid-template-columns:repeat(4,1fr);gap:14px}
.fig{margin:22px 0 26px;padding:0}
.fig img{display:block;width:100%;height:auto;border:1px solid var(--line);border-radius:12px;background:#fff}
.fig figcaption{font-size:12.8px;color:var(--muted);margin-top:9px;line-height:1.5}
.fig figcaption b{color:var(--ink);font-weight:700}
.fig .src{display:block;color:var(--faint);font-size:11.5px;margin-top:3px}
.callout{border-left:4px solid var(--blue);background:#f5f8ff;border-radius:0 12px 12px 0;padding:18px 20px;margin:20px 0}
.callout .ct{font-size:11px;font-weight:800;letter-spacing:.12em;text-transform:uppercase;color:var(--blue);margin-bottom:6px}
.callout p{margin:0;font-size:15px}
.callout.ok{border-color:var(--green);background:#f3fbf5}.callout.ok .ct{color:var(--green)}
.callout.warn{border-color:var(--amber);background:#fffaf0}.callout.warn .ct{color:#b45309}
.callout.red{border-color:var(--red);background:#fff5f7}.callout.red .ct{color:var(--red)}
.pipeline{display:flex;gap:8px;align-items:stretch;margin:18px 0 8px;flex-wrap:wrap}
.step{flex:1 1 150px;min-width:150px;background:#f8fafc;border:1px solid var(--line);border-radius:12px;padding:14px 14px 12px}
.step .sn{font-size:11px;font-weight:800;color:var(--blue);letter-spacing:.08em}
.step b{display:block;font-size:13.5px;margin:5px 0 4px}
.step p{font-size:11.8px;color:var(--muted);margin:0;line-height:1.45}
.arrow{align-self:center;color:#94a3b8;font-size:18px;padding:0 1px}
.horizon{background:linear-gradient(180deg,#fff,#fbfcfe);border:1px solid var(--line);border-radius:14px;padding:18px 18px 16px}
.horizon .tag{font-size:11px;font-weight:800;letter-spacing:.1em;text-transform:uppercase;color:var(--blue)}
.horizon .big{font-size:24px;font-weight:800;letter-spacing:-.02em;margin:4px 0 2px}
.horizon .sub{font-size:12.5px;color:var(--muted);min-height:36px}
.horizon .foot{border-top:1px solid var(--line2);margin-top:12px;padding-top:10px;font-size:12px;color:var(--muted)}
.formula{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:13px;background:#0f172a;color:#e2e8f0;
  border-radius:10px;padding:13px 16px;margin:12px 0;overflow-x:auto;line-height:1.6}
.formula span{color:#93c5fd}
.tablewrap{overflow-x:auto;margin:14px 0 6px;border:1px solid var(--line);border-radius:12px;background:#fff}
.tbl{width:100%;border-collapse:collapse;font-size:12.7px;min-width:640px}
.grid2 .tbl,.grid3 .tbl{min-width:0}
.tbl thead th{position:sticky;top:0;background:#f1f5fb;color:#334155;font-weight:700;padding:9px 10px;border-bottom:1px solid var(--line);
  white-space:nowrap;font-size:11.5px;letter-spacing:.02em}
.tbl td{padding:8px 10px;border-bottom:1px solid var(--line2);vertical-align:middle}
.tbl tbody tr:last-child td{border-bottom:none}
.tbl tbody tr:hover{background:#f8fafc}
.tbl .right{text-align:right;font-variant-numeric:tabular-nums}
.tbl .center{text-align:center}
.tbl .left{text-align:left}
.tbl .best{font-weight:800;color:var(--green)}
.tbl .good{color:var(--green);font-weight:700}
.tbl .bad{color:var(--red);font-weight:700}
.tbl .sig{text-decoration:underline;text-underline-offset:3px}
.tnote{font-size:11.5px;color:var(--faint);padding:9px 12px;border-top:1px solid var(--line2);background:#fbfcfe}
.badge{display:inline-block;font-size:10.5px;font-weight:700;letter-spacing:.02em;border-radius:999px;padding:2px 8px;white-space:nowrap}
.badge.ok{background:var(--greenbg);color:#166534}
.badge.bad{background:var(--redbg);color:#991b1b}
.badge.warn{background:#fef3c7;color:#92400e}
.badge.muted{background:#eef2f7;color:#475569}
.podium{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin:18px 0 6px}
.pod{background:#fff;border:1px solid var(--line);border-radius:14px;padding:15px 16px;box-shadow:var(--shadow)}
.pod .ph{font-size:11px;font-weight:800;letter-spacing:.09em;text-transform:uppercase;color:var(--muted);margin-bottom:10px}
.pod .pl{display:flex;gap:10px;align-items:flex-start;padding:7px 0;border-top:1px dashed var(--line2)}
.pod .pl:first-of-type{border-top:none}
.rank{flex:0 0 26px;height:26px;border-radius:8px;display:grid;place-items:center;font-size:12px;font-weight:800;background:#eef2f7;color:#475569}
.pod .pl.p1 .rank{background:#fef3c7;color:#92400e}
.pod .pl.p2 .rank{background:#e2e8f0;color:#334155}
.pod .pl.p3 .rank{background:#fde8d7;color:#9a3412}
.pod .nm{font-size:12.5px;font-weight:700;line-height:1.25}
.pod .sc{font-size:11.5px;color:var(--muted);font-variant-numeric:tabular-nums}
.bigstat{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin:16px 0}
.stat{background:#f8fafc;border:1px solid var(--line);border-radius:12px;padding:14px 16px}
.stat .v{font-size:26px;font-weight:800;letter-spacing:-.02em;line-height:1.1}
.stat .l{font-size:11.5px;color:var(--muted);margin-top:4px;line-height:1.4}
.stat.blue .v{color:var(--blue)}.stat.red .v{color:var(--red)}.stat.violet .v{color:var(--violet)}
.stat.green .v{color:var(--green)}.stat.amber .v{color:#b45309}
.chartbox{background:#fff;border:1px solid var(--line);border-radius:14px;padding:16px 16px 10px;box-shadow:var(--shadow);margin:18px 0 26px}
.chart-controls{display:flex;flex-wrap:wrap;gap:8px 14px;align-items:center;margin-bottom:8px}
.chart-controls label{display:inline-flex;align-items:center;gap:6px;font-size:12px;font-weight:600;color:#334155;cursor:pointer;user-select:none}
.chart-controls input{accent-color:var(--blue)}
.sw{width:16px;height:3px;border-radius:2px;display:inline-block}
.chart-canvas{position:relative;width:100%}
.chart-canvas svg{display:block;width:100%;height:auto}
.tip{position:absolute;pointer-events:none;background:rgba(15,23,42,.94);color:#fff;border-radius:9px;padding:9px 11px;
  font-size:11.5px;line-height:1.55;box-shadow:0 8px 24px rgba(15,23,42,.25);opacity:0;transition:opacity .12s;min-width:160px;z-index:5}
.tip b{color:#fff}.tip .tv{float:right;font-variant-numeric:tabular-nums;margin-left:14px}
.chart-note{font-size:11.5px;color:var(--faint);padding:8px 2px 2px}
.quote{font-size:21px;line-height:1.45;font-weight:700;letter-spacing:-.01em;margin:6px 0 12px}
.quote em{font-style:normal;color:var(--blue)}
ul.clean{margin:10px 0 0;padding:0;list-style:none}
ul.clean li{position:relative;padding-left:22px;margin-bottom:9px}
ul.clean li:before{content:'';position:absolute;left:2px;top:9px;width:7px;height:7px;border-radius:2px;background:var(--blue)}
ul.check li:before{content:'✓';background:none;color:var(--green);font-weight:800;top:0;left:0;font-size:13px}
ul.cross li:before{content:'✕';background:none;color:var(--red);font-weight:800;top:0;left:0;font-size:12px}
.twocol{columns:2;column-gap:34px}
.twocol li{break-inside:avoid}
details{margin:10px 0 0}
summary{cursor:pointer;font-size:12.5px;font-weight:700;color:var(--blue);padding:6px 0}
.footer{background:#0b1220;color:#94a3b8;margin-top:60px;padding:38px 0 44px;font-size:12.5px}
.footer b{color:#e2e8f0}
.footer a{color:#93c5fd}
.kv{display:grid;grid-template-columns:170px 1fr;gap:6px 16px;font-size:12.8px}
.kv .k{color:var(--muted)}
.kv .v{color:var(--ink)}
@media (max-width:980px){.kpis{grid-template-columns:repeat(2,1fr)}.grid3,.grid4,.podium,.bigstat{grid-template-columns:1fr 1fr}
  .hero h1{font-size:34px}.sechead{grid-template-columns:40px 1fr}.twocol{columns:1}}
@media (max-width:640px){.kpis{grid-template-columns:1fr 1fr}.grid2,.grid3,.grid4,.podium,.bigstat{grid-template-columns:1fr}
  .hero h1{font-size:29px}.wrap{padding:0 16px}.card{padding:20px 18px}.sechead h2{font-size:24px}}
@media print{nav.toc{display:none}.hero{padding:30px 0}.card{box-shadow:none;break-inside:avoid}.fig{break-inside:avoid}
  body{background:#fff}.chartbox{break-inside:avoid}}
"""

# --------------------------------------------------------------------------- JS
JS = """
(function(){
  const D = window.VOL_DATA;
  const defs = [
    {k:'actual',  n:'Реализованная', c:'#0b1220', w:3.0},
    {k:'har',     n:'HAR', c:'#0d9488', w:1.9},
    {k:'gbm_all', n:'Бустинг gbm_all', c:'#7c3aed', w:1.9},
    {k:'harx_all',n:'HAR-X + Deribit + лесенка', c:'#2563eb', w:2.1},
    {k:'mkt_ln_raw',n:'Лесенка: центр', c:'#e11d48', w:2.1},
    {k:'iv_mz',   n:'Deribit / DVOL', c:'#f59e0b', w:1.9}
  ];
  const state = {}; defs.forEach(d=>state[d.k]=true);
  const svg = document.getElementById('volChart');
  const tip = document.getElementById('volTip');
  const legend = document.getElementById('volLegend');
  const NS = 'http://www.w3.org/2000/svg';
  const W=1120,H=430,M={l:58,r:16,t:14,b:34};
  const n = D.weeks.length;
  const xs = i => M.l + i*(W-M.l-M.r)/(n-1);
  const ymax = () => {
    let mx = 0;
    defs.forEach(d=>{ if(!state[d.k]) return; D.series[d.k].forEach(v=>{ if(v!=null && v>mx) mx=v; }); });
    return Math.max(60, Math.ceil(mx*1.12/20)*20);
  };
  const ys = (v,Y) => M.t + (1-v/Y)*(H-M.t-M.b);
  function pathFor(vals,Y){
    let p='', open=false;
    vals.forEach((v,i)=>{
      if(v==null){open=false;return;}
      p += (open?'L':'M') + xs(i).toFixed(1) + ',' + ys(v,Y).toFixed(1) + ' ';
      open=true;
    });
    return p;
  }
  function draw(){
    const Y = ymax();
    svg.innerHTML='';
    // horizontal grid
    for(let v=0; v<=Y; v+=20){
      const y = ys(v,Y);
      const ln = document.createElementNS(NS,'line');
      ln.setAttribute('x1',M.l);ln.setAttribute('x2',W-M.r);ln.setAttribute('y1',y);ln.setAttribute('y2',y);
      ln.setAttribute('stroke', v===0?'#94a3b8':'#e8ecf3'); ln.setAttribute('stroke-width', v===0?1:0.8);
      svg.appendChild(ln);
      const tx = document.createElementNS(NS,'text');
      tx.setAttribute('x',M.l-9); tx.setAttribute('y',y+3.5); tx.setAttribute('text-anchor','end');
      tx.setAttribute('font-size','9.5'); tx.setAttribute('fill','#8a94a6');
      tx.textContent = v;
      svg.appendChild(tx);
    }
    // x labels
    for(let i=0;i<n;i+=4){
      const x=xs(i);
      const tx=document.createElementNS(NS,'text');
      tx.setAttribute('x',x);tx.setAttribute('y',H-13);tx.setAttribute('text-anchor','middle');
      tx.setAttribute('font-size','9.5');tx.setAttribute('fill','#8a94a6');
      tx.textContent=D.weeks[i];
      svg.appendChild(tx);
    }
    // series
    defs.forEach(d=>{
      if(!state[d.k]) return;
      const el=document.createElementNS(NS,'path');
      el.setAttribute('d',pathFor(D.series[d.k],Y));
      el.setAttribute('fill','none'); el.setAttribute('stroke',d.c);
      el.setAttribute('stroke-width',d.w); el.setAttribute('stroke-linejoin','round'); el.setAttribute('stroke-linecap','round');
      el.setAttribute('data-k',d.k);
      svg.appendChild(el);
    });
    // overlay
    const over=document.createElementNS(NS,'rect');
    over.setAttribute('x',M.l);over.setAttribute('y',M.t);over.setAttribute('width',W-M.l-M.r);over.setAttribute('height',H-M.t-M.b);
    over.setAttribute('fill','transparent'); svg.appendChild(over);
    const vline=document.createElementNS(NS,'line');
    vline.setAttribute('stroke','#334155');vline.setAttribute('stroke-width','1');vline.setAttribute('stroke-dasharray','3 3');
    vline.setAttribute('opacity','0'); vline.setAttribute('pointer-events','none'); svg.appendChild(vline);
    const dots=[];
    defs.forEach(d=>{
      const c=document.createElementNS(NS,'circle');
      c.setAttribute('r','3.4');c.setAttribute('fill',d.c);c.setAttribute('stroke','#fff');c.setAttribute('stroke-width','1.5');c.setAttribute('pointer-events','none');
      c.setAttribute('opacity','0'); svg.appendChild(c); dots.push({el:c,k:d.k});
    });
    over.addEventListener('mousemove', ev=>{
      const r=svg.getBoundingClientRect();
      const mx=(ev.clientX-r.left)/r.width*W;
      let i=Math.round((mx-M.l)/(W-M.l-M.r)*(n-1));
      i=Math.max(0,Math.min(n-1,i));
      const x=xs(i);
      vline.setAttribute('x1',x);vline.setAttribute('x2',x);vline.setAttribute('y1',M.t);vline.setAttribute('y2',H-M.b);
      vline.setAttribute('opacity','1');
      let rows='<b>'+D.weeks[i]+'.2026</b>';
      defs.forEach(d=>{
        const v=D.series[d.k][i];
        const dot=dots.find(z=>z.k===d.k).el;
        if(v==null){dot.setAttribute('opacity','0');return;}
        dot.setAttribute('cx',x);dot.setAttribute('cy',ys(v,Y));dot.setAttribute('opacity','1');
        if(state[d.k]) rows+='<div style="margin-top:3px"><span style="color:'+d.c+'">●</span> '+d.n+'<span class="tv">'+v.toFixed(1)+'%</span></div>';
      });
      tip.innerHTML=rows;
      tip.style.opacity='1';
      const px = x/W*100;
      const left = px>72 ? 'auto' : (px+1.5)+'%';
      const right = px>72 ? (100-px+1.5)+'%' : 'auto';
      tip.style.left=left; tip.style.right=right; tip.style.top='18px';
    });
    over.addEventListener('mouseleave', ()=>{
      tip.style.opacity='0'; vline.setAttribute('opacity','0'); dots.forEach(d=>d.el.setAttribute('opacity','0'));
    });
  }
  // legend
  defs.forEach(d=>{
    const lab=document.createElement('label');
    const cb=document.createElement('input'); cb.type='checkbox'; cb.checked=true;
    cb.addEventListener('change',()=>{state[d.k]=cb.checked; draw();});
    const sw=document.createElement('span'); sw.className='sw'; sw.style.background=d.c;
    lab.appendChild(cb); lab.appendChild(sw); lab.appendChild(document.createTextNode(d.n));
    legend.appendChild(lab);
  });
  draw();
})();
"""

# --------------------------------------------------------------------------- build
P = []
P.append(f"""<!doctype html><html lang="ru"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Волатильность биткоина по лесенкам Polymarket и Kalshi — финальный отчёт</title>
<style>{CSS}</style></head><body>""")

# HERO
P.append(f"""
<header class="hero"><div class="wrap">
  <div class="eyebrow">NES · Introduction to Forecasting · финальный отчёт · снимок 31.10.2026</div>
  <h1>Волатильность биткоина<br>по лесенкам <em>Polymarket</em> и <em>Kalshi</em></h1>
  <p class="lead">Прогнозируем реализованную дисперсию BTC на 55 минут, 24 часа и 6 дней; сравниваем с опционами Deribit
  и проверяем, можно ли заработать на разнице после спредов и комиссий. В отчёте — три года минутных данных,
  весь путь «факт против прогноза» по времени и 2 399 сделок живой лаборатории.</p>
  <div class="meta"><b>Александр Шестаков</b><span class="dotsep"></span><b>Александр Шилин</b>
    <span class="dotsep"></span>сдача 01.11.2026<span class="dotsep"></span>данные до 31.10.2026 01:05 UTC</div>
  <div class="kpis">
    <div class="kpi"><div class="v">2,01<small> млн</small></div><div class="l">минут BTC/USDT<br>январь 2023 — октябрь 2026</div></div>
    <div class="kpi"><div class="v">6 891</div><div class="l">часовой снимок<br>Kalshi с лесенкой</div></div>
    <div class="kpi accent"><div class="v">0,329</div><div class="l">CRPS центра лесенки —<br>лучший на 55 минут</div></div>
    <div class="kpi accent"><div class="v">−0,0215</div><div class="l">QLIKE даёт лесенка<br>сверх Deribit (DM −6,1)</div></div>
    <div class="kpi red"><div class="v">0<small> из 53</small></div><div class="l">гипотез прошли Холма<br>с положительным P&amp;L</div></div>
  </div>
</div></header>""")

P.append("""<nav class="toc"><div class="wrap">
<a href="#summary">Резюме</a><a href="#data">Данные</a><a href="#method">Метод</a>
<a href="#paths">Факт и прогноз</a><a href="#scores">Кто точнее</a><a href="#why">Почему лесенка</a>
<a href="#gbm">Бустинг</a><a href="#brier">Вероятности</a><a href="#alpha">Альфа</a>
<a href="#limits">Ограничения</a><a href="#repro">Воспроизводимость</a></div></nav>""")

# ------------------------------------------------------------------ summary
P.append(f"""
<section id="summary"><div class="wrap">
<div class="sechead"><div class="num">00</div><div><h2>Резюме: что мы узнали</h2>
<p class="dek">Четыре вывода, которые держатся на данных, и один — про то, почему «лучший прогноз» не равен «лучшей сделке».</p></div></div>

<div class="grid2">
  <div class="card">
    <h3>1. На 55 минутах лесенка — лучший прогноз волатильности</h3>
    <p>Центр часовой лесенки Kalshi даёт CRPS <b>0,329</b> против <b>0,358</b> у HAR и <b>0,614</b> у Deribit на общей выборке
    (5 822 часа, январь–октябрь 2026). Лучшая регрессия с лесенкой — <b>0,337</b>. На новой выборке 13–30.10
    лесенка и лучшая модель делят первое место: <b>0,341</b> у обеих.</p>
    <div class="bigstat">
      <div class="stat blue"><div class="v">0,329</div><div class="l">CRPS, центр лесенки</div></div>
      <div class="stat"><div class="v">0,337</div><div class="l">HAR-X + Deribit + лесенка</div></div>
      <div class="stat amber"><div class="v">0,702</div><div class="l">Deribit на новой выборке</div></div>
    </div>
  </div>
  <div class="card">
    <h3>2. Лесенка добавляет информацию сверх опционов — но только на часе</h3>
    <p>Добавление лесенки к HAR-X + Deribit снижает QLIKE на <b>0,0215</b> (DM <b>−6,08</b>, p &lt; 0,001) и CRPS на
    <b>0,0131</b> (DM <b>−11,5</b>). На 24 часах эффекта нет (p = 0,42), на 6 днях — тоже (p = 0,33), а точечный
    прогноз дисперсии от лесенки там просто шумит.</p>
    <div class="callout warn" style="margin-bottom:0"><div class="ct">Где ломается интуиция</div>
    <p>Точная дисперсия ≠ точные вероятности страйков. Рынок на 55 минутах остаётся лучше по Brier
    (0,10139 против 0,10162 у лучшей модели), а на 15-минутных рынках — с запасом
    (0,1463 против 0,1522, t −19,3).</p></div>
  </div>
  <div class="card">
    <h3>3. Бустинг на трёх годах минут бьёт HAR внутри дня</h3>
    <p>QLIKE вне выборки: <b>−27%</b> на 15 минутах, <b>−23%</b> на часе, но <b>+22%</b> хуже на сутках. Лучший
    прогноз девяти месяцев — среднее логарифмов бустинга и центра лесенки: CRPS <b>0,3255</b> против 0,3335
    у лесенки (<b>DM −5,3</b>). На новых двух неделях преимущество исчезает: harx_all 0,3347.</p>
    <div class="bigstat">
      <div class="stat violet"><div class="v">−27%</div><div class="l">QLIKE, 15 минут</div></div>
      <div class="stat violet"><div class="v">−23%</div><div class="l">QLIKE, 1 час</div></div>
      <div class="stat"><div class="v">+22%</div><div class="l">QLIKE, 1 день (хуже)</div></div>
    </div>
  </div>
  <div class="card">
    <h3>4. После комиссий не выживает ничего</h3>
    <p>53 правила в каталоге, 19 активных, <b>2 399</b> закрытых сделок на реальных bid/ask — и <b>ни одной</b>
    гипотезы, прошедшей Холма с плюсом. Пять «альф» оказались артефактами данных: устаревший листинг Kalshi,
    пустая сторона стакана как 0, снимок опционов против живого перпа, один месяц вместо истории, метки времени
    с разницей 20–40 секунд.</p>
    <div class="bigstat">
      <div class="stat green"><div class="v">4</div><div class="l">подтверждённые гипотезы R1–R21</div></div>
      <div class="stat amber"><div class="v">5</div><div class="l">пойманных ложных альф</div></div>
      <div class="stat red"><div class="v">0</div><div class="l">торгуемых после издержек</div></div>
    </div>
  </div>
</div>

<div class="callout" style="margin-top:6px"><div class="ct">Одной фразой</div>
<p class="quote">Рынок коротких бинарных контрактов <em>знает внутридневную сезонность волатильности</em> лучше, чем
дневная опционная IV; но как только прогноз превращается в сделку, спред и комиссия съедают всё.</p></div>

<div class="card" style="margin-top:22px">
  <h3>Что добавилось в этом снимке</h3>
  <div class="grid4">
    <div class="stat"><div class="v" style="font-size:20px">389</div><div class="l">часов после 13.10<br>в новой выборке (было 344)</div></div>
    <div class="stat"><div class="v" style="font-size:20px">2 399</div><div class="l">закрытых сделок<br>лаборатории 28–29.10</div></div>
    <div class="stat"><div class="v" style="font-size:20px">111 тыс.</div><div class="l">строк стаканов<br>коротких бинарных рынков</div></div>
    <div class="stat"><div class="v" style="font-size:20px">10</div><div class="l">живых прогнозов<br>в SHA-256 цепочке (13 файлов)</div></div>
  </div>
  <p class="small muted" style="margin-top:14px">Предыдущий замороженный срез заканчивался 28.10. Здесь прогнозы продлены до 31.10 01:05 UTC,
  добавлены живые данные лаборатории за 28–29.10 и пересчитаны все таблицы и графики. Целостность live-цепочки
  проверена: 10 записей, цепочка цела, код совпадает с замороженным (SHA-256).</p>
</div>
</div></section>""")

# ------------------------------------------------------------------ data
P.append(f"""
<section id="data"><div class="wrap">
<div class="sechead"><div class="num">01</div><div><h2>Задача и данные</h2>
<p class="dek">Лесенка бинарных контрактов — это стрип цифровых опционов. Её ширину можно перевести в безмодельную
дисперсию, а её изменение — сравнить с нашими моделями и опционами Deribit.</p></div></div>

<div class="card">
<h3>Как лесенка превращается в прогноз волатильности</h3>
<p>Контракт «BTC выше K в момент T» платит $1, если S<sub>T</sub> &gt; K. Его цена — вероятность, а весь набор страйков —
  распределение цены на T. По формуле Бриттен–Джонса–Нойбергера ожидаемая интегральная дисперсия равна
  цене «дисперсионного свопа», собранной из цифровых опционов:</p>
<div class="formula">E[∫<sub>t</sub><sup>T</sup> σ² du] = 2·E[ S<sub>T</sub>/F − 1 − ln(S<sub>T</sub>/F) ]
  = 2[ ∫<sub>F</sub><sup>∞</sup> d(K)(1/F − 1/K)dK + ∫<sub>0</sub><sup>F</sup> (1 − d(K))(1/K − 1/F)dK ]</div>
<p>Между страйками пробит-CDF интерполируется линейно по ln K, за крайними — экстраполируется наклоном крайнего
  отрезка. Дополнительно считаем логнормальную и Student-t подгонку по страйкам. Для Kalshi цель корректируется
  на расчёт по среднему BRTI за 60 секунд: <span class="small">RV[t, T−1 мин) + ⅓·RV[последняя минута]</span>.</p>
<div class="pipeline">
  <div class="step"><div class="sn">01</div><b>Минуты Binance</b><p>2,01 млн свечей; RV, полудисперсии, скачки, число сделок, объём.</p></div><div class="arrow">→</div>
  <div class="step"><div class="sn">02</div><b>Лесенка</b><p>Цены страйков → CDF → дисперсия BJN / логнорм. / Student-t.</p></div><div class="arrow">→</div>
  <div class="step"><div class="sn">03</div><b>Deribit</b><p>ATM IV ближайших экспираций, DVOL; пересчёт полной дисперсии на окно.</p></div><div class="arrow">→</div>
  <div class="step"><div class="sn">04</div><b>Модели</b><p>HAR, HAR-X, +Deribit, +лесенка, бустинг; скользящая оценка, плотность-смесь.</p></div><div class="arrow">→</div>
  <div class="step"><div class="sn">05</div><b>Оценка</b><p>QLIKE, CRPS, log score, PIT, DM, MCS, Brier по страйкам.</p></div>
</div>
</div>

<div class="grid3" style="margin-bottom:22px">
  <div class="horizon"><div class="tag">Kalshi · 55 минут</div><div class="big">32 761</div>
    <div class="sub">момент прогноза, из них <b>6 891</b> час с лесенкой; окно HH:05 → HH+1:00, расчёт по BRTI.</div>
    <div class="foot">Общая выборка моделей: 5 822 часа · новая: 389 часов</div></div>
  <div class="horizon"><div class="tag">Polymarket · 24 часа</div><div class="big">1 365</div>
    <div class="sub">дневных окон, <b>405</b> с лесенкой и <b>512</b> с ATM Deribit; расчёт 12:00 ET по close Binance.</div>
    <div class="foot">Общая выборка: 291 день · новая: 13 дней</div></div>
  <div class="horizon"><div class="tag">Polymarket · 6 дней</div><div class="big">1 360</div>
    <div class="sub">недельных окон, <b>411</b> с лесенкой и <b>497</b> с ATM Deribit; перекрывающиеся окна.</div>
    <div class="foot">Общая выборка: 335 дней · новая: 11 дней</div></div>
</div>

{img("fig01_data_timeline.png", "Рис. 1. Покрытие данных: три года минут, восемь месяцев лесенок, двое суток лаборатории.",
     "Источник: data/raw, data/interim, data/lab. Для коротких источников ширина полосы на графике искусственно расширена — смотрите подписи.")}

<div class="card">
<h3>Инвентарь данных</h3>
{table(list(inventory.columns), [[esc(x).replace("51 замороженная гипотеза", "53 ID (19 активных)") for x in r] for r in inventory.values.tolist()],
       aligns=["left"]*6, note="Все источники публичные, без аккаунта. Полные пути и объёмы — в reports/final/tables/data_inventory.csv.")}
<p class="small muted" style="margin-top:12px">Снимок кода и данных: <b>31.10.2026</b>. Основные артефакты: 275 485 строк прогнозов kx1h,
  167 863 строки оценок, 1,03 млн строк свечей Kalshi, 30 141 рынок KXBTC15M, 2 399 закрытых сделок лаборатории.
  Суммарно около 200 МБ сырых данных и 97 МБ очищенных таблиц.</p>
</div>

<div class="callout warn"><div class="ct">Дисциплина времени</div>
<p>Прогноз в момент t использует только данные с метками &lt; t. Скользящие окна — 90 дней для часа и 365 для суток,
  переобучение каждые 24 окна / каждый день. Живые прогнозы записаны в SHA-256 цепочку до исхода; правка файлов
  в <code>forecasts_live/</code> запрещена.</p></div>
</div></section>""")

# ------------------------------------------------------------------ method
P.append(f"""
<section id="method"><div class="wrap">
<div class="sechead"><div class="num">02</div><div><h2>Метод: что именно сравниваем</h2>
<p class="dek">Все модели прогнозируют одну и ту же величину — логарифм реализованной дисперсии окна. Разница только
в том, какие источники информации им разрешено использовать.</p></div></div>

<div class="card">
<h3>Модели</h3>
<table class="tbl">
<thead><tr><th class="left">Модель</th><th class="left">Что использует</th><th class="left">Зачем в отчёте</th></tr></thead>
<tbody>
<tr><td><b>naive</b></td><td>ln RV прошлого окна той же длины</td><td>нижняя планка</td></tr>
<tr><td><b>HAR</b></td><td>дневной, недельный и месячный ln RV (+ сезонность для часа)</td><td>базовая модель Corsi (2009)</td></tr>
<tr><td><b>HAR-X</b></td><td>+ полудисперсии, доля скачков, HARQ, макрокалендарь, выходные</td><td>асимметрия и короткие шоки</td></tr>
<tr><td><b>HAR-X + Deribit</b></td><td>+ ln DVOL² и ln ATM IV² окна</td><td>что дают опционы</td></tr>
<tr><td><b>HAR-X + лесенка</b></td><td>+ ln BJN-дисперсия лесенки</td><td>что даёт розничный рынок</td></tr>
<tr><td><b>HAR-X + Deribit + лесенка</b></td><td>оба источника сразу</td><td>ключевой encompassing-тест</td></tr>
<tr><td><b>Лесенка: центр</b></td><td>логнормальная подгонка по центральным страйкам</td><td>рынок как самостоятельный прогноз</td></tr>
<tr><td><b>Лесенка BJN / Student-t</b></td><td>безмодельная дисперсия / толстые хвосты</td><td>альтернативные способы чтения лесенки</td></tr>
<tr><td><b>Deribit IV</b></td><td>ATM IV / DVOL, как есть и после MZ-калибровки</td><td>опционный рынок как прогноз</td></tr>
<tr><td><b>gbm_all</b></td><td>HistGradientBoosting на признаках HAR-X + Deribit + лесенка</td><td>нелинейность</td></tr>
<tr><td><b>Бустинг на 3 годах минут</b></td><td>лаги RV 5 мин / 1 ч / 1 д / 1 нед, число сделок, объём, λ Кайла, поток, час недели</td><td>отдельная гонка моделей</td></tr>
<tr><td><b>Смесь</b></td><td>равные веса HAR-X, Deribit и лесенки</td><td>проверка, что комбинация не бесплатна</td></tr>
</tbody></table>
<p class="small muted">Плотность прогноза — смесь нормальных в ln RV: CRPS и вероятности страйков считаются из неё,
  точка — среднее смеси. Комиссии и спреды в торговых тестах всегда вычитаются: Kalshi taker ⌈0,07·P(1−P)⌉ до цента,
  maker ⌈0,0175·P(1−P)⌉, Polymarket 0,07·P(1−P), перп 5 б.п. за сторону.</p>
</div>

<div class="grid2">
<div class="card"><h3>Как читать графики волатильности</h3>
<p><b>Факт</b> — реализованная дисперсия окна, переведённая в годовое отклонение:
  √(RV·секунд_в_году / длительность_окна). <b>Прогноз</b> — √(E[RV]) из плотности модели. Мы агрегируем
  <i>дисперсию</i>, а уже потом извлекаем корень: среднее из волатильностей смещено вверх.</p>
<p>На часовом горизонте отдельное наблюдение — очень шумная величина: один час может дать 10% и 120% годовых.
  Поэтому на длинных графиках — недельные и месячные средние, а сырые часовые точки показываем только на
  последних двух неделях.</p></div>
<div class="card"><h3>Метрики</h3>
<div class="formula">QLIKE = RV/σ̂² − ln(RV/σ̂²) − 1          <span>// ниже — лучше</span>
CRPS  = ∫ (F(x) − 1{{x ≥ y}})² dx      <span>// ниже — лучше</span>
Brier = (p − outcome)²               <span>// ниже — лучше</span></div>
<p class="small">DM — Диболд–Мариано с HAC Newey–West и поправкой HLN; MCS — Model Confidence Set Хансена;
  для многих гипотез — Холм и BH; для стратегий — deflated Sharpe. Три выборки: <b>history</b> (до 13.10),
  <b>new</b> (после 13.10) и <b>live</b> (записано до исхода).</p></div>
</div>
</div></section>""")

# ------------------------------------------------------------------ paths (core)
P.append(f"""
<section id="paths"><div class="wrap">
<div class="sechead"><div class="num">03</div><div><h2>Факт и прогноз волатильности во времени</h2>
<p class="dek">Главный раздел: как разные методы отслеживают реализованную волатильность BTC — от трёхлетней истории
до последних двух недель по часам.</p></div></div>

<div class="callout"><div class="ct">Что видно сразу</div>
<p>Все модели хорошо ловят <b>уровень</b> и <b>режимы</b>: корреляция недельных лог-рядов с фактом — 0,97 у HAR,
  0,97 у центра лесенки, 0,94 у бустинга. Опционы Deribit на часовом горизонте — 0,73: DVOL почти не меняется
  внутри дня и систематически завышает ночную волатильность.</p></div>

{img("fig02_vol_full_history.png", "Рис. 2. Три года, месячные средние: HAR и бустинг идут почти вплотную; DVOL ошибается в спокойные годы.",
     "Kalshi, 55 минут. Реализованная волатильность — средняя дисперсия месяца, переведённая в годовое отклонение.")}

<div class="grid2">
<div class="card"><h3>Что показывает рис. 2</h3>
<ul class="clean">
<li>Спокойные 2023–2024: DVOL держится около 40–45%, тогда как факт опускается к 25–30%. Это не «ошибка опционов»,
  а премия за риск дисперсии и отсутствие внутридневной сезонности в 30-дневном индексе.</li>
<li>Всплески 2025 и февраля 2026 все методы ловят одновременно; расхождение — в спокойных неделях.</li>
<li>Бустинг ближе к факту, чем HAR, но разница на месячном сглаживании почти не видна — она проявляется в QLIKE
  на часовых точках (раздел 06).</li>
</ul></div>
<div class="card"><h3>Цифры за всем этим</h3>
<div class="kv">
<div class="k">Недельных наблюдений</div><div class="v">192 (2023–2026)</div>
<div class="k">corr log(факт) vs HAR</div><div class="v"><b>0,974</b></div>
<div class="k">corr vs бустинг</div><div class="v">0,942</div>
<div class="k">corr vs центр лесенки</div><div class="v"><b>0,969</b> (42 недели)</div>
<div class="k">corr vs Deribit / DVOL</div><div class="v">0,728</div>
<div class="k">Смещение HAR</div><div class="v">+0,01 лог-пункта (≈0%)</div>
<div class="k">Смещение naive</div><div class="v">−0,215 (завышает дисперсию на 24%)</div>
</div></div>
</div>

{img("fig03_vol_2026.png", "Рис. 3. 2026 год, недельные средние: лесенка и HAR-X + Deribit + лесенка идут вплотную к факту; DVOL отстаёт на спокойных участках.",
     "Kalshi, 55 минут. Заштрихованная зона — новая выборка 13–30.10, на которую модели не настраивались.")}

<div class="chartbox">
  <div style="display:flex;justify-content:space-between;align-items:baseline;gap:12px;flex-wrap:wrap">
    <div><div style="font-size:15px;font-weight:800">Интерактив: включите и выключите методы</div>
    <div style="font-size:12px;color:#5b6472;margin-top:2px">Kalshi, 55 минут, недельные средние 2026 года. Наведите курсор на график.</div></div>
    <div style="font-size:11px;color:#8a94a6">данные: final/tables/weekly_vol_kx1h.csv</div>
  </div>
  <div class="chart-controls" id="volLegend"></div>
  <div class="chart-canvas"><svg id="volChart" viewBox="0 0 1120 430" role="img" aria-label="Прогнозы волатильности по неделям"></svg>
    <div class="tip" id="volTip"></div></div>
  <noscript><p class="small muted">Интерактивному графику нужен JavaScript; статичная версия — рис. 3 выше.</p></noscript>
  <div class="chart-note">Годовая волатильность, %. Каждая точка — средняя дисперсия недели, затем корень. Пропуски — метод ещё не стартовал.</div>
</div>

{img("fig04_vol_last14.png", "Рис. 4. Последние две недели по часам: модель и лесенка повторяют почти каждый пик, DVOL остаётся плоским.",
     "Вверху — уровень; внизу — 6-часовое среднее относительной ошибки. Deribit систематически завышает ночь и занижает всплески.")}

<div class="card">
<h3>Что видно на часовом масштабе</h3>
<div class="grid3">
<div class="stat blue"><div class="v">0,98</div><div class="l">корреляция лог-рядов<br>у центра лесенки</div></div>
<div class="stat violet"><div class="v">0,95</div><div class="l">у бустинга<br>(недельные средние)</div></div>
<div class="stat amber"><div class="v">+30…+120%</div><div class="l">ошибка DVOL ночью<br>в спокойные часы</div></div>
</div>
<p>Пики 29–30.10 (годовая волатильность за час доходила до <b>128%</b>; исторический максимум — 653% 10.10.2025)
  не предсказывает ни одна модель — это
  ровно тот случай, когда реализованная дисперсия измеряется по одной минуте. Но <b>недельное среднее</b>
  все методы держат близко к факту, а лесенка лучше всех повторяет внутридневной профиль.</p>
</div>

{img("fig05_vol_pm.png", "Рис. 5. Polymarket, 24 часа и 6 дней: на сутках рынок и опционы сходятся, на неделе все методы шумят сильнее.",
     "Недельные средние дневных прогнозов; окна перекрываются. Красный выброс на 6 днях — тонкие страйки лесенки в январе 2026.")}

{img("fig06_scatter.png", "Рис. 6. Прогноз против факта на одной выборке часов: у лесенки лучшая линейная связь, у DVOL — худшая.",
     "Hexbin, логарифмическая шкала плотности; красная линия — 1:1. QLIKE и CRPS — на той же выборке 6 158 часов.")}

{img("fig07_bias_heatmap.png", "Рис. 7. Смещение по месяцам: наивная модель завышает дисперсию весь год, Deribit — почти всё время, HAR — почти нигде.",
     "log(средняя фактическая дисперсия / средняя прогнозная); ноль = калиброванный уровень.")}

<div class="grid2">
<div><h4 style="margin-top:0">Режимы волатильности</h4>
{table(["Режим", "Недель", "Средняя годовая волатильность"],
       [[f'{pd.Timestamp(r["from"]).strftime("%d.%m.%Y")} — {pd.Timestamp(r["to"]).strftime("%d.%m.%Y")}',
         num(r.weeks, 0), f"{num(r.ann_vol, 1)}%"] for _, r in regimes.iterrows()],
       aligns=["left", "right", "right"],
       note="Разбивка недельного лог-RV по BIC; границы — не экзогенные события, а статистические сдвиги уровня.")}
</div>
<div><p class="small" style="margin-top:4px">Оптимальная по BIC разбивка делит три года на шесть отрезков со средней
  годовой волатильностью от 25% до 44%. Модели не знают, в каком режиме они находятся, — но HAR с дневными лагами
  подстраивается к смене уровня за несколько дней, поэтому режимы почти не добавляют к прогнозу: R18 не подтверждён,
  R19 проходит только BH на 10%.</p>
  <p class="small muted">Для торговли это означает: нельзя калибровать порог «высокая волатильность» на всей истории —
  в спокойном режиме он будет срабатывать примерно вдвое чаще.</p></div>
</div>

{img("fig16_regimes.png", "Рис. 8. Шесть режимов волатильности, выбранных по BIC: от 25% до 44% годовых.", "Недельные значения реализованной дисперсии, 2023–2026; подписи — средняя годовая волатильность режима.")}

<div class="grid2">
<div class="card"><h3>Факт и прогноз: количественная сводка</h3>
{table(["Горизонт", "Лучшая модель по CRPS", "CRPS", "corr (нед.)", "Смещение"],
       [["55 минут", "Лесенка: центр", "0,329", "0,97", "−0,020"],
        ["55 минут", "HAR-X + Deribit + лесенка", "0,337", "0,97", "−0,003"],
        ["24 часа", "Лесенка: центр", "0,224", "0,89", "−0,057"],
        ["24 часа", "Deribit ATM", "0,224", "0,89", "+0,007"],
        ["6 дней", "Deribit ATM", "0,242", "0,64", "−0,016"],
        ["6 дней", "HAR-X + Deribit + лесенка", "0,247", "0,72", "−0,043"]],
       aligns=["left","left","right","right","right"],
       note="corr — корреляция лога недельных средних; смещение — среднее log(факт/прогноз).")}
</div>
<div class="card"><h3>CRPS по годам: вывод не держится на одном периоде</h3>
{year_table_html()}
</div>
</div>
</div></section>""")

# ------------------------------------------------------------------ scores
P.append(f"""
<section id="scores"><div class="wrap">
<div class="sechead"><div class="num">04</div><div><h2>Кто точнее: сводная таблица и значимость</h2>
<p class="dek">Сравнение на общей выборке, где у каждой модели есть прогноз на каждый момент. Ниже — не только
средние потери, но и тесты на разницу, чтобы отделить сигнал от шума.</p></div></div>

{podium_html()}

<div class="card">
<h3>55 минут, Kalshi — общая выборка 5 822 часа</h3>
{table(SC_HEAD, score_rows("kx1h", 9), aligns=SC_ALIGN,
       note="Отсортировано по CRPS. Зелёным — лучшее значение в столбце. p(MZ) — p-значение регрессии Минцера–Зарновица; &lt; 0,05 = калибровка отвергается.")}
<details><summary>Показать все 15 моделей</summary>
{table(SC_HEAD, score_rows("kx1h", 99, full=True), aligns=SC_ALIGN)}
</details>
</div>

<div class="card">
<h3>24 часа, Polymarket — общая выборка 291 день</h3>
{table(SC_HEAD, score_rows("pm24h", 9), aligns=SC_ALIGN,
       note="На сутках рынок, Deribit и HAR-X + Deribit неразличимы; лесенка BJN в чистом виде хуже, калиброванная — ближе.")}
</div>

<div class="card">
<h3>6 дней, Polymarket — общая выборка 335 дней</h3>
{table(SC_HEAD, score_rows("pm6d", 9), aligns=SC_ALIGN,
       note="На неделе Deribit и HAR-X лучшие; добавление лесенки к HAR-X + Deribit ухудшает QLIKE в 2,4 раза (0,342 против 0,143).")}
</div>

<div class="callout warn"><div class="ct">Ловушка «лучший CRPS»</div>
<p>У центра лесенки на 55 минутах лучший CRPS (0,329) и лучший MSE log (0,374) — то есть лучшая плотность.
  Но её точечный прогноз — среднее логнормального распределения — взрывается на редких широких распределениях,
  поэтому QLIKE 0,660. И наоборот: <b>HAR-X + Deribit + лесенка</b> даёт лучший точечный прогноз (QLIKE 0,248),
  но его CRPS 0,337. Поэтому в отчёте везде две метрики: CRPS — про плотность, QLIKE — про уровень.</p></div>

{img("fig09_scores_heatmap.png", "Рис. 9. Сводная таблица потерь: строки — модели, столбцы — горизонты; в клетках — абсолютные значения, цвет — отклонение от лучшего в горизонте.",
     "Белое/зелёное — лучший результат, красное — хуже. Сравнение только внутри столбца.")}

{img("fig10_dm_forest.png", "Рис. 10. Тесты Диболда–Мариано: что добавляет каждый блок и где разница значима.",
     "CRPS, выборка history. Синий — значимо лучше, красный — значимо хуже, серый — незначимо (p ≥ 0,05).")}

<div class="card"><h3>Ключевые сравнения: CRPS и QLIKE</h3>
{dm_table()}</div>

<div class="card">
<h3>Свежий срез: 13–30 октября (новая выборка)</h3>
<p class="small">Модели не настраивались на эти наблюдения. На 55 минутах появилось 389 часов (было 344 в срезе до 28.10).
  На сутках и неделе наблюдений мало — 13 и 11, поэтому выводы по ним шумные.</p>
<div class="grid3">
  <div><h4>55 минут · n = 389</h4>
  {table(["Модель", "CRPS", "QLIKE", "Δ CRPS к HAR"],
         [[esc(r.model_label), num(r.crps), num(r.qlike), pct(r.crps_vs_har_pct)] for _, r in rec_kx.head(6).iterrows()],
         aligns=["left","right","right","right"], note="Δ &lt; 0 — лучше HAR.")}</div>
  <div><h4>24 часа · n = 13</h4>
  {table(["Модель", "CRPS", "Δ к HAR"],
         [[esc(r.model_label), num(r.crps), pct(r.crps_vs_har_pct)] for _, r in rec_pm24.head(5).iterrows()],
         aligns=["left","right","right"])}</div>
  <div><h4>6 дней · n = 11</h4>
  {table(["Модель", "CRPS", "Δ к HAR"],
         [[esc(r.model_label), num(r.crps), pct(r.crps_vs_har_pct)] for _, r in rec_pm6.head(5).iterrows()],
         aligns=["left","right","right"])}</div>
</div>
<p class="small muted" style="margin-top:12px">На 55 минутах лучшие — HAR-X + Deribit + лесенка и центр лесенки (0,341),
  на 9% лучше HAR (0,375); Deribit — 0,702. На 24 часах формально лидирует HAR-X + лесенка (0,234), но при 13 наблюдениях
  это не результат. На 6 днях Deribit (0,122) снова лучший — как и на всей истории.</p>
</div>

<div class="callout ok"><div class="ct">Что выдерживает проверку</div>
<p>На 55 минутах лесенка добавляет информацию сверх Deribit: Δ QLIKE <b>−0,0215</b> (DM <b>−6,08</b>, p &lt; 0,001),
  Δ CRPS <b>−0,0131</b> (DM <b>−11,5</b>). Это главный положительный результат проекта. На 24 часах и 6 днях
  добавка незначима: рынок и опционы уже содержат одну и ту же информацию.</p></div>
</div></section>""")

# ------------------------------------------------------------------ why
P.append(f"""
<section id="why"><div class="wrap">
<div class="sechead"><div class="num">05</div><div><h2>Почему лесенка выигрывает на часе</h2>
<p class="dek">Не потому, что рынок «умнее» опционов, а потому, что лесенка торгуется на конкретный час и знает
внутридневной профиль волатильности. Дневная опционная IV — нет.</p></div></div>

{img("fig08_seasonality.png", "Рис. 11. Внутридневная сезонность: факт и лесенка растут к открытию США, DVOL остаётся плоским.",
     "Kalshi, декабрь 2025 — октябрь 2026, медианы по часам UTC; 6 891 час с лесенкой.")}

<div class="grid2">
<div class="card"><h3>Цифры сезонности</h3>
<div class="kv">
<div class="k">Реализованная, 14 UTC</div><div class="v"><b>57,3%</b> годовых</div>
<div class="k">Реализованная, 4 UTC</div><div class="v">25,0%</div>
<div class="k">Отношение</div><div class="v"><b>×2,3</b></div>
<div class="k">Лесенка, 14 UTC</div><div class="v">55,9%</div>
<div class="k">Лесенка, 4 UTC</div><div class="v">28,6%</div>
<div class="k">DVOL, 4–16 UTC</div><div class="v">40,2–40,7% (плоско)</div>
</div></div>
<div class="card"><h3>Премия за риск дисперсии</h3>
<p>Среднее log(RV) − log(подразумеваемая дисперсия) на 55 минутах:</p>
<div class="kv">
<div class="k">DVOL</div><div class="v"><b>−0,715</b> (t −30,1): implied в <b>2,04×</b> выше факта</div>
<div class="k">Лесенка BJN</div><div class="v">−0,284 (t −26,1): в 1,33× выше</div>
<div class="k">Лесенка: центр</div><div class="v"><b>+0,037</b> (t 4,4): почти без премии</div>
</div>
<p class="small muted" style="margin-top:10px">Сырая BJN-дисперсия завышена на треть, логнормальный центр — почти
  без смещения. Но именно у центра тяжёлые хвосты точечного прогноза, поэтому его QLIKE хуже (см. раздел 04).</p></div>
</div>

<div class="card">
<h3>На 24 часах преимущество исчезает</h3>
<p>На сутках ATM IV почти совпадает с фактом: отношение подразумеваемой дисперсии к реализованной — <b>1,04</b>
  (t −1,58, незначимо). Лесенка BJN завышает в 1,46 раза, DVOL — в 1,40. Дневная лесенка и Deribit ATM дают
  почти одинаковый уровень — 40,4% и 40,8% годовых при разбросе 3,2 п.п. По исследованию
  <code>studies/vol_models_summary.md</code>, реализованная волатильность ложится между ними, наклон регрессии
  факта на оба прогноза — 0,53, а продажа суточной дисперсии теряла <b>−3 п.п. в день</b> (t −3): премии за риск
  дисперсии в этой выборке нет. На 6 днях картина та же: ATM IV 1,04, BJN 1,24, DVOL 1,21 — лесенка становится
  шумнее из-за тонких страйков и перекрывающихся окон.</p>
</div>

<div class="callout"><div class="ct">Механизм, а не магия</div>
<p>На 55 минутах лесенка — единственный инструмент, который <i>торгуется на этот конкретный час</i> и в ценах
  которого агрегированы ожидания сотен участников о сезонности, макрорелизах и потоке. Deribit DVOL — 30-дневный
  индекс, равномерно размазанный по суткам; пересчёт на час стирает профиль. Это ограничение метода переноса
  дневной IV на час, а не «неэффективность» опционов.</p></div>
</div></section>""")

# ------------------------------------------------------------------ gbm
P.append(f"""
<section id="gbm"><div class="wrap">
<div class="sechead"><div class="num">06</div><div><h2>Бустинг: три года минут против HAR</h2>
<p class="dek">Отдельная гонка на минутных данных Binance, 2024–2026 вне выборки. Здесь нет лесенки — только
прошлая волатильность, активность и календарь.</p></div></div>

{img("fig13_horserace.png", "Рис. 12. QLIKE вне выборки: бустинг выигрывает 23–27% внутри дня и проигрывает 22% на сутках.",
     "Walk-forward, переобучение раз в месяц, DM с Newey–West. Данные: 2,01 млн минут Binance, 2024–2026.")}

<div class="card">
<h3>Таблица гонки</h3>
{table(["Горизонт", "HAR", "HAR без сезонности", "HAR + активность", "GBM", "GARCH(1,1)", "EWMA", "Выигрыш GBM"],
       [["15 минут", "0,459", "0,619", "0,430", "<b>0,335</b>", "—", "0,466", "<b>−27%</b>"],
        ["1 час", "0,330", "0,265", "0,331", "<b>0,253</b>", "0,440", "0,457", "<b>−23%</b>"],
        ["1 день", "0,188", "<b>0,165</b>", "0,198", "0,230", "0,355", "0,322", "+22%"]],
       aligns=["left"]+["right"]*7,
       note="QLIKE, ниже — лучше. Выигрыш GBM — относительное изменение к HAR; отрицательное = лучше.")}
<p class="small">Внутри дня деревья выигрывают каждый год (2024, 2025, 2026): выигрыш к HAR стабилен —
  +10…+16% на 15 минутах и +7…+8% на часе в каждом году. На сутках тысяча наблюдений — слишком мало для нелинейной модели:
  HAR без сезонности лучше.</p>
</div>

<div class="grid2">
<div class="card"><h3>Бустинг + лесенка — лучший прогноз девяти месяцев</h3>
<div class="kv">
<div class="k">GBM + центр лесенки</div><div class="v"><b>CRPS 0,3255</b>, DM к лесенке −5,3</div>
<div class="k">GBM + HAR-X + лесенка</div><div class="v">0,3278, DM −2,3</div>
<div class="k">Центр лесенки</div><div class="v">0,3335</div>
<div class="k">GBM отдельно</div><div class="v">0,3342, DM +0,2</div>
<div class="k">HAR-X + Deribit + лесенка</div><div class="v">0,3392, DM +2,2</div>
<div class="k">Те же модели, 13–28.10</div><div class="v">harx_all <b>0,3347</b>, комбинации — шум</div>
</div>
<p class="small muted" style="margin-top:10px">Выборка 15.01–12.10.2026, 5 744 часа с лесенкой; источник —
  <code>studies/gbm_kx1h.py</code>, <code>results/studies/gbm_kx1h.csv</code>. Среднее берётся по логарифмам,
  а не по уровням.</p></div>
<div class="card"><h3>Что это значит</h3>
<p>Градиентный бустинг — не замена лесенке, а второй, независимый взгляд: трёхлетняя история минут и
  часовой рыночный снимок ошибаются по-разному. Их среднее даёт лучший CRPS, но запас прочности мал: на новых
  двух неделях harx_all снова первый, а транзакционные издержки всё равно больше разницы в качестве прогноза.</p>
<ul class="clean"><li>15 минут: деятельность (число сделок, объём, λ Кайла) даёт ~6% QLIKE.</li>
<li>1 час: активность не помогает — вся информация уже в RV.</li>
<li>1 день: сезонность лучше убрать, чем моделировать деревьями.</li></ul></div>
</div>
</div></section>""")

# ------------------------------------------------------------------ brier
P.append(f"""
<section id="brier"><div class="wrap">
<div class="sechead"><div class="num">07</div><div><h2>Вероятности страйков: рынок не проигрывает</h2>
<p class="dek">Хорошая дисперсия ещё не значит хорошие вероятности. Проверяем Brier и log score по каждому страйку
на тех же снимках лесенки.</p></div></div>

{img("fig11_brier.png", "Рис. 13. Brier на страйк: на 55 минутах рынок значимо лучше моделей, на 24 часах и 6 днях — ничья.",
     "Kalshi 42 987 страйков / 6 555 снимков; Polymarket 3 783 / 342 и 3 683 / 335. DM по снимкам.")}

<div class="grid2">
<div class="card"><h3>Абсолютные значения</h3>
{table(["Горизонт", "Рынок", "HAR-X", "HAR-X + Deribit", "HAR-X + Deribit + лесенка"],
       [["55 минут", "<b>0,10139</b>", "0,10189", "0,10181", "0,10162"],
        ["24 часа", "<b>0,03864</b>", "0,03882", "0,03869", "0,03869"],
        ["6 дней", "0,09743", "<b>0,09633</b>", "0,09666", "0,09651"]],
       aligns=["left"]+["right"]*4, note="Brier на страйк, ниже — лучше.")}
<p class="small muted" style="margin-top:12px">На 55 минутах разрыв с лучшей моделью — 0,00023 Brier
  (DM +1,82, p = 0,069): рынок не хуже. На 6 днях модели даже чуть лучше, но разница незначима (t −1,1).</p></div>
<div class="card"><h3>Калибровка плотности</h3>
<p>PIT-гистограммы ближе всего к равномерной у HAR-X + Deribit + лесенка; у HAR на 6 днях дефицит
  высоких PIT (прогнозы систематически выше факта), у центра лесенки на 6 днях — горб в центре и почти нет
  значений у 0 и 1 (плотность слишком широкая).</p>
<div class="kv">
<div class="k">cov50, harx_all (55 мин)</div><div class="v">0,568 (номинал 0,50)</div>
<div class="k">cov90, harx_all</div><div class="v">0,918 (номинал 0,90)</div>
<div class="k">cov50, лесенка</div><div class="v">0,589</div>
<div class="k">cov90, лесенка</div><div class="v">0,924</div>
</div></div>
</div>

{img("fig12_pit.png", "Рис. 14. PIT: равномерность = калибровка. Пунктир — идеал; горб в центре — слишком широкая плотность.",
     "Все доступные наблюдения 2025–2026; строки — горизонты, столбцы — методы.")}

<div class="callout red"><div class="ct">Почему лучшая дисперсия не помогает</div>
<p>Вероятность страйка зависит сначала от того, где спот относительно страйка, и только потом — от дисперсии.
  Книга Kalshi видит BRTI и поток в реальном времени; наша модель получает минутные свечи. Улучшение дисперсии
  на 2–3% просто не может перебить эту асимметрию информации. На 55 минутах разрыв Brier между рынком и
  HAR-X — 0,00041 (DM +3,67), между рынком и лучшей моделью — 0,00023 (p = 0,069).
  На 15-минутных рынках разрыв ещё больше: Brier книги 0,1463 против 0,1522 у модели (t −19,3,
  <code>results/studies/kx15_summary.md</code>).</p></div>
</div></section>""")

# ------------------------------------------------------------------ alpha
P.append(f"""
<section id="alpha"><div class="wrap">
<div class="sechead"><div class="num">08</div><div><h2>Альфа: 53 гипотезы и ноль выживших</h2>
<p class="dek">Параллельно с курсовым вопросом круглосуточная лаборатория проверяла торговые правила на живых
стаканах семи площадок. Ни одно не прошло поправку на множественность с положительным P&amp;L.</p></div></div>

{img("fig17_alpha_funnel.png", "Рис. 15. Воронка: от 53 замороженных правил до нуля, прошедших Холма с плюсом.",
     "Каталог STRATEGIES.md; 2 399 закрытых сделок в data/lab/ledger.csv (28–29.10.2026).")}

<div class="grid2">
<div class="card"><h3>Лаборатория: что показали живые сделки</h3>
<p>Каждое правило заморожено до первой сделки и получает новый ID при изменении. Вход и выход — по реальным
  bid/ask, комиссии и спред вычтены, журнал дописывается и хешируется SHA-256.</p>
<ul class="clean">
<li><b>P06</b> (Kalshi 15m vs fair value): +6,2¢ на контракт, но всего 5 сделок — не результат.</li>
<li><b>P02</b> (Polymarket 15m): +6,0¢, t 1,5, Холм 1,0 — «ничего».</li>
<li><b>D06</b> (order-book imbalance): −0,10¢, t −8,6 — стабильно убыточно, выключено.</li>
<li><b>M01</b> (maker на Kalshi 15m): −9,4¢ на контракт за 5 сделок — асимметричный отбор не компенсируется
  мейкер-рибейтом; выборки пока нет.</li>
<li><b>O04</b> (стрэддл против прогноза): одна сделка −$469 — ровно тот случай, когда «одна сделка ≠ выборка».</li>
</ul></div>
<div class="card"><h3>Пять ложных альф</h3>
<table class="tbl" style="min-width:0;font-size:12px">
<thead><tr><th class="left">Что выглядело как альфа</th><th class="left">Причина</th></tr></thead>
<tbody>
<tr><td>Модель бьёт рынок 15m, P06 +12¢ (t 4,8)</td><td>листинг Kalshi отстаёт от стакана на 30–60 с</td></tr>
<tr><td>Арбитраж монотонности A11, +86¢</td><td>пустая сторона стакана отдаётся как 0</td></tr>
<tr><td>Паритет пут–колл O02, +$6 139</td><td>снимок опционов 10 мин против живого перпа</td></tr>
<tr><td>Продажа лотереи мейкером, +8,7¢ (t 3,4)</td><td>один месяц; на 220 днях −2¢</td></tr>
<tr><td>Kalshi–Polymarket арбитраж, +2…8¢ (t 10–19)</td><td>метки времени разнятся на 20–40 с</td></tr>
</tbody></table>
<p class="small muted" style="margin-top:10px">Общее правило: сравнивать только котировки одного момента,
  проверять на самой длинной истории и считать независимыми наблюдениями часы и рынки, а не сделки.</p></div>
</div>

<div class="card">
<h3>Микроструктура: спред — это налог</h3>
<div class="grid2">
<div>
{table(["Площадка", "Котировок", "Медиана, ¢", "Среднее, ¢"],
       [["Polymarket ≤ 1,5 д", "330", "1,00", "0,92"],
        ["Polymarket 1,5–3 д", "443", "1,00", "1,10"],
        ["Polymarket 5,5–8 д", "1 141", "2,00", "2,28"],
        ["Kalshi, 1 час", "574", "1,00", "5,24"]],
       aligns=["left","right","right","right"],
       note="Живой логгер, 67 снимков Polymarket и 53 снимка Kalshi, 28–29.10.2026.")}
</div>
<div>
<p class="small">У Kalshi медиана 1¢, но среднее 5,24¢: редкие тонкие страйки с огромным спредом. На 50¢
  тейкер платит 1,75¢ комиссии плюс половину спреда; круговая сделка стоит 4–6¢ — это больше типичного
  преимущества модели (0,1–0,3¢ Brier).</p>
<p class="small muted">Реплей стакана воспроизводит живые прогнозы с корреляцией <b>1,000</b>
  (|Δ| у HAR 0,008, у сырой IV 0,000). Гипотеза «бинарные контракты ведут спот на 45–120 с» не подтвердилась:
  на нулевом сдвиге корреляция 0,70–0,72 для Polymarket и 0,43 для Kalshi, а на ±45…120 с |r| ≤ 0,07.</p>
</div>
</div>
</div>

{img("fig15_lab.png", "Рис. 16. Средний чистый P&amp;L на сделку по правилам с ≥ 5 сделками: всё в пределах шума или минус.",
     "P&L после комиссий и спреда; t — по кластерам входа. Полная таблица — reports/final/tables/lab.csv.")}

<div class="card">
<h3>Живая таблица: 12 правил, о которых стоит знать</h3>
{table(["ID", "Семейство", "Правило", "Сделок", "Средний net, ¢", "t", "Холм p", "Вердикт"],
       [[esc(r.id), esc(r.family), esc(r.hypothesis), num(r.trades, 0), num(r.mean_net * 100, 2),
         num(r.t, 2) if pd.notna(r.t) else "—", pval(r.holm) if pd.notna(r.holm) else "—", live_verdict(r.verdict)]
        for _, r in lab[lab.trades >= 5].sort_values("mean_net", ascending=False).head(12).iterrows()],
       aligns=["left","center","left","right","right","right","right","center"],
       note="Отрицательный P&L — убыток. p — поправка Холма на 53 гипотезы; 1,0 означает, что эффект неотличим от нуля, «—» — слишком мало сделок для теста.")}
</div>

{img("fig14_research.png", "Рис. 17. Двадцать две исследовательские гипотезы: четыре подтверждены после Холма, четыре — на грани (BH 10%), остальные — шум.",
     "reports/research.md; t с Newey–West. R1 — сезонность волатильности, R10 — переоценка хвостов, R13 — всплеск в первую минуту часа, R21 — HAR против GARCH.")}

<div class="card">
<h3>Что подтвердилось в исследованиях (R1–R21, 22 проверки)</h3>
{table(["ID", "Гипотеза", "Эффект", "Ед.", "t", "Холм p", "Вердикт"],
       [[esc(r.id), esc(r.hypothesis_ru), num(r.effect, 2), esc(r.unit), num(r.t, 2), pval(r.holm), verdict(r.verdict)]
        for _, r in research.iterrows()],
       aligns=["left","left","right","left","right","right","center"],
       note="Полная таблица с n и устойчивостью — reports/final/tables/research.csv. BH — Бенджамини–Хохберг.")}
<p class="small muted" style="margin-top:12px"><b>R1</b> (сезонность) — самый сильный эффект: наклон 0,97, t 38,6,
  сохраняется во всех режимах. <b>R10</b> — хвосты лесенок переоценены на 2,6 п.п., но после спреда и комиссии
  торговли нет. <b>R13</b> — волатильность в первую минуту часа выше в 1,10 раза. <b>R21</b> — HAR бьёт GARCH
  на сутках. Отдельно стоит <b>R11 cash-and-carry</b>: Sharpe 6,0, но это сбор фандинга с ограничением ёмкости,
  а не предсказание волатильности.</p>
</div>

<div class="callout warn"><div class="ct">Осторожно с «подтверждено»</div>
<p>R10 статистически подтверждён, но экономически не торгуется: 2,6 п.п. переоценки меньше спреда на дальних
  страйках. R2, R8a, R12, R19 проходят только BH на 10% и не проходят Холма. Это ровно та зона, где раньше
  рождались «альфы» — и умирали на полной выборке.</p></div>
</div></section>""")

# ------------------------------------------------------------------ limits
P.append(f"""
<section id="limits"><div class="wrap">
<div class="sechead"><div class="num">09</div><div><h2>Ограничения и что дальше</h2>
<p class="dek">Что может сломать выводы и какие из них уже сейчас хрупкие.</p></div></div>

<div class="grid2">
<div class="card"><h3>Ограничения</h3>
<ul class="clean">
<li><b>Две недели — не выборка для суток.</b> На 24 часах 13 новых наблюдений, на 6 днях 11; выводы по ним — шум.</li>
<li><b>История Polymarket — без стакана.</b> Торговые тесты на ней — верхняя граница, а не P&amp;L.</li>
<li><b>Базис BRTI/Binance.</b> Kalshi рассчитывается по BRTI, у нас минуты Binance; базис оценивается по прошлым расчётам.</li>
<li><b>Бумажная торговля.</b> Нет собственного влияния на цену и места в очереди, кроме правил с моделью очереди.</li>
<li><b>Юрисдикции.</b> Доступ к Kalshi и Polymarket ограничен; практическая ценность любой стратегии зависит от этого.</li>
<li><b>Один актив.</b> BTC; перенос на ETH и другие серии Kalshi не проверен.</li>
</ul></div>
<div class="card"><h3>Что дальше</h3>
<ul class="check">
<li>Внутридневная торговля гаммой на Deribit: длинные дельта-хеджированные дневные стрэддлы перед открытием США,
  короткие — на азиатскую ночь. Нужна внутридневная история цен опционов.</li>
<li>Живая проверка W02, A12/A12F и V07 — ещё несколько недель.</li>
<li>Полная история Kalshi 15m и Polymarket 5m/15m на одновременных стаканах.</li>
<li>Перенос методики на ETH и другие серии Kalshi.</li>
</ul></div>
</div>

<div class="card">
<h3>Расхождения в снимке</h3>
<p class="small">README фиксирует 51 гипотезу (снимок 29.10), каталог <code>STRATEGIES.md</code> — 53 ID (19 активных,
  34 выведенных). В отчёте используется 53 как более свежее число; live-таблица содержит 46 строк, потому что часть
  правил ещё не совершила ни одной сделки. Предыдущий замороженный срез заканчивался 28.10; здесь данные продлены
  до 31.10 01:05 UTC, поэтому числа в разделе 04 немного отличаются от <code>reports/oos_report.md</code> —
  это не ошибка, а более длинная выборка.</p>
</div>
</div></section>""")

# ------------------------------------------------------------------ repro
P.append(f"""
<section id="repro"><div class="wrap">
<div class="sechead"><div class="num">10</div><div><h2>Воспроизводимость</h2>
<p class="dek">Все числа и графики собираются из файлов репозитория двумя скриптами, без сети.</p></div></div>

<div class="card">
<h3>Как пересобрать</h3>
<div class="formula">python reports/final_report_analysis.py   <span># таблицы + 17 рисунков</span>
python reports/build_final_report.py      <span># HTML + Markdown + интерактив</span></div>
<div class="grid2">
<div><h4>Что читается</h4>
<div class="kv">
<div class="k">Прогнозы</div><div class="v">forecasts_kx1h.parquet, forecasts_pm24h.parquet, forecasts_pm6d.parquet</div>
<div class="k">Оценки</div><div class="v">scores.parquet, results/tables/*.csv</div>
<div class="k">Панели</div><div class="v">data/interim/panel_*.parquet</div>
<div class="k">Лаборатория</div><div class="v">data/lab/ledger.csv, reports/lab.md</div>
<div class="k">Исследования</div><div class="v">reports/research.md</div>
</div></div>
<div><h4>Что создаётся</h4>
<div class="kv">
<div class="k">Отчёт</div><div class="v">reports/final_report.html, reports/final_report.md</div>
<div class="k">Таблицы</div><div class="v">reports/final/tables/*.csv (13 файлов)</div>
<div class="k">Рисунки</div><div class="v">reports/final/figures/*.png (17 файлов)</div>
<div class="k">Интерактив</div><div class="v">weekly_vol_kx1h.csv внутри HTML</div>
</div></div>
</div>
<p class="small muted" style="margin-top:14px">Проверка целостности live-прогнозов: 10 записей, цепочка SHA-256 цела,
  код совпадает с замороженным (<code>reports/oos_report.md</code>, раздел Code check).</p>
</div>

<div class="card">
<h3>Индекс таблиц</h3>
{table_index()}
</div>

<div class="card">
<h3>Быстрая проверка ключевых чисел</h3>
{table(["Что", "Значение", "Где"],
       [["CRPS центра лесенки, 55 мин", "0,3293", "scoreboard.csv"],
        ["CRPS HAR-X + Deribit + лесенка, 55 мин", "0,3374", "scoreboard.csv"],
        ["Δ QLIKE от лесенки сверх Deribit", "−0,0215 (DM −6,08)", "dm_key.csv / dm_forest"],
        ["Δ CRPS от лесенки сверх Deribit", "−0,0131 (DM −11,5)", "oos_dm.csv"],
        ["Brier рынка, 55 мин", "0,10139", "brier.csv"],
        ["Brier лучшей модели, 55 мин", "0,10162", "brier.csv"],
        ["QLIKE GBM, 1 час / HAR", "0,253 / 0,330", "horserace.csv"],
        ["Закрытых сделок лаборатории", "2 399", "ledger.csv"],
        ["Правил, прошедших Холма с плюсом", "0", "lab.csv"]],
       aligns=["left","right","left"])}
</div>
</div></section>""")

# FOOTER
P.append("""<footer class="footer"><div class="wrap">
  <div style="display:flex;justify-content:space-between;gap:24px;flex-wrap:wrap">
    <div><b>Волатильность биткоина по лесенкам Polymarket и Kalshi</b><br>
      Курсовой проект РЭШ «Introduction to Forecasting» · снимок 31.10.2026</div>
    <div>Александр Шестаков · Александр Шилин<br>сдача 01.11.2026</div>
  </div>
  <div style="margin-top:18px;color:#64748b">Отчёт собран из данных репозитория; все графики — авторские,
    источники данных публичные (Binance, Polymarket, Kalshi, Deribit). Не является инвестиционной рекомендацией.</div>
</div></footer>""")

P.append(f"""<script>window.VOL_DATA = {json.dumps(VOL_DATA, ensure_ascii=False)};</script>
<script>{JS}</script></body></html>""")

out = "".join(P).replace("< ", "&lt; ")
(HERE / "final_report.html").write_text(out, encoding="utf-8")
print("html", len(out) / 1e6, "MB")

# --------------------------------------------------------------------------- markdown
md = []
md.append("# Волатильность биткоина по лесенкам Polymarket и Kalshi\n")
md.append("_Курсовой проект РЭШ «Introduction to Forecasting» · Александр Шестаков, Александр Шилин · "
          "снимок данных 31.10.2026 · сдача 01.11.2026_\n")
md.append("**Главный вывод.** На 55 минутах центр лесенки Kalshi — лучший прогноз волатильности "
          "(CRPS 0,329 против 0,358 у HAR и 0,614 у Deribit). Лесенка добавляет информацию сверх опционов "
          "Deribit (Δ QLIKE −0,0215, DM −6,08). На 24 часах и 6 днях рынок и опционы неразличимы. "
          "После комиссий не выживает ни одна из 53 торговых гипотез.\n")
md.append("Полная версия с интерактивным графиком и всеми таблицами: **[final_report.html](final_report.html)**.\n")

md.append("## 1. Ключевые числа\n")
md.append(md_table(pd.DataFrame([
    ["CRPS, 55 мин, центр лесенки", "0,329", "лучший на горизонте"],
    ["CRPS, 55 мин, HAR-X + Deribit + лесенка", "0,337", "лучшая регрессия"],
    ["CRPS, 55 мин, Deribit", "0,614", "худший"],
    ["Δ QLIKE от лесенки сверх Deribit, 55 мин", "−0,0215 (DM −6,08)", "p < 0,001"],
    ["Δ QLIKE от лесенки сверх Deribit, 24 ч", "+0,0017 (DM +0,82)", "незначимо"],
    ["Δ QLIKE от лесенки сверх Deribit, 6 дн", "+0,199 (DM +0,98)", "незначимо, шум"],
    ["GBM: QLIKE к HAR, 15 мин / 1 час / 1 день", "−27% / −23% / +22%", "вне выборки 2024–2026"],
    ["Brier рынка / лучшей модели, 55 мин", "0,10139 / 0,10162", "рынок не хуже (p = 0,069)"],
    ["Живых сделок лаборатории", "2 399", "28–29.10.2026"],
    ["Гипотез, прошедших Холма с плюсом", "0 из 53", "—"],
]), headers=["Показатель", "Значение", "Комментарий"]))

md.append("\n## 2. Факт и прогноз волатильности во времени\n")
for f, cap in [("fig02_vol_full_history.png", "Рис. 1. Три года, месячные средние: HAR, бустинг и DVOL."),
               ("fig03_vol_2026.png", "Рис. 2. 2026 год, недельные средние: все методы против факта."),
               ("fig04_vol_last14.png", "Рис. 3. Последние две недели по часам: факт, HAR-X + Deribit + лесенка, лесенка, DVOL."),
               ("fig05_vol_pm.png", "Рис. 4. Polymarket 24 часа и 6 дней."),
               ("fig06_scatter.png", "Рис. 5. Прогноз против факта на общей выборке часов."),
               ("fig07_bias_heatmap.png", "Рис. 6. Смещение прогнозов по месяцам 2026.")]:
    md.append(f"![{cap}](final/figures/{f})\n\n_{cap}_\n")

md.append("\n## 3. Кто точнее\n")
md.append("### 55 минут, Kalshi (общая выборка 5 822 часа)\n")
_mdsc = sc[sc.horizon == "kx1h"].sort_values("crps").head(9).copy()
_mdsc["mz_p"] = _mdsc.mz_p.map(lambda p: "< 0,001" if p < 0.001 else f"{p:.3f}".replace(".", ","))
md.append(md_table(_mdsc,
                   ["model_label", "n", "crps", "qlike", "logscore", "cov50", "cov90", "mz_b", "mz_p"],
                   ["Модель", "n", "CRPS", "QLIKE", "Log score", "cov50", "cov90", "MZ b", "p(MZ)"]))
md.append("\n### 24 часа и 6 дней\n")
for h, name in [("pm24h", "Polymarket, 24 часа"), ("pm6d", "Polymarket, 6 дней")]:
    md.append(f"**{name}**\n")
    md.append(md_table(sc[sc.horizon == h].sort_values("crps").head(8),
                       ["model_label", "n", "crps", "qlike", "logscore", "cov50", "cov90"],
                       ["Модель", "n", "CRPS", "QLIKE", "Log score", "cov50", "cov90"]))
    md.append("")

md.append("\n## 4. Почему лесенка выигрывает на часе\n")
md.append("![Сезонность](final/figures/fig08_seasonality.png)\n\n_Рис. 7. Внутридневная сезонность: факт, лесенка и плоский DVOL._\n")
md.append("Реализованная волатильность в 14 UTC — 57,3% годовых, в 4 UTC — 25,0% (×2,3). Лесенка повторяет профиль "
          "(55,9% и 28,6%), DVOL — нет (40,2–40,7% в любое время суток). Средняя подразумеваемая дисперсия DVOL "
          "в 2,04 раза выше факта, у лесенки BJN — в 1,33 раза, у логнормального центра — 0,96.\n")

md.append("\n## 5. Бустинг на трёх годах минут\n")
md.append("![Гонка](final/figures/fig13_horserace.png)\n\n_Рис. 8. QLIKE вне выборки 2024–2026._\n")
md.append(md_table(pd.DataFrame([
    ["15 минут", "0,459", "0,619", "0,430", "0,335", "—", "0,466", "−27%"],
    ["1 час", "0,330", "0,265", "0,331", "0,253", "0,440", "0,457", "−23%"],
    ["1 день", "0,188", "0,165", "0,198", "0,230", "0,355", "0,322", "+22%"],
]), headers=["Горизонт", "HAR", "HAR deseas.", "HAR + активность", "GBM", "GARCH", "EWMA", "GBM к HAR"]))
md.append("\nСреднее логарифмов «GBM + центр лесенки» — лучший прогноз 15.01–12.10.2026: CRPS 0,3255 против 0,3335 "
          "у лесенки (DM −5,3, 5 744 часа). На новых двух неделях harx_all снова первый (0,3347), а улучшение "
          "дисперсии не переносится на Brier страйков.\n")

md.append("\n## 6. Вероятности страйков\n")
md.append("![Brier](final/figures/fig11_brier.png)\n\n_Рис. 9. Brier на страйк: рынок против моделей._\n")
md.append(md_table(lad, ["horizon", "source", "n_strikes", "n_snapshots", "brier_per_strike", "logloss_per_strike"],
                   ["Горизонт", "Источник", "Страйков", "Снимков", "Brier", "Log loss"], "{:.5f}"))
md.append("\nНа 55 минутах рынок лучше HAR-X (DM +3,67), HAR-X + Deribit (DM +3,01) и не хуже лучшей модели "
          "HAR-X + Deribit + лесенка (DM +1,82, p = 0,069). На 24 часах и 6 днях разницы нет.\n")

md.append("\n## 7. Альфа\n")
md.append("![Воронка](final/figures/fig17_alpha_funnel.png)\n\n_Рис. 10. 53 правила → 19 активных → 2 399 сделок → 0 выживших._\n")
md.append("![Лаборатория](final/figures/fig15_lab.png)\n\n_Рис. 11. Средний чистый P&L на сделку._\n")
_labmd = lab[lab.trades >= 5].sort_values("mean_net", ascending=False).head(10).copy()
_labmd["verdict"] = _labmd.verdict.map({"nothing": "ничего", "collecting": "собирает данные",
                                        "loses money": "убыточно", "no signal yet": "нет сигнала"}).fillna(_labmd.verdict)
_labmd["holm"] = _labmd.holm.map(lambda p: "" if pd.isna(p) else ("< 0,001" if p < 0.001 else f"{p:.3f}".replace(".", ",")))
md.append(md_table(_labmd,
                   ["id", "family", "trades", "mean_net", "t", "holm", "verdict"],
                   ["ID", "Семейство", "Сделок", "Средний net, $", "t", "Холм p", "Вердикт"], "{:.3f}"))
md.append("\nПять «альф» оказались артефактами: устаревший листинг Kalshi, пустая сторона стакана как 0, снимок "
          "опционов против живого перпа, один месяц вместо истории и рассинхронизация меток времени на 20–40 с.\n")

md.append("\n## 8. Исследования R1–R21\n")
md.append("![Research](final/figures/fig14_research.png)\n\n_Рис. 12. t-статистики 22 гипотез._\n")
md.append("После Холма подтверждены R1 (сезонность), R10 (переоценка хвостов), R13 (всплеск в первую минуту часа), "
          "R21 (HAR лучше GARCH на сутках). R2, R8a, R12, R19 проходят только BH на 10%. Остальные 14 — шум. "
          "R10 не торгуется после спреда; R11 cash-and-carry даёт Sharpe 6,0, но это сбор фандинга с ограниченной ёмкостью.\n")

md.append("\n## 9. Ограничения\n")
md.append("- Две недели новых наблюдений на 24 часах и 6 днях — не выборка.\n"
          "- История Polymarket без стакана; торговые тесты на ней — верхняя граница.\n"
          "- Kalshi рассчитывается по BRTI, у нас минуты Binance; базис модельный.\n"
          "- Бумажная торговля не учитывает влияние на цену и очередь.\n"
          "- Один актив (BTC); перенос на ETH не проверен.\n")

md.append("\n## 10. Воспроизводимость\n")
md.append("```bash\npython reports/final_report_analysis.py   # таблицы и рисунки\n"
          "python reports/build_final_report.py      # HTML + Markdown\n```\n")
md.append("Таблицы: `reports/final/tables/*.csv`; рисунки: `reports/final/figures/*.png`. "
          "Live-прогнозы: 10 записей, цепочка SHA-256 цела, код совпадает с замороженным.\n")

(HERE / "final_report.md").write_text("\n".join(md), encoding="utf-8")
print("md", (HERE / "final_report.md").stat().st_size / 1e3, "KB")
print("done")
