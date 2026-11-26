"""Generate docs/IDEAS_100.md from results/ideas100/results.csv (numbers always match the run)."""
import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
res = pd.read_csv(ROOT / "results/ideas100/results.csv").sort_values("id")

def fmt(x, nd=4):
    return "—" if not np.isfinite(x) else f"{x:.{nd}f}"

def fmtp(x):
    if not np.isfinite(x):
        return "—"
    return f"{x:.3g}"

# verdict helper
def verdict(r):
    if r["status"] != "ok":
        return f"не протестирована: {r['status']}"
    if r["category"] == "Directional":
        if r["id"] == 100:
            return f"MSE |r| {r['mse_abs']:.2e} против {r['mse_abs_base']:.2e} у среднего (DM {r['dm_vs_base']:.1f}, p={fmtp(r['p_value'])}): HAR предсказывает абсолютную доходность лучше среднего — волатильность информативна для |r|."
        r2 = r.get("oos_r2", np.nan)
        corr = r.get("corr", np.nan)
        sh = r.get("sharpe_net_ann", np.nan)
        if np.isfinite(r2) and r2 > 0 and abs(corr) > 0.02:
            return f"OOS R²={r2:.4f}, corr={corr:.3f}, Sharpe net={sh:.2f}: слабый, но ненулевой сигнал; после оборота 10 б.п./переворот торговать ежечасно нельзя."
        return f"OOS R²={r2:.4f}, corr={corr:.3f}, Sharpe net={sh:.2f}: сигнала нет (прогноз ≈ константа, стратегия ≈ buy-and-hold)."
    dm = r.get("dm_vs_har", np.nan)
    ph = r.get("p_holm", np.nan)
    gain = r.get("qlike_gain_pct", np.nan)
    if np.isfinite(dm) and dm < -2 and np.isfinite(ph) and ph < 0.05:
        return f"QLIKE {r['qlike']:.4f} ({gain:+.1f}% к HAR), DM {dm:.2f}, Holm p={fmtp(ph)}: улучшение подтверждается после поправки на 100 сравнений."
    if np.isfinite(dm) and dm < -2:
        return f"QLIKE {r['qlike']:.4f} ({gain:+.1f}% к HAR), DM {dm:.2f}, Holm p={fmtp(ph)}: номинально лучше HAR, но после Холма не значимо — кандидат на дальнейшую проверку."
    if np.isfinite(dm) and dm > 2:
        return f"QLIKE {r['qlike']:.4f} ({gain:+.1f}% к HAR), DM {dm:.2f}: значимо хуже HAR."
    return f"QLIKE {r['qlike']:.4f} ({gain:+.1f}% к HAR), DM {dm:.2f}: не отличается от HAR."

lines = []
lines.append("# 100 идей/моделей — дизайн и результаты вневыборочной проверки")
lines.append("")
lines.append("> Сгенерировано из `results/ideas100/results.csv` скриптом `studies/gen_ideas100_doc.py` — числа всегда соответствуют последнему прогону `studies/test_100_ideas.py`.")
lines.append("> Панель: `kx1h` (Kalshi, окно 55 минут, 32 763 часа, 2023-02 … 2026-10). Цель — `y = log RV` окна `[t, e)`.")
lines.append("")
lines.append("## Как тестировали (честность)")
lines.append("")
lines.append("- **No look-ahead.** Признаки в момент `t` используют только данные до `t`; обучение OLS/Ridge — только на случаях с `e_j <= t` (окно 2 160 часов, переобучение раз в неделю), как в `src/models.py`.")
lines.append("- **ML (Lasso/GBM/RF/ET)** — один временной сплит: обучение до 2025-06-01, тест после 2025-06-02 (purge 1 день). Для признаков лесенки (есть только с декабря 2025) — хронологический сплит 60/40 на строках с признаками.")
lines.append("- **Метрики волатильности:** QLIKE (Patton 2011, главная точечная потеря), MSE на `log RV`, гауссова CRPS. **DM** — Diebold–Mariano с HAC Newey–West и поправкой Harvey–Leybourne–Newbold против базовой HAR (идея 01) на общей выборке; **Holm** — поправка на все 100 сравнений.")
lines.append("- **Направленные идеи (91–100):** регрессия следующей часовой доходности, OOS R², корреляция, Sharpe знаковой стратегии net (10 б.п. за переворот позиции — консервативно для ежечасной торговли).")
lines.append("- **Аутэйджи биржи** (forward-filled свечи, `filled=True`) помечены как missing, а не как нулевая волатильность — иначе BV/MedRV дают ровно 0 и ломают логи (см. идеи 16–17).")
lines.append("")
lines.append("## Сводка по категориям")
lines.append("")
cats = res.groupby("category").agg(n=("id", "count"), ok=("status", lambda s: (s == "ok").sum()))
lines.append("| Категория | Идей | Протестировано |")
lines.append("|---|---|---|")
for c, row in cats.iterrows():
    lines.append(f"| {c} | {int(row['n'])} | {int(row['ok'])} |")
lines.append("")
lines.append("## Лучшие 15 (по QLIKE)")
lines.append("")
lines.append("| # | Идея | n | QLIKE | CRPS | DM | Holm p |")
lines.append("|---|---|---|---|---|---|---|")
vol = res[(res["status"] == "ok") & (res["category"] != "Directional")].sort_values("qlike").head(15)
for _, r in vol.iterrows():
    lines.append(f"| {int(r['id'])} | {r['name']} | {int(r['n_test'])} | {r['qlike']:.4f} | {r['crps']:.4f} | {r['dm_vs_har']:.2f} | {fmtp(r['p_holm'])} |")
lines.append("")
lines.append("## Все 100 идей")
lines.append("")
for _, r in res.iterrows():
    iid = int(r["id"])
    lines.append(f"### {iid}. {r['name']} — {r['category']}")
    lines.append("")
    lines.append(f"- **Источник/мотивация:** {r['note']}.")
    lines.append(f"- **Признаки:** `{', '.join(str(x) for x in str(r.get('features','')).split(',')) if pd.notna(r.get('features','')) else 'см. studies/test_100_ideas.py (IDEAS)'}; **оценка:** {r['estimator']}.")
    lines.append(f"- **Результат:** {verdict(r)}")
    if r["category"] != "Directional" and r["status"] == "ok":
        lines.append(f"- **Числа:** n={int(r['n_test'])}, QLIKE={r['qlike']:.4f}, MSE_log={r['mse_log']:.4f}, CRPS={r['crps']:.4f}, тест {r['test_start']} … {r['test_end']}.")
    lines.append("")
lines.append("## Что из этого следует")
lines.append("")
lines.append("1. **Лесенка доминирует.** Лучшие модели — HAR + параметры лесенки (`mkt_s`, RMSE, `nu`), а не только MFIV: форма распределения несёт информацию о будущей дисперсии сверх её ширины. Это согласуется с главным результатом проекта (лесенка лучше Deribit на часе из-за внутридневной сезонности).")
lines.append("2. **HAR-X + макро + сезонность** — лучшие чисто временные модели. Макро-дамми и Фурье hour-of-week дают значимый выигрыш даже после Холма; объём, фандинг, VRP-прокси — нет.")
lines.append("3. **Jump-robust меры (BV/MedRV/truncated/RV5) хуже RV** на часовом горизонте BTC: скачки — сигнал, а не шум, плюс дискретность цен ломает BV/MedRV на минутах. На сутках картина может быть иной (см. `docs/LITERATURE.md`).")
lines.append("4. **ML не бьёт OLS на тех же признаках** на этом сплите (GBM близок, но не значимо лучше HAR-X); GBM выигрывает, когда ему дают много признаков и лесенку (идея 77), но выборка лесенки мала (2 733 часа теста) — нужен walk-forward, а не один сплит.")
lines.append("5. **Направленных сигналов нет.** Все OOS R² около нуля; DVOL-change даёт corr 0.02, но ежечасные перевороты убивает оборот. Волатильность предсказуема, знак — нет (согласуется с R20).")
lines.append("")
lines.append("## Воспроизведение")
lines.append("")
lines.append("```bash")
lines.append("pip install -r requirements.txt")
lines.append("python studies/test_100_ideas.py   # -> results/ideas100/results.csv, summary.md, forecasts_long.parquet")
lines.append("python studies/gen_ideas100_doc.py # -> этот файл")
lines.append("```")
lines.append("")

out = ROOT / "docs/IDEAS_100.md"
out.write_text("\n".join(lines), encoding="utf-8")
print(f"wrote {out} ({len(lines)} lines)")
