"""Build a shareable Alpha Lab snapshot from aggregate research results."""
from html import escape
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TABLES = {
    "wing_stopping": "Когда входить после появления edge",
    "wing_shapes": "Форма лестницы страйков",
    "wing_sizing": "Размер позиции и просадка",
    "wing_ten": "Десять моделей вероятности выплаты",
    "wing_filters": "Десять фильтров входа",
    "wing_sleep_summary": "Непрерывная работа и пропущенные часы",
    "wing_consensus": "Согласие модельной и рыночной оценок",
    "wing_capacity": "Модели на доступном объёме",
    "wing_ledger_capacity": "Стратегии W01–W12",
}

def main():
    sections = []
    for key, title in TABLES.items():
        file = ROOT / "results/lab" / (key + ".csv")
        frame = pd.read_csv(file)
        sections.append(f'<section><h2>{escape(title)}</h2><a href="../results/lab/{key}.csv">Скачать таблицу</a>'
                        f'<div>{frame.to_html(index=False, na_rep="—")}</div></section>')
    page = """<!doctype html><html lang="ru"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Alpha Lab · стратегии крыльев</title><style>body{font:16px/1.5 system-ui;color:#17342f;background:#f4f7f5;max-width:1440px;margin:auto;padding:24px}section{background:white;padding:18px;margin:18px 0}section div{overflow:auto}table{border-collapse:collapse;font-size:14px}td,th{white-space:nowrap;text-align:right;padding:6px 10px;border:1px solid #dae5df}a{color:#006951}</style>
<h1>Alpha Lab · стратегии крыльев BTC</h1><p>Исследовательский снимок. Периоды и выборки указаны в техническом разборе. Архивные сделки рассчитаны на один контракт с проверкой цены минутой позже. Расчёты по стаканам используют видимый объём с учётом комиссии и исходного ценового лимита.</p>
<nav><a href="wing_atlas.html">Атлас рынка</a> · <a href="../docs/wing-models.html">Технический разбор</a> · <a href="../README.md">О проекте</a></nav>"""
    (ROOT/"reports/alpha-lab.html").write_text(page+"".join(sections)+"</html>", encoding="utf-8")

if __name__ == "__main__":
    main()
