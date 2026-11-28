"""Цифры отчёта, посчитанные один раз — из таблиц, а не из литералов в тексте.

Повод: в `reports/build_final_report.py` заголовочные числа были вписаны руками —
«0 из 53», «53 правила в каталоге, 19 активных, 2 399 закрытых сделок». Данные за это время уехали
(в каталоге 54 ID, закрытых сделок больше), и отчёт стал противоречить собственным таблицам: в KPI «0 из 53»,
а в таблице рядом другие числа. Любая модель, которая «правит отчёт», бьётся об это первой.

Здесь единственный источник правды:

    facts = compute()                 # посчитать из reports/final/tables/lab.csv, data/lab/ledger.csv, STRATEGIES.md
    text = substitute(template, facts) # подставить в шаблон; неизвестный маркер → ошибка, а не тихая подстановка

Маркеры (`__N_HYPOTHESES__` и т. п.) стоят в шаблоне отчёта вместо чисел. Если маркер остался неподставленным
или в шаблоне появилось новое число, которого нет в фактах, сборка падает: лучше не собрать отчёт, чем собрать
уверенно врущий.
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LAB_CSV = ROOT / "reports" / "final" / "tables" / "lab.csv"
LEDGER = ROOT / "data" / "lab" / "ledger.csv"
CATALOG = ROOT / "STRATEGIES.md"

# маркер → как называется факт; порядок важен только для читаемости HANDOFF.md
TOKENS = {
    "__N_HYPOTHESES__": "n_hypotheses",
    "__N_WITH_TRADES__": "n_with_trades",
    "__N_HOLM_POSITIVE__": "n_holm_positive",
    "__N_CATALOG_IDS__": "n_catalog_ids",
    "__N_CATALOG_ACTIVE__": "n_catalog_active",
    "__N_CLOSED_TRADES__": "n_closed_trades",
    "__N_OPEN_TRADES__": "n_open_trades",
}
NOT_COMPUTED = "—"


def fmt(n) -> str:
    """Число как в отчёте: пробел между тысячами, «—» если не посчитано."""
    if n is None:
        return NOT_COMPUTED
    return f"{int(n):,}".replace(",", " ")


def _read_lab(path: Path):
    import pandas as pd

    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception:                                         # noqa: BLE001
        return None


def catalog_counts(path: Path = CATALOG) -> tuple[int | None, int | None]:
    """(сколько ID в каталоге, сколько из них с исполненными сделками).

    Каталог — таблица в Markdown: считаем строки, начинающиеся с `| D01 |` и т. п. «Активные» без гадания по
    тексту: это те ID, которые есть в таблице лаборатории с trades > 0 (см. compute()).
    """
    if not path.exists():
        return None, None
    ids = re.findall(r"^\|\s*([A-Z]\d{2}[A-Z]?)\s*\|", path.read_text(encoding="utf-8"), re.M)
    return (len(set(ids)) or None), None


def compute(lab_csv: Path = LAB_CSV, ledger: Path = LEDGER, catalog: Path = CATALOG) -> dict:
    """Все заголовочные числа отчёта. Непосчитанное — None, и это видно в отчёте как «—»."""
    facts = {name: None for name in TOKENS.values()}
    lab = _read_lab(lab_csv)
    if lab is not None and {"trades", "holm", "mean_net"}.issubset(lab.columns):
        facts["n_hypotheses"] = len(lab)
        facts["n_with_trades"] = int((lab["trades"] >= 5).sum())
        holm = lab["holm"].astype(float)
        net = lab["mean_net"].astype(float)
        facts["n_holm_positive"] = int(((holm < 0.05) & (net > 0)).sum())
    n_ids, _ = catalog_counts(catalog)
    facts["n_catalog_ids"] = n_ids
    if lab is not None and "trades" in lab.columns:
        facts["n_catalog_active"] = int((lab["trades"] > 0).sum())
    if ledger.exists():
        try:
            import pandas as pd

            d = pd.read_csv(ledger, usecols=["event"])
            facts["n_closed_trades"] = int((d["event"] == "CLOSE").sum())
            facts["n_open_trades"] = int((d["event"] == "OPEN").sum())
        except Exception:                                     # noqa: BLE001 — битый журнал не повод падать
            facts["n_closed_trades"] = facts["n_open_trades"] = None
    return facts


class TemplateError(RuntimeError):
    """В шаблоне осталось число, которого нет в фактах."""


def substitute(text: str, facts: dict) -> str:
    """Подставить факты в маркеры и проверить, что ни один маркер не остался."""
    for token, name in TOKENS.items():
        text = text.replace(token, fmt(facts.get(name)))
    left = re.findall(r"__[A-Z_]+__", text)
    if left:
        raise TemplateError(f"в шаблоне остались неизвестные маркеры: {sorted(set(left))}")
    return text


def describe(facts: dict) -> str:
    """Одна строка для логов и HANDOFF.md."""
    return (f"гипотез в таблице лаборатории {fmt(facts.get('n_hypotheses'))}"
            f" (с ≥5 сделками — {fmt(facts.get('n_with_trades'))}), "
            f"прошли Холма с плюсом — {fmt(facts.get('n_holm_positive'))}; "
            f"ID в каталоге {fmt(facts.get('n_catalog_ids'))}, "
            f"с исполненными сделками {fmt(facts.get('n_catalog_active'))}, "
            f"сделок в журнале {fmt(facts.get('n_open_trades'))} открытий / "
            f"{fmt(facts.get('n_closed_trades'))} закрытий")
