"""Чтение часовых лесенок Kalshi — одно место для всех изучений.

Два урока с машины пользователя (Python 3.14 / pandas 3 / pyarrow):

* маркеры «этот день перекачан с фильтром по страйкам» когда-то писались как `2026-01-19.csv.dense`, то есть
  подходили под маску `*.csv*`, которой изучения ищут данные, и `pd.read_csv(usecols=...)` падал с
  `Usecols do not match columns` — так умерли двадцать изучений;
* `strike`/`expiration_value`/`yes_bid` приезжают строками (`large_string`), и сравнение `strike > value`
  падало с `Invalid comparison between dtype=str and ndarray`.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

LADDER_COLS = ["ticker", "end_ts", "yes_bid", "yes_ask", "volume", "strike",
               "strike_type", "result", "expiration_value", "close_time", "event"]
NUMERIC = ("strike", "expiration_value", "yes_bid", "yes_ask", "volume")


def ladder_files(directory) -> list[Path]:
    """Only real data files under a `*.csv*` mask: `.csv` and `.csv.gz`, never `.csv.dense` or anything else."""
    return sorted(p for p in Path(directory).glob("*.csv*") if p.name.endswith((".csv", ".csv.gz")))


def read_ladder_file(path, usecols=LADDER_COLS):
    """One ladder file, or None with a printed reason when the file is not a ladder at all."""
    try:
        header = pd.read_csv(path, nrows=0).columns
    except Exception as exc:                                   # noqa: BLE001 — any garbage on disk
        print(f"  пропуск {Path(path).name}: не читается как csv ({type(exc).__name__})", flush=True)
        return None
    missing = [c for c in usecols if c not in header]
    if missing:
        print(f"  пропуск {Path(path).name}: нет колонок {missing[:3]}", flush=True)
        return None
    d = pd.read_csv(path, usecols=usecols)
    for col in NUMERIC:
        if col in d.columns and not pd.api.types.is_numeric_dtype(d[col]):
            d[col] = pd.to_numeric(d[col], errors="coerce")
    return d


def no_data(what: str):
    """Raise the one phrase the runner turns into «нет данных в окне» instead of a traceback."""
    raise ValueError(f"нет данных в окне: {what}")
