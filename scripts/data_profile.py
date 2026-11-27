"""What is actually on this disk — одной страницей текста вместо выгрузки сотен мегабайт.

    python scripts/data_profile.py                     напечатать профиль
    python scripts/data_profile.py --out profile.txt   то же в файл (его удобно прислать)

Зачем: чтобы разбирать проблемы с данными, не выгружая данные. Здесь только факты о файлах — сколько,
за какой период, сколько строк, есть ли повторы ключей. Этого хватает, чтобы понять, например, что в
часовых лесенках есть повторные (тикер, минута) и поэтому падает `wings_ten`.

Скрипт ничего не меняет и не падает: каждая секция обёрнута в try/except и печатает «нет данных», если
датасета на диске нет.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

DATA = ROOT / "data"
# (путь, что это, читать ли пары ключей для проверки повторов)
DATASETS = [
    ("raw/kalshi_kxbtcd_60m", "часовые лесенки KXBTCD", True),
    ("raw/wing_gap_candles", "ноябрьские снимки книги", True),
    ("lab/markets", "журнал книги лаборатории", True),
    ("raw/kx15", "ленты KXBTC15M", False),
    ("raw/kalshi_kxbtc_60m", "часовые range-бакеты", False),
    ("raw/binance", "минуты и свечи Binance", False),
    ("raw/deribit", "опционы Deribit", False),
    ("raw/polymarket", "Polymarket", False),
    ("interim", "промежуточные parquet", False),
]
SIZE_WARN = 3 * 1024 ** 3        # больше 3 ГБ — повод сказать вслух


def _human(n: float) -> str:
    for unit in ("Б", "КБ", "МБ", "ГБ"):
        if n < 1024 or unit == "ГБ":
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} ГБ"


def _dups(paths: list[Path]) -> tuple[int, int]:
    """(строк всего, повторных пар (ticker, end_ts)) — только для файлов, где эти колонки есть."""
    total = dups = 0
    for f in paths:
        try:
            d = pd.read_csv(f, usecols=["ticker", "end_ts"], engine="pyarrow")
        except Exception:                                  # noqa: BLE001 — не наш формат, пропускаем
            continue
        total += len(d)
        dups += int(d.duplicated().sum())
    return total, dups


def profile() -> list[str]:
    L: list[str] = []
    L.append(f"Python {sys.version.split()[0]} · pandas {pd.__version__} · {sys.platform}")
    L.append(f"папка: {ROOT}")
    try:
        import study_all                                            # noqa: PLC0415

        L.append(f"версия кода: {study_all.code_stamp()}")
    except Exception:                                               # noqa: BLE001
        L.append("версия кода: не определена")
    L.append("")
    L.append(f"{'датасет':44s}{'файлов':>8s}{'период':>27s}{'строк':>13s}{'повторов':>10s}{'объём':>10s}")
    for rel, what, check in DATASETS:
        base = DATA / rel
        name = f"{rel} ({what})"
        try:
            files = [f for f in sorted(base.rglob("*")) if f.is_file() and not f.name.startswith(".")]
            if not files:
                L.append(f"{name[:44]:44s}{'—':>8s}{'нет на диске':>27s}")
                continue
            size = sum(f.stat().st_size for f in files)
            csvs = [f for f in files if f.suffix in (".csv", ".gz", ".parquet")]
            period = ""
            if csvs and rel != "interim":                 # у parquet имена — это названия таблиц, не даты
                stems = sorted(f.name.split(".")[0] for f in csvs)
                dates = [s for s in stems if re.fullmatch(r"\d{4}-?\d{2}-?\d{2}(\d{2}(\d{2})?)?", s)]
                if dates:
                    period = f"{dates[0]} … {dates[-1]}" if len(dates) > 1 else dates[0]
            rows = dups = 0
            if check and csvs:
                rows, dups = _dups(csvs)
            L.append(f"{name[:44]:44s}{len(files):>8d}{period:>27s}"
                     f"{(format(rows, ',') if rows else '—'):>12s}{(format(dups, ',') if dups else '—'):>10s}"
                     f"{_human(size):>10s}")
            if dups:
                L.append(f"    ! {dups} повторных (ticker, end_ts): из-за них падает wings_ten; в коде уже дедупликация")
        except Exception as exc:                                    # noqa: BLE001 — профиль не должен падать
            L.append(f"{name[:44]:44s}  ошибка чтения: {type(exc).__name__}: {exc}")
    L.append("")
    total = sum(f.stat().st_size for f in DATA.rglob("*") if f.is_file()) if DATA.exists() else 0
    L.append(f"всего в data/: {_human(total)}" + ("  (много — выгружать такое на GitHub не нужно)"
                                                  if total > SIZE_WARN else ""))
    L.append("")
    L.append("Этот файл (обычно 20–40 строк) и есть то, что нужно прислать вместо данных:")
    L.append("по нему видно, чего не хватает, за какой период и есть ли битые/повторные строки.")
    return L


def main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="что лежит в data/: файлы, периоды, строки, повторы")
    ap.add_argument("--out", default=None, help="записать в файл (например profile.txt)")
    a = ap.parse_args(argv)
    text = "\n".join(profile())
    print(text)
    if a.out:
        Path(a.out).write_text(text + "\n", encoding="utf-8")
        print(f"\nзаписано: {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
