"""Чтение часовых лесенок: файлы-маркеры и битые файлы не должны валить изучения.

Повод: на машине пользователя двадцать изучений падали с
`ValueError: Usecols do not match columns, columns expected but not found: ['yes_bid', ...]`.
Причина — маркеры «этот день перекачан с фильтром по страйкам» писались как `2026-01-19.csv.dense`,
то есть попадали под маску `*.csv*`, которой изучения ищут данные, и читались как таблица.
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from scripts.study_all import migrate_dense_markers  # noqa: E402
from src.download.kalshi import _marker, _sparse  # noqa: E402
from studies.wing_panel import LADDER_COLS, read_ladder_file  # noqa: E402
from studies.wings_two_sided import utc_index  # noqa: E402

LADDER = (
    "ticker,end_ts,yes_bid,yes_ask,volume,strike,strike_type,result,expiration_value,close_time,event\n"
    "KXBTCD-26JUL0112-T100000,1782900000,0.41,0.43,120,100000.5,greater,no,,2026-07-01T13:00:00Z,KXBTCD-26JUL0112\n"
    "KXBTCD-26JUL0112-T101000,1782900000,0.11,0.13,40,,greater,yes,100850.25,2026-07-01T13:00:00Z,KXBTCD-26JUL0112\n"
)


def _write(dir_: Path, name: str, text: str) -> Path:
    path = dir_ / name
    path.write_text(text, encoding="utf-8")
    return path


class LadderReaderTest(unittest.TestCase):
    def test_real_ladder_is_read_with_numeric_columns(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write(Path(tmp), "2026-07-01.csv", LADDER)
            d = read_ladder_file(path)
        self.assertEqual(len(d), 2)
        self.assertTrue(pd.api.types.is_numeric_dtype(d.strike))
        self.assertTrue(pd.api.types.is_numeric_dtype(d.expiration_value))
        self.assertEqual(d.strike.iloc[0], 100000.5)
        self.assertTrue(pd.isna(d.strike.iloc[1]))
        self.assertEqual(d.expiration_value.iloc[1], 100850.25)

    def test_marker_file_is_skipped_not_parsed(self):
        """Тот самый `*.csv.dense`: маска `*.csv*` его ловит, но читать его как данные нельзя."""
        with tempfile.TemporaryDirectory() as tmp:
            path = _write(Path(tmp), "2026-07-01.csv.dense",
                          "re-downloaded with the spacing-aware strike filter")
            self.assertIsNone(read_ladder_file(path))
            # и он действительно попадает под маску, из-за которой всё ломалось
            self.assertEqual([p.name for p in Path(tmp).glob("*.csv*")], ["2026-07-01.csv.dense"])

    def test_truncated_file_is_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write(Path(tmp), "2026-07-02.csv", "ticker,end_ts\nKXBTCD-x,1\n")
            self.assertIsNone(read_ladder_file(path))

    def test_required_columns_are_the_ladder_ones(self):
        for col in ("yes_bid", "yes_ask", "strike", "expiration_value", "close_time", "end_ts"):
            self.assertIn(col, LADDER_COLS)

    def test_every_csv_glob_in_studies_filters_marker_names(self):
        """Статическая проверка: любая маска `*.csv*` рядом должна отбрасывать не-данные."""
        offenders, files = [], sorted((ROOT / "studies").glob("*.py")) + sorted((ROOT / "src").rglob("*.py"))
        for f in files:
            lines = f.read_text(encoding="utf-8").splitlines()
            for i, line in enumerate(lines):
                if 'glob("*.csv*")' not in line:
                    continue
                window = " ".join(lines[i:i + 3])
                if "endswith" not in window:
                    offenders.append(f"{f.relative_to(ROOT)}:{i + 1}")
        self.assertEqual(offenders, [], f"маска *.csv* без фильтра имён: {offenders}")


class MarkerNameTest(unittest.TestCase):
    def test_new_markers_never_look_like_csv(self):
        self.assertEqual(_marker(Path("d/2026-01-19.csv.gz")).name, "2026-01-19.dense")
        self.assertEqual(_marker(Path("d/2026-01-19.csv")).name, "2026-01-19.dense")
        self.assertNotIn(".csv", _marker(Path("d/2026-01-19.csv.gz")).name)

    def test_sparse_recognises_legacy_and_new_markers(self):
        sparse = ("event,ticker\nKXBTCD-26JAN1912,KXBTCD-26JAN1912-T100000\n"
                  "KXBTCD-26JAN1912,KXBTCD-26JAN1912-T101000\n")
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp) / "2026-01-19.csv"                     # .gz пришлось бы писать по-настоящему сжатым
            raw.write_text(sparse, encoding="utf-8")
            self.assertTrue(_sparse(raw))                           # 2 страйка на событие — день перекачать
            (Path(tmp) / "2026-01-19.dense").write_text("ok", encoding="utf-8")
            self.assertFalse(_sparse(raw))                          # новый маркер
            (Path(tmp) / "2026-01-19.dense").unlink()
            (Path(tmp) / "2026-01-19.csv.dense").write_text("ok", encoding="utf-8")
            self.assertFalse(_sparse(raw))                          # старый маркер тоже понимаем


class MigrateMarkersTest(unittest.TestCase):
    def test_old_markers_are_renamed_out_of_the_csv_namespace(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            (d / "2026-01-19.csv.dense").write_text("ok", encoding="utf-8")
            (d / "2026-01-20.csv.gz").write_text("x", encoding="utf-8")
            (d / "2026-01-20.csv.dense").write_text("ok", encoding="utf-8")
            (d / "2026-01-21.dense").write_text("ok", encoding="utf-8")       # уже новое имя
            (d / "2026-01-21.csv.dense").write_text("ok", encoding="utf-8")   # дубль — просто убрать

            self.assertEqual(migrate_dense_markers(d), 3)
            self.assertEqual(sorted(p.name for p in d.iterdir()),
                             ["2026-01-19.dense", "2026-01-20.csv.gz", "2026-01-20.dense", "2026-01-21.dense"])
            self.assertEqual(migrate_dense_markers(d), 0)                     # идемпотентно


class NoDataTest(unittest.TestCase):
    def test_phrase_is_reported_as_missing_data_not_as_a_crash(self):
        from scripts.study_all import classify
        from studies.ladders import no_data

        with self.assertRaises(ValueError):
            no_data("нет снимков книги")
        log = ('Traceback (most recent call last):\n'
               '  File "studies/x.py", line 1, in main\n'
               '    no_data(f"нет данных в окне: {what}")\n'
               'ValueError: нет данных в окне: нет снимков книги\n')
        kind, reason = classify(1, log)
        self.assertEqual(kind, "nodata")
        self.assertEqual(reason, "нет данных в окне: нет снимков книги")


class UtcIndexTest(unittest.TestCase):
    def test_naive_and_aware_both_become_utc(self):
        naive = pd.DatetimeIndex(pd.to_datetime(["2026-09-01 00:00", "2026-09-01 01:00"]))
        self.assertIsNotNone(utc_index(naive).tz)
        self.assertEqual(str(utc_index(naive).tz), "UTC")
        aware = naive.tz_localize("UTC")
        self.assertTrue((utc_index(aware) >= pd.Timestamp("2026-09-01", tz="UTC")).all())


if __name__ == "__main__":
    unittest.main()
