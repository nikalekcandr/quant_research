"""Binance 1-second candles: parsing both archive flavours and the monthly parquet build."""
import io
import zipfile
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import pandas as pd

from src.download import binance_1s as b1s

MS = 1759395600000          # 2025-10-02 09:00:00 UTC in milliseconds
US = MS * 1000              # the archives switched to microseconds in 2025


def zip_bytes(rows, header=False) -> bytes:
    lines = [",".join(b1s.COLS)] if header else []
    lines += [",".join(str(x) for x in r) for r in rows]
    text = "\n".join(lines) + "\n"
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("BTCUSDT-1s-2026-11-01.csv", text)
    return buf.getvalue()


def row(ts, price=110000.0, vol=0.5, trades=3):
    return [ts, price, price + 1, price - 1, price, vol, ts + 999, price * vol, trades, vol / 2, price * vol / 2, 0]


class ReadZipTest(unittest.TestCase):
    def test_millisecond_file_without_header(self):
        df = b1s.read_zip(zip_bytes([row(MS), row(MS + 1000, 110010.0)]))
        self.assertEqual(list(df.columns), b1s.KEEP)
        self.assertEqual(len(df), 2)
        self.assertEqual(df["ts"].iloc[0], pd.Timestamp("2025-10-02 09:00:00", tz="UTC"))
        self.assertAlmostEqual(float(df["close"].iloc[1]), 110010.0)
        self.assertEqual(int(df["trades"].iloc[0]), 3)

    def test_microsecond_file_with_header(self):
        df = b1s.read_zip(zip_bytes([row(US)], header=True))
        self.assertEqual(len(df), 1)
        self.assertEqual(df["ts"].iloc[0], pd.Timestamp("2025-10-02 09:00:00", tz="UTC"))

    def test_one_second_spacing_is_kept(self):
        df = b1s.read_zip(zip_bytes([row(US), row(US + 1_000_000), row(US + 2_000_000)]))
        self.assertEqual(df["ts"].diff().dropna().unique().tolist(), [pd.Timedelta(seconds=1)])

    def test_dtypes_are_small(self):
        df = b1s.read_zip(zip_bytes([row(US)]))
        self.assertEqual(str(df["close"].dtype), "float32")       # 2.6M rows a month: keep it compact
        self.assertEqual(str(df["trades"].dtype), "int32")


class BuildTest(unittest.TestCase):
    def test_build_writes_one_parquet_per_month(self):
        with TemporaryDirectory() as d:
            raw, interim = Path(d) / "raw", Path(d) / "interim"
            raw.mkdir()
            (raw / "BTCUSDT-1s-2026-11.zip").write_bytes(zip_bytes([row(MS), row(MS + 1000)]))
            (raw / "BTCUSDT-1s-2026-12.zip").write_bytes(zip_bytes([row(MS + 86400 * 1000)]))
            old = (b1s.RAW_DIR, b1s.INTERIM)
            b1s.RAW_DIR, b1s.INTERIM = raw, interim
            try:
                df = b1s.build_interim()
                self.assertEqual(sorted(p.name for p in interim.glob("*.parquet")),
                                 ["2026-11.parquet", "2026-12.parquet"])
                self.assertEqual(len(df), 3)
                self.assertTrue((interim / "2026-11.parquet").exists())
                again = b1s.build_interim()                     # idempotent, reads the parquet back
                self.assertEqual(len(again), 3)
            finally:
                b1s.RAW_DIR, b1s.INTERIM = old

    def test_empty_is_an_empty_frame(self):
        with TemporaryDirectory() as d:
            old = b1s.RAW_DIR
            b1s.RAW_DIR = Path(d) / "nothing"
            try:
                self.assertTrue(b1s.build_interim().empty)
            finally:
                b1s.RAW_DIR = old


class DownloadTest(unittest.TestCase):
    def test_missing_days_are_requested_and_existing_skipped(self):
        with TemporaryDirectory() as d:
            raw = Path(d)
            (raw / "BTCUSDT-1s-2026-11-02.zip").write_bytes(b"x")
            asked = []

            class FakeHttp:
                def get(self, url, params=None, raw_=False, **kw):
                    asked.append(url)
                    return zip_bytes([row(MS)])

            old = b1s.RAW_DIR
            b1s.RAW_DIR = raw
            try:
                n = b1s.download_days("2026-11-01", "2026-11-04", FakeHttp())
                self.assertEqual(n, 2)                           # 11-01 and 11-03
                self.assertEqual(len(asked), 2)
                self.assertTrue((raw / "BTCUSDT-1s-2026-11-03.zip").exists())
            finally:
                b1s.RAW_DIR = old


if __name__ == "__main__":
    unittest.main()
