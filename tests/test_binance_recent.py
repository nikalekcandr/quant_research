"""`binance: ok за 0.0 мin` must never hide a month of missing minutes again."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.download import binance  # noqa: E402


def kline(ts_ms, close=100.0):
    return [ts_ms, str(close), str(close), str(close), str(close), "1.0", ts_ms + 59_999, "1.0", 3, "1", "1", "0"]


class RecentTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.old = binance.RAW_DIR
        binance.RAW_DIR = Path(self.tmp.name)
        self.now = pd.Timestamp("2026-11-27T12:00:00Z")

    def tearDown(self):
        binance.RAW_DIR = self.old
        self.tmp.cleanup()

    def _write_recent(self, times):
        df = pd.DataFrame({"open_time": [int(t.timestamp() * 1000) for t in times],
                           "close": 1.0, "volume": 1.0})
        df.to_csv(binance.RAW_DIR / "recent.csv.gz", index=False)

    def _http(self, rows):
        class Http:
            def __init__(self):
                self.calls = []

            def get(self, url, params=None, raw=False):
                self.calls.append((url, dict(params or {})))
                return rows

        return Http()

    def test_new_minutes_are_appended_not_replaced(self):
        old = [self.now - pd.Timedelta(minutes=m) for m in (30, 29, 28)]
        self._write_recent(old)
        fresh = [kline(int(self.now.timestamp() * 1000) - 60_000 * m) for m in (2, 1)]
        out = binance._recent(self._http(fresh))
        got = pd.read_csv(binance.RAW_DIR / "recent.csv.gz")
        self.assertEqual(len(got), 5)                     # 3 old + 2 new, nothing lost
        self.assertEqual(out["rows"], 2)

    def test_a_hole_of_weeks_is_filled_from_the_last_known_minute(self):
        start = self.now - pd.Timedelta(days=20)
        self._write_recent([start])
        chunk = [kline(int((start + pd.Timedelta(minutes=i)).timestamp() * 1000)) for i in range(1, 1001)]

        class Http:
            def __init__(self):
                self.calls = []

            def get(self, url, params=None, raw=False):
                self.calls.append(params)
                return chunk if len(self.calls) == 1 else chunk[:5]

        http = Http()
        binance._recent(http)
        self.assertEqual(http.calls[0]["startTime"], int((start + pd.Timedelta(minutes=1)).timestamp() * 1000))
        self.assertEqual(len(http.calls), 2)              # walked in 1000-minute chunks, not one shot

    def test_rest_failure_keeps_the_file(self):
        self._write_recent([self.now - pd.Timedelta(minutes=5)])
        before = (binance.RAW_DIR / "recent.csv.gz").read_bytes()
        out = binance._recent(self._http([]))
        self.assertEqual(out["rows"], 0)
        self.assertEqual((binance.RAW_DIR / "recent.csv.gz").read_bytes(), before)

    def test_a_dead_rest_fails_the_step_instead_of_pretending(self):
        class Http:
            stats = {"requests": 0, "429": 0}

            def get(self, url, params=None, raw=False):
                return None                                   # every archive missing, REST empty

        with self.assertRaises(RuntimeError):
            binance.download("2026-11-01", http=Http())


if __name__ == "__main__":
    unittest.main()
