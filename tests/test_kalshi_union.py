"""Kalshi: the pipeline reads every ladder dataset on disk and keeps the finest rows."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import pandas as pd

from src.download import kalshi


def frame(event, minutes, start_min=0):
    rows = []
    for i in range(start_min, minutes):
        rows.append({"ticker": f"{event}-T110000", "end_ts": 1767225600 + 60 * i, "yes_bid": 40 + i,
                     "yes_ask": 42 + i, "last": 41 + i, "volume": 1, "open_interest": 100,
                     "strike": 110000.0, "cap_strike": None, "strike_type": "greater", "result": "yes",
                     "expiration_value": 110000.0, "open_time": "2026-01-01T00:00:00Z",
                     "close_time": "2026-01-01T01:00:00Z", "volume_total": 5, "event": event})
    return pd.DataFrame(rows)


class UnionTest(unittest.TestCase):
    def test_whole_hour_rows_win_over_the_first_ten_minutes(self):
        with TemporaryDirectory() as d:
            coarse, fine = Path(d) / "kalshi", Path(d) / "kalshi_kxbtcd_60m"
            coarse.mkdir(), fine.mkdir()
            # day A: only the 10-minute download exists
            frame("KXBTCD-26JAN0101", 10).to_csv(coarse / "2026-01-01.csv.gz", index=False)
            # day B: the same first 10 minutes twice (coarse + fine) and 50 more minutes in the fine copy
            frame("KXBTCD-26JAN0201", 10).to_csv(coarse / "2026-01-02.csv.gz", index=False)
            frame("KXBTCD-26JAN0201", 60).to_csv(fine / "2026-01-02.csv.gz", index=False)

            old = kalshi.ladder_dirs
            kalshi.ladder_dirs = lambda: [coarse, fine]
            try:
                df = kalshi.build_interim()
            finally:
                kalshi.ladder_dirs = old

            self.assertEqual(len(df), 10 + 60)                     # no duplicated first ten minutes
            day_b = df[df["event"] == "KXBTCD-26JAN0201"]
            self.assertEqual(len(day_b), 60)
            self.assertEqual(int(day_b["yes_bid"].max()), 40 + 59)  # the fine copy, not the coarse one
            self.assertIn("candle_end", df.columns)
            self.assertNotIn("end_ts", df.columns)


if __name__ == "__main__":
    unittest.main()
