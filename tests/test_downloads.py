import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.download.gaps import largest_hole, ticker_date, _last_file_date
from src.download.macro import merge, parse_bls, parse_fed
from src.download.perp import parse_klines, parse_premium, parse_zip


class PerpParseTest(unittest.TestCase):
    def test_klines_and_premium_share_the_time_convention(self):
        row = [1793145600000, "90000.0", "90100.0", "89900.0", "90050.0", "12.5", 0, 0, 10, 0, 0, 0]
        k = parse_klines([row])
        self.assertEqual(k.close.iloc[0], 90050.0)
        self.assertEqual(k.open_time.iloc[0], pd.Timestamp("2026-10-28", tz="UTC"))
        p = parse_premium([[1793145600000, "90000", "90100", "89900", "0.00042", "0", 0, 0, 0, 0, 0, 0]])
        self.assertAlmostEqual(p.premium.iloc[0], 0.00042)
        self.assertEqual(p.open_time.iloc[0], k.open_time.iloc[0])

    def test_microsecond_archives_are_detected(self):
        row = [1793145600000000, "1", "1", "1", "90000", "1", 0, 0, 0, 0, 0, 0]
        self.assertEqual(parse_klines([row]).open_time.iloc[0], pd.Timestamp("2026-10-28", tz="UTC"))

    def test_empty_payload(self):
        self.assertTrue(parse_klines([]).empty)
        self.assertTrue(parse_premium([]).empty)

    def test_zip_with_and_without_header(self):
        import io
        import zipfile
        body = "1793145600000,1,1,1,90000,1,0,0,0,0,0,0\n"
        for text in (body, "open_time,open,high,low,close,volume,close_time,quote_volume,trades,tb,tq,ignore\n" + body):
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w") as z:
                z.writestr("x.csv", text)
            one = parse_zip(buf.getvalue())
            self.assertEqual(len(one), 1)
            self.assertEqual(one.close.iloc[0], 90000.0)


class MacroParseTest(unittest.TestCase):
    BLS = """<html><body><table>
      <tr><td><a href="/cpi">Consumer Price Index</a></td><td>November 12, 2026</td></tr>
      <tr><td><a href="/empsit">Employment Situation</a></td><td>December 4, 2026</td></tr>
      <tr><td>Something else</td><td>November 1, 2026</td></tr>
    </table></body></html>"""

    FED = """<html><body><h3>2026 FOMC Meetings</h3>
      <div>January 27-28</div><div>March 17-18</div><div>December 8-9</div>
      <h3>2027 FOMC Meetings</h3><div>January 26-27</div></body></html>"""

    def test_bls_events_and_times(self):
        d = parse_bls(self.BLS)
        self.assertEqual(set(d.event), {"CPI", "NFP"})
        self.assertTrue((d.time_et == "08:30").all())
        self.assertIn(pd.Timestamp("2026-11-12").date(), set(d.date))

    def test_fed_decision_is_the_second_day(self):
        d = parse_fed(self.FED)
        dates = set(d.date)
        self.assertIn(pd.Timestamp("2026-01-28").date(), dates)     # 27-28 -> 28
        self.assertIn(pd.Timestamp("2026-12-09").date(), dates)
        self.assertIn(pd.Timestamp("2027-01-27").date(), dates)     # next year block is read too
        self.assertTrue((d.time_et == "14:00").all())

    def test_merge_is_append_only_and_deduplicated(self):
        old = pd.DataFrame([{"event": "CPI", "date": "2026-10-13", "time_et": "08:30"}])
        new = pd.DataFrame([{"event": "CPI", "date": "2026-10-13", "time_et": "08:30"},
                            {"event": "FOMC", "date": "2026-10-28", "time_et": "14:00"}])
        out = merge(new, old, now=pd.Timestamp("2026-11-26"))
        self.assertEqual(len(out), 2)
        self.assertEqual(list(out.columns), ["event", "date", "time_et"])
        self.assertEqual(out.date.iloc[0], pd.Timestamp("2026-10-13").date())

    def test_absurd_future_date_is_refused(self):
        old = pd.DataFrame([{"event": "CPI", "date": "2026-10-13", "time_et": "08:30"}])
        new = pd.DataFrame([{"event": "CPI", "date": "2031-01-01", "time_et": "08:30"}])
        with self.assertRaises(ValueError):
            merge(new, old, now=pd.Timestamp("2026-11-26"))


class GapAuditTest(unittest.TestCase):
    def test_ticker_date(self):
        self.assertEqual(ticker_date("KXBTCD-26NOV1819-T63099.99.csv.gz"),
                         pd.Timestamp("2026-11-18", tz="UTC"))

    def test_three_file_name_conventions(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            (d / "BTCUSDT-1m-2026-10-27.zip").write_bytes(b"")
            (d / "2026-11.csv.gz").write_bytes(b"")
            (d / "2026110812.csv.gz").write_bytes(b"")
            self.assertEqual(_last_file_date("*.zip", d), pd.Timestamp("2026-10-27", tz="UTC"))
            self.assertEqual(_last_file_date("*.csv.gz", d), pd.Timestamp("2026-11-08", tz="UTC"))

    def test_largest_hole(self):
        self.assertEqual(largest_hole(["2026-10-01", "2026-10-08", "2026-11-01"]), 24)
        self.assertIsNone(largest_hole(["2026-10-01"]))


if __name__ == "__main__":
    unittest.main()
