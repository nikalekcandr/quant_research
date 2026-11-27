"""Kalshi: `--workers N` fetches several days at once, and must produce byte-identical files.

The venue is replaced by a local HTTP server that answers the two endpoints the live path needs
(`/markets` and `/markets/candlesticks`) with a fixed latency, so the measurement is about our own
scheduling, not about the network.
"""
import json
import threading
import time
import unittest
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.parse import parse_qs, urlparse

import pandas as pd

from src.download import kalshi
from src.utils import Http

LATENCY = 0.25          # seconds per request: roughly one Kalshi round-trip
# 7 strikes: below 6 per event the pipeline treats a day as "sparse" and re-downloads it
STRIKES = [109700, 109800, 109900, 110000, 110100, 110200, 110300]


def _markets(event: str):
    day = datetime(2026, 10, 2, 10, tzinfo=timezone.utc)
    return {"markets": [{
        "ticker": f"{event}-T{s}", "floor_strike": s, "cap_strike": s, "strike_type": "greater",
        "result": "yes" if s <= 110000 else "no", "expiration_value": 110000,
        "open_time": "2026-10-02T09:00:00Z", "close_time": "2026-10-02T10:00:00Z", "volume": 12,
    } for s in STRIKES]}


def _candles(ticker: str):
    return [{"end_period_ts": 1759395600 + 60 * i,
             "yes_bid": {"close": 40 + i}, "yes_ask": {"close": 42 + i}, "price": {"close": 41 + i},
             "volume": 3 + i, "open_interest": 100 + i} for i in range(10)]


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        time.sleep(LATENCY)
        u = urlparse(self.path)
        q = parse_qs(u.query)
        if u.path.endswith("/markets"):
            body = _markets(q["event_ticker"][0])
        elif u.path.endswith("/markets/candlesticks"):
            body = {"markets": [{"market_ticker": tk, "candlesticks": _candles(tk)}
                                for tk in q["market_tickers"][0].split(",")]}
        else:
            body = {}
        raw = json.dumps(body).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


class WorkersTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        cls.thread = threading.Thread(target=cls.srv.serve_forever, daemon=True)
        cls.thread.start()
        cls.base = f"http://127.0.0.1:{cls.srv.server_address[1]}/trade-api/v2"

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()
        cls.srv.server_close()

    def run_days(self, workers, days, out_dir):
        old_base, old_raw = kalshi.BASE, kalshi.RAW_DIR
        kalshi.BASE, kalshi.RAW_DIR = self.base, Path(out_dir)
        try:
            t0 = time.time()
            kalshi.download_history("2026-10-02", days[-1], hour_step=6, http=Http(min_interval=0.0),
                                    workers=workers)
            return time.time() - t0, sorted(p.name for p in Path(out_dir).glob("*.csv.gz"))
        finally:
            kalshi.BASE, kalshi.RAW_DIR = old_base, old_raw

    def test_parallel_is_faster_and_identical(self):
        days = ["2026-10-02", "2026-10-03", "2026-10-04", "2026-10-05"]
        with TemporaryDirectory() as a, TemporaryDirectory() as b:
            seq, files_seq = self.run_days(1, days, a)
            par, files_par = self.run_days(4, days, b)
            self.assertEqual(files_seq, files_par)          # same days written
            self.assertEqual(len(files_seq), 4)
            for name in files_seq:                          # and the same rows, row by row
                left = pd.read_csv(Path(a) / name).sort_values(["ticker"]).reset_index(drop=True)
                right = pd.read_csv(Path(b) / name).sort_values(["ticker"]).reset_index(drop=True)
                pd.testing.assert_frame_equal(left, right, check_like=True)
            self.assertGreater(seq / max(par, 1e-6), 1.8, f"sequential {seq:.1f}s vs parallel {par:.1f}s")

    def test_progress_line_reports_days_requests_and_eta(self):
        import contextlib
        import io
        out = io.StringIO()
        with TemporaryDirectory() as d:
            old_base, old_raw = kalshi.BASE, kalshi.RAW_DIR
            kalshi.BASE, kalshi.RAW_DIR = self.base, Path(d)
            try:
                with contextlib.redirect_stdout(out):
                    kalshi.download_history("2026-10-02", "2026-10-03", hour_step=6,
                                            http=Http(min_interval=0.0), workers=2, report_every=0.2)
            finally:
                kalshi.BASE, kalshi.RAW_DIR = old_base, old_raw
        text = out.getvalue()
        self.assertRegex(text, r"kalshi: \d+/2 days done")
        self.assertIn("requests", text)
        self.assertIn("ETA", text)
        self.assertRegex(text, r"kalshi: finished 2/2 days in .* \d+ requests")

    def test_existing_days_are_skipped(self):
        with TemporaryDirectory() as d:
            first, _ = self.run_days(2, ["2026-10-02", "2026-10-03"], d)
            second, files = self.run_days(2, ["2026-10-02", "2026-10-03"], d)
            self.assertLess(second, first / 2)              # nothing to fetch the second time
            self.assertEqual(len(files), 2)


class PlumbingTest(unittest.TestCase):
    def test_update_passes_the_flags_through(self):
        from src.download import update
        self.assertEqual(update.main(["--audit", "--workers", "3", "--min-interval", "0.05"]), 0)
        self.assertEqual(update.WORKERS, 3)
        self.assertEqual(update.MIN_INTERVAL, 0.05)
        update.WORKERS, update.MIN_INTERVAL = 4, 0.15     # leave the defaults behind


if __name__ == "__main__":
    unittest.main()
