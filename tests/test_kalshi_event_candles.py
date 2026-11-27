"""One request per event instead of one per strike: the difference between ~4 h and ~30 h for the archive."""
from __future__ import annotations

import sys
import tempfile
import threading
import unittest
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.download import kalshi  # noqa: E402

T0, T1 = 1_790_000_000, 1_790_003_600


def candle(ts, close=0.5):
    return {"end_period_ts": ts, "yes_bid": {"close_dollars": "0.48"}, "yes_ask": {"close_dollars": "0.52"},
            "price": {"close_dollars": "0.50"}, "volume_fp": "12.00", "open_interest_fp": "400.00"}


class StubHttp:
    """Answers like Kalshi and counts the round-trips."""

    def __init__(self, event_pages=None, per_ticker=None, markets=None):
        self.event_pages = event_pages if event_pages is not None else []
        self.per_ticker = per_ticker
        self.market_rows = markets
        self.calls = []

    def get(self, url, params=None, raw=False):
        self.calls.append((url, dict(params or {})))
        if "/series/" in url and "/events/" in url:
            page = int((params or {}).get("start_ts", T0))
            for start, payload in self.event_pages:
                if start == page:
                    return payload
            return self.event_pages[0][1] if len(self.event_pages) == 1 else {"market_tickers": [],
                                                                             "market_candlesticks": []}
        if "/historical/markets/" in url and url.endswith("/candlesticks"):
            return {"candlesticks": self.per_ticker or []}
        if url.endswith("/markets"):
            return {"markets": self.market_rows or []}
        if url.endswith("/historical/markets"):
            return {"markets": []}
        raise AssertionError(f"unexpected url {url}")


class CandlesEventTest(unittest.TestCase):
    def test_every_strike_of_an_event_in_one_response(self):
        http = StubHttp(event_pages=[(T0, {
            "market_tickers": ["A-1", "A-2"],
            "market_candlesticks": [[candle(T0 + 60), candle(T0 + 120)], [candle(T0 + 60)]],
            "adjusted_end_ts": None})])
        rows = kalshi.candles_event(http, "KXBTCD", "KXBTCD-26NOV2712", T0, T1)
        self.assertEqual(len(http.calls), 1)
        self.assertEqual({r["ticker"] for r in rows}, {"A-1", "A-2"})
        self.assertEqual(sorted(r["end_ts"] for r in rows), [T0 + 60, T0 + 60, T0 + 120])
        self.assertAlmostEqual(rows[0]["yes_bid"], 0.48)
        self.assertAlmostEqual(rows[0]["open_interest"], 400.0)

    def test_candles_outside_the_window_are_dropped(self):
        http = StubHttp(event_pages=[(T0, {"market_tickers": ["A-1"],
                                           "market_candlesticks": [[candle(T0 - 3600), candle(T0 + 60),
                                                                    candle(T1 + 60)]]})])
        rows = kalshi.candles_event(http, "KXBTCD", "E", T0, T1)
        self.assertEqual([r["end_ts"] for r in rows], [T0 + 60])

    def test_pagination_follows_adjusted_end_ts(self):
        mid = T0 + 1800
        pages = [(T0, {"market_tickers": ["A-1"], "market_candlesticks": [[candle(T0 + 60)]],
                       "adjusted_end_ts": mid}),
                 (mid, {"market_tickers": ["A-1"], "market_candlesticks": [[candle(mid + 60)]],
                        "adjusted_end_ts": None})]
        http = StubHttp(event_pages=pages)
        rows = kalshi.candles_event(http, "KXBTCD", "E", T0, T1)
        self.assertEqual([r["end_ts"] for r in rows], [T0 + 60, mid + 60])
        self.assertEqual([c[1]["start_ts"] for c in http.calls], [T0, mid])

    def test_event_level_keeps_only_the_selected_strikes(self):
        http = StubHttp(event_pages=[(T0, {"market_tickers": ["A-1", "A-2", "FAR"],
                                           "market_candlesticks": [[candle(T0 + 60)], [candle(T0 + 60)],
                                                                   [candle(T0 + 60)]]})])
        rows = kalshi._event_candles(http, "KXBTCD", "E", T0, T1, ["A-1", "A-2"], live=False)
        self.assertEqual({r["ticker"] for r in rows}, {"A-1", "A-2"})

    def test_falls_back_to_one_request_per_strike(self):
        http = StubHttp(event_pages=[(T0, {"market_tickers": [], "market_candlesticks": []})],
                        per_ticker=[candle(T0 + 60)])
        rows = kalshi._event_candles(http, "KXBTCD", "E", T0, T1, ["A-1", "A-2"], live=False)
        self.assertEqual(len(rows), 2)                     # one candle per ticker, from the fallback
        self.assertTrue(any("/historical/markets/A-1/candlesticks" in c[0] for c in http.calls))


class OneDayCostTest(unittest.TestCase):
    """A day must cost ~2 requests per event, not one per strike."""

    T_OPEN = int(pd.Timestamp("2026-01-19T11:00:00Z").timestamp())

    def _strikes(self, ev):
        return [{"ticker": f"{ev}-T{k}", "floor_strike": 100_000 + 250 * k, "cap_strike": None,
                 "strike_type": "greater", "result": "no", "expiration_value": "101000",
                 "open_time": "2026-01-19T11:00:00Z", "close_time": "2026-01-19T12:00:00Z", "volume": 40}
                for k in range(9)]

    def test_two_requests_per_event(self):
        t_open = self.T_OPEN
        strikes_of = self._strikes

        class Http:
            def __init__(self):
                self.calls = []

            def get(self, url, params=None, raw=False):
                self.calls.append((url, dict(params or {})))
                if url.endswith("/markets"):
                    return {"markets": strikes_of(params["event_ticker"])}
                if "/events/" in url:
                    ev = url.split("/events/")[1].split("/")[0]
                    mk = strikes_of(ev)
                    return {"market_tickers": [m["ticker"] for m in mk],
                            "market_candlesticks": [[candle(t_open + 60 * i) for i in range(1, 4)] for _ in mk],
                            "adjusted_end_ts": None}
                return {"markets": []}

        http = Http()
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            kalshi._one_day_impl(http, pd.Timestamp("2026-01-19").date(), minutes=60, series="KXBTCD",
                                 hour_step=12, out_dir=out, center={"v": None}, lock=threading.Lock())
            files = list(out.glob("*.csv.gz"))
            self.assertEqual(len(files), 1)
            rows = pd.read_csv(files[0])
        self.assertGreater(len(rows), 0)
        self.assertEqual(len(http.calls), 4)               # 2 events x (markets + event candles)
        self.assertFalse(any("/historical/markets/" in u for u, _ in http.calls))


if __name__ == "__main__":
    unittest.main()
