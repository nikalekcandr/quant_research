"""Parsers against abbreviated copies of real API responses (captured October 2026).

    python -m pytest tests -q      or      python tests/test_parsers.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.download import binance, deribit, kalshi, polymarket  # noqa: E402
from src.measure import deribit_iv, ladder  # noqa: E402

PM_EVENT = {
    "id": "55023", "slug": "bitcoin-above-on-october-13-2026", "title": "Bitcoin above ___ on October 13?",
    "startDate": "2026-10-06T16:00:14Z", "endDate": "2026-10-13T16:00:00Z", "closed": False,
    "markets": [
        {"question": f"Will the price of Bitcoin be above ${k:,} on October 13?", "groupItemTitle": f"{k:,}",
         "slug": f"bitcoin-above-{k // 1000}k-on-october-13-2026", "conditionId": "0x1",
         "clobTokenIds": json.dumps([f"{i}551853009754", f"{i}999"]), "startDate": "2026-10-06T16:00:15Z",
         "outcomePrices": json.dumps([str(p), str(round(1 - p, 4))]), "volumeNum": 1000.0, "closed": False,
         "bestBid": p - 0.005, "bestAsk": p + 0.005}
        for i, (k, p) in enumerate(zip(range(76000, 98000, 2000),
                                       [0.998, 0.9915, 0.973, 0.775, 0.195, 0.023, 0.0035, 0.0015, 0.0005, 0.0005, 0.0005]))
    ],
}
KALSHI_MARKET = {"close_time": "2026-10-12T14:00:00Z", "event_ticker": "KXBTCD-26OCT1210", "expiration_value": "82880.82",
                 "floor_strike": 82999.99, "open_time": "2026-10-12T13:00:00Z", "result": "no", "strike_type": "greater",
                 "ticker": "KXBTCD-26OCT1210-T82999.99", "volume_fp": "53110.00", "yes_ask_dollars": "0.2100",
                 "yes_bid_dollars": "0.2000", "yes_ask_size_fp": "944.85", "yes_bid_size_fp": "150.00"}
KALSHI_CANDLE_LIVE = {"end_period_ts": 1791810060, "open_interest_fp": "10388.45",
                      "price": {"close_dollars": "0.2000"}, "volume_fp": "12980.45",
                      "yes_ask": {"close_dollars": "0.2100"}, "yes_bid": {"close_dollars": "0.2000"}}
KALSHI_CANDLE_HIST = {"end_period_ts": 1786572060, "open_interest": "2344.75", "price": {"close": "0.8000"},
                      "volume": "2607.72", "yes_ask": {"close": "0.8300"}, "yes_bid": {"close": "0.8000"}}
DERIBIT_TRADES = [
    {"timestamp": 1791792006991, "price": 0.049, "mark_price": 0.04945833, "iv": 41.58,
     "instrument_name": "BTC-16OCT26-80000-C", "index_price": 83855.05, "direction": "sell", "amount": 2.0},
    {"timestamp": 1791792031494, "price": 0.0925, "iv": 38.26, "instrument_name": "BTC-25JUN27-95000-C",
     "index_price": 83867.37, "direction": "buy", "amount": 2.0},
]


def test_polymarket_event():
    lad = polymarket.parse_event(PM_EVENT)
    assert len(lad) == 11
    assert lad["strike"].tolist()[:3] == [76000.0, 78000.0, 80000.0]
    assert lad["settle_ts"].iloc[0] == pd.Timestamp("2026-10-13 16:00", tz="UTC")
    assert lad["yes_token"].iloc[0] == "0551853009754"


def test_polymarket_slugs():
    from datetime import date
    assert polymarket.slug_candidates(date(2026, 10, 13)) == ["bitcoin-above-on-october-13-2026", "bitcoin-above-on-october-13"]


def test_kalshi_fields():
    assert kalshi.event_ticker(pd.Timestamp("2026-10-12 10:00").to_pydatetime()) == "KXBTCD-26OCT1210"
    assert kalshi._num(KALSHI_MARKET, "yes_bid") == 0.20
    assert kalshi._num(KALSHI_MARKET, "volume") == 53110.0
    assert kalshi._num(KALSHI_CANDLE_LIVE["yes_ask"], "close") == 0.21
    assert kalshi._num(KALSHI_CANDLE_HIST["yes_ask"], "close") == 0.83
    assert kalshi._num(KALSHI_CANDLE_HIST, "volume") == 2607.72


def test_deribit_instrument():
    e, k, cp = deribit.parse_instrument("BTC-16OCT26-80000-C")
    assert e == pd.Timestamp("2026-10-16 08:00", tz="UTC") and k == 80000 and cp == "C"


def test_binance_timestamps():
    us = pd.Series([1791814920000000])   # microseconds (spot archives since 2025)
    msec = pd.Series([1672531200000])     # milliseconds (2023)
    assert binance._to_ts(us).iloc[0] == pd.Timestamp("2026-10-12 14:22", tz="UTC")
    assert binance._to_ts(msec).iloc[0] == pd.Timestamp("2023-01-01", tz="UTC")


def test_ladder_fit_on_real_prices():
    lad = polymarket.parse_event(PM_EVENT)
    p = np.array([0.998, 0.9915, 0.973, 0.775, 0.195, 0.023, 0.0035, 0.0015, 0.0005, 0.0005, 0.0005])
    fit = ladder.fit_snapshot(pd.DataFrame({"strike": lad["strike"], "p": p}))
    vol = fit["mkt_s"] / np.sqrt(1 / 365) * 100     # roughly one day before settlement
    assert 10 < vol < 120, vol
    assert 82000 < np.exp(fit["mkt_m"]) < 84000


def test_deribit_atm():
    k = np.log(np.array([80000, 82000, 84000, 86000]) / 83000)
    iv = np.array([42.0, 39.0, 38.0, 39.5])
    a = deribit_iv.atm_iv(k, iv, np.ones(4))
    assert 37 < a < 40


def test_bjn_variance_recovers_lognormal():
    from scipy import stats
    F, v = 83000.0, 3e-4
    K = np.arange(70000, 98000, 2000.0)
    d = stats.norm.cdf((np.log(F / K) - v / 2) / np.sqrt(v))
    r = ladder.mfiv(K, d)
    assert abs(r["mfiv"] / v - 1) < 0.01, r
    assert r["mfiv_trunc"] <= r["mfiv"]


def test_bjn_on_real_ladder_with_two_central_strikes():
    # Polymarket, Oct 11 2026, 24h before settlement: only 82k and 84k are between 2 and 98 cents
    K = np.arange(76000, 98000, 2000.0)
    p = np.array([0.9995, 0.9985, 0.9925, 0.935, 0.095, 0.0085, 0.0005, 0.0005, 0.0005, 0.0005, 0.0005])
    r = ladder.mfiv(K, p)
    vol = np.sqrt(r["mfiv"] * 365) * 100
    assert 15 < vol < 35, vol


if __name__ == "__main__":
    for name, f in list(globals().items()):
        if name.startswith("test_"):
            f()
            print("ok", name)
