"""Offline tests of the v6 additions.   python tests/test_v6.py   (one 'ok' line per test)"""
import json
import pathlib
import sys
import tempfile

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import config  # noqa: E402
import alpha_live as A  # noqa: E402
from src.download import kalshi, venues  # noqa: E402


def test_kalshi_fee_rounding():
    assert A.kalshi_fee(0.5) == 0.02          # 0.07 * 0.25 = 1.75c -> 2c
    assert A.kalshi_fee(0.05) == 0.01         # 0.33c -> 1c
    assert A.kalshi_fee(0.01) == 0.01


def test_box_arbitrage():
    above = pd.DataFrame({"strike": [79999.99, 80499.99], "bid": [0.60, 0.30], "ask": [0.61, 0.31]})
    fair = pd.DataFrame({"floor": [80000, 80250], "cap": [80249.99, 80499.99], "bid": [0.14, 0.14], "ask": [0.15, 0.16]})
    assert A.box_arbitrage(above, fair) == []                 # consistent prices: no arbitrage after fees
    cheap = fair.assign(bid=[0.10, 0.10], ask=[0.11, 0.11])
    r = A.box_arbitrage(above, cheap)
    assert len(r) == 1 and r[0]["how"] == "short box" and abs(r[0]["edge"] - 0.01) < 1e-9


def test_smile_flat_equals_black():
    snap = pd.Timestamp("2026-10-26 12:00", tz="UTC")
    rows = []
    for exp in ("27OCT26", "28OCT26"):
        for k in range(70000, 92000, 1000):
            for cp in "CP":
                rows.append({"instrument": f"BTC-{exp}-{k}-{cp}", "mark_iv": 40.0, "underlying": 80000.0})
    sm = A.Smile(pd.DataFrame(rows), snap)
    T = pd.Timestamp("2026-10-27 16:00", tz="UTC")
    K = np.array([78000.0, 80000.0, 83000.0])
    tau = (T - snap).total_seconds() / (365 * 86400)
    w = 0.4 ** 2 * tau
    expect = stats.norm.cdf((np.log(80000 / K) - w / 2) / np.sqrt(w))
    got = sm.digital(T, K)
    assert np.max(np.abs(got - expect)) < 2e-3, (got, expect)


def test_smile_skew_raises_downside_digital():
    snap = pd.Timestamp("2026-10-26 12:00", tz="UTC")
    rows = []
    for exp in ("27OCT26", "28OCT26"):
        for k in range(70000, 92000, 1000):
            iv = 40 + 0.004 * (80000 - k) / 10      # higher vol for low strikes (put skew)
            for cp in "CP":
                rows.append({"instrument": f"BTC-{exp}-{k}-{cp}", "mark_iv": iv, "underlying": 80000.0})
    flat = [dict(r, mark_iv=40.0) for r in rows]
    T = pd.Timestamp("2026-10-27 16:00", tz="UTC")
    d_skew = A.Smile(pd.DataFrame(rows), snap).digital(T, np.array([80000.0]))
    d_flat = A.Smile(pd.DataFrame(flat), snap).digital(T, np.array([80000.0]))
    assert d_skew[0] > d_flat[0]               # negative skew: P(S > K) at the money is higher than lognormal


def test_interp_ladder():
    K = np.array([78000, 80000, 82000])
    p = stats.norm.sf(np.log(K / 80000) / 0.02)
    mid = A.interp_ladder(K, p, np.array([81000.0]))
    assert abs(mid[0] - stats.norm.sf(np.log(81000 / 80000) / 0.02)) < 1e-6
    assert np.isnan(A.interp_ladder(K, p, np.array([90000.0]))[0])


def test_classify_polymarket():
    assert venues.classify_pm("Bitcoin Up or Down - October 28, 11AM ET", "") == "updown"
    assert venues.classify_pm("Bitcoin Up or Down - October 28, 11:15AM-11:30AM ET", "") == "updown_15m"
    assert venues.classify_pm("What price will Bitcoin hit in October?", "Will Bitcoin reach $130,000?") == "barrier"
    assert venues.classify_pm("Bitcoin price on October 29?", "between $80,000 and $82,000") == "range"
    assert venues.parse_levels("between $80,000 and $82,000") == [80000.0, 82000.0]


def test_kalshi_strike_filter():
    wide = pd.DataFrame({"strike": [94749.99, 94999.99, 95249.99]})
    narrow = pd.DataFrame({"strike": [72999.99, 73099.99, 73199.99]})
    assert kalshi._step_ok(wide, False).all()
    assert kalshi._step_ok(narrow, False).tolist() == [True, False, True]


def test_ledger_chain():
    from src import live_forecast as lf
    old = config.FORECASTS_LIVE
    with tempfile.TemporaryDirectory() as d:
        config.FORECASTS_LIVE = pathlib.Path(d)
        for i in range(3):
            p = config.FORECASTS_LIVE / "kx1h" / f"2026102{i}T1005.json"
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps({"i": i}))
            lf._ledger_append(p, "kx1h", pd.Timestamp("2026-10-28 10:05", tz="UTC"), pd.Timestamp("2026-10-28 11:00", tz="UTC"), None)
        assert lf.verify_ledger()[0]
        (config.FORECASTS_LIVE / "kx1h" / "20261021T1005.json").write_text("{}")
        assert not lf.verify_ledger()[0]
    config.FORECASTS_LIVE = old


def test_binance_fallback():
    from src.download import binance

    class H:
        def get(self, url, params=None, raw=False):
            if "api.binance.com" in url:
                raise RuntimeError("451")
            return [[1]]
    binance._GOOD_HOST[:] = []
    assert binance.api_get(H(), "/api/v3/klines", {}) == [[1]]
    assert binance._GOOD_HOST[0].endswith("binance.vision")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
