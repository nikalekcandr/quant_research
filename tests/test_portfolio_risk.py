"""Tests for src/portfolio.py and src/risk.py — deterministic, no network.

    pytest tests/test_portfolio_risk.py -q
    python tests/test_portfolio_risk.py
"""
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import portfolio as pf_mod
from src import risk as risk_mod


# ------------------------------------------------------------------ portfolio
def test_kelly_shrinks_when_n_small():
    # High mean/var but n=5 -> strong shrink (n/(n+20)=0.2)
    assert pf_mod._kelly_fraction(0.05, 0.01, n=5) < pf_mod._kelly_fraction(0.05, 0.01, n=500)
    # Cap
    assert abs(pf_mod._kelly_fraction(10.0, 0.01, n=500)) <= pf_mod.KELLY_CAP + 1e-9


def test_kelly_zero_var():
    assert pf_mod._kelly_fraction(0.05, 0.0, n=100) == 0.0
    assert pf_mod._kelly_fraction(0.05, float("nan"), n=100) == 0.0


def test_portfolio_scales_correlated_binaries():
    from dataclasses import dataclass

    @dataclass
    class FakeOrder:
        hyp: str
        kind: str
        instrument: str
        settle_ts: str = "2026-10-28"

        @property
        def info(self):
            return {"event": "same_event"}

    orders = [FakeOrder("P01", "binary", f"m{i}") for i in range(4)]
    # Four binaries on same event: corr scale = 1/sqrt(4)=0.5
    pf = pf_mod.Portfolio()
    scales = pf.corr_scale_for(orders)
    assert all(abs(v - 0.5) < 1e-9 for v in scales.values()), scales

    # Perps bucketed together, binary separate
    orders2 = [FakeOrder("D01", "perp", "BTC"), FakeOrder("D01", "perp", "BTC"), FakeOrder("P01", "binary", "m2")]
    scales2 = pf.corr_scale_for(orders2)
    assert abs(scales2[0] - 1 / math.sqrt(2)) < 1e-9
    assert scales2[2] == 1.0


def test_dd_brake():
    # No DD -> full size
    pf = pf_mod.Portfolio()
    pf.cum = pd.Series([0.0, 0.01, 0.02, 0.015])
    pf.daily = pd.Series([0.01, 0.01, -0.005, 0.005])
    # Small DD (<5%) -> no brake
    assert pf.dd_scale() == 1.0

    # 15% DD -> 0.25 (linear between 10%->0.5 and 20%->0.0)
    pf.cum = pd.Series([0.0, 0.10, 0.05, -0.05])  # peak 0.10, now -0.05 -> log DD 0.15 -> 1-exp(-0.15)=0.139
    s = pf.dd_scale()
    assert 0.2 < s < 0.5, s

    # 25% DD -> freeze 0.0
    pf.cum = pd.Series([0.0, 0.5, 0.2])
    assert pf.dd_scale() == 0.0


def test_vol_scale_bounds():
    pf = pf_mod.Portfolio()
    # Low vol -> scale up, but capped at 3.0
    pf.daily = pd.Series([0.001] * 30 + [0.0011] * 10)
    assert 0.2 <= pf.vol_scale() <= 3.0
    # No history -> 1.0
    pf2 = pf_mod.Portfolio()
    assert pf2.vol_scale() == 1.0


def test_portfolio_ingest_and_summary():
    led = pd.DataFrame([
        {"event": "CLOSE", "hyp": "P01", "pnl": 0.05, "ts": "2026-10-28T10:00:00Z"},
        {"event": "CLOSE", "hyp": "P01", "pnl": -0.02, "ts": "2026-10-28T11:00:00Z"},
        {"event": "CLOSE", "hyp": "D06", "pnl": -0.01, "ts": "2026-10-28T12:00:00Z"},
        {"event": "OPEN", "hyp": "P01", "pnl": None, "ts": "2026-10-28T13:00:00Z"},
    ])
    pf = pf_mod.Portfolio(ledger=led)
    assert "P01" in pf.stats
    assert pf.stats["P01"]["n"] == 2
    summ = pf.summary()
    assert len(summ) == 2
    assert "kelly" in summ.columns
    stats = pf.portfolio_stats()
    assert "realised_vol_ann" in stats or stats == {}


# ------------------------------------------------------------------ risk
def test_var_es_basic():
    rets = pd.Series(np.random.default_rng(0).normal(0, 0.01, 100))
    ve = risk_mod.var_es(rets)
    assert ve["VaR_95"] < 0
    assert ve["ES_95"] <= ve["VaR_95"]  # ES more extreme (more negative)
    assert ve["n"] == 100
    # Too few -> nan
    ve2 = risk_mod.var_es(pd.Series([0.01] * 5))
    assert math.isnan(ve2["VaR_95"])


def test_drawdown_stats():
    s = pd.Series([0.0, 0.05, 0.10, 0.08, 0.12, 0.09], index=pd.date_range("2026-01-01", periods=6))
    dd = risk_mod.drawdown_stats(s)
    assert dd["max_drawdown"] > 0
    assert dd["max_dd_days"] >= 1
    assert dd["total_return"] == 0.09
    assert "calmar" in dd


def test_pit_histogram_uniform():
    pits = pd.Series(np.random.default_rng(1).uniform(0, 1, 200))
    df = risk_mod.pit_histogram(pits, bins=10)
    assert len(df) == 10
    assert hasattr(df, "attrs")
    # Uniform should have high p-value
    assert df.attrs["p_uniform"] > 0.01

    pits_bad = pd.Series([0.5] * 100)  # all in middle bin -> chi2 huge
    df2 = risk_mod.pit_histogram(pits_bad, bins=10)
    assert df2.attrs["p_uniform"] < 1e-6


def test_christoffersen():
    # Correctly calibrated: ~5% violations, iid -> high p-values
    pits = pd.Series(np.random.default_rng(2).uniform(0, 1, 300))
    res = risk_mod.christoffersen(pits, alpha=0.05)
    assert res["n"] == 300
    assert "p_cc" in res
    # Degenerate: no violations
    pits_ok = pd.Series([0.5] * 100)
    res2 = risk_mod.christoffersen(pits_ok, alpha=0.05)
    assert "note" in res2 or "p_cc" in res2


def test_qlike_decomposition():
    y = pd.Series([0.0001, 0.0002, 0.00015, 0.0003] * 10)
    f = y * 1.1  # biased up 10%
    res = risk_mod.qlike_decomposition(y, f)
    assert res["n"] == 40
    assert res["bias_log"] > 0  # f > y on average
    assert abs(res["corr_log"] - 1.0) < 0.01  # perfectly correlated
    assert res["mz_R2"] > 0.9


def test_position_risk():
    led = pd.DataFrame([
        {"event": "OPEN", "id": "1", "hyp": "D01", "venue": "binance-perp", "side": "long"},
        {"event": "OPEN", "id": "2", "hyp": "P01", "venue": "polymarket", "side": "yes"},
        {"event": "CLOSE", "id": "1", "hyp": "D01", "venue": "binance-perp", "side": "long"},
        {"event": "OPEN", "id": "3", "hyp": "P01", "venue": "polymarket", "side": "no"},
    ])
    pr = risk_mod.position_risk(led)
    # id=1 closed, so only 2+3 remain (actually 2 rows OPEN not closed: id=2,3)
    assert len(pr) >= 1
    assert "open_positions" in pr.columns


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
