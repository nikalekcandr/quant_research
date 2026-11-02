"""When to believe the wing book: a sequential probability ratio test (Wald 1945) on hourly P&L.

H0: the book earns 0 per hour.  H1: it earns what the out-of-sample backtest says (Sep–Oct 2026, W03+W04, one
contract each, taker: +3.28¢ per active hour, sd 29.9¢; studies/wings_execution.py). After every settled hour
    LLR += (μ1/σ²)·x − μ1²/(2σ²)
and the test stops at LLR ≥ ln((1−β)/α) (accept H1: the edge is there) or LLR ≤ ln(β/(1−α)) (accept H0: stop
trading the idea). With α = 5%, β = 20% that takes ≈ 320 active hours under H1 and ≈ 220 under H0 — about three
weeks of the lab, versus a fixed-sample t-test that would need the whole month regardless of how things go.
"""
from __future__ import annotations

import math

import pandas as pd

MU1, SD = 0.0328, 0.299            # $ per active hour, one contract per slot
ALPHA, BETA = 0.05, 0.20
SLOT = {"W03": "late", "W05": "late", "W04": "early", "W06": "early"}


def bounds(alpha: float = ALPHA, beta: float = BETA) -> tuple[float, float]:
    return math.log((1 - beta) / alpha), math.log(beta / (1 - alpha))


def sprt(hourly: pd.Series, mu1: float = MU1, sd: float = SD) -> dict:
    """hourly: P&L per active hour in $ for one contract per slot."""
    a, b = bounds()
    llr = (mu1 / sd ** 2 * hourly - mu1 ** 2 / (2 * sd ** 2)).cumsum()
    state = "edge confirmed (accept H1)" if len(llr) and llr.iloc[-1] >= a else \
        "no edge (accept H0): stop" if len(llr) and llr.iloc[-1] <= b else "continue"
    first = llr[(llr >= a) | (llr <= b)]
    return {"hours": len(hourly), "mean ¢/hour": 100 * hourly.mean() if len(hourly) else float("nan"),
            "LLR": float(llr.iloc[-1]) if len(llr) else 0.0, "upper": a, "lower": b, "state": state,
            "crossed_at": None if first.empty else str(first.index[0])}


def hourly_from_trades(trades: pd.DataFrame, hyp_col="hyp", event_col="event", pnl_col="pnl_per_contract",
                       t_col="settle_ts") -> pd.Series:
    """First trade per (slot, event) — W05 duplicates W03, W06 duplicates W04 — summed per settlement hour."""
    x = trades.assign(slot=trades[hyp_col].map(SLOT)).dropna(subset=["slot"])
    x = x.sort_values("t" if "t" in x else t_col).drop_duplicates(["slot", event_col])
    return x.groupby(pd.to_datetime(x[t_col], utc=True))[pnl_col].sum().sort_index()
