"""Trade intents for the live executor (trader/).

The lab decides, the executor executes. Whenever a wing rule (family W, Kalshi) books a paper trade or posts /
withdraws a paper maker quote, one JSON line is appended to data/lab/intents.jsonl. The rules themselves are
untouched (they stay frozen); this is only an outbound copy of their decisions.

kinds
  take        buy `side` now at up to `price` (taker); W03/W04 always, W05/W06 at the fallback minute
  quote       post-only bid for `side` at `price` (= selling the cheap side at its ask) until `expires`
  cancel      withdraw the quote of (hyp, event)
  paper_fill  the lab counted its paper quote as filled (information only)
Every line carries the lab's fair value of the side we buy, the margin of the rule and the USD spot at decision
time, so the executor can re-check the edge on the live book and refuse stale decisions (AGENTS.md rule 7).
"""
from __future__ import annotations

import json
import math

import pandas as pd

from lab.core import LAB

INTENTS = LAB / "intents.jsonl"
MARGIN = {"W03": 0.01, "W04": 0.0, "W05": 0.01, "W06": 0.0}
ENABLED = set(MARGIN)
SPOT: float | None = None          # set every loop by lab.wings.Wings.step: Coinbase mid of the latest tick


def _num(x):
    try:
        x = float(x)
        return x if math.isfinite(x) else None
    except (TypeError, ValueError):
        return None


def emit(kind: str, hyp: str, ticker: str, event: str, side: str, price: float, now, fair_side=None,
         settle_ts=None, expires=None, **extra):
    """Append one intent; never raises (the lab must not stop because of the executor)."""
    if hyp not in ENABLED:
        return
    try:
        rec = {"id": f"{hyp}|{event}|{kind}|{pd.Timestamp(now).value // 10**9}", "ts": str(pd.Timestamp(now)),
               "kind": kind, "hyp": hyp, "ticker": ticker, "event": event, "side": side, "price": round(float(price), 4),
               "fair_side": _num(fair_side), "margin": MARGIN[hyp], "spot": _num(SPOT),
               "settle_ts": None if settle_ts is None else str(settle_ts),
               "expires": None if expires is None else str(expires)}
        rec.update({k: (_num(v) if isinstance(v, (int, float)) else v) for k, v in extra.items()})
        with open(INTENTS, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, separators=(",", ":")) + "\n")
    except Exception:  # noqa: BLE001
        pass


def from_order(o, now):
    """Called by Book.submit for every accepted order: wing orders become `take` / `paper_fill` intents."""
    if o.kind != "binary" or o.venue != "kalshi" or o.hyp not in ENABLED:
        return
    info = o.info or {}
    legs = info.get("legs") or [{}]
    emit("paper_fill" if info.get("maker") else "take", o.hyp, o.instrument, legs[0].get("event", ""), o.side, o.price,
         now, fair_side=o.fair, settle_ts=o.settle_ts, edge=o.edge, wing_price=info.get("wing_price"),
         left_min=info.get("left_min"))
