"""M01 / M02: simulated passive quotes on Kalshi, filled only by real prints from the trade tape.

Each minute, for the 15-minute market and up to four hourly-ladder strikes near the money, we post (on paper) a YES
bid at fair − 4¢ and a YES ask at fair + 4¢ (selling YES = buying NO at 1 − ask), only if the quote sits inside or
behind the current spread (otherwise it would be a taker order). The quote lives 60 seconds. It counts as filled only
if a real trade printed STRICTLY through our price during that minute — at our price we might have been behind the
queue. Fills are held to settlement like every other binary; maker fee 0.0175·P(1−P) rounded up to the cent.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from lab import signals
from lab.core import Order, log
from src.download import kalshi

KALSHI = "https://api.elections.kalshi.com/trade-api/v2"
HALF = 0.04


def maker_fee(p: float) -> float:
    return math.ceil(round(0.0175 * p * (1 - p) * 100, 6)) / 100


def _px(t):
    v = kalshi._num(t, "yes_price")
    if v is None:
        return None
    return v / 100 if v > 1 else v


class Maker:
    def __init__(self):
        self.quotes = []          # dicts: hid, market, event, kind, side, price, t0, end

    def step(self, ctx, lab):
        now = ctx.now
        # 1. fills of the quotes posted in the previous minute
        live, todo = [], {}
        for q in self.quotes:
            if now - q["t0"] >= pd.Timedelta(seconds=55):
                todo.setdefault(q["market"], []).append(q)
            else:
                live.append(q)
        for market, qs in todo.items():
            t0 = min(q["t0"] for q in qs)
            try:
                res = lab.http.get(f"{KALSHI}/markets/trades", {"ticker": market, "min_ts": int(t0.timestamp()),
                                                                "max_ts": int((t0 + pd.Timedelta(seconds=60)).timestamp()),
                                                                "limit": 1000}) or {}
                prints = [p for p in (_px(t) for t in res.get("trades") or []) if p is not None]
            except Exception as e:
                log(f"maker: trades {market} failed ({str(e)[:100]})")
                continue
            for q in qs:
                hit = any(p < q["price"] - 1e-9 for p in prints) if q["side"] == "yes" else \
                    any(p > q["price"] + 1e-9 for p in prints)
                if not hit:
                    continue
                px = q["price"] if q["side"] == "yes" else 1 - q["price"]
                fee = maker_fee(px)
                o = Order(hyp=q["hid"], kind="binary", venue="kalshi", instrument=market, side=q["side"], price=px, fee=fee,
                          fair=q["fair"] if q["side"] == "yes" else 1 - q["fair"],
                          edge=(q["fair"] - px if q["side"] == "yes" else (1 - q["fair"]) - px) - fee,
                          settle_ts=str(q["end"]), unit="$/contract", key=f"{q['hid']}|{market}|{q['side']}",
                          info={"legs": [{"venue": "kalshi", "kind": q["kind"], "market": market, "event": q["event"],
                                          "side": q["side"], "price": px, "fee": fee}], "maker": True,
                                "quote": q["price"], "prints": len(prints)})
                if ctx.book.submit(o, now, "M"):
                    log(f"MAKER FILL {q['hid']} {q['side']} {market} @ {px:.2f}")
        self.quotes = live
        # 2. new quotes
        m = ctx.markets
        if m is None or m.empty:
            return
        cands = []
        for _, r in m[m["kind"] == "kx_15m"].iterrows():
            cands.append(("M01", r))
        h = m[m["kind"] == "kx_ladder_1h"].dropna(subset=["bid", "ask"])
        if len(h):
            left = (pd.Timestamp(h["end"].iloc[0]) - now).total_seconds() / 60
            if left >= 10:
                fr = [(signals.fair_value(ctx, r), r) for _, r in h.iterrows()]
                fr = [(p, r) for p, r in fr if p is not None and 0.2 <= p <= 0.8]
                fr.sort(key=lambda x: abs(x[0] - 0.5))
                cands += [("M02", r) for _, r in fr[:4]]
        for hid, r in cands:
            if (pd.Timestamp(r["end"]) - now).total_seconds() < 120:
                continue
            p = signals.fair_value(ctx, r)
            bid, ask = r["bid"], r["ask"]
            if p is None or not (np.isfinite(bid) and np.isfinite(ask)):
                continue
            b = math.floor((p - HALF) * 100) / 100
            a = math.ceil((p + HALF) * 100) / 100
            base = {"hid": hid, "market": r["market"], "event": r["event"], "kind": r["kind"], "t0": now,
                    "end": pd.Timestamp(r["end"]), "fair": p}
            if 0.02 <= b < ask and not ctx.book.has_key(f"{hid}|{r['market']}|yes"):
                self.quotes.append({**base, "side": "yes", "price": b})
            if a <= 0.98 and a > bid and not ctx.book.has_key(f"{hid}|{r['market']}|no"):
                self.quotes.append({**base, "side": "no", "price": a})
