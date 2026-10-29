"""Trade-tape strategies on Kalshi (frozen 29.10.2026 17:40 MSK). Run once a minute from Lab.minute.

M03  Lottery seller: on the 15-minute market and on hourly-ladder strikes, when one side costs 2–20¢ (the "cheap"
     side) and ≥ 2 minutes are left, we join the best offer of the cheap side with one contract (paper). Queue: the
     size displayed at that price when we join is ahead of us. Over the next 60 s we are filled if prints strictly
     through our price occur, or if takers buy at our price more contracts than the queue ahead. Held to settlement,
     maker fee 0.0175·P(1−P).
M04  Same quote, spot-aware: the quote is cancelled at the first 15-s spot sample that moved ≥ 3 bp toward the cheap
     side winning (BTC up for a cheap YES, down for a cheap NO); prints after the cancel do not fill us.
T01  Copy the favourite buyer: if in the last 60 s takers bought ≥ 200 contracts of the favourite side at 65–95¢ on
     the 15-minute market, we buy the favourite at the current book ask (taker fee). One per market.
T02  Fade the lottery burst: if in the last 60 s takers bought ≥ 500 contracts of the cheap side at ≤ 20¢, we buy
     the favourite at the current book ask (taker fee). One per market.
Backtest of the same rules on sampled history: studies/tape_strategies.py.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from lab.core import Order, kalshi_fee, log

KALSHI = "https://api.elections.kalshi.com/trade-api/v2"
CHEAP = (0.02, 0.20)
CANCEL_BP = 3e-4


def maker_fee(p):
    return math.ceil(round(0.0175 * p * (1 - p) * 100, 6)) / 100


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def tape(http, ticker, t0, t1, pages=5):
    """Prints of one market in [t0, t1]: list of (ts, yes_price, contracts, taker_side)."""
    out, cur = [], None
    for _ in range(pages):
        p = {"ticker": ticker, "min_ts": int(t0.timestamp()), "max_ts": int(t1.timestamp()), "limit": 1000}
        if cur:
            p["cursor"] = cur
        res = http.get(f"{KALSHI}/markets/trades", p) or {}
        tr = res.get("trades") or []
        for t in tr:
            yp = _f(t.get("yes_price_dollars", t.get("yes_price")))
            if yp is None:
                continue
            yp = yp / 100 if yp > 1 else yp
            out.append((pd.Timestamp(t.get("created_time")), yp, _f(t.get("count_fp", t.get("count"))) or 1.0,
                        t.get("taker_side")))
        cur = res.get("cursor")
        if not cur or not tr:
            break
    return out


class Tape:
    def __init__(self):
        self.quotes = []

    # ------------------------------------------------------------------ fills
    def _filled(self, q, prints, cancel_at):
        """q: we sell the cheap side at q['ask'] (its own price). A taker who buys the cheap side pays ≥ our price."""
        ahead, got = q["queue"], 0.0
        for ts, yp, n, side in sorted(prints):
            if cancel_at is not None and ts >= cancel_at:
                break
            if side != q["cheap"]:
                continue
            paid = yp if side == "yes" else 1 - yp
            if paid > q["ask"] + 1e-9:          # through our price: we were ahead of that level
                return True
            if abs(paid - q["ask"]) < 1e-9:
                got += n
                if got > ahead:
                    return True
        return False

    def _cancel_time(self, lab, q):
        """First 15-s sample after the quote where Binance moved ≥ 3 bp toward the cheap side winning."""
        s0 = q.get("spot")
        if not s0:
            return None
        sgn = 1.0 if q["cheap"] == "yes" else -1.0
        for r in list(getattr(lab, "fast", []) or []):
            ts, bn = r.get("ts"), r.get("bn")
            if ts is None or bn is None or ts <= q["t0"] or ts > q["t0"] + pd.Timedelta(seconds=60):
                continue
            if sgn * math.log(bn / s0) >= CANCEL_BP:
                return ts
        return None

    def _settle_quotes(self, ctx, lab):
        now = ctx.now
        live, todo = [], {}
        for q in self.quotes:
            (todo.setdefault(q["market"], []) if now - q["t0"] >= pd.Timedelta(seconds=55) else live).append(q)
        for market, qs in todo.items():
            t0 = min(q["t0"] for q in qs)
            try:
                prints = tape(lab.http, market, t0, t0 + pd.Timedelta(seconds=60))
            except Exception as e:
                log(f"tape: trades {market} failed ({str(e)[:100]})")
                continue
            for q in qs:
                cancel = self._cancel_time(lab, q) if q["hid"] == "M04" else None
                if not self._filled(q, prints, cancel):
                    continue
                # selling the cheap side at a = buying the other side at 1 − a
                side = "no" if q["cheap"] == "yes" else "yes"
                px = 1 - q["ask"]
                fee = maker_fee(px)
                o = Order(hyp=q["hid"], kind="binary", venue="kalshi", instrument=market, side=side, price=px, fee=fee,
                          settle_ts=str(q["end"]), unit="$/contract", key=f"{q['hid']}|{market}",
                          info={"legs": [{"venue": "kalshi", "kind": q["kind"], "market": market, "event": q["event"],
                                          "side": side, "price": px, "fee": fee}], "maker": True, "cheap_price": q["ask"],
                                "queue": q["queue"], "prints": len(prints), "cancelled": str(cancel) if cancel else None,
                                "left_min": round((q["end"] - q["t0"]).total_seconds() / 60, 1)})
                if ctx.book.submit(o, now, "M"):
                    log(f"TAPE FILL {q['hid']} sold {q['cheap']} {market} @ {q['ask']:.2f}")
        self.quotes = live

    # ------------------------------------------------------------------ new quotes and taker signals
    def step(self, ctx, lab):
        self._settle_quotes(ctx, lab)
        m = ctx.markets
        if m is None or m.empty:
            return
        now = ctx.now
        k = m[(m["venue"] == "kalshi") & m["kind"].isin(["kx_15m", "kx_ladder_1h"])]
        spot = (ctx.tick or {}).get("bn_mid")
        for _, r in k.iterrows():
            bid, ask = r["bid"], r["ask"]
            if not (np.isfinite(bid) and np.isfinite(ask)) or (pd.Timestamp(r["end"]) - now).total_seconds() < 120:
                continue
            if ask - bid > 0.05:
                continue
            for cheap, a, q_ahead in (("yes", ask, r.get("ask_size")), ("no", 1 - bid, r.get("bid_size"))):
                if not (CHEAP[0] <= a <= CHEAP[1]):
                    continue
                for hid in ("M03", "M04"):
                    if ctx.book.has_key(f"{hid}|{r['market']}"):
                        continue
                    self.quotes.append({"hid": hid, "market": r["market"], "event": r["event"], "kind": r["kind"],
                                        "cheap": cheap, "ask": round(float(a), 4),
                                        "queue": float(q_ahead) if q_ahead is not None and np.isfinite(q_ahead) else 1e9,
                                        "t0": now, "end": pd.Timestamp(r["end"]), "spot": spot})
        # taker signals on the 15-minute market
        for _, r in k[k["kind"] == "kx_15m"].iterrows():
            bid, ask = r["bid"], r["ask"]
            left = (pd.Timestamp(r["end"]) - now).total_seconds() / 60
            if not (np.isfinite(bid) and np.isfinite(ask)) or left < 2 or ask - bid > 0.05:
                continue
            try:
                prints = tape(lab.http, r["market"], now - pd.Timedelta(seconds=60), now)
            except Exception:
                continue
            fav_buy = sum(n for _, yp, n, s in prints if (yp if s == "yes" else 1 - yp) >= 0.65 and
                          (yp if s == "yes" else 1 - yp) <= 0.95)
            cheap_buy = sum(n for _, yp, n, s in prints if (yp if s == "yes" else 1 - yp) <= 0.20)
            mid = (bid + ask) / 2
            fav, px = ("yes", ask) if mid >= 0.5 else ("no", 1 - bid)
            for hid, ok in (("T01", fav_buy >= 200), ("T02", cheap_buy >= 500)):
                if not ok or ctx.book.has_key(f"{hid}|{r['market']}") or not (0.5 <= px <= 0.98):
                    continue
                o = Order(hyp=hid, kind="binary", venue="kalshi", instrument=r["market"], side=fav, price=px,
                          fee=kalshi_fee(px), settle_ts=str(r["end"]), unit="$/contract", key=f"{hid}|{r['market']}",
                          info={"legs": [{"venue": "kalshi", "kind": r["kind"], "market": r["market"], "event": r["event"],
                                          "side": fav, "price": px, "fee": kalshi_fee(px)}], "mid": mid,
                                "fav_buy_60s": fav_buy, "cheap_buy_60s": cheap_buy, "left_min": round(left, 1),
                                "size": r.get("ask_size") if fav == "yes" else r.get("bid_size")})
                if ctx.book.submit(o, now, "T"):
                    log(f"OPEN {hid} {fav} {r['market']} @ {px:.2f} (fav buys {fav_buy:.0f}, cheap buys {cheap_buy:.0f})")
