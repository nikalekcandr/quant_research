"""W05 / W06: wing selling with everything that helped on history (frozen 29.10.2026 22:00 MSK).

Chosen on Jul–Aug 2026 Kalshi hourly-ladder candles, checked on Sep–Oct (studies/wings_ladder.py + wings_improve):
  * fair value with a blended variance: geometric mean of the lab's seasonal EWMA forecast and the gradient-boosted
    forecast trained on three years of minutes (`gbm_long` in the vol zoo)  — Sep–Oct W04-window +3.5¢, t 4.5
  * no trading in the hours settling 13:00–15:00 UTC (the US open)          — helps in every window and half
  * one position per window and hour, the strike with the largest edge
  * maker first: offer the cheap side at its ask (post-only, paper) and count it filled only by prints through our
    price or by taker volume at our price beyond the queue displayed when we joined; if still unfilled at the
    fallback minute and the taker edge (bid − fair − fee) is still ≥ the margin, sell at the bid (taker fee).
W05: cheap side 10–20¢, 5–15 min left, margin 1¢, fallback at 7 min left.
W06: cheap side 6–10¢, 15–30 min left, margin 0, fallback at 17 min left.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from lab.core import Order, kalshi_fee, log
from lab.signals import RETIRED, fair_value
from lab.tape import maker_fee, tape

RULES = {"W05": dict(lo=0.10, hi=0.20, l0=5, l1=15, margin=0.01, fallback=7),
         "W06": dict(lo=0.06, hi=0.10, l0=15, l1=30, margin=0.0, fallback=17)}
US_OPEN_HOURS = {13, 14, 15}         # settlement hour (UTC) excluded


def _gbm_var(ctx, end):
    """Variance to `end` from the zoo's gbm_long: predicted log RV of the coming hour, scaled to the minutes left."""
    lab = ctx.lab
    fit = getattr(getattr(lab, "zoo", None), "fits", {}).get("gbm_long") if lab is not None else None
    if fit is None:
        return None
    key = ("gbm_h", end.floor("h"))
    if key not in ctx.cache:
        g, smear, _ = fit
        r = ctx.minutes.returns(10)
        rv = (r ** 2).groupby(r.index.floor("h")).agg(["sum", "count"])
        rv = rv[(rv["count"] >= 50) & (rv["sum"] > 0)]["sum"]
        if len(rv) < 168:
            return None
        h = np.log(rv)
        hh = end - pd.Timedelta(hours=1)
        x = np.array([[h.iloc[-1], h.iloc[-24:].mean(), h.iloc[-168:].mean(), hh.hour, hh.dayofweek]])
        ctx.cache[key] = math.exp(g.predict(x)[0]) * smear / 60          # per minute
    left = max((end - ctx.now).total_seconds() / 60 - 2 / 3, 1 / 3)
    return ctx.cache[key] * left


def _fair_blend(ctx, r):
    """P(YES) with variance = sqrt(var_lab × var_gbm); falls back to the lab variance alone."""
    p_lab = fair_value(ctx, r, "ln")
    if p_lab is None:
        return None
    end = pd.Timestamp(r["end"])
    v_lab = ctx.minutes.forecast_var(ctx.now, end - pd.Timedelta(seconds=30))
    v_gbm = _gbm_var(ctx, end)
    if v_gbm is None or not (np.isfinite(v_lab) and v_lab > 0):
        return p_lab
    # rescale the lab's z-score to the blended variance
    z = float(np.clip(np.sqrt(2) * _erfinv(2 * p_lab - 1), -8, 8))
    return float(0.5 * (1 + math.erf(z * math.sqrt(v_lab / math.sqrt(v_lab * v_gbm)) / math.sqrt(2))))


def _erfinv(y):
    from scipy.special import erfinv
    return float(erfinv(np.clip(y, -0.999999, 0.999999)))


class Wings:
    def __init__(self):
        self.quotes = {}          # (hid, event) -> quote dict

    def step(self, ctx, lab):
        if all(h in RETIRED for h in RULES):
            return
        m = ctx.markets
        if m is None or m.empty:
            return
        now = ctx.now
        k = m[(m["venue"] == "kalshi") & (m["kind"] == "kx_ladder_1h")]
        # 1. live quotes: filled by the tape? at the fallback minute: take or drop
        for key, q in list(self.quotes.items()):
            hid, event = key
            rule = RULES[hid]
            left = (q["end"] - now).total_seconds() / 60
            try:
                prints = tape(lab.http, q["market"], q["t0"], now)
            except Exception:
                prints = []
            if self._filled(q, prints):
                self._book(ctx, hid, q, q["ask"], maker=True, left=left)
                del self.quotes[key]
                continue
            if left <= rule["fallback"]:
                row = k[k["market"] == q["market"]]
                if len(row):
                    r = row.iloc[0]
                    c_bid = r["bid"] if q["cheap"] == "yes" else 1 - r["ask"]
                    p = _fair_blend(ctx, r)
                    if p is not None and np.isfinite(c_bid):
                        fair = p if q["cheap"] == "yes" else 1 - p
                        if c_bid - fair - kalshi_fee(1 - c_bid) >= rule["margin"]:
                            self._book(ctx, hid, q, c_bid, maker=False, left=left, fair=fair)
                del self.quotes[key]
        # 2. new quotes: best strike per window and hour
        for hid, rule in RULES.items():
            if hid in RETIRED:
                continue
            best = None
            for _, r in k.iterrows():
                end = pd.Timestamp(r["end"])
                if end.hour in US_OPEN_HOURS or (hid, r["event"]) in self.quotes or ctx.book.has_key(f"{hid}|{r['event']}"):
                    continue
                left = (end - now).total_seconds() / 60
                bid, ask = r["bid"], r["ask"]
                if not (rule["l0"] <= left <= rule["l1"]) or not (np.isfinite(bid) and np.isfinite(ask)) or ask - bid > 0.03:
                    continue
                p = _fair_blend(ctx, r)
                if p is None:
                    continue
                for cheap, c_bid, c_ask, q_ahead, fair in (("yes", bid, ask, r.get("ask_size"), p),
                                                            ("no", 1 - ask, 1 - bid, r.get("bid_size"), 1 - p)):
                    if not (rule["lo"] <= c_bid <= rule["hi"]):
                        continue
                    edge = c_bid - fair - kalshi_fee(1 - c_bid)          # taker edge; the maker edge is larger
                    if edge >= rule["margin"] and (best is None or edge > best[0]):
                        best = (edge, r, cheap, round(float(c_ask), 4), q_ahead, fair, end)
            if best is not None:
                edge, r, cheap, c_ask, q_ahead, fair, end = best
                self.quotes[(hid, r["event"])] = {"market": r["market"], "event": r["event"], "kind": r["kind"],
                                                  "cheap": cheap, "ask": c_ask, "fair": fair, "end": end, "t0": now,
                                                  "queue": float(q_ahead) if q_ahead is not None and np.isfinite(q_ahead) else 1e9}

    @staticmethod
    def _filled(q, prints):
        ahead, got = q["queue"], 0.0
        for ts, yp, n, side in sorted(prints):
            if side != q["cheap"]:
                continue
            paid = yp if side == "yes" else 1 - yp
            if paid > q["ask"] + 1e-9:
                return True
            if abs(paid - q["ask"]) < 1e-9:
                got += n
                if got > ahead:
                    return True
        return False

    @staticmethod
    def _book(ctx, hid, q, cheap_px, maker, left, fair=None):
        side = "no" if q["cheap"] == "yes" else "yes"
        px = 1 - cheap_px
        fee = maker_fee(px) if maker else kalshi_fee(px)
        fair = q["fair"] if fair is None else fair
        o = Order(hyp=hid, kind="binary", venue="kalshi", instrument=q["market"], side=side, price=px, fee=fee,
                  fair=1 - fair, edge=cheap_px - fair - fee, settle_ts=str(q["end"]), unit="$/contract",
                  key=f"{hid}|{q['event']}",
                  info={"legs": [{"venue": "kalshi", "kind": q["kind"], "market": q["market"], "event": q["event"],
                                  "side": side, "price": px, "fee": fee}], "maker": maker, "wing_price": cheap_px,
                        "quoted_at": str(q["t0"]), "left_min": round(left, 1), "queue": q["queue"]})
        if ctx.book.submit(o, ctx.now, "W"):
            log(f"WINGS {hid} {'maker' if maker else 'taker'} sold {q['cheap']} {q['market']} @ {cheap_px:.2f}")
