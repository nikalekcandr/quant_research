"""The executor: turns the lab's wing intents into Kalshi orders, with every check a desk would want.

Flow per intent (data/lab/intents.jsonl, written by lab/intents.py):
  1. refuse if: rule not enabled · kill switch · daily loss limit · too many positions / trades this hour ·
     already in this (rule, event) · intent older than max_intent_age_s · spot moved > max_spot_move_bp since the
     lab decided · < min_minutes_left to the settlement window
  2. read the LIVE production order book (rule 2: never the /markets listing) and re-check: spread ≤ max_spread,
     price still ≤ the lab's price + slippage, edge = fair − price − fee ≥ the rule's margin
  3. size = min(max_contracts, risk_per_trade / price, max_book_share × size at the best level)
  4. shadow: record what would have been bought at the live book · demo: send the order to the demo exchange ·
     live: send it to the production exchange
     take  → limit IOC at the checked price (never worse; the unfilled rest is cancelled by the exchange)
     quote → limit post-only GTC at the lab's maker price, expiring at the rule's fallback minute; cancelled early
             if spot moves > 2 × max_spot_move_bp against us
  5. reconcile: order status and fills from the API (not from the response alone); settlement from the market's
     result; every step is a line in data/trader/journal_<mode>.jsonl
"""
from __future__ import annotations

import json
import math
import os
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import requests

from trader.client import Book, Kalshi, KalshiError, dollars, fill_count
from trader.settings import DATA, INTENTS, Settings


# ------------------------------------------------------------------ fees (Kalshi, per order, rounded up to the cent)
def taker_fee(price: float, count: int = 1) -> float:
    return math.ceil(round(0.07 * count * price * (1 - price) * 100, 6)) / 100


def maker_fee(price: float, count: int = 1) -> float:
    return math.ceil(round(0.0175 * count * price * (1 - price) * 100, 6)) / 100


def strike_of(ticker: str) -> float | None:
    """KXBTCD-26NOV0213-T76999.99 -> 76999.99 (the ladder's 'greater than' strike)."""
    try:
        return float(ticker.rsplit("-T", 1)[1])
    except (IndexError, ValueError):
        return None


def refresh_fair(fair_side: float, side: str, ticker: str, s0: float, s1: float) -> float | None:
    """Move the lab's fair value of the side we buy to the current spot.

    The cheap side's probability c = 1 − fair_side was computed by the lab at spot s0. Under the lab's (log-normal)
    model c = Φ(±ln(s0/K) / sd), so sd = ±ln(s0/K) / Φ⁻¹(c); the same sd at spot s1 gives the new c. For a wing with
    10–30 minutes left one basis point of spot moves c by about 1¢ — the size of the whole edge — which is why a
    stale fair value must be refreshed rather than trusted.
    """
    from scipy.stats import norm
    k = strike_of(ticker)
    c = 1 - fair_side
    if k is None or not (0 < c < 1) or not (s0 > 0 and s1 > 0):
        return None
    sign = 1.0 if side == "no" else -1.0          # buying NO: the cheap side is YES = "above K"
    z = norm.ppf(c)
    x0 = sign * math.log(s0 / k)
    if abs(z) < 1e-6 or x0 / z <= 0:
        return None
    sd = x0 / z
    return 1 - float(norm.cdf(sign * math.log(s1 / k) / sd))


# Rules that bet on the same thing share one slot per hourly event (studies/wings_execution.py, 03.11.2026):
# W05 picks the same strike and side as W03 in 100% of common hours, W06 the same as W04. W03 and W04 differ in
# strike but bet the same way in 70% of shared hours, and a W03 loss raises the chance of a W04 loss 2.7×.
SLOT = {"W03": "late", "W05": "late", "W04": "early", "W06": "early"}


def slot_key(hyp: str, event: str) -> str:
    return f"{SLOT.get(hyp, hyp)}|{event}"


def now_utc() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC")


# ------------------------------------------------------------------ journal and state
class Journal:
    def __init__(self, mode: str, root: Path = DATA):
        root.mkdir(parents=True, exist_ok=True)
        self.path = root / f"journal_{mode}.jsonl"

    def write(self, what: str, **kw):
        rec = {"t": str(now_utc()), "what": what, **kw}
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, default=str, separators=(",", ":")) + "\n")
        return rec


@dataclass
class State:
    offset: int = 0                                   # bytes of intents.jsonl already read
    positions: dict = field(default_factory=dict)     # "hyp|event" -> position record
    resting: dict = field(default_factory=dict)       # "hyp|event" -> resting maker order record
    trades: list = field(default_factory=list)        # entry times (for the per-hour limit)
    realised: dict = field(default_factory=dict)      # day -> realised P&L in $

    @classmethod
    def load(cls, path: Path) -> "State":
        if path.exists():
            raw = json.loads(path.read_text(encoding="utf-8"))
            return cls(**{k: v for k, v in raw.items() if k in cls.__dataclass_fields__})
        return cls()

    def save(self, path: Path):
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.__dict__, default=str, indent=1), encoding="utf-8")
        tmp.replace(path)


# ------------------------------------------------------------------ spot (proxy of the BRTI the hourly ladder settles on)
class Spot:
    def __init__(self, url: str, max_age_s: float = 2.0):
        self.url, self.max_age, self.t, self.v = url, max_age_s, 0.0, float("nan")

    def get(self) -> float:
        if time.time() - self.t > self.max_age:
            try:
                r = requests.get(self.url, timeout=3)
                self.v, self.t = float(r.json()["price"]), time.time()
            except Exception:  # noqa: BLE001
                return float("nan")
        return self.v


# ------------------------------------------------------------------ the executor
class Executor:
    def __init__(self, cfg: Settings, venue: Kalshi | None, market_data: Kalshi, spot: Spot | None = None,
                 intents_path: Path = INTENTS, root: Path = DATA):
        """venue: the exchange orders go to (None in shadow mode); market_data: production public data."""
        self.cfg, self.venue, self.md = cfg, venue, market_data
        self.mode = cfg.mode
        self.spot = spot or Spot(cfg.spot_url)
        self.intents_path = intents_path
        self.root = root
        self.state_path = root / f"state_{self.mode}.json"
        self.st = State.load(self.state_path)
        self.j = Journal(self.mode, root)
        self.kill_file = root / "KILL"
        if self.st.offset == 0 and intents_path.exists():
            self.st.offset = intents_path.stat().st_size   # first start: ignore old decisions
        # one executor per mode: two copies would send every order twice
        self.lock = root / f"lock_{self.mode}.json"
        if self.lock.exists():
            try:
                other = json.loads(self.lock.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                other = {}
            if other.get("pid") != os.getpid() and time.time() - self.lock.stat().st_mtime < 90:
                raise RuntimeError(f"another {self.mode} executor is running (pid {other.get('pid')}): {self.lock}")
        self._touch_lock()

    def _touch_lock(self):
        self.lock.write_text(json.dumps({"pid": os.getpid(), "t": str(now_utc())}), encoding="utf-8")

    # -------------------------------------------------------------- intents
    def read_intents(self) -> list[dict]:
        if not self.intents_path.exists():
            return []
        size = self.intents_path.stat().st_size
        if size < self.st.offset:                          # file rotated
            self.st.offset = 0
        out = []
        with open(self.intents_path, "rb") as f:
            f.seek(self.st.offset)
            chunk = f.read()
        text = chunk.decode("utf-8", errors="replace")
        consumed = text.rfind("\n") + 1                    # only complete lines
        for line in text[:consumed].splitlines():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        self.st.offset += len(text[:consumed].encode("utf-8"))
        return out

    # -------------------------------------------------------------- limits
    def _today(self) -> str:
        return now_utc().strftime("%Y-%m-%d")

    def refuse_reason(self, it: dict) -> str | None:
        cfg = self.cfg
        key = slot_key(it["hyp"], it["event"])
        if self.kill_file.exists():
            return "kill switch"
        if it["hyp"] not in cfg.rules:
            return "rule not enabled"
        if self.st.realised.get(self._today(), 0.0) <= -cfg.daily_loss_limit_usd:
            return "daily loss limit"
        if key in self.st.positions:
            return "already in this slot and event"
        if it["kind"] == "quote" and key in self.st.resting:
            return "quote already resting"
        if len(self.st.positions) >= cfg.max_open_positions:
            return "max open positions"
        hour = now_utc() - pd.Timedelta(hours=1)
        if sum(pd.Timestamp(t) > hour for t in self.st.trades) >= cfg.max_trades_per_hour:
            return "max trades per hour"
        age = (now_utc() - pd.Timestamp(it["ts"])).total_seconds()
        if age > cfg.max_intent_age_s:
            return f"stale intent ({age:.0f} s)"
        if it.get("settle_ts"):
            left = (pd.Timestamp(it["settle_ts"]) - now_utc()).total_seconds() / 60
            if left < cfg.min_minutes_left:
                return f"too close to settlement ({left:.1f} min)"
        s0, s1 = it.get("spot"), self.spot.get()
        if s0 and math.isfinite(s1) and abs(s1 / s0 - 1) * 1e4 > cfg.max_spot_move_bp:
            return f"spot moved {1e4 * (s1 / s0 - 1):+.1f} bp since the decision"
        if not s0 or not math.isfinite(s1):
            return "no spot to refresh the fair value"
        return None

    def size(self, price: float, best_size: float) -> int:
        cfg = self.cfg
        if self.mode == "demo":
            return cfg.demo_contracts
        n = min(cfg.max_contracts, math.floor(cfg.risk_per_trade_usd / price), math.floor(cfg.max_book_share * best_size))
        return max(int(n), 0)

    # -------------------------------------------------------------- handlers
    def handle(self, it: dict):
        kind = it.get("kind")
        if kind == "paper_fill":
            self.j.write("lab_paper_fill", intent=it["id"])
            return
        if kind == "cancel":
            return self.cancel_quote(it)
        why = self.refuse_reason(it)
        if why:
            self.j.write("skip", intent=it["id"], reason=why)
            return
        try:
            book = self.md.book(it["ticker"])
        except Exception as e:  # noqa: BLE001
            self.j.write("skip", intent=it["id"], reason=f"book error {e}")
            return
        if kind == "take":
            key = slot_key(it["hyp"], it["event"])
            if key in self.st.resting:                       # our real maker order may already have filled
                self.cancel_quote(it)
                if key in self.st.positions:
                    return
            return self.take(it, book)
        if kind == "quote" and it["hyp"] in self.cfg.maker_rules:
            return self.quote(it, book)

    def spread_reason(self, book: Book) -> str | None:
        yes_spread = book.ask("yes")[0] - book.bid("yes")[0]
        if not (yes_spread == yes_spread) or yes_spread > self.cfg.max_spread + 1e-9:
            return f"spread {yes_spread:.2f}"
        return None

    def check_edge(self, it: dict, book: Book, price: float, fee_per: float) -> str | None:
        why = self.spread_reason(book)
        if why:
            return why
        fair = it.get("fair_side")
        if fair is None:
            return "no fair value in the intent"
        s1 = self.spot.get()
        if it.get("spot") and math.isfinite(s1):
            f1 = refresh_fair(fair, it["side"], it["ticker"], it["spot"], s1)
            if f1 is not None:
                it["fair_now"] = f1
                fair = f1
        edge = fair - price - fee_per
        if edge < it.get("margin", 0.0) - 1e-9:
            move = (1e4 * (s1 / it["spot"] - 1)) if it.get("spot") and math.isfinite(s1) else float("nan")
            return (f"edge {100 * edge:+.2f}¢ < margin {100 * it.get('margin', 0):.0f}¢ at {price:.2f} "
                    f"(lab fair {it['fair_side']:.4f} → {fair:.4f}, spot {move:+.1f} bp)")
        return None

    def take(self, it: dict, book: Book):
        side = it["side"]
        ask, ask_size = book.ask(side)
        limit = round(it["price"] + 0.01 * self.cfg.slippage_ticks, 2)
        if not (ask == ask):
            self.j.write("skip", intent=it["id"], reason="no offer")
            return
        if self.spread_reason(book):
            self.j.write("skip", intent=it["id"], reason=self.spread_reason(book))
            return
        if ask > limit + 1e-9:
            self.j.write("skip", intent=it["id"], reason=f"price moved: ask {ask:.2f} > limit {limit:.2f}")
            return
        why = self.check_edge(it, book, ask, taker_fee(ask))
        if why:
            self.j.write("skip", intent=it["id"], reason=why)
            return
        # walk the book (EXECUTION.md §3): take each level whose own edge is still ≥ the margin, never more than
        # max_book_share of a level, up to the risk budget; the IOC limit is the worst level accepted
        fair = it.get("fair_now", it.get("fair_side"))
        target = self.cfg.demo_contracts if self.mode == "demo" else \
            min(self.cfg.max_contracts, math.floor(self.cfg.risk_per_trade_usd / ask))
        n, limit_px, cost = 0, ask, 0.0
        for px, qty in book.asks(side):
            if n >= target or px > limit + 1e-9 or fair - px - taker_fee(px) < it.get("margin", 0.0) - 1e-9:
                break
            cap = int(self.cfg.max_book_share * qty)
            if cap < 1 and qty >= 1 and target == 1:
                cap = 1                                  # a 1-contract order may take a thin level
            take_n = min(target - n, cap) if self.mode != "demo" else target - n
            if take_n < 1:
                break
            n, limit_px, cost = n + take_n, px, cost + take_n * px
        if n < 1:
            self.j.write("skip", intent=it["id"], reason=f"size 0 (best level {ask_size:.0f})")
            return
        vwap = cost / n
        rec = {"intent": it["id"], "hyp": it["hyp"], "ticker": it["ticker"], "event": it["event"], "side": side,
               "price": limit_px, "vwap": round(vwap, 4), "count": n, "fair": it.get("fair_side"), "fair_now": it.get("fair_now"),
               "settle_ts": it.get("settle_ts"), "book_ask_size": ask_size, "lab_price": it["price"], "decision_age_s":
               (now_utc() - pd.Timestamp(it["ts"])).total_seconds()}
        if self.mode == "shadow":
            self._open(rec | {"price": vwap, "filled": n, "fee": taker_fee(vwap, n), "maker": False})
            self.j.write("shadow_fill", **rec)
            return
        self.send(it, rec, tif="immediate_or_cancel")

    def quote(self, it: dict, book: Book):
        side = it["side"]
        price = round(it["price"], 2)
        ask = book.ask(side)[0]
        if ask == ask and price >= ask - 1e-9:
            self.j.write("skip", intent=it["id"], reason=f"post-only would cross: {price:.2f} ≥ ask {ask:.2f}")
            return
        why = self.check_edge(it, book, price, maker_fee(price))
        if why:
            self.j.write("skip", intent=it["id"], reason=why)
            return
        n = self.cfg.demo_contracts if self.mode == "demo" else \
            min(self.cfg.max_contracts, math.floor(self.cfg.risk_per_trade_usd / price))   # a bid adds depth
        if n < 1:
            self.j.write("skip", intent=it["id"], reason="size 0")
            return
        rec = {"intent": it["id"], "hyp": it["hyp"], "ticker": it["ticker"], "event": it["event"], "side": side,
               "price": price, "count": n, "fair": it.get("fair_side"), "settle_ts": it.get("settle_ts"),
               "expires": it.get("expires"), "spot0": it.get("spot")}
        key = slot_key(it["hyp"], it["event"])
        if self.mode == "shadow":
            self.st.resting[key] = rec | {"order_id": None, "posted": str(now_utc())}
            self.j.write("shadow_quote", **rec)
            return
        exp = int(pd.Timestamp(it["expires"]).timestamp()) if it.get("expires") else None
        o = self.send(it, rec, tif="good_till_canceled", post_only=True, expiration_ts=exp, open_position=False)
        if o is not None and o.get("status") == "resting":
            self.st.resting[key] = rec | {"order_id": o["order_id"], "posted": str(now_utc())}

    def send(self, it, rec, tif, post_only=False, expiration_ts=None, open_position=True):
        coid = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{self.mode}|{it['id']}"))
        book_side = rec["side"]
        price = rec["price"]
        if self.mode == "demo":                       # demo books are not the real market
            try:
                db = self.venue.book(rec["ticker"])
            except KalshiError as e:
                self.j.write("skip", intent=it["id"], reason=f"demo market missing: {e.status}")
                return None
            dask = db.ask(book_side)[0]
            if tif == "immediate_or_cancel":
                if not (dask == dask):
                    self.j.write("skip", intent=it["id"], reason="demo: no offer")
                    return None
                if not self.cfg.demo_ignore_edge and dask > price + 1e-9:
                    self.j.write("skip", intent=it["id"], reason=f"demo ask {dask:.2f} > {price:.2f}")
                    return None
                price = dask if self.cfg.demo_ignore_edge else price
            elif dask == dask and price >= dask:
                price = round(dask - 0.01, 2)
                if price < 0.01:
                    return None
        self.j.write("order_sent", **rec, price_sent=price, tif=tif, post_only=post_only, client_order_id=coid)
        try:
            o = self.venue.create_order(rec["ticker"], book_side, rec["count"], price, tif=tif, post_only=post_only,
                                        expiration_ts=expiration_ts, client_order_id=coid)
        except KalshiError as e:
            self.j.write("order_rejected", intent=it["id"], status=e.status, body=e.body[:500])
            return None
        except Exception as e:  # noqa: BLE001  network: did it arrive?
            o = self._find_by_coid(rec["ticker"], coid)
            self.j.write("order_unknown" if o is None else "order_recovered", intent=it["id"], error=str(e)[:200])
            if o is None:
                return None
        oid = o.get("order_id")
        try:
            o = self.venue.order(oid) or o           # the exchange's view, not the POST response
        except Exception:  # noqa: BLE001
            pass
        filled = fill_count(o)
        fees = dollars(o.get("taker_fees_dollars", o.get("maker_fees_dollars", 0))) or 0.0
        if not fees and o.get("average_fee_paid_dollars") is not None:     # V2 create-order response (04.11)
            fees = round(dollars(o["average_fee_paid_dollars"]) * fill_count(o), 4)
        self.j.write("order_result", intent=it["id"], order_id=oid, status=o.get("status"), filled=filled,
                     fees=fees)
        if open_position and filled > 0:
            self._open(rec | {"filled": filled, "fee": fees, "price": price, "order_id": oid, "maker": post_only})
        return o

    def _find_by_coid(self, ticker, coid):
        try:
            for o in self.venue.orders(ticker=ticker):
                if o.get("client_order_id") == coid:
                    return o
        except Exception:  # noqa: BLE001
            return None
        return None

    def _open(self, rec):
        key = slot_key(rec["hyp"], rec["event"])
        self.st.positions[key] = rec | {"opened": str(now_utc())}
        self.st.trades.append(str(now_utc()))
        self.st.trades = self.st.trades[-50:]
        self.j.write("position_open", **self.st.positions[key])

    def cancel_quote(self, it):
        key = slot_key(it["hyp"], it["event"])
        q = self.st.resting.pop(key, None)
        if q is None:
            return
        if self.mode == "shadow" or not q.get("order_id"):
            self.j.write("shadow_cancel", intent=it.get("id"), key=key)
            return
        try:
            self.venue.cancel(q["order_id"])
        except KalshiError as e:
            if e.status != 404:
                self.j.write("cancel_error", key=key, status=e.status, body=e.body[:200])
        try:
            o = self.venue.order(q["order_id"])
        except Exception:  # noqa: BLE001
            o = {}
        filled = fill_count(o)
        self.j.write("quote_cancelled", key=key, order_id=q["order_id"], filled=filled)
        if filled > 0:
            self._open(q | {"count": filled, "filled": filled, "maker": True,
                            "fee": dollars(o.get("maker_fees_dollars", 0)) or maker_fee(q["price"], int(filled))})

    # -------------------------------------------------------------- periodic work
    def watch_quotes(self):
        """Resting maker orders: count fills; cancel if spot ran against us or the quote expired."""
        s1 = self.spot.get()
        for key, q in list(self.st.resting.items()):
            exp = pd.Timestamp(q["expires"]) if q.get("expires") else None
            adverse = False
            if q.get("spot0") and math.isfinite(s1):
                move = (s1 / q["spot0"] - 1) * 1e4
                # buying NO on "above K" loses if BTC rises; buying YES loses if it falls
                adverse = (move > self.cfg.quote_cancel_bp) if q["side"] == "no" else (move < -self.cfg.quote_cancel_bp)
            if self.mode == "shadow":
                if (exp is not None and now_utc() >= exp) or adverse:
                    self.st.resting.pop(key)
                    self.j.write("shadow_quote_end", key=key, reason="adverse spot" if adverse else "expired")
                continue
            try:
                o = self.venue.order(q["order_id"])
            except Exception:  # noqa: BLE001
                continue
            filled, status = fill_count(o), o.get("status")
            if status != "resting" or adverse or (exp is not None and now_utc() >= exp):
                if status == "resting":
                    try:
                        self.venue.cancel(q["order_id"])
                    except KalshiError:
                        pass
                    o = self.venue.order(q["order_id"])
                    filled = fill_count(o)
                self.st.resting.pop(key, None)
                self.j.write("quote_end", key=key, status=o.get("status"), filled=filled,
                             reason="adverse spot" if adverse else ("expired" if exp is not None and now_utc() >= exp else status))
                if filled > 0:
                    self._open(q | {"count": filled, "filled": filled, "maker": True,
                                    "fee": dollars(o.get("maker_fees_dollars", 0)) or maker_fee(q["price"], int(filled))})

    def settle(self):
        """Close positions whose market has a result; realised P&L feeds the daily loss limit."""
        for key, p in list(self.st.positions.items()):
            if not p.get("settle_ts") or now_utc() < pd.Timestamp(p["settle_ts"]) + pd.Timedelta(minutes=2):
                continue
            try:
                # demo positions settle on the demo exchange's own result
                m = (self.venue if self.mode == "demo" and self.venue is not None else self.md).market(p["ticker"])
            except Exception:  # noqa: BLE001
                continue
            res = (m.get("result") or "").lower()
            if res not in ("yes", "no"):
                continue
            n = float(p.get("filled") or p.get("count") or 0)
            pnl = n * ((1.0 if res == p["side"] else 0.0) - p["price"]) - float(p.get("fee") or 0.0)
            day = self._today()
            self.st.realised[day] = round(self.st.realised.get(day, 0.0) + pnl, 4)
            self.st.positions.pop(key)
            self.j.write("settled", key=key, ticker=p["ticker"], side=p["side"], result=res, count=n,
                         price=p["price"], fee=p.get("fee"), pnl=round(pnl, 4), maker=p.get("maker", False))

    def step(self):
        for it in self.read_intents():
            try:
                self.handle(it)
            except Exception as e:  # noqa: BLE001
                self.j.write("error", intent=it.get("id"), error=repr(e)[:300])
        try:
            self.watch_quotes()
            self.settle()
        except Exception as e:  # noqa: BLE001
            self.j.write("error", where="periodic", error=repr(e)[:300])
        self.st.save(self.state_path)
        self._touch_lock()
        (self.root / f"heartbeat_{self.mode}.json").write_text(json.dumps({
            "t": str(now_utc()), "open": len(self.st.positions), "resting": len(self.st.resting),
            "realised_today": self.st.realised.get(self._today(), 0.0)}), encoding="utf-8")

    def run(self, once: bool = False):
        self.j.write("start", mode=self.mode, rules=self.cfg.rules, settings=self.cfg.__dict__)
        while True:
            if self.kill_file.exists():
                n = self.venue.cancel_all() if self.venue is not None else 0
                self.j.write("kill", cancelled=n)
                self.st.save(self.state_path)
                return
            self.step()
            if once:
                return
            time.sleep(self.cfg.poll_s)
