"""Tests of the Kalshi executor without network: signing, book parsing, fees, every refusal rule, demo order flow,
maker quotes, settlement and the kill switch.   python -m pytest tests/test_trader.py -q"""
from __future__ import annotations

import base64
import json
import sys
import time
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from trader.client import Book, parse_book, price_str, sign  # noqa: E402
from trader.executor import Executor, maker_fee, taker_fee  # noqa: E402
from trader.settings import Settings  # noqa: E402


# ------------------------------------------------------------------ fakes
class FakeSpot:
    def __init__(self, v=78000.0):
        self.v = v

    def get(self):
        return self.v


class FakeKalshi:
    def __init__(self, book: Book, fill: str = "all", result: str = ""):
        self._book, self.fill, self.result = book, fill, result
        self.sent, self.cancelled, self._orders = [], [], {}

    def book(self, ticker, depth=10):
        return Book(ticker, self._book.yes_bids, self._book.no_bids, time.time())

    def market(self, ticker):
        return {"ticker": ticker, "result": self.result}

    def create_order(self, ticker, side, count, price, **kw):
        oid = f"o{len(self.sent) + 1}"
        filled = count if self.fill == "all" else (0 if self.fill == "none" else 1)
        status = "resting" if kw.get("post_only") and filled < count else ("executed" if filled else "canceled")
        o = {"order_id": oid, "ticker": ticker, "side": side, "status": status, "fill_count": filled,
             "remaining_count": count - filled, "taker_fees_dollars": str(taker_fee(price, filled)) if filled else "0",
             "client_order_id": kw.get("client_order_id")}
        self.sent.append({"ticker": ticker, "side": side, "count": count, "price": price, **kw})
        self._orders[oid] = o
        return o

    def order(self, oid):
        return self._orders[oid]

    def orders(self, **kw):
        return list(self._orders.values())

    def cancel(self, oid):
        self.cancelled.append(oid)
        self._orders[oid]["status"] = "canceled"
        return {"order": self._orders[oid]}

    def cancel_all(self):
        return 0


def book(yes_bid=0.08, no_bid=0.91, size=500):
    """YES bid 8¢, NO bid 91¢ → YES ask 9¢, NO ask 92¢ (cheap side = YES, we buy NO at 0.92)."""
    return Book("KXBTCD-26NOV0213-T79999.99", [(yes_bid, size)], [(no_bid, size)], time.time())


def intent(kind="take", hyp="W04", age_s=2, price=0.92, fair=0.95, spot=78000.0, left_min=20, **kw):
    now = pd.Timestamp.now(tz="UTC")
    return {"id": f"{hyp}|E1|{kind}|{now.value}", "ts": str(now - pd.Timedelta(seconds=age_s)), "kind": kind,
            "hyp": hyp, "ticker": "KXBTCD-26NOV0213-T79999.99", "event": "KXBTCD-26NOV0213", "side": "no",
            "price": price, "fair_side": fair, "margin": {"W03": 0.01, "W04": 0.0, "W05": 0.01, "W06": 0.0}[hyp],
            "spot": spot, "settle_ts": str(now + pd.Timedelta(minutes=left_min)),
            "expires": str(now + pd.Timedelta(minutes=left_min - 7)), **kw}


def make(tmp_path, mode="shadow", fill="all", result="", **cfg_kw):
    tmp_path.mkdir(parents=True, exist_ok=True)
    cfg = Settings(mode=mode, **cfg_kw)
    v = FakeKalshi(book(), fill=fill, result=result)
    ex = Executor(cfg, None if mode == "shadow" else v, v, FakeSpot(), intents_path=tmp_path / "intents.jsonl",
                  root=tmp_path)
    return ex, v


def events(ex):
    return [json.loads(x) for x in ex.j.path.read_text().splitlines()]


# ------------------------------------------------------------------ client pieces
def test_rsa_pss_signature_verifies():
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.asymmetric import padding, rsa
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    msg = "1700000000000GET/trade-api/v2/portfolio/balance"
    sig = base64.b64decode(sign(key, msg))
    key.public_key().verify(sig, msg.encode(), padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=32),
                            hashes.SHA256())


def test_ed25519_signature_verifies():
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    key = Ed25519PrivateKey.generate()
    msg = "1700000000000POST/trade-api/v2/portfolio/orders"
    key.public_key().verify(base64.b64decode(sign(key, msg)), msg.encode())


def test_parse_book_fixed_point_and_legacy_and_empty_sides():
    b = parse_book("T", {"orderbook_fp": {"yes_dollars": [["0.0700", "10.00"], ["0.0800", "500.00"], ["0.0000", "5"]],
                                          "no_dollars": [["0.9100", "300.00"]]}})
    assert b.yes_bid == (0.08, 500.0) and b.ask("yes") == (pytest.approx(0.09), 300.0)
    assert b.ask("no") == (pytest.approx(0.92), 500.0)
    legacy = parse_book("T", {"orderbook": {"yes": [[8, 500]], "no": None}})
    assert legacy.yes_bid == (0.08, 500) and legacy.ask("yes")[0] != legacy.ask("yes")[0]   # NaN: no offer


def test_price_string_and_fees():
    assert price_str(0.92) == "0.9200"
    with pytest.raises(ValueError):
        price_str(1.0)
    assert taker_fee(0.92) == 0.01 and taker_fee(0.92, 10) == 0.06      # ceil(0.07·10·0.92·0.08 = 5.15¢)
    assert maker_fee(0.92, 10) == 0.02


# ------------------------------------------------------------------ refusals
def test_shadow_take_records_a_fill(tmp_path):
    ex, _ = make(tmp_path)
    ex.handle(intent())
    assert "early|KXBTCD-26NOV0213" in ex.st.positions
    assert ex.st.positions["early|KXBTCD-26NOV0213"]["filled"] == 5      # $5 / 0.92 → 5 contracts


@pytest.mark.parametrize("kw,reason", [
    ({"age_s": 60}, "stale intent"),
    ({"spot": 79000.0}, "spot moved"),               # > 20 bp: refuse outright
    ({"price": 0.89}, "price moved"),                 # the ask (0.92) is 3¢ above the lab's price: beyond 2 ticks
    ({"fair": 0.92}, "edge"),                          # fair − price − fee < 0
    ({"left_min": 1}, "too close to settlement"),
    ({"hyp": "W03", "fair": 0.935}, "edge"),          # W03 needs ≥ 1¢
])
def test_refusals(tmp_path, kw, reason):
    ex, _ = make(tmp_path)
    ex.handle(intent(**kw))
    assert not ex.st.positions
    assert reason in events(ex)[-1]["reason"]


def test_fair_value_is_refreshed_to_the_current_spot(tmp_path):
    """W03 decided at spot 77 900 with fair 0.945 (edge 1.5¢ ≥ 1¢). Spot is now 78 000 (+12.8 bp, towards the strike
    79 999.99): the YES wing we sell is now worth more, the refreshed fair is ≈ 0.936 and the edge 0.6¢ < 1¢."""
    from trader.executor import refresh_fair
    f1 = refresh_fair(0.945, "no", "KXBTCD-26NOV0213-T79999.99", 77900.0, 78000.0)
    assert 0.93 < f1 < 0.94
    assert refresh_fair(0.945, "no", "KXBTCD-26NOV0213-T79999.99", 77900.0, 77900.0) == pytest.approx(0.945)
    ex, _ = make(tmp_path)
    ex.handle(intent(hyp="W03", fair=0.945, spot=77900.0))
    assert "edge" in events(ex)[-1]["reason"]
    ex2, _ = make(tmp_path / "b")
    ex2.handle(intent(hyp="W03", fair=0.945, spot=78050.0))      # spot fell since the decision: edge grew
    assert "late|KXBTCD-26NOV0213" in ex2.st.positions


def test_one_position_per_rule_and_event_and_hour_limit(tmp_path):
    ex, _ = make(tmp_path, max_trades_per_hour=1)
    ex.handle(intent())
    ex.handle(intent())
    assert events(ex)[-1]["reason"] == "already in this slot and event"
    ex.handle(intent(hyp="W03", fair=0.96))
    assert events(ex)[-1]["reason"] == "max trades per hour"


def test_wide_spread_refused(tmp_path):
    ex, v = make(tmp_path)
    v._book = book(yes_bid=0.04, no_bid=0.91)         # YES 4¢ / 9¢
    ex.handle(intent())
    assert events(ex)[-1]["reason"].startswith("spread")


def test_kill_switch(tmp_path):
    ex, _ = make(tmp_path)
    (tmp_path / "KILL").write_text("x")
    ex.handle(intent())
    assert events(ex)[-1]["reason"] == "kill switch"


# ------------------------------------------------------------------ demo order flow, settlement, loss limit
def test_demo_ioc_fill_and_settlement_and_daily_limit(tmp_path):
    ex, v = make(tmp_path, mode="demo", result="no", daily_loss_limit_usd=0.5)
    ex.handle(intent())
    assert v.sent and v.sent[0]["tif"] == "immediate_or_cancel" and v.sent[0]["count"] == 1
    assert v.sent[0]["client_order_id"]
    p = ex.st.positions["early|KXBTCD-26NOV0213"]
    p["settle_ts"] = str(pd.Timestamp.now(tz="UTC") - pd.Timedelta(minutes=5))
    ex.settle()
    st = [e for e in events(ex) if e["what"] == "settled"][0]
    assert st["pnl"] == pytest.approx(1 - 0.92 - 0.01)
    # a losing settlement trips the daily limit
    ex.handle(intent(hyp="W03", fair=0.96))
    v.result = "yes"
    ex.st.positions["late|KXBTCD-26NOV0213"]["settle_ts"] = str(pd.Timestamp.now(tz="UTC") - pd.Timedelta(minutes=5))
    ex.settle()
    ex.handle(intent(hyp="W05", fair=0.96))
    assert events(ex)[-1]["reason"] == "daily loss limit"


def test_demo_unfilled_ioc_opens_nothing(tmp_path):
    ex, v = make(tmp_path, mode="demo", fill="none")
    ex.handle(intent())
    assert v.sent and not ex.st.positions


def test_maker_quote_then_cancel_with_partial_fill(tmp_path):
    ex, v = make(tmp_path, mode="demo", fill="partial", demo_contracts=3, maker_rules=["W05"])
    ex.handle(intent(kind="quote", hyp="W05", price=0.91, fair=0.94))   # bid for NO at 0.91 = sell YES at 9¢
    assert v.sent[0]["post_only"] and v.sent[0]["tif"] == "good_till_canceled"
    assert "late|KXBTCD-26NOV0213" in ex.st.resting
    ex.handle(intent(kind="cancel", hyp="W05"))
    assert v.cancelled and ex.st.positions["late|KXBTCD-26NOV0213"]["filled"] == 1


def test_post_only_never_crosses(tmp_path):
    ex, v = make(tmp_path, mode="demo", maker_rules=["W05"])
    ex.handle(intent(kind="quote", hyp="W05", price=0.92, fair=0.96))   # 0.92 = the NO ask: would take
    assert not v.sent and "cross" in events(ex)[-1]["reason"]


def test_intent_reader_skips_partial_last_line(tmp_path):
    ex, _ = make(tmp_path)
    p = tmp_path / "intents.jsonl"
    p.write_text(json.dumps(intent()) + "\n" + '{"id": "half')
    assert len(ex.read_intents()) == 1
    with open(p, "a") as f:
        f.write('-line"}\n')
    assert ex.read_intents() == [{"id": "half-line"}]


def test_lab_emits_intents(tmp_path, monkeypatch):
    from lab import intents
    from lab.core import Order
    monkeypatch.setattr(intents, "INTENTS", tmp_path / "i.jsonl")
    intents.SPOT = 78000.0
    o = Order(hyp="W04", kind="binary", venue="kalshi", instrument="KX-T1", side="no", price=0.92, fee=0.01,
              fair=0.95, edge=0.02, settle_ts="2026-11-02 14:00:00+00:00",
              info={"legs": [{"event": "KX"}], "wing_price": 0.08})
    intents.from_order(o, pd.Timestamp("2026-11-02 13:40", tz="UTC"))
    rec = json.loads((tmp_path / "i.jsonl").read_text())
    assert rec["kind"] == "take" and rec["fair_side"] == 0.95 and rec["spot"] == 78000.0 and rec["margin"] == 0.0


def test_maker_quotes_are_off_by_default(tmp_path):
    ex, v = make(tmp_path, mode="demo")
    ex.handle(intent(kind="quote", hyp="W05", price=0.91, fair=0.94))
    assert not v.sent and not ex.st.resting


def test_w03_and_w05_share_one_slot(tmp_path):
    """W05 bets on the same strike and side as W03: a second position in the same hour is refused."""
    ex, _ = make(tmp_path)
    ex.handle(intent(hyp="W03", fair=0.96))
    ex.handle(intent(hyp="W05", fair=0.96))
    assert events(ex)[-1]["reason"] == "already in this slot and event"
    ex.handle(intent(hyp="W04"))                                   # the other slot is free
    assert "early|KXBTCD-26NOV0213" in ex.st.positions


def test_book_sweep_stops_where_the_edge_ends(tmp_path):
    """Asks for NO at 0.92 (40 contracts) and 0.93 (400): $50 budget wants 54; 25% of level one = 10, then the
    second level has edge 0.95 − 0.93 − 0.01 = 1¢ ≥ 0 for W04 → 54 − 10 = 44 more, capped at 25% of 400 = 100."""
    ex, v = make(tmp_path, risk_per_trade_usd=50, max_contracts=100)
    v._book = Book("T", [(0.08, 40), (0.07, 400)], [(0.91, 500)], time.time())
    ex.handle(intent(price=0.93))
    p = ex.st.positions["early|KXBTCD-26NOV0213"]
    assert p["filled"] == 54 and p["vwap"] == pytest.approx((10 * 0.92 + 44 * 0.93) / 54, abs=1e-4)
    ex2, v2 = make(tmp_path / "b", risk_per_trade_usd=50, max_contracts=100)
    v2._book = Book("T", [(0.08, 40), (0.07, 400)], [(0.91, 500)], time.time())
    ex2.handle(intent(price=0.93, fair=0.94))                      # level two: 0.94 − 0.93 − 0.01 = 0 → still ok
    ex3, v3 = make(tmp_path / "c", risk_per_trade_usd=50, max_contracts=100)
    v3._book = Book("T", [(0.08, 40), (0.07, 400)], [(0.91, 500)], time.time())
    ex3.handle(intent(price=0.93, fair=0.939))                     # level two edge < 0: only level one, 10 contracts
    assert ex3.st.positions["early|KXBTCD-26NOV0213"]["filled"] == 10


def test_credentials_from_key_files(tmp_path, monkeypatch):
    import trader.settings as S
    for k in ("KALSHI_DEMO_KEY_ID", "KALSHI_DEMO_KEY_FILE", "KALSHI_KEY_ID", "KALSHI_KEY_FILE"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr(S, "KEYS", tmp_path)
    assert S.credentials("demo") == (None, None)
    (tmp_path / "demo_key_id.txt").write_text("abc-123\n")
    (tmp_path / "demo.pem").write_text("x")
    assert S.credentials("demo") == ("abc-123", str(tmp_path / "demo.pem"))
    assert S.credentials("live") == (None, None)                      # demo keys never serve live


def test_profiles_load_on_top_of_settings(tmp_path):
    from trader.settings import Settings
    live = Settings.load(tmp_path / "s.json", profile="live_1contract")
    assert live.mode == "live" and live.max_contracts == 1 and live.daily_loss_limit_usd == 3.0
    demo = Settings.load(tmp_path / "s.json", profile="demo_plumbing")
    assert demo.mode == "demo" and demo.demo_ignore_edge and demo.demo_contracts == 1
    assert Settings.load(tmp_path / "s.json").mode == "shadow"


def test_one_contract_profile_takes_a_thin_level(tmp_path):
    ex, v = make(tmp_path, risk_per_trade_usd=1.0, max_contracts=1)
    v._book = Book("T", [(0.08, 2)], [(0.91, 500)], time.time())       # only 2 contracts offered at 0.92
    ex.handle(intent())
    assert ex.st.positions["early|KXBTCD-26NOV0213"]["filled"] == 1


def test_scan_key_dir_finds_keys_under_any_name(tmp_path):
    from trader.settings import scan_key_dir
    d = tmp_path / "demo"
    d.mkdir()
    (d / "my-kalshi-download.txt").write_text("-----BEGIN RSA PRIVATE KEY-----\nabc\n-----END RSA PRIVATE KEY-----\n")
    (d / "id.txt").write_text("key id: 4b0c8a8e-1f1e-4c8a-9d2b-0a1b2c3d4e5f\n")
    kid, kf = scan_key_dir(d)
    assert kid == "4b0c8a8e-1f1e-4c8a-9d2b-0a1b2c3d4e5f" and kf.endswith("my-kalshi-download.txt")
    assert scan_key_dir(tmp_path / "prod") == (None, None)


def test_second_executor_for_the_same_mode_refuses_to_start(tmp_path):
    ex, v = make(tmp_path)
    (tmp_path / "lock_shadow.json").write_text(json.dumps({"pid": -1, "t": "now"}))   # another live process
    with pytest.raises(RuntimeError):
        Executor(ex.cfg, None, v, FakeSpot(), intents_path=tmp_path / "intents.jsonl", root=tmp_path)


def test_pays_a_tick_more_when_the_edge_still_clears(tmp_path):
    """Lab decided at 0.92, the ask is now 0.93: with slippage_ticks = 2 the order goes if fair − 0.93 − fee ≥ margin."""
    ex, v = make(tmp_path)
    v._book = book(yes_bid=0.07, no_bid=0.91)                      # NO ask 0.93
    ex.handle(intent(price=0.92, fair=0.95))
    assert ex.st.positions["early|KXBTCD-26NOV0213"]["price"] == pytest.approx(0.93)


def test_v2_order_mapping_and_cancel_routing():
    """V2 create-order wants the YES leg only; buy NO 0.92 = ask YES 0.08; cancel carries the ticker."""
    from trader.client import Kalshi, KalshiError
    k = Kalshi("demo")
    calls = []

    def fake(method, path, params=None, body=None, auth=True, retries=2):
        calls.append((method, path, params, body))
        if method == "POST":
            return {"order_id": "o1", "client_order_id": body["client_order_id"], "fill_count": "1.00",
                    "remaining_count": "0.00", "average_fill_price": "0.0800", "average_fee_paid": "0.01"}
        if method == "GET":
            raise KalshiError(404, "{}", method, path)
        return {"order_id": "o1", "reduced_by": "0.00"}
    k.request = fake
    o = k.create_order("KXBTCD-X", "no", 1, 0.92, tif="immediate_or_cancel", client_order_id="c1")
    m, path, _, body = calls[0]
    assert path == "/portfolio/events/orders" and body["side"] == "ask" and body["price"] == "0.0800"
    assert body["count"] == "1" and body["time_in_force"] == "immediate_or_cancel" and "expiration_time" not in body
    assert o["status"] == "executed" and o["fill_count_fp"] == "1.00" and o["average_fill_price_dollars"] == "0.9200"
    assert k.order("o1")["order_id"] == "o1"                       # GET 404 -> what the POST returned
    k.cancel("o1")
    assert calls[-1][1] == "/portfolio/events/orders/o1" and calls[-1][2] == {"market_ticker": "KXBTCD-X"}
    k.create_order("KXBTCD-X", "yes", 1, 0.01, tif="good_till_canceled", post_only=True, expiration_ts=99)
    b = calls[-1][3]
    assert b["side"] == "bid" and b["price"] == "0.0100" and b["post_only"] and b["expiration_time"] == 99
