"""Kalshi Trade API v2 client: request signing, retries, rate limiting, typed helpers.

Auth (docs.kalshi.com/getting_started/api_keys): every private request carries
    KALSHI-ACCESS-KEY        key id
    KALSHI-ACCESS-TIMESTAMP  milliseconds since the epoch
    KALSHI-ACCESS-SIGNATURE  base64( sign(timestamp + METHOD + path_without_query) )
with RSA-PSS / SHA-256 / MGF1-SHA-256 / salt = 32 bytes, or Ed25519, depending on the key type. The path includes
the /trade-api/v2 prefix. Prices on the wire are fixed-point dollar strings ("0.9100"), counts are integers on
orders and fixed-point strings ("10.00") on fills and positions.
"""
from __future__ import annotations

import base64
import json
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlencode, urlparse

import requests

BASES = {
    "demo": "https://external-api.demo.kalshi.co/trade-api/v2",
    "prod": "https://api.elections.kalshi.com/trade-api/v2",
}


class KalshiError(RuntimeError):
    def __init__(self, status: int, body: str, method: str, path: str):
        super().__init__(f"HTTP {status} {method} {path}: {body[:300]}")
        self.status, self.body = status, body


def load_private_key(path: str | Path):
    """PEM, or a bare base64 PKCS#8 body without the BEGIN/END lines (some downloads and copy-pastes lose them)."""
    import base64
    from cryptography.hazmat.primitives import serialization
    raw = Path(path).read_bytes()
    if b"PRIVATE KEY" in raw:
        return serialization.load_pem_private_key(raw, password=None)
    return serialization.load_der_private_key(base64.b64decode(b"".join(raw.split())), password=None)


def sign(key, message: str) -> str:
    """base64 signature of the message with an RSA (PSS, SHA-256, salt = digest length) or Ed25519 key."""
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.asymmetric import padding
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    data = message.encode()
    if isinstance(key, Ed25519PrivateKey):
        sig = key.sign(data)
    else:
        sig = key.sign(data, padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.DIGEST_LENGTH),
                       hashes.SHA256())
    return base64.b64encode(sig).decode()


class TokenBucket:
    """Client-side rate limit (Kalshi Basic tier: 200 read / 100 write tokens per second; we stay far below)."""

    def __init__(self, rate: float, burst: float):
        self.rate, self.burst, self.tokens, self.t = rate, burst, burst, time.monotonic()
        self.lock = threading.Lock()

    def take(self, n: float = 1.0):
        while True:
            with self.lock:
                now = time.monotonic()
                self.tokens = min(self.burst, self.tokens + (now - self.t) * self.rate)
                self.t = now
                if self.tokens >= n:
                    self.tokens -= n
                    return
                wait = (n - self.tokens) / self.rate
            time.sleep(wait)


def dollars(x) -> float:
    """'0.9100' / 0.91 / None -> float (NaN for missing)."""
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def price_str(p: float) -> str:
    """Price in dollars -> the fixed-point string Kalshi expects, whole cents only (the BTC ladders tick in 1¢)."""
    c = int(round(p * 100))
    if not 1 <= c <= 99:
        raise ValueError(f"price {p} outside 0.01..0.99")
    return f"{c / 100:.4f}"


@dataclass
class Book:
    """Top of book for one market. Kalshi returns bids only: YES bids and NO bids; YES ask = 1 − best NO bid."""
    ticker: str
    yes_bids: list            # [(price, size)] best first
    no_bids: list
    t: float                  # time.time() when received

    @property
    def yes_bid(self):
        return self.yes_bids[0] if self.yes_bids else (float("nan"), 0.0)

    @property
    def no_bid(self):
        return self.no_bids[0] if self.no_bids else (float("nan"), 0.0)

    def ask(self, side: str):
        """(price, size) to BUY `side` now: the best bid of the other side, mirrored."""
        p, q = self.no_bid if side == "yes" else self.yes_bid
        return (round(1 - p, 4), q) if p == p else (float("nan"), 0.0)

    def bid(self, side: str):
        return self.yes_bid if side == "yes" else self.no_bid

    def asks(self, side: str):
        lv = self.no_bids if side == "yes" else self.yes_bids
        return [(round(1 - p, 4), q) for p, q in lv]


def parse_book(ticker: str, res: dict) -> Book:
    ob = res.get("orderbook_fp") or res.get("orderbook") or {}

    def lv(x, legacy_cents=False):
        out = []
        for row in x or []:
            p, q = dollars(row[0]), dollars(row[1])
            if legacy_cents:
                p /= 100
            if p > 0 and q > 0:                       # rule 3: an empty side may come back as 0
                out.append((round(p, 4), q))
        return sorted(out, key=lambda r: -r[0])

    if "yes_dollars" in ob or "no_dollars" in ob:
        y, n = lv(ob.get("yes_dollars")), lv(ob.get("no_dollars"))
    else:                                            # legacy format: integer cents
        y, n = lv(ob.get("yes"), True), lv(ob.get("no"), True)
    return Book(ticker, y, n, time.time())


class Kalshi:
    def __init__(self, env: str = "demo", key_id: str | None = None, key_file: str | None = None,
                 base: str | None = None, timeout: float = 5.0):
        if env not in BASES and not base:
            raise ValueError(f"env must be one of {list(BASES)}")
        self.env = env
        self.base = (base or BASES[env]).rstrip("/")
        self.prefix = urlparse(self.base).path            # /trade-api/v2
        self.key_id = key_id
        self.key = load_private_key(key_file) if key_file else None
        self.s = requests.Session()
        self.timeout = timeout
        self.read_bucket = TokenBucket(rate=10, burst=20)
        self.write_bucket = TokenBucket(rate=5, burst=10)
        self.clock_offset_ms = 0                          # server − local, from the Date header

    # -------------------------------------------------------------- transport
    def _headers(self, method: str, path: str) -> dict:
        if self.key is None or not self.key_id:
            return {}
        ts = str(int(time.time() * 1000 + self.clock_offset_ms))
        return {"KALSHI-ACCESS-KEY": self.key_id, "KALSHI-ACCESS-TIMESTAMP": ts,
                "KALSHI-ACCESS-SIGNATURE": sign(self.key, ts + method + self.prefix + path)}

    def request(self, method: str, path: str, params: dict | None = None, body: dict | None = None,
                auth: bool = True, retries: int = 2):
        bucket = self.read_bucket if method == "GET" else self.write_bucket
        url = self.base + path + (("?" + urlencode({k: v for k, v in params.items() if v is not None})) if params else "")
        last = None
        for attempt in range(retries + 1):
            bucket.take()
            h = {"Content-Type": "application/json", "Accept": "application/json"}
            if auth:
                h.update(self._headers(method, path))
            try:
                r = self.s.request(method, url, headers=h, data=json.dumps(body) if body is not None else None,
                                   timeout=self.timeout)
            except requests.RequestException as e:
                last = e
                time.sleep(0.5 * (attempt + 1))
                continue
            if r.status_code == 429 or r.status_code >= 500:
                last = KalshiError(r.status_code, r.text, method, path)
                time.sleep(1.0 * (attempt + 1))
                continue
            if r.status_code >= 400:
                raise KalshiError(r.status_code, r.text, method, path)
            return r.json() if r.text else {}
        raise last if isinstance(last, Exception) else RuntimeError("request failed")

    # -------------------------------------------------------------- public market data
    def exchange_status(self) -> dict:
        return self.request("GET", "/exchange/status", auth=False)

    def server_clock_check(self) -> float:
        """Seconds local clock is behind the server (from the HTTP Date header); stored as an offset for signing."""
        r = self.s.get(self.base + "/exchange/status", timeout=self.timeout)
        from email.utils import parsedate_to_datetime
        srv = parsedate_to_datetime(r.headers["Date"]).timestamp()
        off = srv - time.time()
        self.clock_offset_ms = int(off * 1000) if abs(off) > 1.0 else 0
        return off

    def market(self, ticker: str) -> dict:
        return self.request("GET", f"/markets/{ticker}", auth=False).get("market", {})

    def markets(self, **params) -> list:
        out, cursor = [], None
        for _ in range(20):
            res = self.request("GET", "/markets", {**params, "cursor": cursor, "limit": 1000}, auth=False)
            out += res.get("markets", [])
            cursor = res.get("cursor")
            if not cursor:
                break
        return out

    def book(self, ticker: str, depth: int = 10) -> Book:
        return parse_book(ticker, self.request("GET", f"/markets/{ticker}/orderbook", {"depth": depth}, auth=False))

    # -------------------------------------------------------------- portfolio (signed)
    def balance(self) -> dict:
        """Cash and portfolio value in dollars (accepts both the cents and the *_dollars field styles)."""
        r = self.request("GET", "/portfolio/balance")

        def usd(name):
            if r.get(f"{name}_dollars") is not None:
                return dollars(r[f"{name}_dollars"])
            return dollars(r.get(name)) / 100 if r.get(name) is not None else float("nan")
        return {"cash": usd("balance"), "portfolio_value": usd("portfolio_value"), "raw": r}

    def positions(self, **params) -> list:
        out, cursor = [], None
        for _ in range(20):
            res = self.request("GET", "/portfolio/positions", {**params, "cursor": cursor, "limit": 1000})
            out += res.get("market_positions", [])
            cursor = res.get("cursor")
            if not cursor:
                break
        return out

    def orders(self, **params) -> list:
        out, cursor = [], None
        for _ in range(20):
            res = self.request("GET", "/portfolio/orders", {**params, "cursor": cursor, "limit": 1000})
            out += res.get("orders", [])
            cursor = res.get("cursor")
            if not cursor:
                break
        return out

    def order(self, order_id: str) -> dict:
        return self.request("GET", f"/portfolio/orders/{order_id}").get("order", {})

    def fills(self, **params) -> list:
        out, cursor = [], None
        for _ in range(20):
            res = self.request("GET", "/portfolio/fills", {**params, "cursor": cursor, "limit": 1000})
            out += res.get("fills", [])
            cursor = res.get("cursor")
            if not cursor:
                break
        return out

    def create_order(self, ticker: str, side: str, count: int, price: float, *, action: str = "buy",
                     tif: str = "immediate_or_cancel", post_only: bool = False, expiration_ts: int | None = None,
                     client_order_id: str | None = None, reduce_only: bool = False) -> dict:
        """Limit order. tif: immediate_or_cancel (taker, never worse than `price`), good_till_canceled (with
        post_only=True: maker; rejected instead of crossing), fill_or_kill."""
        if side not in ("yes", "no") or action not in ("buy", "sell") or count < 1:
            raise ValueError("bad order")
        body = {"ticker": ticker, "side": side, "action": action, "count": int(count), "type": "limit",
                f"{side}_price_dollars": price_str(price), "client_order_id": client_order_id or str(uuid.uuid4()),
                "cancel_order_on_pause": True, "self_trade_prevention_type": "taker_at_cross"}
        if tif != "good_till_canceled":
            body["time_in_force"] = tif
        if post_only:
            body["post_only"] = True
        if expiration_ts:
            body["expiration_ts"] = int(expiration_ts)
        if reduce_only:
            body["reduce_only"] = True
        return self.request("POST", "/portfolio/orders", body=body, retries=0).get("order", {})

    def cancel(self, order_id: str) -> dict:
        return self.request("DELETE", f"/portfolio/orders/{order_id}", retries=1)

    def cancel_all(self) -> int:
        n = 0
        for o in self.orders(status="resting"):
            try:
                self.cancel(o["order_id"])
                n += 1
            except KalshiError:
                pass
        return n


def fill_count(order: dict) -> float:
    for k in ("fill_count_fp", "fill_count"):
        if order.get(k) is not None:
            return dollars(order[k])
    return 0.0
