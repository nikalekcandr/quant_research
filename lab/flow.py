"""Real-time trade flow and liquidations of the BTC perpetual (websocket), in 15-second buckets.

Binance USDT-M combined stream btcusdt@aggTrade / btcusdt@forceOrder; if Binance futures are unreachable from this
network, Bybit's public linear stream (publicTrade.BTCUSDT, allLiquidation.BTCUSDT). Needs `websocket-client`
(installed on first use). Buckets go to data/lab/flow/YYYY-MM-DD.csv and stay 6 hours in memory.
"""
from __future__ import annotations

import json
import subprocess
import sys
import threading
import time
from collections import deque

import pandas as pd

from lab.core import LAB, append_rows, day_file, log, log_exc, utcnow

FLOW = LAB / "flow"
BUCKET_S = 15


def _ws_module():
    try:
        import websocket  # noqa: F401
        return websocket
    except ImportError:
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", "websocket-client"], timeout=180)
            import websocket  # noqa: F401
            return websocket
        except Exception as e:
            log(f"flow: websocket-client not available ({e!r}); flow rules stay silent")
            return None


class Flow:
    def __init__(self):
        self.lock = threading.Lock()
        self.cur = None
        self.buckets = deque(maxlen=6 * 3600 // BUCKET_S)
        self.source = None
        self.last_msg = 0.0

    # -------------------------------------------------------------- aggregation
    def _bucket(self, ts: float):
        b = int(ts // BUCKET_S) * BUCKET_S
        if self.cur is None or self.cur["t"] != b:
            if self.cur is not None:
                done = dict(self.cur)
                done["ts"] = pd.Timestamp(done.pop("t"), unit="s", tz="UTC")
                done["source"] = self.source
                self.buckets.append(done)
                try:
                    append_rows(day_file(FLOW, done["ts"]), [done])
                except Exception:
                    pass
            self.cur = {"t": b, "buy_usd": 0.0, "sell_usd": 0.0, "n": 0, "liq_long_usd": 0.0, "liq_short_usd": 0.0,
                        "last": None}
        return self.cur

    def trade(self, ts, price, qty, seller_aggressor):
        with self.lock:
            c = self._bucket(ts)
            usd = price * qty
            c["sell_usd" if seller_aggressor else "buy_usd"] += usd
            c["n"] += 1
            c["last"] = price
            self.last_msg = time.time()

    def liquidation(self, ts, side_sell, price, qty):
        """side_sell: the forced order SELLS (a long was liquidated)."""
        with self.lock:
            c = self._bucket(ts)
            c["liq_long_usd" if side_sell else "liq_short_usd"] += price * qty
            self.last_msg = time.time()

    def frame(self, minutes: float = 360) -> pd.DataFrame:
        with self.lock:
            rows = list(self.buckets)
        d = pd.DataFrame(rows)
        if d.empty:
            return d
        return d[d["ts"] > utcnow() - pd.Timedelta(minutes=minutes)]

    # -------------------------------------------------------------- websocket
    def run(self, stop: threading.Event):
        ws_mod = _ws_module()
        if ws_mod is None:
            return
        attempts = [("binance", "wss://fstream.binance.com/stream?streams=btcusdt@aggTrade/btcusdt@forceOrder"),
                    ("bybit", "wss://stream.bybit.com/v5/public/linear")]
        i = 0
        while not stop.is_set():
            name, url = attempts[i % 2]
            self.source = name
            t0 = time.time()
            try:
                ws = ws_mod.create_connection(url, timeout=20)
                if name == "bybit":
                    ws.send(json.dumps({"op": "subscribe", "args": ["publicTrade.BTCUSDT", "allLiquidation.BTCUSDT"]}))
                log(f"flow: connected to {name}")
                last_ping = time.time()
                while not stop.is_set():
                    msg = ws.recv()
                    if not msg:
                        break
                    self._handle(name, json.loads(msg))
                    if name == "bybit" and time.time() - last_ping > 20:
                        ws.send(json.dumps({"op": "ping"}))
                        last_ping = time.time()
                ws.close()
            except Exception as e:
                log(f"flow: {name} stream dropped ({str(e)[:120]})")
            # a connection that died within a minute counts as a failure of this venue: try the other one
            if time.time() - t0 < 60:
                i += 1
            time.sleep(5)

    def _handle(self, name, m):
        try:
            if name == "binance":
                d = m.get("data") or {}
                if d.get("e") == "aggTrade":
                    self.trade(d["T"] / 1000, float(d["p"]), float(d["q"]), bool(d["m"]))
                elif d.get("e") == "forceOrder":
                    o = d["o"]
                    self.liquidation(o["T"] / 1000, o["S"] == "SELL", float(o.get("ap") or o["p"]), float(o["q"]))
            else:
                topic = m.get("topic", "")
                if topic.startswith("publicTrade"):
                    for t in m.get("data") or []:
                        self.trade(t["T"] / 1000, float(t["p"]), float(t["v"]), t["S"] == "Sell")
                elif topic.startswith("allLiquidation") or topic.startswith("liquidation"):
                    for t in (m.get("data") if isinstance(m.get("data"), list) else [m.get("data")]):
                        if t:
                            # Bybit: S = side of the liquidated POSITION ("Buy" = a long was liquidated)
                            self.liquidation(t["T"] / 1000, t["S"] == "Buy", float(t["p"]), float(t["v"]))
        except Exception:
            log_exc("flow message")
