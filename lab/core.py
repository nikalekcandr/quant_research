"""Paths, logging, the hash-chained paper-trading ledger, open positions, fee models."""
from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import time
import traceback
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import config
from src.utils import Http

FROZEN = "2026-10-29"                     # rules of every hypothesis fixed on this date, before live outcomes
LAB = config.DATA / "lab"
TICKS = LAB / "ticks"                     # one wide row per minute: quotes, funding, OI, flows of ~7 venues
FAST = LAB / "fast"                       # 15-second mids of Binance / Coinbase / Bybit (lead-lag)
MARKETS = LAB / "markets"                 # every minute: quotes of short-dated Polymarket / Kalshi BTC markets
LEDGER = LAB / "ledger.csv"               # append-only: OPEN / CLOSE / INSTANT rows, SHA-256 chained
POSITIONS = LAB / "positions.json"
ZOO = LAB / "volzoo.csv"                  # volatility forecasts of the model zoo (scored when realised)
HEARTBEAT = LAB / "heartbeat.json"
REPORT_MD = config.REPORTS / "lab.md"
REPORT_HTML = config.REPORTS / "lab.html"
RESULTS = config.RESULTS / "lab"

PERP_FEE = 0.0005          # Binance USDT-M taker 0.05% per side
PERP_FEE_MAKER = 0.0002
OPTION_FEE_RATE = 0.0003   # Deribit / OKX / Bybit / Binance options taker: 0.03% of the underlying ...
OPTION_FEE_CAP = 0.125     # ... capped at 12.5% of the option price


def _jd(o):
    """JSON fallback: numpy numbers as numbers (np.int64 is not a Python int), everything else as text."""
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return str(o)


def code_version() -> str:
    """Hash of the lab's rule files: every trade records which version of the rules produced it."""
    h = hashlib.sha256()
    for f in sorted((config.ROOT / "lab").glob("*.py")):
        h.update(f.read_bytes())
    return h.hexdigest()[:12]


# ------------------------------------------------------------------ logging (one file per day, rotated)
def log(msg: str):
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    try:
        print(line, flush=True)
    except Exception:
        pass
    try:
        p = config.ROOT / "logs" / f"lab_{datetime.now():%Y%m%d}.log"
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


def log_exc(where: str):
    log(f"!!! {where}: {traceback.format_exc(limit=6)}")


def utcnow() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC")


def lab_http() -> Http:
    """Short retries: a slow venue must not stall the minute loop."""
    h = Http(min_interval=0.04, retries=2)
    return h


def atomic_write(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def append_rows(path: Path, rows: list[dict]):
    """Append dict rows to a daily CSV; new columns are allowed (the header is rewritten when they appear)."""
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            header = f.readline().strip().split(",")
        new = [c for c in df.columns if c not in header]
        if new:   # rare: a source added a field -> rewrite the file with the union of columns
            old = pd.read_csv(path)
            pd.concat([old, df], ignore_index=True).to_csv(path, index=False)
            return
        df.reindex(columns=header).to_csv(path, mode="a", header=False, index=False)
    else:
        df.to_csv(path, index=False)


def day_file(folder: Path, ts: pd.Timestamp) -> Path:
    return folder / f"{ts:%Y-%m-%d}.csv"


# ------------------------------------------------------------------ fees
def kalshi_fee(p: float) -> float:
    """Kalshi taker fee per contract: 0.07 * P * (1 - P), rounded up to the cent."""
    return math.ceil(round(config.KALSHI_FEE_RATE * p * (1 - p) * 100, 6)) / 100


def pm_fee(p: float, rate: float = 0.07) -> float:
    """Polymarket crypto_fees_v2 (taker only): rate * P * (1 - P) per share."""
    return rate * p * (1 - p)


def option_fee(price_usd: float, spot: float) -> float:
    return min(OPTION_FEE_RATE * spot, OPTION_FEE_CAP * max(price_usd, 0.0))


# ------------------------------------------------------------------ ledger
LEDGER_COLS = ["event", "ts", "id", "hyp", "family", "venue", "instrument", "side", "qty", "price", "fee",
               "fair", "edge", "exit_ts", "exit_price", "pnl", "pnl_gross", "unit", "info", "code", "prev", "hash"]


def _last_hash() -> str:
    if not LEDGER.exists():
        return "GENESIS"
    with open(LEDGER, "rb") as f:
        f.seek(0, 2)
        size = f.tell()
        f.seek(max(0, size - 4096))
        tail = f.read().decode("utf-8", "ignore").strip().splitlines()
    if len(tail) < 1 or tail[-1].startswith("event,"):
        return "GENESIS"
    return tail[-1].rsplit(",", 1)[-1]


def ledger_write(row: dict):
    row = {k: row.get(k) for k in LEDGER_COLS}
    row["prev"] = _last_hash()
    body = json.dumps({k: (None if row[k] is None else str(row[k])) for k in LEDGER_COLS if k != "hash"}, sort_keys=True)
    row["hash"] = hashlib.sha256(body.encode()).hexdigest()[:20]
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    new = not LEDGER.exists()
    pd.DataFrame([row], columns=LEDGER_COLS).to_csv(LEDGER, mode="a", header=new, index=False)


def read_ledger() -> pd.DataFrame:
    if not LEDGER.exists():
        return pd.DataFrame(columns=LEDGER_COLS)
    d = pd.read_csv(LEDGER, dtype={"id": str, "info": str})
    for c in ("ts", "exit_ts"):
        d[c] = pd.to_datetime(d[c], utc=True, errors="coerce")
    return d


@dataclass
class Order:
    """A paper order. kind: 'perp' (closed after hold_min at the quotes of that moment), 'binary' (held to the
    venue's settlement), 'instant' (an arbitrage whose P&L is locked at entry)."""
    hyp: str
    kind: str
    venue: str
    instrument: str
    side: str                 # 'long' / 'short' (perp), 'yes' / 'no' (binary), 'arb'
    price: float              # entry price paid (ask for buys, bid for sells), already the taker side
    qty: float = 1.0
    fee: float = 0.0
    fair: float | None = None
    edge: float | None = None
    hold_min: float | None = None   # perp
    settle_ts: str | None = None    # binary
    pnl: float | None = None        # instant
    pnl_gross: float | None = None
    unit: str = ""
    key: str | None = None          # de-duplication key: one position per key
    info: dict = field(default_factory=dict)


class Book:
    """Open positions (JSON on disk) + ledger writes."""

    def __init__(self):
        self.pos: dict = {}
        self.keys: set = set()
        try:
            data = json.loads(POSITIONS.read_text(encoding="utf-8"))
            self.pos = data.get("open", {})
            self.keys = set(data.get("keys", []))
        except Exception:
            pass
        self.code = code_version()

    def save(self):
        keys = sorted(self.keys)[-20000:]
        atomic_write(POSITIONS, json.dumps({"open": self.pos, "keys": keys}, default=_jd))

    def has_key(self, key: str) -> bool:
        return key in self.keys

    def open_count(self, hyp: str) -> int:
        return sum(1 for p in self.pos.values() if p["hyp"] == hyp)

    def submit(self, o: Order, now: pd.Timestamp, family: str):
        key = o.key or f"{o.hyp}|{o.instrument}|{o.side}"
        if key in self.keys:
            return False
        self.keys.add(key)
        pid = hashlib.sha1(f"{key}|{now}".encode()).hexdigest()[:12]
        info = json.dumps(o.info, default=_jd, separators=(",", ":"))
        if o.kind == "instant":
            ledger_write({"event": "INSTANT", "ts": now, "id": pid, "hyp": o.hyp, "family": family, "venue": o.venue,
                          "instrument": o.instrument, "side": o.side, "qty": o.qty, "price": o.price, "fee": o.fee,
                          "fair": o.fair, "edge": o.edge, "exit_ts": now, "pnl": o.pnl, "pnl_gross": o.pnl_gross,
                          "unit": o.unit, "info": info, "code": self.code})
            return True
        p = asdict(o)
        p.update({"id": pid, "family": family, "open_ts": str(now)})
        if o.kind in ("perp", "fastperp", "perp_pair"):
            p["exit_due"] = str(now + pd.Timedelta(minutes=o.hold_min or 60))
        self.pos[pid] = p
        ledger_write({"event": "OPEN", "ts": now, "id": pid, "hyp": o.hyp, "family": family, "venue": o.venue,
                      "instrument": o.instrument, "side": o.side, "qty": o.qty, "price": o.price, "fee": o.fee,
                      "fair": o.fair, "edge": o.edge, "unit": o.unit, "info": info, "code": self.code})
        return True

    def close(self, pid: str, now: pd.Timestamp, exit_price: float, pnl: float, pnl_gross: float, extra: dict | None = None):
        p = self.pos.pop(pid, None)
        if p is None:
            return
        info = json.dumps({**(p.get("info") or {}), **(extra or {})}, default=_jd, separators=(",", ":"))
        ledger_write({"event": "CLOSE", "ts": pd.Timestamp(p["open_ts"]), "id": pid, "hyp": p["hyp"], "family": p["family"],
                      "venue": p["venue"], "instrument": p["instrument"], "side": p["side"], "qty": p["qty"],
                      "price": p["price"], "fee": p["fee"], "fair": p.get("fair"), "edge": p.get("edge"),
                      "exit_ts": now, "exit_price": exit_price, "pnl": pnl, "pnl_gross": pnl_gross,
                      "unit": p.get("unit"), "info": info, "code": self.code})


def verify_ledger() -> tuple[bool, int]:
    d = pd.read_csv(LEDGER, dtype=str, keep_default_na=False) if LEDGER.exists() else pd.DataFrame()
    prev = "GENESIS"
    for i, r in d.iterrows():
        row = {k: (None if r[k] == "" else r[k]) for k in LEDGER_COLS}
        if row["prev"] != prev:
            return False, i
        body = json.dumps({k: row[k] for k in LEDGER_COLS if k != "hash"}, sort_keys=True)
        # values were stringified when hashed; '' and None both mean empty
        prev = row["hash"]
    return True, len(d)
