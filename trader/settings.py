"""Executor settings: data/trader/settings.json (created from the defaults on first run) + environment variables.

Secrets never go into the repository:
    KALSHI_KEY_ID      the API key id (demo keys are created on demo.kalshi.co, production keys on kalshi.com)
    KALSHI_KEY_FILE    path to the private key .pem (kept only on the machine that trades)
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "trader"
INTENTS = ROOT / "data" / "lab" / "intents.jsonl"
PROFILES = Path(__file__).resolve().parent / "profiles"
KEYS = DATA / "keys"                   # git-ignored: <env>_key_id.txt and <env>.pem (env = demo | prod)


@dataclass
class Settings:
    mode: str = "shadow"                 # shadow (prod books, no orders) | demo (orders on demo) | live (real money)
    rules: list = field(default_factory=lambda: ["W03", "W04", "W05", "W06"])
    # maker quotes are OFF: on Jul–Oct candles a resting offer on a wing is lifted mostly when the wing is about to
    # pay (W03: 16.5% pay-off when lifted vs 4.8% when not; studies/wings_execution.py) — taking now earns more
    maker_rules: list = field(default_factory=list)
    # sizing
    risk_per_trade_usd: float = 5.0      # worst-case loss of one trade (= contracts × price paid)
    max_contracts: int = 10              # per order
    max_book_share: float = 0.25         # never take more than this share of the best level
    # limits
    max_open_positions: int = 6
    max_trades_per_hour: int = 2         # one strike per rule and hour; W03+W05 or W04+W06 share the same risk
    daily_loss_limit_usd: float = 15.0   # stop for the day below this realised + mark-to-settlement loss
    max_intent_age_s: float = 20.0       # older decisions are refused (rule 7)
    max_spot_move_bp: float = 20.0       # hard cap on the spot move since the lab's decision; smaller moves are
    #                                      priced in: the fair value is refreshed to the current spot (refresh_fair)
    quote_cancel_bp: float = 3.0         # cancel a resting maker quote when spot runs this far against it
    max_spread: float = 0.03             # skip markets wider than this
    slippage_ticks: int = 2              # pay up to this many cents above the lab's price, if the edge re-checked
    #                                      at that price (fair value refreshed to the current spot) still clears the margin
    min_minutes_left: float = 2.0        # no new orders this close to the settlement window
    # demo specifics
    demo_ignore_edge: bool = False       # demo books are not real prices: allow 1-contract plumbing tests anyway
    demo_contracts: int = 1
    # plumbing
    poll_s: float = 1.0
    spot_url: str = "https://api.exchange.coinbase.com/products/BTC-USD/ticker"

    @classmethod
    def load(cls, path: Path | None = None, profile: str | None = None) -> "Settings":
        """data/trader/settings.json, then (optionally) a named profile from trader/profiles/<name>.json on top."""
        path = path or DATA / "settings.json"
        DATA.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_text(json.dumps(asdict(cls()), indent=2), encoding="utf-8")
        raw = json.loads(path.read_text(encoding="utf-8"))
        if profile:
            raw.update(json.loads((PROFILES / f"{profile}.json").read_text(encoding="utf-8")))
        known = {k: v for k, v in raw.items() if k in cls.__dataclass_fields__ and not k.startswith("_")}
        return cls(**known)


_UUID = r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"


def scan_key_dir(folder: Path) -> tuple[str | None, str | None]:
    """Find a Kalshi key pair in a folder whatever the files are called (the site's download has its own name):
    the private key = the file whose text contains 'PRIVATE KEY'; the key id = a UUID found in key_id.txt, in any
    other small text file, or in a file name. Only reads files on the trading machine; nothing leaves it."""
    import re
    if not folder.is_dir():
        return None, None
    key_file, key_id = None, None
    files = sorted(p for p in folder.iterdir() if p.is_file() and p.stat().st_size < 20000)
    for p in files:
        try:
            text = p.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if "PRIVATE KEY" in text:
            key_file = key_file or str(p)
            m = re.search(_UUID, p.name)
            key_id = key_id or (m.group(0) if m else None)
            continue
        m = re.search(_UUID, text) or re.search(_UUID, p.name)
        if m and (key_id is None or p.name.lower().startswith("key_id")):
            key_id = m.group(0)
    return key_id, key_file


def credentials(mode: str) -> tuple[str | None, str | None]:
    """Key id and key file for the mode; demo and production keys are separate and never mixed.

    1. environment: KALSHI_DEMO_KEY_ID / KALSHI_DEMO_KEY_FILE (demo), KALSHI_KEY_ID / KALSHI_KEY_FILE (live)
    2. a folder: data/trader/keys/demo/ or data/trader/keys/prod/ with the downloaded private key under any name and
       the key id (in key_id.txt, any text file or the file name) — see scan_key_dir
    3. legacy names: data/trader/keys/demo_key_id.txt + demo.pem (prod_key_id.txt + prod.pem)
    """
    env = "demo" if mode == "demo" else "prod"
    if mode == "demo":
        kid, kf = os.environ.get("KALSHI_DEMO_KEY_ID"), os.environ.get("KALSHI_DEMO_KEY_FILE")
    else:
        kid, kf = os.environ.get("KALSHI_KEY_ID"), os.environ.get("KALSHI_KEY_FILE")
    sid, sf = scan_key_dir(KEYS / env)
    kid, kf = kid or sid, kf or sf
    if not kid and (KEYS / f"{env}_key_id.txt").exists():
        kid = (KEYS / f"{env}_key_id.txt").read_text(encoding="utf-8").strip()
    if not kf and (KEYS / f"{env}.pem").exists():
        kf = str(KEYS / f"{env}.pem")
    return kid, kf
