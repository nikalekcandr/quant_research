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
    slippage_ticks: int = 0              # extra cents allowed above the lab's price on a taker order
    min_minutes_left: float = 2.0        # no new orders this close to the settlement window
    # demo specifics
    demo_ignore_edge: bool = False       # demo books are not real prices: allow 1-contract plumbing tests anyway
    demo_contracts: int = 1
    # plumbing
    poll_s: float = 1.0
    spot_url: str = "https://api.exchange.coinbase.com/products/BTC-USD/ticker"

    @classmethod
    def load(cls, path: Path | None = None) -> "Settings":
        path = path or DATA / "settings.json"
        DATA.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_text(json.dumps(asdict(cls()), indent=2), encoding="utf-8")
        raw = json.loads(path.read_text(encoding="utf-8"))
        known = {k: v for k, v in raw.items() if k in cls.__dataclass_fields__}
        return cls(**known)


def credentials(mode: str) -> tuple[str | None, str | None]:
    """Key id and key file for the mode; demo and production keys are separate (KALSHI_DEMO_* overrides)."""
    if mode == "demo":
        return (os.environ.get("KALSHI_DEMO_KEY_ID") or os.environ.get("KALSHI_KEY_ID"),
                os.environ.get("KALSHI_DEMO_KEY_FILE") or os.environ.get("KALSHI_KEY_FILE"))
    return os.environ.get("KALSHI_KEY_ID"), os.environ.get("KALSHI_KEY_FILE")
