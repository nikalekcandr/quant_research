"""Project-wide settings. Every script reads its parameters from here.

Change dates and paths here, not inside the modules.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RAW = DATA / "raw"            # untouched API responses and archives
INTERIM = DATA / "interim"    # cleaned tables built from RAW
LIVE = DATA / "live"          # snapshots written by the live logger
MANUAL = DATA / "manual"      # hand-maintained inputs (macro calendar)
RESULTS = ROOT / "results"
TABLES = RESULTS / "tables"
FIGURES = RESULTS / "figures"
FORECASTS_LIVE = ROOT / "forecasts_live"   # committed to git before outcomes are known
REPORTS = ROOT / "reports"

# ---------------------------------------------------------------- sample
# Binance 1-minute candles and Deribit DVOL: long history for HAR estimation.
BINANCE_START = "2023-01-01"
DVOL_START = "2023-01-01"
# Polymarket daily "Bitcoin above ___ on <date>" ladders (noon ET) start in Sept 2025.
POLYMARKET_START = "2025-09-01"
# Kalshi KXBTCD hourly ladders. Each hourly event costs ~2 requests (~50 requests per day of history).
KALSHI_START = "2026-01-01"
KALSHI_HOUR_STEP = 1          # 1 = every hour, 2 = every second hour, ...
# Deribit option trades are downloaded only in a window before each daily snapshot.
DERIBIT_START = "2025-06-01"
DERIBIT_WINDOW_MIN = 60       # minutes of trades before the snapshot used for ATM IV

# Pseudo-out-of-sample evaluation starts here (models are estimated on data before each date).
EVAL_START = "2026-01-01"
# True out-of-sample period: forecasts committed to git before the outcome.
LIVE_START = "2026-10-09"

# ---------------------------------------------------------------- forecast design
# Horizons, in hours. "pm" ladders settle at 12:00 ET; "kx" ladders settle at the top of each hour.
HORIZONS = {
    "kx1h": {"source": "kalshi", "hours": 55 / 60},  # snapshot 5 minutes after the hourly market opens
    "pm24h": {"source": "polymarket", "hours": 24},
    "pm6d": {"source": "polymarket", "hours": 144},
}

# Ladder fitting
CENTRAL_BAND = (0.05, 0.95)   # strikes used for the central lognormal fit
CENTRAL_BAND_WIDE = (0.02, 0.98)  # fallback when fewer than MIN_STRIKES fall in CENTRAL_BAND
TAIL_BAND = (0.002, 0.998)    # strikes used for the Student-t fit
MIN_STRIKES = 3
PM_MAX_STALENESS_MIN = 30     # ignore a Polymarket price older than this at the snapshot

# Rolling estimation windows (number of forecast occasions)
ROLL_WINDOW = {"kx1h": 24 * 90, "pm24h": 365, "pm6d": 365}
MIN_TRAIN = {"kx1h": 24 * 30, "pm24h": 120, "pm6d": 120}
REFIT_EVERY = {"kx1h": 24, "pm24h": 1, "pm6d": 1}
# Models that use prediction-market features are estimated only where those features exist.
MIN_TRAIN_MARKET = {"kx1h": 24 * 14, "pm24h": 60, "pm6d": 60}

# Trading rule
EDGE_THRESHOLDS = [0.02, 0.05]
PM_HALF_SPREAD = 0.005        # assumed half-spread for Polymarket history without books (5 tenths of a cent)
KALSHI_FEE_RATE = 0.07        # taker fee = rate * P * (1 - P) per contract

# Network
USER_AGENT = "btcvol-course-project/1.0 (academic research)"
HTTP_TIMEOUT = 20

# ---------------------------------------------------------------- v6: frozen specification and new data
# The model specification (features, windows, priors, scoring) was fixed on 15 October 2026 with data up to
# 12 October. Everything from NEW_DATA_START on is data the specification never saw: the honest out-of-sample
# check. oos_report.py verifies that the model code is byte-identical to the frozen version (FROZEN_CODE_SHA256).
SPEC_FROZEN = "2026-10-15"
NEW_DATA_START = "2026-10-13"
FROZEN_CODE_SHA256 = {
    "src/models.py": "b2032433df42cf7c423e1ffa96008dad021a1db72b361d479aa2de655d27fcf3",
    # build.py: one documented change on 28 Oct — the "every second strike" rule of the Kalshi ladder keeps every
    # strike when Kalshi listed them $250+ apart (before 18 March 2026). For $100 spacing (all of July-October,
    # the whole evaluation of v5 and every new or live ladder) the selection is identical; v5 tables reproduce exactly.
    "src/build.py": "fdb5a6c94dd9f425b15d5c009d5e1afdb921308d23346e465da390e8001222a1",
    "src/evaluate.py": "9d3099ef10815d5f51c902174e24200cdd4b63a8753678ed63416f6a07c719e4",
    "src/measure/ladder.py": "6988dc20d85888195250dcaf6629352dfe17db72831ebdc33b492a3111f8c7c0",
    "src/measure/realized.py": "e224086ce0e0e28b4c08515f7220d2e47be301b45e594c07376163d0b7fc02e0",
    "src/measure/deribit_iv.py": "97c02b077650330f7eb1e24fbc280faed02ed14ee590f0c78f5e9290696e087a",
}

# Binance REST hosts, tried in order. api.binance.com answers HTTP 451 from some networks; data-api.binance.vision
# is Binance's public market-data mirror with the same /api/v3 endpoints.
BINANCE_HOSTS = ["https://api.binance.com", "https://data-api.binance.vision", "https://api1.binance.com",
                 "https://api2.binance.com", "https://api3.binance.com"]

# Earliest dates found by `python backfill.py --probe` are stored in data/manual/starts.json and override the
# defaults above, so that every step downloads and uses all the history that exists.
_STARTS = MANUAL / "starts.json"
if _STARTS.exists():
    import json as _json
    for _k, _v in _json.loads(_STARTS.read_text(encoding="utf-8")).items():
        if _k in ("POLYMARKET_START", "KALSHI_START", "DERIBIT_START", "EVAL_START") and _v:
            globals()[_k] = _v
