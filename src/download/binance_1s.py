"""Binance BTC/USDT 1-second candles: the fine grid for settlement and execution work.

Why it exists at all: the hourly Kalshi events settle on the average of 60 BRTI prints in the last minute, and
the wing strategy lives and dies on what happens in the last minutes before expiry. One-minute candles cannot
see that; one-second candles can.

Archives (same host as the 1-minute files):
    https://data.binance.vision/data/spot/monthly/klines/BTCUSDT/1s/BTCUSDT-1s-YYYY-MM.zip
    https://data.binance.vision/data/spot/daily/klines/BTCUSDT/1s/BTCUSDT-1s-YYYY-MM-DD.zip   (current month)

Size is the reason this is opt-in and not part of the default download: one day of 1-second candles is 86,400
rows, about 60 times a day of 1-minute candles. A month is ~50 MB zipped; five months is ~250 MB. Keep it out of
git (`.gitignore`) and treat it as a cache that can be rebuilt from the exchange at any time.

    python -m src.download.binance_1s 2026-11-01 2026-11-26      # daily files for a window
    python -m src.download.binance_1s --months 2026-07 2026-11   # whole months
"""
from __future__ import annotations

import io
import zipfile
from datetime import timedelta

import pandas as pd

# --- runnable both ways -------------------------------------------------------
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
# -----------------------------------------------------------------------------

import config
from src.utils import Http, ensure, log, utcnow

VISION = "https://data.binance.vision/data/spot"
RAW_DIR = config.RAW / "binance_1s"
INTERIM = config.INTERIM / "binance_1s"
COLS = ["open_time", "open", "high", "low", "close", "volume", "close_time", "quote_volume", "trades",
        "taker_base", "taker_quote", "ignore"]
KEEP = ["ts", "open", "high", "low", "close", "volume", "trades"]


def _ts(x: pd.Series) -> pd.Series:
    """Milliseconds or microseconds since epoch -> UTC timestamps (the archives changed the unit in 2025)."""
    x = pd.to_numeric(x).astype("int64")
    x = x.where(x <= 10 ** 14, x // 1000)
    return pd.to_datetime(x, unit="ms", utc=True)


def read_zip(content: bytes) -> pd.DataFrame:
    with zipfile.ZipFile(io.BytesIO(content)) as z:
        df = pd.read_csv(z.open(z.namelist()[0]), header=None)
    if not str(df.iloc[0, 0]).isdigit():          # some files come with a header row
        df = df.iloc[1:]
    df = df.iloc[:, :12]
    df.columns = COLS
    out = pd.DataFrame({
        "ts": _ts(df["open_time"]),
        "open": pd.to_numeric(df["open"], errors="coerce").astype("float32"),
        "high": pd.to_numeric(df["high"], errors="coerce").astype("float32"),
        "low": pd.to_numeric(df["low"], errors="coerce").astype("float32"),
        "close": pd.to_numeric(df["close"], errors="coerce").astype("float32"),
        "volume": pd.to_numeric(df["volume"], errors="coerce").astype("float32"),
        "trades": pd.to_numeric(df["trades"], errors="coerce").astype("int32"),
    })
    return out


def _save(content: bytes, path) :
    path.write_bytes(content)
    return path


def download_days(since: str, until: str | None = None, http: Http | None = None) -> int:
    """Daily 1-second files for [since, until). Skips what is already on disk; returns the number downloaded."""
    http = http or Http(min_interval=0.10)
    ensure(RAW_DIR)
    d = pd.Timestamp(since).normalize()
    stop = (pd.Timestamp(until).normalize() if until else pd.Timestamp(utcnow()).normalize())
    n = 0
    while d < stop:
        f = RAW_DIR / f"BTCUSDT-1s-{d:%Y-%m-%d}.zip"
        if not f.exists():
            content = http.get(f"{VISION}/daily/klines/BTCUSDT/1s/BTCUSDT-1s-{d:%Y-%m-%d}.zip", raw=True)
            if content is None:
                log(f"binance_1s: {d:%Y-%m-%d} не опубликован, пропускаю")
            else:
                _save(content, f)
                n += 1
                log(f"binance_1s: {d:%Y-%m-%d} ({len(content) / 1024:.0f} КБ)")
        d += timedelta(days=1)
    log(f"binance_1s: скачано {n} дневных файлов в {RAW_DIR}")
    return n


def download_months(since: str, until: str | None = None, http: Http | None = None) -> int:
    """Whole-month archives — cheaper per day than daily files for complete months."""
    http = http or Http(min_interval=0.10)
    ensure(RAW_DIR)
    month = pd.Timestamp(since).normalize().replace(day=1)
    stop = (pd.Timestamp(until) if until else pd.Timestamp(utcnow())).normalize().replace(day=1)
    n = 0
    while month < stop:
        f = RAW_DIR / f"BTCUSDT-1s-{month:%Y-%m}.zip"
        if not f.exists():
            content = http.get(f"{VISION}/monthly/klines/BTCUSDT/1s/BTCUSDT-1s-{month:%Y-%m}.zip", raw=True)
            if content is None:
                log(f"binance_1s: месяца {month:%Y-%m} на архиве нет, пропускаю")
            else:
                _save(content, f)
                n += 1
                log(f"binance_1s: {month:%Y-%m} ({len(content) / 1024 / 1024:.1f} МБ)")
        month += pd.offsets.MonthBegin(1)
    log(f"binance_1s: скачано {n} месячных архивов в {RAW_DIR}")
    return n


def build_interim() -> pd.DataFrame:
    """One parquet per month in data/interim/binance_1s — fast to read, ~4x smaller than the zipped CSV."""
    files = sorted(RAW_DIR.glob("BTCUSDT-1s-*.zip"))
    if not files:
        return pd.DataFrame()
    ensure(INTERIM)
    frames = []
    for f in files:
        out = INTERIM / f"{f.stem.split('BTCUSDT-1s-')[1]}.parquet"
        if not out.exists():
            df = read_zip(f.read_bytes())
            df.to_parquet(out, index=False)
            log(f"binance_1s: собрал {out.name} ({len(df)} строк)")
        frames.append(out)
    return frames[-1] if len(frames) == 1 else pd.concat([pd.read_parquet(p) for p in frames], ignore_index=True)


def main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Binance 1-second candles (opt-in: ~60x the size of 1-minute)")
    ap.add_argument("since", nargs="?", default="2026-11-01", help="first day (YYYY-MM-DD)")
    ap.add_argument("until", nargs="?", default=None, help="last day, exclusive (default: today)")
    ap.add_argument("--months", nargs=2, metavar=("SINCE", "UNTIL"), default=None,
                    help="download whole-month archives instead of daily files")
    ap.add_argument("--min-interval", type=float, default=0.10)
    a = ap.parse_args(argv)
    http = Http(min_interval=a.min_interval)
    from src.runner import keep_awake
    keep_awake()
    if a.months:
        download_months(a.months[0], a.months[1], http)
    else:
        download_days(a.since, a.until, http)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
