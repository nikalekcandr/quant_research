"""Shared helpers: HTTP with retries, time zones, small IO utilities."""
from __future__ import annotations

import gzip
import json
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import requests

import config

UTC = timezone.utc
ET = ZoneInfo("America/New_York")


# ------------------------------------------------------------------ HTTP
class Blocked(RuntimeError):
    """The server refuses this network (HTTP 401/403), e.g. a CDN blocking the IP address. Retrying does not help."""


class Http:
    """requests.Session with retries, exponential backoff and a minimum gap between calls.

    min_interval protects us from the public rate limits (Kalshi ~20 req/s, Deribit ~20 req/s).
    """

    def __init__(self, min_interval: float = 0.06, retries: int = 6):
        self.s = requests.Session()
        self.s.headers.update({"User-Agent": config.USER_AGENT, "Accept": "application/json"})
        self.min_interval = min_interval
        self.retries = retries
        self._last = 0.0
        self._lock = threading.Lock()   # several download threads share one rate limit

    def _wait(self):
        with self._lock:
            gap = time.monotonic() - self._last
            if gap < self.min_interval:
                time.sleep(self.min_interval - gap)
            self._last = time.monotonic()

    def get(self, url: str, params: dict | None = None, raw: bool = False):
        err = None
        for attempt in range(self.retries):
            self._wait()
            try:
                r = self.s.get(url, params=params, timeout=config.HTTP_TIMEOUT)
                if r.status_code == 404:
                    return None
                if r.status_code in (401, 403):
                    raise Blocked(f"HTTP {r.status_code} from {url.split('/')[2]}: the server refuses requests from this network")
                if r.status_code in (429, 500, 502, 503, 504):
                    err = f"HTTP {r.status_code}"
                    time.sleep(min(60, 2 ** attempt))
                    continue
                r.raise_for_status()
                return r.content if raw else r.json()
            except (requests.ConnectionError, requests.Timeout, ValueError) as e:
                err = repr(e)
                time.sleep(min(60, 2 ** attempt))
        raise RuntimeError(f"GET failed after {self.retries} attempts: {url} {params} ({err})")

    def post(self, url: str, payload):
        err = None
        for attempt in range(self.retries):
            self._wait()
            try:
                r = self.s.post(url, json=payload, timeout=config.HTTP_TIMEOUT)
                if r.status_code in (429, 500, 502, 503, 504):
                    err = f"HTTP {r.status_code}"
                    time.sleep(min(60, 2 ** attempt))
                    continue
                r.raise_for_status()
                return r.json()
            except (requests.ConnectionError, requests.Timeout, ValueError) as e:
                err = repr(e)
                time.sleep(min(60, 2 ** attempt))
        raise RuntimeError(f"POST failed after {self.retries} attempts: {url} ({err})")


# ------------------------------------------------------------------ time
def utcnow() -> datetime:
    return datetime.now(UTC)


def to_utc(ts) -> pd.Timestamp:
    t = pd.Timestamp(ts)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def noon_et_utc(day) -> pd.Timestamp:
    """12:00 New York time on calendar date `day`, expressed in UTC (16:00 in summer, 17:00 in winter)."""
    d = pd.Timestamp(day).date()
    return pd.Timestamp(datetime(d.year, d.month, d.day, 12, 0, tzinfo=ET)).tz_convert("UTC")


def et_hour_utc(day, hour: int) -> pd.Timestamp:
    d = pd.Timestamp(day).date()
    return pd.Timestamp(datetime(d.year, d.month, d.day, hour, 0, tzinfo=ET)).tz_convert("UTC")


def ms(ts) -> int:
    return int(to_utc(ts).value // 1_000_000)


def sec(ts) -> int:
    return int(to_utc(ts).value // 1_000_000_000)


# ------------------------------------------------------------------ IO
def ensure(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(obj, path: Path, compress: bool = True):
    ensure(path.parent)
    if compress:
        with gzip.open(str(path) + ("" if str(path).endswith(".gz") else ".gz"), "wt", encoding="utf-8") as f:
            json.dump(obj, f)
    else:
        path.write_text(json.dumps(obj), encoding="utf-8")


def load_json(path: Path):
    p = Path(path)
    if not p.exists() and Path(str(p) + ".gz").exists():
        p = Path(str(p) + ".gz")
    if str(p).endswith(".gz"):
        with gzip.open(p, "rt", encoding="utf-8") as f:
            return json.load(f)
    return json.loads(p.read_text(encoding="utf-8"))


def exists_json(path: Path) -> bool:
    p = Path(path)
    return p.exists() or Path(str(p) + ".gz").exists()


def append_csv_gz(df: pd.DataFrame, path: Path):
    """Append rows to a gzip CSV. Each call adds a gzip member; pandas reads the result as one file."""
    if df is None or df.empty:
        return
    ensure(path.parent)
    header = not path.exists()
    with gzip.open(path, "at", encoding="utf-8", newline="") as f:
        df.to_csv(f, header=header, index=False)


def write_table(df: pd.DataFrame, path: Path):
    ensure(path.parent)
    df = norm_dt(df.copy())
    try:
        df.to_parquet(path.with_suffix(".parquet"))
    except Exception:  # pyarrow missing: fall back to CSV
        df.to_csv(path.with_suffix(".csv.gz"), index=False)


def norm_dt(df: pd.DataFrame) -> pd.DataFrame:
    """Store every datetime column as UTC nanoseconds (pandas 2 and 3 pick different units by default,
    and merge_asof refuses to join columns with different units)."""
    for c in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[c]):
            col = df[c]
            if col.dt.tz is None:
                col = col.dt.tz_localize("UTC")
            df[c] = col.dt.tz_convert("UTC").dt.as_unit("ns")
    return df


def read_table(path: Path) -> pd.DataFrame:
    p = Path(path)
    if p.with_suffix(".parquet").exists():
        return norm_dt(pd.read_parquet(p.with_suffix(".parquet")))
    if p.with_suffix(".csv.gz").exists():
        df = pd.read_csv(p.with_suffix(".csv.gz"))
        for c in df.columns:
            if c.endswith("_ts") or c in ("t", "T", "e", "ts", "open_time", "candle_end"):
                df[c] = pd.to_datetime(df[c], utc=True)
        return norm_dt(df)
    raise FileNotFoundError(p)


def log(msg: str):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)
