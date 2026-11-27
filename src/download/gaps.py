"""What is missing, and the exact command that fills it.

The repository versions its data, so the first question of any new study is "how fresh is each tape". This
module answers it without touching the network: every source is inspected on disk, the size of the hole is
printed next to the command that closes it. `--run` then executes the fillers that are safe to run unattended
(spot minutes, perp, funding, macro calendar); the Kalshi ladder is opt-in because it costs one request per
hour of history.

    python -m src.download.gaps                 audit only, nothing is fetched
    python -m src.download.gaps --run           fetch everything that is filled by a cheap downloader
    python -m src.download.gaps --run --kalshi  also extend the Kalshi hourly ladder
    python -m src.download.gaps --as-of 2026-11-26   freeze "now" (reproducible audit)

Sources checked
    spot minutes      data/raw/binance/*.zip + recent.csv.gz          src.download.binance
    perp + basis      data/raw/perp/*.zip + recent_*.csv.gz            src.download.perp      (new)
    funding           data/raw/funding/btc_funding.csv.gz              src.download.funding
    Kalshi ladder     data/raw/kalshi_kxbtcd_60m/*.csv.gz              src.download.kalshi
    November books    data/raw/wing_gap_candles/*.csv.gz               (live logger snapshots, not downloadable)
    macro calendar    data/manual/us_macro_calendar.csv                src.download.macro     (new)
    Deribit dailies   data/raw/deribit/dailies/*.csv.gz                src.download.deribit
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys

import pandas as pd

# --- runnable both ways -------------------------------------------------------
# `python -m src.download.NAME` and `python src/download/NAME.py` (an IDE "Run" button) must both work:
# when this file is executed as a script, Python puts src/download/ on sys.path instead of the repository
# root, and `import config` fails. Put the root back before importing anything from the project.
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
# -----------------------------------------------------------------------------

import config

ROOT = config.ROOT


KALSHI_TICKER = re.compile(r"-(\d{2})([A-Z]{3})(\d{2})\d*")
MONTHS3 = {m: i for i, m in enumerate(
    ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"], start=1)}


def ticker_date(name: str) -> pd.Timestamp | None:
    """`KXBTCD-26NOV1819-T63099.99.csv.gz` -> 2026-11-18 (the ticker carries the settlement day)."""
    m = KALSHI_TICKER.search(name)
    if not m:
        return None
    return pd.Timestamp(year=2000 + int(m.group(1)), month=MONTHS3.get(m.group(2), 1), day=int(m.group(3)),
                        tz="UTC")


def _last_file_date(pattern: str, directory) -> pd.Timestamp | None:
    """Latest date encoded in a file name: `2026-10-27`, `2026-11` or a Kalshi ticker date."""
    dates = []
    for f in directory.glob(pattern):
        m = re.search(r"(\d{4}-\d{2}-\d{2}|\d{4}-\d{2})", f.name)
        if m:
            t = pd.to_datetime(m.group(1), utc=True, errors="coerce")
        else:
            compact = re.match(r"^(\d{8})(\d{2})?", f.name)      # live logger: 2026110812.csv.gz
            t = (pd.to_datetime(compact.group(1), format="%Y%m%d", utc=True)
                 if compact else ticker_date(f.name))
        if t is not None and pd.notna(t):
            dates.append(t)
    return max(dates) if dates else None


def largest_hole(dates) -> int | None:
    """Biggest gap in days between consecutive entries of a calendar (a stale tail is not the only problem)."""
    d = pd.Series(pd.to_datetime(list(dates), utc=True)).sort_values().diff().dt.days
    return int(d.max()) if len(d.dropna()) else None


def _last_ts_csv(path, column: int = 0, unit: str = "ms") -> pd.Timestamp | None:
    if not path.exists():
        return None
    try:
        d = pd.read_csv(path, header=None, usecols=[column], compression="infer")
        v = pd.to_numeric(d.iloc[-1, 0], errors="coerce")
        if pd.isna(v):
            return None
        if v > 10 ** 14:
            unit = "us"
        return pd.to_datetime(int(v), unit=unit, utc=True)
    except Exception:
        return None


def audit(as_of: str | None = None) -> pd.DataFrame:
    """One row per source: what is on disk, how old it is, and what would fill the hole."""
    now = pd.to_datetime(as_of, utc=True) if as_of else pd.Timestamp.now("UTC")
    rows = []

    def add(source, path, last, command, note=""):
        gap = (now - last).days if last is not None else None
        rows.append({"source": source, "path": str(path).replace(str(ROOT) + "/", ""),
                     "last": "" if last is None else f"{last:%Y-%m-%d %H:%M}",
                     "gap days": gap, "command": command, "note": note})

    binance = config.RAW / "binance"
    spot = _last_file_date("BTCUSDT-1m-*.zip", binance)
    recent = _last_ts_csv(binance / "recent.csv.gz")
    add("spot 1m", binance, max([x for x in (spot, recent) if x is not None], default=None),
        "python -m src.download.update --only binance")

    perp = config.RAW / "perp"
    p_arch = _last_file_date("BTCUSDT-1m-*.zip", perp) if perp.exists() else None
    p_rec = _last_ts_csv(perp / "recent_klines.csv.gz")
    add("perp 1m + basis", perp, max([x for x in (p_arch, p_rec) if x is not None], default=None),
        "python -m src.download.update --only perp --build", "" if perp.exists() else "never downloaded")

    funding = config.RAW / "funding/btc_funding.csv.gz"
    last_f = None
    if funding.exists():
        f = pd.read_csv(funding)
        last_f = pd.to_datetime(f.ts.max(), unit="ms", utc=True) if len(f) else None
    add("funding 8h", funding, last_f, "python -m src.download.update --only funding")

    ladder = config.RAW / "kalshi_kxbtcd_60m"
    add("Kalshi ladder", ladder, _last_file_date("*.csv.gz", ladder),
        "python -m src.download.update --only kalshi", "one request per hour of history (~50 per day)")

    books = config.RAW / "wing_gap_candles"
    add("November books", books, _last_file_date("*.csv.gz", books), "(live logger)",
        "book snapshots cannot be downloaded after the fact; only what live.py saved")

    cal = config.MANUAL / "us_macro_calendar.csv"
    last_c, hole = None, None
    if cal.exists():
        c = pd.read_csv(cal)
        last_c = pd.to_datetime(c.date.max(), utc=True)
        hole = largest_hole(c.date)
    add("macro calendar", cal, last_c, "python -m src.download.update --only macro",
        f"largest hole {hole} days" if hole else "")

    der = config.RAW / "deribit/dailies"
    add("Deribit dailies", der, _last_file_date("*.csv.gz", der),
        "python -m src.download.update --only deribit")
    return pd.DataFrame(rows)


FILLERS = [("src.download.binance", []), ("src.download.perp", ["--interim"]),
           ("src.download.funding", []), ("src.download.macro", [])]


def run(fillers, kalshi: bool = False) -> None:
    todo = list(fillers)
    if kalshi:
        todo.append(("src.download.kalshi", ["--hours"]))
    for module, args in todo:
        print(f"\n=== {module} {' '.join(args)}")
        res = subprocess.run([sys.executable, "-m", module, *args], cwd=ROOT)
        if res.returncode:
            print(f"!! {module} failed with code {res.returncode}; the rest continues")


def main() -> None:
    ap = argparse.ArgumentParser(description="audit and fill the data gaps")
    ap.add_argument("--run", action="store_true", help="run the cheap fillers")
    ap.add_argument("--kalshi", action="store_true", help="also extend the Kalshi hourly ladder")
    ap.add_argument("--as-of", default=None, help="audit against this date instead of now")
    a = ap.parse_args()
    table = audit(a.as_of)
    with pd.option_context("display.width", 200, "display.max_colwidth",60):
        print(table.to_string(index=False))
    stale = table[(table["gap days"].notna()) & (table["gap days"] > 1)]
    print(f"\n{len(stale)} source(s) more than a day behind.")
    if a.run:
        run(FILLERS, a.kalshi)


if __name__ == "__main__":
    main()
