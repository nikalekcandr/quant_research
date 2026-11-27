"""US macro release calendar: fills the hole in data/manual/us_macro_calendar.csv.

Why this download exists
    The only conditioning gate that repeated across halves of 2026 in `studies/wings_scale.py` is "do not trade
    the 13-15 UTC US open", and the natural next question — does a scheduled release inside the contract hour
    change the wing's edge — cannot be asked for November because the hand-kept calendar stops on 28 October.
    This module rebuilds that calendar from the two official schedules instead of by hand.

Sources (both publish the whole year in advance, no API key)
    BLS  https://www.bls.gov/schedule/news_release/          CPI, Employment Situation (NFP), PPI, ...
    Fed  https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm   FOMC decision days
Times, ET: 08:30 for BLS releases, 14:00 for the FOMC statement.

Rules
    * append-only: existing rows are never rewritten or dropped, new rows are merged by (event, date);
    * nothing is invented: if a page does not parse, the module raises and writes nothing;
    * rows dated more than 400 days ahead are rejected (a parsing accident, not a calendar).

    python -m src.download.macro            fetch and merge
    python -m src.download.macro --dry-run  show what would be added, write nothing
"""
from __future__ import annotations

import argparse
import os
import re

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
from src.utils import Http, ensure, log, utcnow

CAL = config.MANUAL / "us_macro_calendar.csv"
BLS = "https://www.bls.gov/schedule/news_release/"
FED = "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm"
EVENTS = {"Consumer Price Index": "CPI", "Employment Situation": "NFP", "Producer Price Index": "PPI"}
MONTHS = {m: i for i, m in enumerate(
    ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
     "November", "December"], start=1)}
MONTH_RE = "|".join(MONTHS)
HORIZON_DAYS = 400


def _text(html: str) -> str:
    """Crude tag strip that keeps the words in reading order (the schedules are tables, not JavaScript)."""
    s = re.sub(r"<script.*?</script>|<style.*?</style>", " ", html, flags=re.S | re.I)
    s = re.sub(r"<[^>]+>", " ", s)
    s = s.replace("&nbsp;", " ").replace("&amp;", "&")
    return re.sub(r"[ \t]+", " ", s)


def parse_bls(html: str) -> pd.DataFrame:
    """CPI / NFP / PPI rows from the BLS schedule page: `<event> ... <Month DD, YYYY>` in the same block."""
    text = _text(html)
    rows = []
    for m in re.finditer(r"(" + "|".join(map(re.escape, EVENTS)) + r")(.{0,400}?)"
                         r"(" + MONTH_RE + r")\s+(\d{1,2}),\s*(\d{4})", text, flags=re.S):
        event, date = EVENTS[m.group(1)], pd.Timestamp(year=int(m.group(5)), month=MONTHS[m.group(3)],
                                                       day=int(m.group(4)))
        rows.append({"event": event, "date": date.date(), "time_et": "08:30"})
    return pd.DataFrame(rows).drop_duplicates()


def parse_fed(html: str) -> pd.DataFrame:
    """FOMC statement days: `January 27-28` / `March 18` blocks under a year heading."""
    text = _text(html)
    rows = []
    for year in re.finditer(r"(20\d{2})\s+FOMC\s+Meetings", text):
        y = int(year.group(1))
        block = text[year.end(): year.end() + 2000]
        block = block[: block.find("FOMC Meetings")] if "FOMC Meetings" in block else block
        for m in re.finditer(r"(" + MONTH_RE + r")\s+(\d{1,2})\s*(?:-\s*(\d{1,2}))?", block):
            day = int(m.group(3) or m.group(2))          # two-day meetings decide on the second day
            try:
                date = pd.Timestamp(year=y, month=MONTHS[m.group(1)], day=day)
            except ValueError:                            # e.g. an annotation row, not a date
                continue
            rows.append({"event": "FOMC", "date": date.date(), "time_et": "14:00"})
    return pd.DataFrame(rows).drop_duplicates()


def merge(new: pd.DataFrame, existing: pd.DataFrame, now: pd.Timestamp | None = None) -> pd.DataFrame:
    """Union of the two calendars, deduplicated, sorted, with absurd dates rejected."""
    now = now or pd.Timestamp(utcnow()).tz_localize(None)
    both = pd.concat([existing, new], ignore_index=True)
    both["date"] = pd.to_datetime(both.date).dt.date
    if (pd.to_datetime(both.date) > now + pd.Timedelta(days=HORIZON_DAYS)).any():
        raise ValueError("refusing to write a release more than 400 days ahead: the page did not parse")
    both = both.drop_duplicates(["event", "date"]).sort_values(["date", "event", "time_et"])
    return both[["event", "date", "time_et"]].reset_index(drop=True)


def download(http: Http | None = None, dry_run: bool = False) -> pd.DataFrame:
    http = http or Http(min_interval=0.5)
    html_bls = http.get(BLS, raw=True)
    html_fed = http.get(FED, raw=True)
    bls = parse_bls(html_bls.decode("utf-8", "ignore") if isinstance(html_bls, bytes) else html_bls)
    fed = parse_fed(html_fed.decode("utf-8", "ignore") if isinstance(html_fed, bytes) else html_fed)
    if bls.empty or fed.empty:
        raise RuntimeError(f"calendar parse failed: {len(bls)} BLS rows, {len(fed)} FOMC rows — nothing written")
    existing = pd.read_csv(CAL) if CAL.exists() else pd.DataFrame(columns=["event", "date", "time_et"])
    out = merge(pd.concat([bls, fed], ignore_index=True), existing)
    added = out.merge(existing, how="left", indicator=True)
    log(f"macro: {int((added._merge == 'left_only').sum())} new rows, {len(out)} total")
    if not dry_run:
        ensure(CAL.parent)
        tmp = CAL.with_suffix(".tmp.csv")
        out.to_csv(tmp, index=False)
        os.replace(tmp, CAL)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="rebuild the US macro release calendar")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    out = download(dry_run=a.dry_run)
    print(out.tail(20).to_string(index=False))


if __name__ == "__main__":
    main()
