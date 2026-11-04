"""Polymarket hourly ladders ("Bitcoin above K on <date>, <h> ET", settles on the Binance BTC/USDT 1-hour candle
close) for the wing study, 04.11.2026.
1. Live book rows the lab logged 28.10–03.11 (kind pm_ladder_1h in data/lab/markets/*.csv: real bid/ask/sizes)
   -> data/raw/pm_ladder_1h/live_books.parquet
2. History: every hourly event since 2026-07-01 (newest first, resumable by day): each strike's outcome and the
   1-minute price history of its YES token over the last 65 minutes -> data/raw/pm_ladder_1h/hist/<day>.csv
3. Packs both into _export/pm_ladder_1h.tar.gz."""
import json
import re
import sys
import tarfile
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "raw" / "pm_ladder_1h"
(OUT / "hist").mkdir(parents=True, exist_ok=True)
GAMMA, CLOB = "https://gamma-api.polymarket.com", "https://clob.polymarket.com"
ET = ZoneInfo("America/New_York")
S = requests.Session()
S.headers["User-Agent"] = "btcvol-research"


def say(m):
    print(f"[{datetime.now():%H:%M:%S}] {m}", flush=True)


def get(url, params, tries=5):
    for i in range(tries):
        try:
            r = S.get(url, params=params, timeout=20)
            if r.status_code == 429:
                time.sleep(2 + 2 * i)
                continue
            if r.status_code == 404:
                return None
            r.raise_for_status()
            return r.json()
        except Exception:  # noqa: BLE001
            time.sleep(1 + i)
    return None


# ---------------------------------------------------------------- 1. live rows
rows = []
for f in sorted((ROOT / "data" / "lab" / "markets").glob("*.csv")):
    try:
        for ch in pd.read_csv(f, chunksize=200000, low_memory=False):
            if "kind" in ch:
                rows.append(ch[ch["kind"] == "pm_ladder_1h"])
    except Exception as e:  # noqa: BLE001
        say(f"{f.name}: {e}")
live = pd.concat(rows) if rows else pd.DataFrame()
live.to_parquet(OUT / "live_books.parquet")
say(f"live pm_ladder_1h rows: {len(live)}; events: {live['event'].nunique() if len(live) else 0}")
examples = sorted(live["event"].dropna().unique())[:5] if len(live) else []
say(f"slug examples: {examples}")


# ---------------------------------------------------------------- 2. history
def slugs(end_et):
    mon = end_et.strftime("%B").lower()
    h12 = end_et.hour % 12 or 12
    ap = "am" if end_et.hour < 12 else "pm"
    # the event is named after the hour it starts (end − 1 h) in examples like "...-on-november-2-3pm-et";
    out = []
    for t in (end_et - timedelta(hours=1), end_et):
        mon, day = t.strftime("%B").lower(), t.day
        h12, ap = (t.hour % 12 or 12), ("am" if t.hour < 12 else "pm")
        out.append(f"bitcoin-above-on-{mon}-{day}-{h12}{ap}-et")
    return out


def event_rows(end_utc):
    end_et = end_utc.astimezone(ET)
    for slug in slugs(end_et):
        evs = get(f"{GAMMA}/events", {"slug": slug})
        if evs:
            break
    else:
        return []
    ev = evs[0]
    out = []
    for m in ev.get("markets") or []:
        mend = pd.Timestamp(m.get("endDate") or ev.get("endDate"))
        if abs((mend - pd.Timestamp(end_utc)).total_seconds()) > 3600:
            continue
        toks = json.loads(m["clobTokenIds"]) if m.get("clobTokenIds") else []
        lv = [float(x.replace(",", "")) for x in re.findall(r"\d{1,3}(?:,\d{3})+", m.get("groupItemTitle") or m.get("question") or "")]
        try:
            op = [float(x) for x in json.loads(m.get("outcomePrices") or "[]")]
        except Exception:  # noqa: BLE001
            op = []
        outcome = (1.0 if op[0] >= 0.99 else 0.0 if op[0] <= 0.01 else None) if op else None
        if not toks or not lv:
            continue
        t1 = int(mend.timestamp())
        h = (get(f"{CLOB}/prices-history", {"market": toks[0], "startTs": t1 - 65 * 60, "endTs": t1, "fidelity": 1}) or {}).get("history") or []
        base = {"event": ev.get("slug"), "market": m.get("slug"), "end": mend.isoformat(), "strike": lv[0],
                "outcome": outcome, "volume": m.get("volume"), "fee": json.dumps(m.get("feeSchedule")) if m.get("feeSchedule") else None,
                "fees_enabled": m.get("feesEnabled")}
        out += [dict(base, t=x.get("t"), p=x.get("p")) for x in h] or [dict(base, t=None, p=None)]
    return out


today = datetime.now(timezone.utc).date()
d = today - timedelta(days=1)
empty_days = 0
while d >= datetime(2026, 7, 1).date():
    f = OUT / "hist" / f"{d}.csv"
    if not f.exists():
        ends = [datetime(d.year, d.month, d.day, tzinfo=timezone.utc) + timedelta(hours=h + 1) for h in range(24)]
        with ThreadPoolExecutor(max_workers=6) as ex:
            rows = [r for part in ex.map(event_rows, ends) for r in part]
        pd.DataFrame(rows).to_csv(f, index=False)
        n_ev = len({r["event"] for r in rows})
        n_pts = sum(1 for r in rows if r.get("p") is not None)
        say(f"{d}: {n_ev} events, {len({r['market'] for r in rows})} strikes, {n_pts} price points")
        empty_days = empty_days + 1 if n_pts == 0 else 0
        if empty_days >= 4:
            say("4 days without price history in a row: the API keeps no older history; stopping")
            break
    d -= timedelta(days=1)

exp = ROOT / "_export"
exp.mkdir(exist_ok=True)
with tarfile.open(exp / "pm_ladder_1h.tar.gz", "w:gz") as tar:
    tar.add(OUT, arcname="pm_ladder_1h")
say(f"packed {(exp / 'pm_ladder_1h.tar.gz').stat().st_size} bytes")
