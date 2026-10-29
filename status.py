"""Health check of the live setup: is the logger writing, are forecasts made on time, did the nightly update run.

    python status.py
"""
from __future__ import annotations

import json

import pandas as pd

import config
from src.utils import utcnow


def last_rows(src: str):
    files = sorted((config.LIVE / src).glob("*.csv.gz"))
    if not files:
        return None, 0
    try:
        d = pd.read_csv(files[-1])
        ts = pd.to_datetime(d["snap_ts"], utc=True, format="mixed")
        return ts.max(), ts.nunique()
    except Exception:
        return None, 0


def main():
    now = pd.Timestamp(utcnow())
    print(f"now: {now:%Y-%m-%d %H:%M} UTC\n")
    print("order-book snapshots (today's file):")
    for src in ("pm_books", "kalshi", "kalshi_range", "deribit_chain", "binance"):
        last, n = last_rows(src)
        age = f"{(now - last).total_seconds() / 60:.0f} min ago" if last is not None else "—"
        flag = "" if last is not None and now - last < pd.Timedelta(minutes=25) else "   <-- not writing!"
        print(f"  {src:14s} last {age:>12s}, {n:3d} snapshots today{flag}")
    print("\nlive forecasts:")
    for h in ("kx1h", "pm24h", "pm6d"):
        files = sorted((config.FORECASTS_LIVE / h).glob("*.json"))
        late = 0
        for f in files[-500:]:
            try:
                late += bool(json.loads(f.read_text(encoding="utf-8")).get("late"))
            except Exception:
                pass
        last = files[-1].stem if files else "—"
        print(f"  {h:6s} {len(files):4d} files, {late} late, last t = {last}")
    try:
        from src.live_forecast import verify_ledger
        print("  ledger:", verify_ledger()[1])
    except Exception as e:
        print("  ledger: check failed", e)
    print("\nlogs:")
    logs = sorted((config.ROOT / "logs").glob("*.log"))
    for f in logs[-4:]:
        tail = f.read_text(encoding="utf-8", errors="replace").strip().splitlines()[-1:] or [""]
        print(f"  {f.name}: {tail[0][:110]}")
    err = config.LIVE / "errors.log"
    if err.exists():
        lines = [x for x in err.read_text(encoding="utf-8", errors="replace").splitlines() if x[:4].isdigit()]
        recent = [x for x in lines if x[:19] >= f"{now - pd.Timedelta(hours=6):%Y-%m-%d %H:%M:%S}"]
        print(f"\nerrors.log: {len(recent)} entries in the last 6 hours" + (f"; last: {recent[-1][:110]}" if recent else ""))
    rep = config.REPORTS / "oos_report.md"
    if rep.exists():
        print(f"\nOOS report: {rep} (updated {pd.Timestamp(rep.stat().st_mtime, unit='s'):%Y-%m-%d %H:%M} UTC)")


if __name__ == "__main__":
    main()
