"""Command line for the Kalshi executor.

    python -m trader check     [--mode demo|live]   keys, clock, exchange status, balance, a sample BTC hourly book
    python -m trader test-order [--mode demo]       post a 1-contract bid far below the market, read it back, cancel
    python -m trader run       [--mode shadow|demo|live] [--once]
    python -m trader status    [--mode ...]         open positions, resting quotes, today's and total P&L
    python -m trader report    [--mode ...]         every settled trade + the sequential test (trader/monitor.py)
    python -m trader lab-sprt                       the sequential test on the lab's paper wing trades
    python -m trader cancel-all [--mode demo|live]
    python -m trader kill | unkill                  the kill switch file data/trader/KILL (run loop cancels and exits)
Live trading needs all three: settings mode "live", the flag --real-money and the environment KALSHI_ALLOW_LIVE=1.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import pandas as pd

from trader.client import Kalshi, KalshiError
from trader.executor import Executor, Spot
from trader.monitor import hourly_from_trades, sprt
from trader.settings import DATA, Settings, credentials


def venue_for(mode: str) -> Kalshi | None:
    if mode == "shadow":
        return None
    key_id, key_file = credentials(mode)
    if not key_id or not key_file:
        sys.exit(f"{mode}: set KALSHI_{'DEMO_' if mode == 'demo' else ''}KEY_ID and ..._KEY_FILE first (docs/TRADER.md)")
    return Kalshi("demo" if mode == "demo" else "prod", key_id, key_file)


def cmd_check(cfg, a):
    md = Kalshi("prod")
    print("production exchange status:", md.exchange_status())
    ev = md.markets(series_ticker="KXBTCD", status="open")
    ev = sorted(ev, key=lambda m: m.get("close_time", ""))
    if ev:
        m = ev[0]
        b = md.book(m["ticker"])
        print(f"sample book {m['ticker']}: YES bid {b.yes_bid} / ask {b.ask('yes')}")
    if a.mode in ("demo", "live"):
        v = venue_for(a.mode)
        off = v.server_clock_check()
        print(f"{a.mode}: clock offset {off:+.2f} s; balance {v.balance()}")
        print(f"{a.mode}: open positions {len(v.positions())}, resting orders {len(v.orders(status='resting'))}")
        if a.mode == "demo":
            dm = v.markets(series_ticker="KXBTCD", status="open")
            print(f"demo: {len(dm)} open KXBTCD markets (the BTC hourly ladder) — "
                  + ("found" if dm else "NOT listed on demo: run uses demo_ignore_edge plumbing only on markets that exist"))


def cmd_test_order(cfg, a):
    v = venue_for(a.mode)
    if a.mode != "demo":
        sys.exit("test-order is for the demo environment")
    v.server_clock_check()
    ms = sorted(v.markets(series_ticker="KXBTCD", status="open"), key=lambda m: m.get("close_time", "")) or \
        v.markets(status="open")[:50]
    if not ms:
        sys.exit("no open demo markets")
    m = ms[-1]
    o = v.create_order(m["ticker"], "yes", 1, 0.01, tif="good_till_canceled", post_only=True,
                       expiration_ts=int(time.time()) + 300)
    print("posted:", {k: o.get(k) for k in ("order_id", "status", "yes_price_dollars", "client_order_id")})
    print("read back:", {k: v.order(o["order_id"]).get(k) for k in ("status", "remaining_count", "fill_count")})
    print("cancel:", v.cancel(o["order_id"]).get("order", {}).get("status"))


def cmd_run(cfg, a):
    cfg.mode = a.mode or cfg.mode
    if cfg.mode == "live":
        if not (a.real_money and os.environ.get("KALSHI_ALLOW_LIVE") == "1"):
            sys.exit("live needs --real-money and KALSHI_ALLOW_LIVE=1")
    venue = venue_for(cfg.mode)
    if venue is not None:
        off = venue.server_clock_check()
        print(f"clock offset {off:+.2f} s")
    ex = Executor(cfg, venue, Kalshi("prod"), Spot(cfg.spot_url))
    print(f"executor running: mode={cfg.mode} rules={cfg.rules} journal={ex.j.path}")
    ex.run(once=a.once)


def _journal(mode):
    p = DATA / f"journal_{mode}.jsonl"
    if not p.exists():
        return pd.DataFrame()
    return pd.DataFrame([json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()])


def cmd_status(cfg, a):
    mode = a.mode or cfg.mode
    st = DATA / f"state_{mode}.json"
    s = json.loads(st.read_text()) if st.exists() else {}
    print(f"mode {mode}: open {len(s.get('positions', {}))}, resting {len(s.get('resting', {}))}, "
          f"realised by day {s.get('realised', {})}")
    for k, p in (s.get("positions") or {}).items():
        print(f"  {k}: {p['side']} {p.get('filled')} @ {p['price']:.2f} settles {p.get('settle_ts')}")
    j = _journal(mode)
    if not j.empty:
        print("\nlast events:")
        print(j.groupby("what").size().to_string())
        sk = j[j.what == "skip"]
        if len(sk):
            print("\nwhy intents were skipped:")
            print(sk["reason"].str.replace(r"[\d.+−-]+", "#", regex=True).value_counts().head(12).to_string())


def cmd_report(cfg, a):
    mode = a.mode or cfg.mode
    j = _journal(mode)
    s = j[j.what == "settled"] if not j.empty else j
    if s.empty:
        print("no settled trades yet")
        return
    s = s.assign(per=s.pnl / s["count"])
    print(s[["t", "key", "side", "price", "count", "result", "pnl", "maker"]].to_string(index=False))
    print(f"\n{len(s)} trades, hit {100 * (s.pnl > 0).mean():.0f}%, P&L ${s.pnl.sum():+.2f}, "
          f"{100 * s.per.mean():+.2f}¢ per contract")
    hourly = s.groupby(pd.to_datetime(s["t"], utc=True).dt.floor("h"))["per"].sum()
    print("sequential test (trader/monitor.py):", sprt(hourly))


def cmd_lab_sprt(cfg, a):
    """The same sequential test on the lab's paper wing trades (data/lab/ledger.csv)."""
    import json as _json
    from trader.settings import ROOT
    led = pd.read_csv(ROOT / "data" / "lab" / "ledger.csv")
    w = led[(led["event"] == "CLOSE") & led["hyp"].isin(["W03", "W04", "W05", "W06"])].copy()
    w["ev"] = w["info"].map(lambda x: _json.loads(x)["legs"][0]["event"])
    w["pnl_per_contract"] = w["pnl"] / w["qty"]
    w["t"] = pd.to_datetime(w["ts"], utc=True, format="mixed")
    w["settle_ts"] = pd.to_datetime(w["exit_ts"], utc=True, format="mixed").dt.floor("h")
    print(f"{len(w)} closed paper wing trades {w.groupby('hyp').size().to_dict()}")
    print(sprt(hourly_from_trades(w, event_col="ev")))


def main():
    ap = argparse.ArgumentParser(prog="python -m trader")
    ap.add_argument("cmd", choices=["check", "test-order", "run", "status", "report", "lab-sprt", "cancel-all", "kill", "unkill"])
    ap.add_argument("--mode", choices=["shadow", "demo", "live"], default=None)
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--real-money", action="store_true")
    a = ap.parse_args()
    cfg = Settings.load()
    if a.cmd == "kill":
        (DATA / "KILL").write_text(str(pd.Timestamp.now(tz="UTC")))
        print("kill switch ON: the run loop cancels resting orders and exits")
    elif a.cmd == "unkill":
        (DATA / "KILL").unlink(missing_ok=True)
        print("kill switch OFF")
    elif a.cmd == "cancel-all":
        print("cancelled", venue_for(a.mode or "demo").cancel_all())
    else:
        a.mode = a.mode or cfg.mode
        try:
            {"check": cmd_check, "test-order": cmd_test_order, "run": cmd_run, "status": cmd_status,
             "report": cmd_report, "lab-sprt": cmd_lab_sprt}[a.cmd](cfg, a)
        except KalshiError as e:
            sys.exit(str(e))


if __name__ == "__main__":
    main()
