"""The lab's loop: every minute collect, price, decide, close and settle; every 15 s the fast tick (lead-lag);
every hour the volatility zoo and settlement of all observed markets; every 6 h the ML retrain; every 15 min
the report."""
from __future__ import annotations

import json
import math
import os
import threading
import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

import config
from lab import feeds, markets as mk, signals
from lab.flow import Flow
from lab.maker import Maker
from lab.tape import Tape
from lab.core import (FAST, HEARTBEAT, LAB, MARKETS, PERP_FEE, TICKS, Book, Order, append_rows, atomic_write, day_file,
                      lab_http, log, log_exc, utcnow)
from lab.history import Minutes
from lab.ml import ML
from lab.volzoo import VolZoo

RESOLUTIONS = LAB / "resolutions.json"
MARKETS_FAST = LAB / "markets_fast"


def add_fairs(ctx, m):
    """All fair-value models for every quoted binary (calibration comparison in the report)."""
    for model in signals.FAIR_MODELS:
        col = "fair" if model == "ln" else f"fair_{model}"
        m[col] = [signals.fair_value(ctx, r, model) for _, r in m.iterrows()]


@dataclass
class Ctx:
    now: pd.Timestamp
    tick: dict
    ticks: pd.DataFrame | None
    minutes: Minutes
    markets: pd.DataFrame | None
    book: Book
    ml: ML | None
    cache: dict
    lab: object = None
    flow: object = None

    def fapi(self, path, params=None):
        return feeds._fapi(self.lab.http, path, params)

    def dvol_hourly(self):
        c = self.cache.get("dvol_h")
        if c and self.now - c[0] < pd.Timedelta(minutes=30):
            return c[1]
        end = int(time.time() * 1000)
        try:
            r = self.lab.http.get("https://www.deribit.com/api/v2/public/get_volatility_index_data",
                                  {"currency": "BTC", "resolution": "3600", "start_timestamp": end - 30 * 86400 * 1000,
                                   "end_timestamp": end})
            data = (r.get("result") or {}).get("data") or []
            s = pd.Series([float(x[4]) for x in data], index=pd.to_datetime([x[0] for x in data], unit="ms", utc=True)).sort_index()
        except Exception:
            s = None
        self.cache["dvol_h"] = (self.now, s)
        return s

    def opt_stamp(self):
        day = self.now.strftime("%Y-%m-%d")
        st = []
        for v in ("deribit_chain", "binance_options", "okx_options", "bybit_options"):
            f = config.LIVE / v / f"{day}.csv.gz"
            st.append(f.stat().st_mtime if f.exists() else 0)
        return tuple(st)


class Lab:
    def __init__(self):
        LAB.mkdir(parents=True, exist_ok=True)
        self.http = lab_http()
        self.http_fast = lab_http()
        self.minutes = Minutes()
        self.book = Book()
        self.lock = threading.RLock()
        self.ml = ML()
        self.zoo = VolZoo()
        self.cache = {}
        self.ticks = self._load_ticks()
        self.fast = deque(maxlen=80)
        self.flow = Flow()
        self.maker = Maker()
        self.tape = Tape()
        self.http_mk = lab_http()
        self.markets = None
        self.last_report = None
        self.last_hourly = None
        self.errors = deque(maxlen=20)
        self.started = utcnow()
        try:
            self.res = json.loads(RESOLUTIONS.read_text(encoding="utf-8"))
        except Exception:
            self.res = {}
        log(f"lab: started, {len(self.book.pos)} open positions, rules version {self.book.code}, "
            f"minutes until {self.minutes.df.index[-1] if len(self.minutes.df) else 'none'}")

    def _load_ticks(self):
        parts = []
        for d in (2, 1, 0):
            f = day_file(TICKS, utcnow() - pd.Timedelta(days=d))
            if f.exists():
                try:
                    parts.append(pd.read_csv(f))
                except Exception:
                    pass
        if not parts:
            return None
        t = pd.concat(parts, ignore_index=True)
        t["ts"] = pd.to_datetime(t["ts"], utc=True)
        return t

    # ============================================================== the minute
    def minute(self, now):
        t0 = time.time()
        try:
            self.minutes.update(self.http)
        except Exception:
            log_exc("minutes update")
        tick = feeds.tick(self.http, now)
        append_rows(day_file(TICKS, now), [tick])
        tr = pd.DataFrame([tick])
        self.ticks = tr if self.ticks is None else pd.concat([self.ticks, tr], ignore_index=True)
        self.ticks = self.ticks[self.ticks["ts"] > now - pd.Timedelta(days=3)]
        try:
            m = mk.snapshot(self.http, now)
        except Exception:
            log_exc("markets")
            m = pd.DataFrame()
        ctx = Ctx(now=now, tick=tick, ticks=self.ticks, minutes=self.minutes, markets=m, book=self.book, ml=self.ml,
                  cache=self.cache, lab=self, flow=self.flow)
        if tick.get("dvol") is not None and np.isfinite(tick.get("dvol")):
            self.cache["last_dvol"] = tick["dvol"]
        try:
            lv = signals.kalshi_ladder_level(ctx)
            if lv is not None and np.isfinite(lv):
                self.cache["kx_level_last"] = (now, lv)
        except Exception:
            log_exc("kalshi ladder level")
        if len(m):
            add_fairs(ctx, m)
            m["spot_bn"] = tick.get("bn_mid")
            m["spot_usd"] = tick.get("usd_mid")
            append_rows(day_file(MARKETS, now), m.to_dict("records"))
        self.markets = m
        with self.lock:
            # a broken position must never stop the loop (it did once: a strike stored as text)
            for step in (self.close_due, self.settle, self.zoo5_step):
                try:
                    step(ctx)
                except Exception:
                    self.errors.append(f"{now:%H:%M} {step.__name__}")
                    log_exc(step.__name__)
            n_new = 0
            for h in signals.REGISTRY:
                if h.loop != "minute" or h.retired:
                    continue
                try:
                    for o in h.fn(ctx) or []:
                        if self.book.submit(o, now, h.family):
                            n_new += 1
                            log(f"OPEN {o.hyp} {o.side} {o.instrument} @ {o.price:.4f}" +
                                (f" edge {o.edge:.4f}" if o.edge is not None else ""))
                except Exception:
                    self.errors.append(f"{now:%H:%M} {h.id}")
                    log_exc(f"rule {h.id}")
            try:
                self.maker.step(ctx, self)
            except Exception:
                log_exc("maker")
            try:
                self.tape.step(ctx, self)
            except Exception:
                log_exc("tape")
            self.book.save()
        self.heartbeat(now, time.time() - t0)

    def close_due(self, ctx):
        now, t = ctx.now, ctx.tick
        for pid, p in list(self.book.pos.items()):
            if p["kind"] == "perp" and pd.Timestamp(p.get("exit_due") or now) <= now:
                b, a = signals.perp_quotes(ctx)
                if b is None or not np.isfinite(b):
                    continue
                entry, sgn = p["price"], (1 if p["side"] == "long" else -1)
                exit_px = b if sgn > 0 else a
                pnl = sgn * (exit_px / entry - 1) - 2 * PERP_FEE
                mid0 = (p.get("info") or {}).get("mid", entry)
                gross = sgn * (((a + b) / 2) / mid0 - 1)
                self.book.close(pid, now, exit_px, pnl, gross)
                log(f"CLOSE {p['hyp']} {p['side']} pnl {pnl * 1e4:+.1f} bp")
            elif p["kind"] == "perp_pair" and pd.Timestamp(p.get("exit_due") or now) <= now:
                i = p["info"]
                s, l = i["short"], i["long"]
                qs, ql = (t.get(f"{s}_bid"), t.get(f"{s}_ask")), (t.get(f"{l}_bid"), t.get(f"{l}_ask"))
                if any(x is None or not np.isfinite(x) for x in (*qs, *ql)):
                    continue
                pnl = (i["short_px"] - qs[1]) / i["short_px"] + (ql[0] - i["long_px"]) / i["long_px"] + i["funding_spread"] - 4 * PERP_FEE
                gross = (i["short_mid"] - sum(qs) / 2) / i["short_mid"] + (sum(ql) / 2 - i["long_mid"]) / i["long_mid"] + i["funding_spread"]
                self.book.close(pid, now, None, pnl, gross)
                log(f"CLOSE F01 pnl {pnl * 1e4:+.1f} bp")

    def resolution(self, venue, kind, market, event):
        key = f"{venue}|{market}"
        if key in self.res:
            return self.res[key]
        try:
            o = mk.resolve(self.http, venue, kind, market, event)
        except Exception:
            o = None
        if o is not None:
            self.res[key] = o
            atomic_write(RESOLUTIONS, json.dumps(self.res))
        return o

    def settle(self, ctx):
        for pid, p in list(self.book.pos.items()):
            try:
                self._settle_one(ctx, pid, p)
            except Exception:
                log_exc(f"settle {p.get('hyp')} {p.get('instrument')}")

    def _settle_one(self, ctx, pid, p):
        now = ctx.now
        if True:
            if p["kind"] == "binary":
                st = pd.Timestamp(p["settle_ts"])
                if now < st + pd.Timedelta(minutes=2):
                    return
                legs = p["info"]["legs"]
                outs = [self.resolution(g["venue"], g["kind"], g["market"], g["event"]) for g in legs]
                if any(o is None for o in outs):
                    if now - st > pd.Timedelta(hours=12):
                        self.book.close(pid, now, None, np.nan, np.nan, {"unresolved": True})
                    return
                pay = sum((o if g["side"] == "yes" else 1 - o) for o, g in zip(outs, legs))
                cost = sum(g["price"] for g in legs)
                fee = sum(g["fee"] for g in legs)
                mid = p["info"].get("mid")
                if len(legs) == 1 and mid is not None:
                    gross = pay - (mid if legs[0]["side"] == "yes" else 1 - mid)
                else:
                    gross = pay - cost
                self.book.close(pid, now, pay, pay - cost - fee, gross, {"outcomes": outs})
                log(f"SETTLE {p['hyp']} {p['instrument'][:40]} pays {pay:.0f} pnl {pay - cost - fee:+.3f}")
            elif p["kind"] == "option":
                exp = pd.Timestamp(p["settle_ts"])
                if now < exp + pd.Timedelta(minutes=3):
                    return
                c = self.minutes.close
                w = c[(c.index >= exp - pd.Timedelta(minutes=30)) & (c.index < exp)]
                if len(w) < 25:
                    if now - exp > pd.Timedelta(hours=6):
                        self.book.close(pid, now, None, np.nan, np.nan, {"unresolved": True})
                    return
                ST = float(w.mean())
                k = float(p["info"]["strike"])          # numpy numbers were stored as strings in the JSON
                pay = abs(ST - k)
                sgn = 1 if p["side"] == "long" else -1
                pnl = sgn * (pay - float(p["price"])) - float(p["fee"])
                mid = (float(p["info"]["bid"]) + float(p["info"]["ask"])) / 2
                self.book.close(pid, now, pay, pnl, sgn * (pay - mid), {"S_T": ST})
                log(f"SETTLE O04 {p['venue']} {p['side']} straddle pnl ${pnl:+.0f}")

    # ============================================================== fast loop (thread)
    def fast_loop(self, stop):
        nxt = time.time()
        while not stop.is_set():
            nxt += 15
            try:
                r = feeds.fast_tick(self.http_fast)
                self.fast.append(r)
                append_rows(day_file(FAST, r["ts"]), [r])
                if "D09" not in signals.RETIRED:
                    self.lead_lag(r)
            except Exception:
                log_exc("fast tick")
            time.sleep(max(0.5, nxt - time.time()))

    def fast_markets_loop(self, stop):
        """Every 15 s: quotes of the 5/15-minute binaries, all fair values, the 'fast' rules (V*, A12F)."""
        nxt = time.time()
        while not stop.is_set():
            nxt += 15
            try:
                self.fast_markets_step()
            except Exception:
                log_exc("fast markets")
            time.sleep(max(0.5, nxt - time.time()))

    def fast_markets_step(self):
        now = utcnow()
        if not self.fast or self.ticks is None:
            return
        r = self.fast[-1]
        if (now - r["ts"]).total_seconds() > 30 or "bn" not in r:
            return
        last = self.ticks.iloc[-1].to_dict()
        # spot from the 15-s tick; USD spot = Coinbase, shifted by the minute tick's Coinbase/Kraken basis
        usd = r.get("cb")
        if usd is not None and last.get("usd_mid") and last.get("cb_mid"):
            usd = usd * last["usd_mid"] / last["cb_mid"]
        tick = {**{k: v for k, v in last.items() if k in ("dvol", "usdt_usd")}, "ts": now,
                "bn_bid": r["bn"], "bn_ask": r["bn"], "bn_mid": r["bn"], "usd_mid": usd}
        m = mk.snapshot(self.http_mk, now, fast=True)
        if m.empty:
            return
        ctx = Ctx(now=now, tick=tick, ticks=self.ticks, minutes=self.minutes, markets=m, book=self.book, ml=None,
                  cache=self.cache, lab=self, flow=self.flow)
        add_fairs(ctx, m)
        m["spot_bn"], m["spot_usd"] = r["bn"], usd
        append_rows(day_file(MARKETS_FAST, now), m.to_dict("records"))
        with self.lock:
            for h in signals.REGISTRY:
                if h.loop != "fast" or h.retired:
                    continue
                try:
                    for o in h.fn(ctx) or []:
                        if self.book.submit(o, now, h.family):
                            log(f"OPEN {o.hyp} {o.side} {o.instrument} @ {o.price:.4f}" +
                                (f" edge {o.edge:.4f}" if o.edge is not None else ""))
                except Exception:
                    self.errors.append(f"{now:%H:%M:%S} {h.id}")
                    log_exc(f"rule {h.id}")
            self.book.save()
        mids = {row["market"]: (row["bid"] + row["ask"]) / 2 for _, row in m.iterrows()
                if np.isfinite(row["bid"]) and np.isfinite(row["ask"])}
        # one spot per settlement source would be cleaner; the stale test only needs the size of the move
        self.cache["fast_prev"] = ({"usd": usd, "bn": r["bn"]}, mids)

    def lead_lag(self, r):
        """D09: Coinbase moved more than Binance over the last 15 s by > 3 bp → bet Binance catches up for 60 s."""
        now = r["ts"]
        with self.lock:
            for pid, p in list(self.book.pos.items()):
                if p["kind"] == "fastperp" and pd.Timestamp(p.get("exit_due") or p["open_ts"]) + (pd.Timedelta(0) if p.get("exit_due") else pd.Timedelta(minutes=1)) <= now and "bn" in r:
                    sgn = 1 if p["side"] == "long" else -1
                    gross = sgn * (r["bn"] / p["price"] - 1)
                    self.book.close(pid, now, r["bn"], gross - 2 * PERP_FEE, gross)
            if len(self.fast) < 2 or self.book.open_count("D09"):
                return
            a, b = self.fast[-2], self.fast[-1]
            if not all(k in a and k in b for k in ("bn", "cb")) or (b["ts"] - a["ts"]).total_seconds() > 25:
                return
            d = math.log(b["cb"] / a["cb"]) - math.log(b["bn"] / a["bn"])
            if abs(d) <= 0.0003:
                return
            side = "long" if d > 0 else "short"
            o = Order(hyp="D09", kind="fastperp", venue="binance", instrument="BTCUSDT", side=side, price=b["bn"],
                      fee=PERP_FEE, hold_min=1, unit="return", key=f"D09|{now.floor('s')}", info={"gap": d})
            self.book.submit(o, now, "D")
            self.book.save()

    # ============================================================== hourly and reports
    def zoo5_step(self, ctx):
        self.zoo5(ctx.now)

    def zoo5(self, now):
        """Every 5 minutes: rolling 60-minute window and the window to the next full hour."""
        t0 = now.floor("5min")
        if self.cache.get("zoo5") == t0:
            return
        self.cache["zoo5"] = t0
        wins = [("60m", t0, t0 + pd.Timedelta(minutes=60))]
        nh = t0.floor("h") + pd.Timedelta(hours=1)
        if (nh - t0) >= pd.Timedelta(minutes=10):
            wins.append(("to_hour", t0, nh))
        kv = self.cache.get("kx_level_last")
        kx = kv[1] if kv and now - kv[0] < pd.Timedelta(minutes=3) else None
        try:
            self.zoo.forecast(self.minutes, now, self.cache.get("last_dvol"), windows=wins, kx_level=kx)
        except Exception:
            log_exc("volzoo 5-min")

    def hourly(self, now):
        if self.last_hourly is not None and now - self.last_hourly < pd.Timedelta(minutes=55):
            return
        if not (5 <= now.minute < 9):
            return
        self.last_hourly = now
        try:
            self.zoo.forecast(self.minutes, now, self.ticks["dvol"].dropna().iloc[-1] if self.ticks is not None and "dvol" in self.ticks and self.ticks["dvol"].notna().any() else None)
        except Exception:
            log_exc("volzoo")
        try:
            if not {"D13", "D14"} <= set(signals.RETIRED):     # ML is only used by D13/D14
                self.ml.maybe_train(self.minutes, now)
        except Exception:
            log_exc("ml")
        try:
            self.resolve_observed(now)
        except Exception:
            log_exc("resolve observed markets")
        try:
            from lab import analysis
            analysis.run(self)
        except Exception:
            log_exc("analysis")

    def resolve_observed(self, now, max_calls=150):
        """Settle every market we quoted (not only traded) for the model-vs-market calibration."""
        seen = []
        for folder in (MARKETS, MARKETS_FAST):
            for d in (1, 0):
                f = day_file(folder, now - pd.Timedelta(days=d))
                if f.exists():
                    try:
                        seen.append(pd.read_csv(f, usecols=["venue", "kind", "market", "event", "end"]))
                    except Exception:
                        pass
        if not seen:
            return
        s = pd.concat(seen).drop_duplicates("market")
        s["end"] = pd.to_datetime(s["end"], utc=True)
        s = s[s["end"] < now - pd.Timedelta(minutes=3)]
        calls = 0
        for _, r in s.iterrows():
            if f"{r['venue']}|{r['market']}" in self.res:
                continue
            self.resolution(r["venue"], r["kind"], r["market"], r["event"])
            calls += 1
            if calls >= max_calls:
                break

    def heartbeat(self, now, secs):
        atomic_write(HEARTBEAT, json.dumps({"ts": str(now), "pid": os.getpid(), "loop_seconds": round(secs, 1),
                                            "open_positions": len(self.book.pos), "started": str(self.started),
                                            "markets_quoted": 0 if self.markets is None else len(self.markets),
                                            "recent_rule_errors": list(self.errors)}))

    def report(self, now, force=False):
        if not force and self.last_report is not None and now - self.last_report < pd.Timedelta(minutes=15):
            return
        self.last_report = now
        try:
            from lab import report
            report.write(self, now)
        except Exception:
            log_exc("report")
