"""Alpha on REAL order books: every hypothesis is tested at the quotes the logger actually saw, with fees,
and settled on the actual outcome. All rules and thresholds below were fixed on 28 October 2026, before any
of the books they are applied to existed (PREREGISTERED_ON); nothing here is tuned on the results.

Hypotheses
  A1  no-arbitrage inside one ladder: bid("above K2") > ask("above K1") for K2 > K1 (Polymarket, Kalshi)
  A2  Kalshi "above K" (KXBTCD) vs the sum of Kalshi range buckets above K (KXBTC), same event      [risk-free]
  B1  same settlement, two venues: Kalshi's 12:00 New York hourly ladder prices Polymarket's daily ladder
      in the last hour before noon (both settle at 12:00 ET; BRTI vs Binance basis corrected)
  B2  Polymarket ladder vs the digital price implied by the Deribit volatility SMILE (not just ATM)
  B3  the same with Binance, OKX and Bybit options as the options venue (one test per venue)
  C1  favourite-longshot bias at real bids: sell every YES quoted at <= 5c (and buy every YES at >= 95c)
  D1  other Bitcoin markets (ranges, barriers, up/down, other Kalshi series) vs the Deribit smile
  E1  stale quotes (fast logger): quote vs fair price from the current Binance price; 30 s / 60 s mark-outs
  A3  Limitless vs Polymarket on the SAME Up-or-Down market (Limitless mirrors Polymarket): crossing quotes
  F1  Limitless quotes vs Polymarket's mid on the same market, traded at Limitless, held to the candle close
Trading rules: a signal needs |fair - quote| >= threshold beyond the bid/ask; only the FIRST signal per
(event, strike, side) is traded, held to settlement; profit per contract after fees; t-statistics over
settlement events (HAC); Holm correction across the trading hypotheses.

    python alpha_live.py        -> results/tables/alpha_live_*.csv, reports/alpha_live.md
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import config
from src import evaluate
from src.measure.ladder import monotone
from src.utils import Http, ensure, log, read_table, utcnow

PREREGISTERED_ON = "2026-10-28"
TH = {"B1": 0.03, "B2": 0.03, "B3": 0.03, "C1": 0.05, "D1": 0.05, "E1": 0.02, "F1": 0.03}
KALSHI_FEE_RATE = 0.07
PM_FEE = 0.0            # Polymarket charges no fee on these markets (set here if that changes)
LIMITLESS_FEE = 0.0     # Limitless taker fee per contract is not public in the API: edges are reported before it
MAX_QUOTE_AGE = pd.Timedelta(minutes=10)


def kalshi_fee(p):
    """Kalshi taker fee for a one-contract order: 0.07 * P * (1 - P), rounded UP to the next cent."""
    p = np.asarray(p, float)
    return np.ceil(KALSHI_FEE_RATE * p * (1 - p) * 100 - 1e-9) / 100


def fee(venue, p):
    if venue == "kalshi":
        return kalshi_fee(p)
    return np.full(np.shape(p), LIMITLESS_FEE if venue == "limitless" else PM_FEE, float)


# ------------------------------------------------------------------ loading the logs
def load(src: str) -> pd.DataFrame:
    files = sorted((config.LIVE / src).glob("*.csv.gz")) + sorted((config.DATA / "live_logger_root" / src).glob("*.csv.gz"))
    parts = []
    for f in files:
        try:
            parts.append(pd.read_csv(f, low_memory=False))
        except Exception as e:
            log(f"skip {f.name}: {e!r}")
    if not parts:
        return pd.DataFrame()
    d = pd.concat(parts, ignore_index=True).drop_duplicates()
    for c in d.columns:
        if c in ("snap_ts", "ts", "settle_ts", "close_ts", "end_ts", "start_ts", "expiry"):
            d[c] = pd.to_datetime(d[c], utc=True, errors="coerce", format="mixed")
    return d


# ------------------------------------------------------------------ outcomes
class Outcomes:
    """Settlement values, fetched once and cached in data/live/outcomes.json."""

    def __init__(self):
        self.path = config.LIVE / "outcomes.json"
        try:
            self.c = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            self.c = {}
        self.http = Http(min_interval=0.06, retries=2)
        self._minutes = None
        self.dirty = False

    def save(self):
        if self.dirty:
            ensure(self.path.parent)
            self.path.write_text(json.dumps(self.c), encoding="utf-8")

    def binance_close(self, open_ts: pd.Timestamp):
        """Close of the Binance BTC/USDT 1-minute candle that opens at open_ts (Polymarket's settlement)."""
        key = f"binance|{open_ts:%Y-%m-%dT%H:%M}"
        if key in self.c:
            return self.c[key]
        if open_ts + pd.Timedelta(minutes=1) > pd.Timestamp(utcnow()):
            return None
        val = None
        try:
            if self._minutes is None:
                self._minutes = read_table(config.INTERIM / "minutes").set_index("open_time")["close"]
            if open_ts in self._minutes.index:
                val = float(self._minutes.loc[open_ts])
        except Exception:
            pass
        if val is None:
            try:
                from src.download import binance
                res = binance.api_get(self.http, "/api/v3/klines", {"symbol": "BTCUSDT", "interval": "1m",
                                                                   "startTime": int(open_ts.value // 1_000_000), "limit": 1})
                if res and int(res[0][0]) // 1000 == int(open_ts.value // 1_000_000_000):
                    val = float(res[0][4])
            except Exception as e:
                log(f"binance close {open_ts}: {e!r}"[:150])
        if val is not None:
            self.c[key] = val
            self.dirty = True
        return val

    def kalshi_event(self, event: str) -> dict:
        """{strike_or_ticker: 1/0} plus '_value' (BRTI settlement) for a settled Kalshi event."""
        key = f"kalshi|{event}"
        if key in self.c:
            return self.c[key]
        from src.download import kalshi
        try:
            mk, _ = kalshi.markets(self.http, event)
        except Exception as e:
            log(f"kalshi {event}: {e!r}"[:150])
            return {}
        out = {}
        for m in mk:
            if m.get("result") in ("yes", "no"):
                out[m["ticker"]] = 1.0 if m["result"] == "yes" else 0.0
            if m.get("expiration_value") not in (None, ""):
                try:
                    out["_value"] = float(m["expiration_value"])
                except (TypeError, ValueError):
                    pass
        if out:
            self.c[key] = out
            self.dirty = True
        return out

    def pm_market(self, slug: str, token: str):
        key = f"pm|{token}"
        if key in self.c:
            return self.c[key]
        try:
            res = self.http.get("https://gamma-api.polymarket.com/markets", {"clob_token_ids": token}) or []
            m = res[0] if res else None
            if m and m.get("closed") and m.get("outcomePrices"):
                op = [float(x) for x in json.loads(m["outcomePrices"])]
                if max(op) >= 0.99:
                    self.c[key] = 1.0 if op[0] >= 0.99 else 0.0
                    self.dirty = True
                    return self.c[key]
        except Exception as e:
            log(f"pm outcome {slug}: {e!r}"[:150])
        return None


# ------------------------------------------------------------------ fair prices
def black_call(F, K, w):
    w = np.maximum(w, 1e-12)
    d1 = (np.log(F / K) + w / 2) / np.sqrt(w)
    return F * stats.norm.cdf(d1) - K * stats.norm.cdf(d1 - np.sqrt(w))


class Smile:
    """Implied-volatility surface from one option-chain snapshot. Total variance is interpolated linearly in
    time at fixed log-moneyness between the expiries around T; P(S_T > K) = -dC/dK on that surface (so the
    skew enters the digital price, unlike a lognormal with the ATM volatility)."""

    def __init__(self, chain: pd.DataFrame, snap: pd.Timestamp, venue: str = "deribit"):
        self.snap, self.exp = snap, []
        if chain is None or chain.empty:
            return
        c = chain.copy()
        if venue == "deribit":
            parts = c["instrument"].str.split("-", expand=True)
            c["expiry"] = pd.to_datetime(parts[1], format="%d%b%y", utc=True, errors="coerce") + pd.Timedelta(hours=8)
            c["strike"] = pd.to_numeric(parts[2], errors="coerce")
            c["cp"] = parts[3]
            c["iv"] = c["mark_iv"] / 100
            c["F"] = c["underlying"]
        elif venue == "binance":
            c["iv"] = pd.to_numeric(c["markIV"], errors="coerce")
            c["iv"] = np.where(c["iv"] > 3, c["iv"] / 100, c["iv"])
            c["F"] = pd.to_numeric(c.get("indexPrice"), errors="coerce")
        else:  # normalised chains (okx, bybit): expiry, strike, cp, iv, F already there
            c["iv"] = pd.to_numeric(c["iv"], errors="coerce")
            c["iv"] = np.where(c["iv"] > 3, c["iv"] / 100, c["iv"])
            c["F"] = pd.to_numeric(c["F"], errors="coerce")
        c = c.dropna(subset=["expiry", "strike", "iv", "F"])
        c = c[(c["iv"] > 0.05) & (c["iv"] < 5) & (c["expiry"] > snap + pd.Timedelta(minutes=30))]
        for E, g in c.groupby("expiry"):
            F = float(g["F"].median())
            otm = g[((g["cp"] == "C") & (g["strike"] >= F)) | ((g["cp"] == "P") & (g["strike"] < F))]
            if len(otm) < 4:
                continue
            sm = otm.groupby("strike")["iv"].mean().sort_index()
            tau = (E - snap).total_seconds() / (365 * 86400)
            self.exp.append({"E": E, "F": F, "k": np.log(sm.index.to_numpy(float) / F),
                             "w": sm.to_numpy(float) ** 2 * tau, "tau": tau})
        self.exp.sort(key=lambda x: x["E"])

    def _w(self, e, k):
        return np.interp(k, e["k"], e["w"])   # flat extrapolation in total variance beyond the listed strikes

    def digital(self, T: pd.Timestamp, K: np.ndarray, spot_scale: float = 1.0) -> np.ndarray:
        K = np.asarray(K, float)
        if not self.exp or T <= self.snap:
            return np.full(len(K), np.nan)
        after = [e for e in self.exp if e["E"] >= T]
        before = [e for e in self.exp if e["E"] < T]
        if not after:
            return np.full(len(K), np.nan)
        e2 = after[0]
        tau_T = (T - self.snap).total_seconds() / (365 * 86400)
        if before:
            e1 = before[-1]
            a = (T - e1["E"]).total_seconds() / (e2["E"] - e1["E"]).total_seconds()
            F = (e1["F"] + a * (e2["F"] - e1["F"])) * spot_scale
            wfun = lambda kk: self._w(e1, kk) + a * (self._w(e2, kk) - self._w(e1, kk))   # noqa: E731
        else:
            F = e2["F"] * spot_scale
            wfun = lambda kk: self._w(e2, kk) * tau_T / e2["tau"]   # noqa: E731
        h = 0.0015 * K
        Ku, Kd = K + h, K - h
        cu = black_call(F, Ku, wfun(np.log(Ku / F)))
        cd = black_call(F, Kd, wfun(np.log(Kd / F)))
        return np.clip(-(cu - cd) / (2 * h), 0, 1)


def interp_ladder(strikes, probs, K):
    """Interpolate P(S > K) inside a ladder: probit linear in log strike (as in the BJN code); NaN outside."""
    s = np.asarray(strikes, float)
    o = np.argsort(s)
    s, p = s[o], monotone(np.asarray(probs, float)[o])
    ok = (p > 0.003) & (p < 0.997)
    if ok.sum() < 2:
        return np.full(np.shape(K), np.nan)
    x, z = np.log(s[ok]), stats.norm.ppf(1 - p[ok])
    z = np.maximum.accumulate(z + np.arange(len(z)) * 1e-9)
    K = np.asarray(K, float)
    zk = np.interp(np.log(K), x, z, left=np.nan, right=np.nan)
    return 1 - stats.norm.cdf(zk)


# ------------------------------------------------------------------ trades and statistics
def first_signals(sig: pd.DataFrame) -> pd.DataFrame:
    """Only the first qualifying signal per (event, strike, side) is traded and held to settlement."""
    if sig.empty:
        return sig
    return sig.sort_values("snap_ts").groupby(["event", "strike", "side"], as_index=False).head(1)


def make_signals(d: pd.DataFrame, th: float, hyp: str) -> pd.DataFrame:
    """d: venue, event, strike, snap_ts, settle_ts, bid, ask, fair. Buy YES at ask when fair - ask >= th + fee,
    sell YES at bid when bid - fair >= th + fee."""
    if d.empty:
        return pd.DataFrame()
    d = d.dropna(subset=["fair"]).copy()
    fb = np.array([fee(v, a) for v, a in zip(d["venue"], d["ask"].fillna(0.5))]).ravel()
    fs = np.array([fee(v, b) for v, b in zip(d["venue"], d["bid"].fillna(0.5))]).ravel()
    buy = d[(d["ask"] > 0) & (d["ask"] < 1) & (d["fair"] - d["ask"] - fb >= th)].assign(side="buy", price=lambda x: x["ask"])
    sell = d[(d["bid"] > 0) & (d["bid"] < 1) & (d["bid"] - d["fair"] - fs >= th)].assign(side="sell", price=lambda x: x["bid"])
    s = pd.concat([buy, sell], ignore_index=True)
    s["hyp"] = hyp
    return first_signals(s)


def settle(tr: pd.DataFrame) -> pd.DataFrame:
    if tr.empty or "outcome" not in tr:
        return tr
    tr = tr.copy()
    f = np.array([fee(v, p) for v, p in zip(tr["venue"], tr["price"])]).ravel()
    tr["fee"] = f
    tr["pnl"] = np.where(tr["side"] == "buy", tr["outcome"] - tr["price"] - f, tr["price"] - tr["outcome"] - f)
    return tr


def summarize(tr: pd.DataFrame, by=("hyp", "venue")) -> pd.DataFrame:
    rows = []
    if tr.empty:
        return pd.DataFrame()
    done = tr.dropna(subset=["pnl"])
    for key, g in tr.groupby(list(by)):
        gd = done[np.all([done[c] == v for c, v in zip(by, key if isinstance(key, tuple) else (key,))], axis=0)]
        per_event = gd.groupby("event")["pnl"].sum().reindex(gd.sort_values("settle_ts")["event"].unique())
        m, se, n = evaluate.hac_mean(per_event.to_numpy(float), evaluate.nw_lag(len(per_event), 1)) if len(per_event) else (np.nan, np.nan, 0)
        t = m / se if se and np.isfinite(se) and se > 0 else np.nan
        rows.append({**dict(zip(by, key if isinstance(key, tuple) else (key,))), "signals": len(g), "settled": len(gd),
                     "events": int(gd["event"].nunique()), "pnl_per_contract": gd["pnl"].mean() if len(gd) else np.nan,
                     "hit_rate": (gd["pnl"] > 0).mean() if len(gd) else np.nan, "pnl_total": gd["pnl"].sum(),
                     "pnl_per_event": m, "t": t, "p_one_sided": (1 - stats.norm.cdf(t)) if np.isfinite(t) else np.nan})
    out = pd.DataFrame(rows)
    # Holm across every tested family that has a p-value
    ok = out["p_one_sided"].notna()
    if ok.any():
        p = out.loc[ok, "p_one_sided"].to_numpy()
        order = np.argsort(p)
        m = len(p)
        adj = np.empty(m)
        run = 0.0
        for rank, i in enumerate(order):
            run = max(run, min(1.0, (m - rank) * p[i]))
            adj[i] = run
        out.loc[ok, "p_holm"] = adj
    return out


# ------------------------------------------------------------------ hypotheses
def ladders_pm(pm: pd.DataFrame) -> pd.DataFrame:
    if pm.empty:
        return pm
    d = pm.rename(columns={"event_slug": "event"}).copy()
    d["venue"] = "polymarket"
    return d[["venue", "event", "snap_ts", "settle_ts", "strike", "bid", "ask", "bid_size", "ask_size"]]


def ladders_pm_other(po: pd.DataFrame, lp: pd.DataFrame) -> pd.DataFrame:
    """Polymarket "above" ladders seen through the gamma API (best bid / ask): the HOURLY ladders, and any daily
    ladder snapshot that pm_books (full CLOB books) does not already have."""
    d = po[po["kind"] == "above_ladder"].rename(columns={"event_slug": "event", "best_bid": "bid", "best_ask": "ask",
                                                         "end_ts": "settle_ts", "level1": "strike"}).copy()
    if d.empty:
        return pd.DataFrame()
    d["venue"] = "polymarket"
    d["bid_size"] = np.nan
    d["ask_size"] = np.nan
    if len(lp):
        have = set(zip(lp["event"], lp["snap_ts"]))
        d = d[[(e, t) not in have for e, t in zip(d["event"], d["snap_ts"])]]
    return d[["venue", "event", "snap_ts", "settle_ts", "strike", "bid", "ask", "bid_size", "ask_size", "yes_token"]]


def f1_limitless(lim: pd.DataFrame, po: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Limitless mirrors Polymarket's Up-or-Down markets (same Binance candle; metadata.externalSlug is the
    Polymarket slug). A3: crossing quotes between the two are risk-free (buy Up where cheap, sell where dear).
    F1: trade Limitless at its quotes when Polymarket's mid differs by >= 3c, hold to the candle close."""
    if lim.empty or po.empty:
        return pd.DataFrame(), pd.DataFrame()
    L = lim[lim["kind"] == "updown"].dropna(subset=["external_slug"]).copy()
    P = po[po["kind"].isin(["updown", "updown_15m"])][["snap_ts", "event_slug", "best_bid", "best_ask"]]
    j = L.merge(P, left_on=["external_slug", "snap_ts"], right_on=["event_slug", "snap_ts"], how="inner")
    if j.empty:
        return pd.DataFrame(), pd.DataFrame()
    arb = []
    for _, r in j.iterrows():
        for edge, how in ((r["best_bid"] - r["yes_ask"] - LIMITLESS_FEE, "buy Up on Limitless, sell on Polymarket"),
                          (r["yes_bid"] - r["best_ask"] - LIMITLESS_FEE, "buy Up on Polymarket, sell on Limitless")):
            if np.isfinite(edge) and edge > 0:
                arb.append({"hyp": "A3", "venue": "limitless+polymarket", "event": r["external_slug"],
                            "snap_ts": r["snap_ts"], "duration": r["duration"], "how": how, "edge": edge,
                            "profit_usd": edge})
    q = j.rename(columns={"external_slug": "event", "yes_bid": "bid", "yes_ask": "ask", "end_ts": "settle_ts"})
    q["venue"] = "limitless"
    q["fair"] = (q["best_bid"] + q["best_ask"]) / 2
    q["strike"] = q["open_price"]          # Up = Binance close >= open of the candle
    sig = make_signals(q[["venue", "event", "snap_ts", "settle_ts", "strike", "bid", "ask", "fair", "window_open",
                          "duration"]], TH["F1"], "F1")
    return sig, pd.DataFrame(arb)


def ladders_kx(kx: pd.DataFrame) -> pd.DataFrame:
    if kx.empty:
        return kx
    d = kx.rename(columns={"yes_bid": "bid", "yes_ask": "ask", "yes_bid_size": "bid_size", "yes_ask_size": "ask_size"}).copy()
    d["venue"] = "kalshi"
    return d[["venue", "event", "ticker", "snap_ts", "settle_ts", "strike", "bid", "ask", "bid_size", "ask_size"]]


def a1_monotonicity(lad: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (v, ev, ts), g in lad.dropna(subset=["bid", "ask"]).groupby(["venue", "event", "snap_ts"]):
        g = g.sort_values("strike")
        k, b, a = g["strike"].to_numpy(), g["bid"].to_numpy(), g["ask"].to_numpy()
        bs = g["bid_size"].to_numpy() if "bid_size" in g else np.full(len(g), np.nan)
        as_ = g["ask_size"].to_numpy() if "ask_size" in g else np.full(len(g), np.nan)
        for i in range(len(g)):
            for j in range(i + 1, len(g)):
                edge = b[j] - a[i] - float(fee(v, b[j])) - float(fee(v, a[i]))
                if edge > 0:
                    rows.append({"hyp": "A1", "venue": v, "event": ev, "snap_ts": ts, "k_low": k[i], "k_high": k[j],
                                 "buy_ask_low": a[i], "sell_bid_high": b[j], "edge": edge,
                                 "size": np.nanmin([bs[j], as_[i]]),
                                 "profit_usd": edge * np.nanmin([bs[j], as_[i]])})
    return pd.DataFrame(rows)


def box_arbitrage(above: pd.DataFrame, buckets: pd.DataFrame, tol: float = 1.0, feef=None) -> list[dict]:
    """One snapshot of one event. 'above' columns strike, bid, ask; 'buckets' floor, cap, bid, ask.
    1{K1 < S <= K2} = above(K1) - above(K2) = sum of the buckets that tile (K1, K2], so
      long  box: buy above K1, sell above K2, sell the buckets  -> sum(bucket bids) - ask(K1) + bid(K2) - fees
      short box: sell above K1, buy above K2, buy the buckets   -> bid(K1) - ask(K2) - sum(bucket asks) - fees
    Both are risk-free when positive. Kalshi fees are paid on every leg."""
    out = []
    kalshi_fee = feef or globals()["kalshi_fee"]     # the fee of the venue (Polymarket: none)
    a = above.dropna(subset=["strike"]).sort_values("strike")
    bk = buckets.dropna(subset=["floor", "cap"]).sort_values("floor")
    if a.empty or bk.empty:
        return out
    ks = a["strike"].to_numpy(float)
    for i in range(len(ks)):
        for j in range(i + 1, min(i + 8, len(ks))):
            k1, k2 = ks[i], ks[j]
            inside = bk[(bk["floor"] >= k1 - tol) & (bk["cap"] <= k2 + tol)]
            if inside.empty:
                continue
            fl, cp = inside["floor"].to_numpy(float), inside["cap"].to_numpy(float)
            tiles = abs(fl[0] - k1) <= tol and abs(cp[-1] - k2) <= tol and np.all(np.abs(fl[1:] - cp[:-1]) <= tol)
            if not tiles:
                continue
            r1, r2 = a.iloc[i], a.iloc[j]
            long_ = inside["bid"].sum() - r1["ask"] + r2["bid"] - kalshi_fee(inside["bid"]).sum() - kalshi_fee(r1["ask"]) - kalshi_fee(r2["bid"])
            short = r1["bid"] - r2["ask"] - inside["ask"].sum() - kalshi_fee(inside["ask"]).sum() - kalshi_fee(r1["bid"]) - kalshi_fee(r2["ask"])
            for edge, how in ((long_, "long box"), (short, "short box")):
                if np.isfinite(edge) and edge > 0:
                    out.append({"k1": k1, "k2": k2, "buckets": len(inside), "how": how, "edge": float(edge)})
    return out


def a2_ranges(kx: pd.DataFrame, kr: pd.DataFrame) -> pd.DataFrame:
    """Kalshi "above" ladder (KXBTCD) vs range buckets (KXBTC) of the same event: box arbitrage."""
    if kx.empty or kr.empty:
        return pd.DataFrame()
    rows = []
    kr = kr.copy()
    kr["floor"] = kr["strike"]
    kr["cap"] = pd.to_numeric(kr.get("cap_strike"), errors="coerce")
    kr = kr.rename(columns={"yes_bid": "bid", "yes_ask": "ask"})
    k = kx.rename(columns={"yes_bid": "bid", "yes_ask": "ask"})
    for (ev, ts), g in k.groupby(["event", "snap_ts"]):
        r = kr[(kr["event"] == str(ev).replace("KXBTCD", "KXBTC")) & (kr["snap_ts"] == ts)]
        for x in box_arbitrage(g, r):
            rows.append({"hyp": "A2", "venue": "kalshi", "event": ev, "snap_ts": ts, **x, "profit_usd": x["edge"]})
    return pd.DataFrame(rows)


def a4_pm_ranges(lp: pd.DataFrame, po: pd.DataFrame) -> pd.DataFrame:
    """Polymarket's own two products on one settlement: "Bitcoin price on <date>" range buckets vs the
    "Bitcoin above ___ on <date>" ladder. Same box as A2, Polymarket fees (none)."""
    if lp.empty or po.empty:
        return pd.DataFrame()
    r = po[(po["kind"] == "range") & po["title"].astype(str).str.lower().str.contains("bitcoin price on")].copy()
    if r.empty:
        return pd.DataFrame()
    r = r.rename(columns={"best_bid": "bid", "best_ask": "ask", "level1": "floor", "level2": "cap", "end_ts": "settle_ts"})
    rows = []
    zero = lambda p: np.zeros(np.shape(p)) if np.ndim(p) else 0.0   # noqa: E731
    for (T, ts), g in lp[lp["venue"] == "polymarket"].groupby(["settle_ts", "snap_ts"]):
        rr = r[(r["snap_ts"] == ts) & ((r["settle_ts"] - T).abs() <= pd.Timedelta(minutes=5))]
        if rr.empty:
            continue
        for x in box_arbitrage(g, rr.dropna(subset=["floor", "cap"]), tol=1.0, feef=zero):
            rows.append({"hyp": "A4", "venue": "polymarket", "event": g["event"].iloc[0], "snap_ts": ts, **x,
                         "profit_usd": x["edge"]})
    return pd.DataFrame(rows)


def b1_cross_venue(pm: pd.DataFrame, kx: pd.DataFrame, basis: float) -> pd.DataFrame:
    """Polymarket daily ladder priced by the Kalshi hourly ladder with the same settlement time."""
    if pm.empty or kx.empty:
        return pd.DataFrame()
    out = []
    for (ev, ts), g in pm.groupby(["event", "snap_ts"]):
        T = g["settle_ts"].iloc[0]
        if not (pd.Timedelta(0) < T - ts <= pd.Timedelta(minutes=60)):
            continue
        k = kx[(kx["settle_ts"] == T) & ((kx["snap_ts"] - ts).abs() <= MAX_QUOTE_AGE)]
        if k.empty:
            continue
        k = k[k["snap_ts"] == k["snap_ts"].iloc[(k["snap_ts"] - ts).abs().argmin()]]
        mid = ((k["bid"] + k["ask"]) / 2).to_numpy()
        g = g.copy()
        # Polymarket pays on Binance > K; Kalshi on BRTI > K' with BRTI ~ basis * Binance
        g["fair"] = interp_ladder(k["strike"], mid, g["strike"].to_numpy(float) * basis)
        out.append(g)
    return make_signals(pd.concat(out, ignore_index=True), TH["B1"], "B1") if out else pd.DataFrame()


def b2_options(pm: pd.DataFrame, chain: pd.DataFrame, px: pd.DataFrame, venue: str, hyp: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Polymarket ladder vs the options smile at the nearest chain snapshot (<= 10 min). Also returns every
    quote with its fair price, for the accuracy comparison."""
    if pm.empty or chain.empty:
        return pd.DataFrame(), pd.DataFrame()
    snaps = pd.Series(pd.to_datetime(chain["snap_ts"].dropna().unique(), utc=True)).sort_values().reset_index(drop=True)
    out, cache = [], {}
    for ts, g in pm.groupby("snap_ts"):
        if snaps.empty:
            break
        j = (snaps - ts).abs().idxmin()
        cs = snaps[j]
        if abs(cs - ts) > MAX_QUOTE_AGE:
            continue
        if cs not in cache:
            cache[cs] = Smile(chain[chain["snap_ts"] == cs], cs, venue)
        sm = cache[cs]
        # Binance (settlement) / options index (USD) at the snapshot
        scale = 1.0
        if px is not None and len(px) and sm.exp:
            p = px.iloc[(px["snap_ts"] - ts).abs().argmin()]
            if abs(p["snap_ts"] - ts) <= MAX_QUOTE_AGE:
                F0 = sm.exp[0]["F"]
                scale = float(p["price"]) / F0 if F0 else 1.0
                scale = scale if 0.98 < scale < 1.02 else 1.0
        for T, gg in g.groupby("settle_ts"):
            gg = gg.copy()
            gg["fair"] = sm.digital(T, gg["strike"].to_numpy(float), scale)
            gg["horizon_h"] = (T - ts).total_seconds() / 3600
            out.append(gg)
    allq = pd.concat(out, ignore_index=True) if out else pd.DataFrame()
    if allq.empty:
        return pd.DataFrame(), allq
    allq = allq[allq["horizon_h"] >= 2]          # the last hours belong to B1
    return make_signals(allq, TH[hyp], hyp), allq


def c1_wings(lad: pd.DataFrame) -> pd.DataFrame:
    d = lad.dropna(subset=["bid", "ask"]).copy()
    sell = d[(d["bid"] > 0) & (d["bid"] <= 0.05)].assign(side="sell", price=lambda x: x["bid"])
    buy = d[(d["ask"] >= 0.95) & (d["ask"] < 1)].assign(side="buy", price=lambda x: x["ask"])
    s = pd.concat([sell, buy], ignore_index=True)
    s["hyp"] = "C1"
    return first_signals(s)


def d1_other(po: pd.DataFrame, ko: pd.DataFrame, chain: pd.DataFrame, px: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Other Bitcoin markets priced by the Deribit smile: ranges = D(K1) - D(K2); 'above' = D(K);
    barriers: reflection approximation P(max >= B) ~ 2 P(S_T >= B) for B above the price (and the mirror
    for B below). Up/down markets need the opening price and are left to the descriptive table."""
    rows = []
    if po is not None and len(po):
        q = po.rename(columns={"event_slug": "event", "best_bid": "bid", "best_ask": "ask", "end_ts": "settle_ts"}).copy()
        q["venue"] = "polymarket"
        q["strike"] = q["level1"]
        rows.append(q[["venue", "event", "kind", "snap_ts", "settle_ts", "strike", "level2", "bid", "ask", "yes_token", "question"]])
    if ko is not None and len(ko):
        q = ko.rename(columns={"yes_bid": "bid", "yes_ask": "ask", "close_ts": "settle_ts"}).copy()
        q["venue"] = "kalshi"
        q["kind"] = np.where(q["strike_type"].astype(str).str.contains("between"), "range",
                             np.where(q["strike_type"].astype(str).str.contains("greater"), "above_other",
                                      np.where(q["strike_type"].astype(str).str.contains("less"), "below_other", "other")))
        q["strike"] = q["floor_strike"].fillna(q["cap_strike"])
        q["level2"] = q["cap_strike"]
        q["question"] = q["title"]
        rows.append(q[["venue", "event", "kind", "snap_ts", "settle_ts", "strike", "level2", "bid", "ask", "ticker", "question"]])
    if not rows or chain.empty:
        return pd.DataFrame(), pd.DataFrame()
    q = pd.concat(rows, ignore_index=True)
    q = q[q["kind"].isin(["range", "above_other", "below_other", "barrier"]) & q["strike"].notna()]
    snaps = pd.Series(pd.to_datetime(chain["snap_ts"].unique(), utc=True)).sort_values().reset_index(drop=True)
    out, cache = [], {}
    for ts, g in q.groupby("snap_ts"):
        j = (snaps - ts).abs().idxmin() if len(snaps) else None
        if j is None or abs(snaps[j] - ts) > MAX_QUOTE_AGE:
            continue
        cs = snaps[j]
        if cs not in cache:
            cache[cs] = Smile(chain[chain["snap_ts"] == cs], cs)
        sm = cache[cs]
        spot = sm.exp[0]["F"] if sm.exp else np.nan
        for T, gg in g.dropna(subset=["settle_ts"]).groupby("settle_ts"):
            gg = gg.copy()
            d1 = sm.digital(T, gg["strike"].to_numpy(float))
            d2 = sm.digital(T, gg["level2"].fillna(gg["strike"]).to_numpy(float))
            fair = np.where(gg["kind"] == "range", d1 - d2, d1)
            fair = np.where(gg["kind"] == "below_other", 1 - d1, fair)
            up = gg["strike"].to_numpy(float) >= spot
            fair = np.where(gg["kind"] == "barrier", np.where(up, np.minimum(1, 2 * d1), np.minimum(1, 2 * (1 - d1))), fair)
            gg["fair"] = fair
            out.append(gg)
    allq = pd.concat(out, ignore_index=True) if out else pd.DataFrame()
    if allq.empty:
        return pd.DataFrame(), allq
    return make_signals(allq, TH["D1"], "D1"), allq


def e1_stale(fast: pd.DataFrame) -> pd.DataFrame:
    """Fast-logger quotes vs the fair price from the CURRENT Binance price, with the width of the ladder's own
    implied distribution from the same snapshot (so only the centre is 'news'). Mark-outs after 30 and 60 s."""
    if fast.empty:
        return pd.DataFrame()
    q = fast[fast["venue"] != "binance"].dropna(subset=["bid", "ask", "strike"]).copy()
    q["mid"] = (q["bid"] + q["ask"]) / 2
    rows = []
    for (v, ev, ts), g in q.groupby(["venue", "event", "ts"]):
        S = float(g["btc"].iloc[0])
        T = pd.to_datetime(g["settle_ts"].iloc[0], utc=True)
        tau = (T - ts).total_seconds() / (365 * 86400)
        if tau <= 0 or len(g) < 3:
            continue
        mid = g["mid"].to_numpy(float)
        ok = (mid > 0.03) & (mid < 0.97)
        if ok.sum() < 2:
            continue
        z = stats.norm.ppf(1 - mid[ok])
        slope = np.polyfit(np.log(g["strike"].to_numpy(float)[ok]), z, 1)[0]   # 1 / (sigma sqrt(tau))
        if slope <= 0:
            continue
        sd = 1 / slope
        g = g.copy()
        g["fair"] = 1 - stats.norm.cdf((np.log(g["strike"].to_numpy(float) / S) + sd ** 2 / 2) / sd)
        rows.append(g)
    if not rows:
        return pd.DataFrame()
    d = pd.concat(rows, ignore_index=True)
    d = d.sort_values("ts")
    for hsec in (30, 60):
        fut = d[["venue", "event", "strike", "ts", "mid"]].copy()
        fut["ts"] = fut["ts"] - pd.Timedelta(seconds=hsec)
        d = pd.merge_asof(d.sort_values("ts"), fut.sort_values("ts").rename(columns={"mid": f"mid_{hsec}s"}),
                          on="ts", by=["venue", "event", "strike"], direction="forward", tolerance=pd.Timedelta(seconds=15))
    fb = fee_vec(d["venue"], d["ask"])
    fs = fee_vec(d["venue"], d["bid"])
    buy = d["fair"] - d["ask"] - fb >= TH["E1"]
    sell = d["bid"] - d["fair"] - fs >= TH["E1"]
    d = d[buy | sell].copy()
    d["side"] = np.where(buy[buy | sell], "buy", "sell")
    for hsec in (30, 60):
        d[f"markout_{hsec}s"] = np.where(d["side"] == "buy", d[f"mid_{hsec}s"] - d["ask"], d["bid"] - d[f"mid_{hsec}s"])
    d["hyp"] = "E1"
    return d


def fee_vec(venue, p):
    return np.array([float(fee(v, x)) for v, x in zip(venue, p.fillna(0.5))])


# ------------------------------------------------------------------ main
def basis_estimate(kx: pd.DataFrame, oc: "Outcomes") -> float:
    """Median BRTI / Binance over the Kalshi settlements in the logs (BRTI = Kalshi's expiration value, Binance =
    close of the minute before settlement). Falls back to 1 with fewer than 5 settlements."""
    if kx.empty:
        return 1.0
    ratios = []
    now = pd.Timestamp(utcnow())
    for ev, g in kx.groupby("event"):
        T = g["settle_ts"].iloc[0]
        if pd.isna(T) or T > now:
            continue
        v = oc.kalshi_event(str(ev)).get("_value")
        b = oc.binance_close(T - pd.Timedelta(minutes=1))
        if v and b:
            ratios.append(v / b)
    return float(np.median(ratios)) if len(ratios) >= 5 else 1.0


def attach_outcomes(tr: pd.DataFrame, oc: Outcomes) -> pd.DataFrame:
    if tr.empty:
        return tr
    tr = tr.copy()
    vals = []
    now = pd.Timestamp(utcnow())
    for _, r in tr.iterrows():
        y = np.nan
        T = r.get("settle_ts")
        if pd.isna(T) or T > now:
            vals.append(y)
            continue
        if r["venue"] == "limitless" and pd.notna(r.get("strike")):
            S = oc.binance_close(T - pd.Timedelta(minutes=1))      # close of the candle = close of its last minute
            y = float(S >= r["strike"]) if S is not None else np.nan
        elif r["venue"] == "polymarket" and str(r["hyp"]).split()[0] in ("B1", "B2", "B3", "C1", "A1") and pd.notna(r.get("strike")):
            S = oc.binance_close(T)
            y = float(S > r["strike"]) if S is not None else np.nan
        elif r["venue"] == "polymarket" and pd.notna(r.get("yes_token")):
            v = oc.pm_market(r.get("event"), str(r["yes_token"]))
            y = v if v is not None else np.nan
        elif r["venue"] == "kalshi":
            res = oc.kalshi_event(str(r["event"]))
            if "ticker" in r and pd.notna(r.get("ticker")) and r["ticker"] in res:
                y = res[r["ticker"]]
            elif "_value" in res and pd.notna(r.get("strike")):
                y = float(res["_value"] > r["strike"])
        vals.append(y)
    tr["outcome"] = vals
    return tr


def accuracy(allq: pd.DataFrame, label: str) -> pd.DataFrame:
    """Brier of the market mid and of the fair price on the same settled quotes."""
    if allq is None or allq.empty or "outcome" not in allq:
        return pd.DataFrame()
    d = allq.dropna(subset=["outcome", "fair", "bid", "ask"]).copy()
    if d.empty:
        return pd.DataFrame()
    d["mid"] = (d["bid"] + d["ask"]) / 2
    return pd.DataFrame([{"comparison": label, "quotes": len(d), "events": d["event"].nunique(),
                          "brier_market_mid": ((d["mid"] - d["outcome"]) ** 2).mean(),
                          "brier_fair": ((d["fair"] - d["outcome"]) ** 2).mean(),
                          "mean_gap_mid_minus_fair": (d["mid"] - d["fair"]).mean()}])


def main(argv=None):
    oc = Outcomes()
    pm, kx, kr = load("pm_books"), load("kalshi"), load("kalshi_range")
    chain, px, bo = load("deribit_chain"), load("binance"), load("binance_options")
    po, ko, fast = load("pm_other"), load("kalshi_other"), load("fast")
    log(f"alpha_live: pm {len(pm)}, kalshi {len(kx)}, kalshi_range {len(kr)}, deribit {len(chain)}, binance_options {len(bo)}, "
        f"pm_other {len(po)}, kalshi_other {len(ko)}, fast {len(fast)} rows")
    lp, lk = ladders_pm(pm), ladders_kx(kx)
    lp = pd.concat([lp, ladders_pm_other(po, lp)], ignore_index=True) if len(po) else lp
    lad = pd.concat([lp, lk], ignore_index=True) if len(lp) or len(lk) else pd.DataFrame()
    trades, allq = [], {}
    arbs = []
    if len(lad):
        arbs.append(a1_monotonicity(lad))
        trades.append(c1_wings(lad))
    arbs.append(a2_ranges(kx, kr))
    if len(po):
        arbs.append(a4_pm_ranges(lp, po))
    if len(lp) and len(lk):
        trades.append(b1_cross_venue(lp, lk, basis_estimate(lk, oc)))
    if len(lp) and len(chain):
        t, q = b2_options(lp, chain, px, "deribit", "B2")
        trades.append(t)
        allq["Polymarket ladder vs Deribit smile"] = q
    for vname, chain_v, label in (("binance", bo, "Binance options"), ("okx", load("okx_options"), "OKX options"),
                                  ("bybit", load("bybit_options"), "Bybit options")):
        if len(lp) and len(chain_v):
            t, q = b2_options(lp, chain_v, px, vname, "B3")
            if len(t):
                t["options_venue"] = vname
            trades.append(t)
            allq[f"Polymarket ladder vs {label}"] = q
    if (len(po) or len(ko)) and len(chain):
        t, q = d1_other(po, ko, chain, px)
        trades.append(t)
        allq["other Bitcoin markets vs Deribit smile"] = q
    lim = load("limitless")
    if len(lim) and len(po):
        t, a3 = f1_limitless(lim, po)
        trades.append(t)
        arbs.append(a3)
    tr = pd.concat([t for t in trades if t is not None and len(t)], ignore_index=True) if any(t is not None and len(t) for t in trades) else pd.DataFrame()
    tr = settle(attach_outcomes(tr, oc))
    acc = []
    for label, q in allq.items():
        if q is not None and len(q):
            q = q.assign(hyp="B2" if "Deribit" in label and "ladder" in label else ("B3" if "options" in label else "D1"))
            acc.append(accuracy(attach_outcomes(q, oc), label))
    oc.save()
    stale = e1_stale(fast)
    ensure(config.TABLES)
    if len(tr) and "options_venue" in tr:
        tr["hyp"] = np.where(tr["hyp"] == "B3", "B3 " + tr["options_venue"].fillna(""), tr["hyp"])
    summ = summarize(tr) if len(tr) else pd.DataFrame()
    if len(stale):
        st_rows = []
        for v, g in stale.groupby("venue"):
            for h in (30, 60):
                x = g[f"markout_{h}s"].dropna().to_numpy(float)
                m, se, n = evaluate.hac_mean(x, evaluate.nw_lag(len(x), 1))
                st_rows.append({"hyp": f"E1 mark-out {h}s", "venue": v, "signals": len(g), "settled": n,
                                "pnl_per_contract": m, "t": m / se if se else np.nan})
        summ = pd.concat([summ, pd.DataFrame(st_rows)], ignore_index=True)
    arb = pd.concat([a for a in arbs if a is not None and len(a)], ignore_index=True) if any(a is not None and len(a) for a in arbs) else pd.DataFrame()
    tr.to_csv(config.TABLES / "alpha_live_trades.csv", index=False)
    summ.to_csv(config.TABLES / "alpha_live_summary.csv", index=False)
    arb.to_csv(config.TABLES / "alpha_live_arbs.csv", index=False)
    accd = pd.concat(acc, ignore_index=True) if acc else pd.DataFrame()
    accd.to_csv(config.TABLES / "alpha_live_accuracy.csv", index=False)
    if len(stale):
        stale.to_csv(config.TABLES / "alpha_live_stale.csv", index=False)
    # report
    from oos_report import md
    L = ["# Alpha on real order books", "",
         f"Rules fixed on {PREREGISTERED_ON}; thresholds {TH}; Kalshi fee 0.07·P(1−P) rounded up to the cent; "
         "first signal per (event, strike, side), held to settlement; t over settlement events; Holm across rows.", "",
         f"Books: Polymarket ladders {pm['snap_ts'].nunique() if len(pm) else 0} snapshots, Kalshi {kx['snap_ts'].nunique() if len(kx) else 0}, "
         f"Kalshi ranges {kr['snap_ts'].nunique() if len(kr) else 0}, Deribit {chain['snap_ts'].nunique() if len(chain) else 0}, "
         f"Binance options {bo['snap_ts'].nunique() if len(bo) else 0}, other markets {po['snap_ts'].nunique() if len(po) else 0}, "
         f"fast logger {fast['ts'].nunique() if len(fast) else 0}.", "",
         "## Trading hypotheses", md(summ, 4),
         "## Risk-free arbitrage found (A1 ladder monotonicity, A2 Kalshi above vs ranges, A3 Limitless vs Polymarket)",
         (md(arb.groupby(["hyp", "venue"]).agg(cases=("edge", "size"), snapshots=("snap_ts", "nunique"),
                                              median_edge=("edge", "median"), max_edge=("edge", "max"),
                                              max_profit_usd=("profit_usd", "max") if "profit_usd" in arb else ("edge", "max")).reset_index(), 4)
          if len(arb) else "_none: every quoted ladder was free of executable arbitrage after fees_\n"),
         "## Who is more accurate on settled quotes: the market or the fair price?", md(accd, 4)]
    ensure(config.REPORTS)
    (config.REPORTS / "alpha_live.md").write_text("\n".join(L), encoding="utf-8")
    log(f"alpha_live: {len(tr)} signals, {int(tr['pnl'].notna().sum()) if 'pnl' in tr else 0} settled, {len(arb)} arbitrage cases -> reports/alpha_live.md")


if __name__ == "__main__":
    main(sys.argv[1:])
