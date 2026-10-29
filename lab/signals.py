"""The hypotheses. Every rule below was fixed on lab.core.FROZEN, before any live outcome was seen.

Families
    D  directional BTC (Binance USDT-M perpetual, entry/exit at the real bid/ask, 5 bp taker fee per side)
    P  prediction markets: our fair value vs the real book, held to the venue's settlement, real fees
    A  no-arbitrage violations in ladders / across venues (held to settlement: the profit is what it pays)
    O  options: cross-venue and static arbitrage (locked at entry), straddles vs our volatility forecast
    F  cross-venue funding carry
Units: D and F trades in return on notional; P and A in dollars per $1 contract; O in dollars per 1 BTC option.

Each rule is a function f(ctx) -> list[Order]. The daemon calls every function once a minute; a rule
decides itself whether its time has come. A position key makes every signal fire at most once.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

import config
from lab.core import PERP_FEE, Order, kalshi_fee, log, option_fee, pm_fee

ET = "America/New_York"
MARGIN = 0.03            # minimum edge after fees for every fair-value bet on a binary
NYSE_HOLIDAYS = {"2026-11-26", "2026-12-25", "2027-01-01", "2027-01-18"}


@dataclass
class Hyp:
    id: str
    family: str
    title: str
    rule: str
    source: str
    fn: object
    loop: str = "minute"      # "minute" (60-s loop) or "fast" (15-s loop of the short binaries)
    frozen: str = "2026-10-29"
    retired: str = ""         # non-empty = no longer run (reason); its trades stay in the ledger and the report


REGISTRY: list[Hyp] = []

# Retired 29.10.2026 17:45 MSK: significantly losing live, or the idea is disproved on history (taker against our
# fair value: studies/kx15_study.py, 20 317 Kalshi 15-minute markets, model worse than the book, t −19)
RETIRED = {
    "D06": "loses: −10 bp per trade, t −8.7, hit 8%",
    "D07": "loses: t −2.8",
    "M01": "symmetric maker at fair ± 4¢ gets picked off: −6.6¢, t −6.1; replaced by M03/M04",
    "M02": "as M01 on the hourly ladder: −3.1¢; replaced by M03/M04",
    "P01": "taker against our fair value: disproved on history (kx15_study)",
    "P04": "loses: −15¢, t −2.3",
    "P06": "taker against our fair value on Kalshi 15m: −2…−3¢ on 20 317 markets",
    "P07": "loses: −3.9¢, t −2.1",
    "V01": "15-s fair value, ≈ 0 on 203 trades; the book is better than the model",
    "V02": "−0.9¢ on 205 trades",
    "V03": "−1.8¢ on 157 trades",
    "V04": "≈ 0 on 197 trades",
    "V05": "−0.6¢ on 211 trades",
    # retired 29.10.2026 18:25 MSK on the 220-day print sample (1 303 markets, studies/tape_strategies.py): the
    # October edge of the lottery seller was one month; over Mar–Oct makers break even and takers lose the fee
    "M03": "220-day backtest −1.9¢ (t −1.7), October +8.5¢ was the outlier month; ladder strikes uncapped (5 correlated losses in one hour live)",
    "M04": "as M03: −2.0¢ (t −1.8) on 1 192 markets",
    "T01": "220-day backtest −3.7¢ (t −2.9)",
    "T02": "220-day backtest −2.4¢ (t −2.3)",
    "W01": "−9.9¢ live (12 losses of 94), ≈ 0 on history; W02 keeps the model-filtered version",
}


def hyp(id, family, title, rule, source, loop="minute", frozen="2026-10-29"):
    def deco(fn):
        REGISTRY.append(Hyp(id, family, title, rule, source, fn, loop, frozen, RETIRED.get(id, "")))
        return fn
    return deco


# ------------------------------------------------------------------ helpers
def perp_quotes(ctx):
    t = ctx.tick
    b, a = t.get("bf_bid"), t.get("bf_ask")
    if b is None or a is None or not (np.isfinite(b) and np.isfinite(a)):
        b, a = t.get("bn_bid"), t.get("bn_ask")
    return b, a


def perp_order(ctx, hid, side, hold_min, key, **info):
    b, a = perp_quotes(ctx)
    if b is None or a is None or not np.isfinite(b):
        return []
    price = a if side == "long" else b
    return [Order(hyp=hid, kind="perp", venue="binance-perp", instrument="BTCUSDT", side=side, price=price,
                  fee=PERP_FEE, hold_min=hold_min, unit="return", key=key,
                  info={"mid": (a + b) / 2, **info})]


def in_window(now, minute, width=3):
    return minute <= now.minute < minute + width


def et_now(now):
    return now.tz_convert(ET)


def trading_day(d) -> bool:
    return d.dayofweek < 5 and d.strftime("%Y-%m-%d") not in NYSE_HOLIDAYS


def phi(x):
    return float(stats.norm.cdf(x))


# ================================================================== D: directional
@hyp("D01", "D", "Hourly reversal", "at HH:00 take the opposite side of the last hour's return; hold 60 min",
     "short-term reversal (Jegadeesh 1990; for crypto hourly: R4c in research.md)")
def d01(ctx):
    if not in_window(ctx.now, 0):
        return []
    H = ctx.now.floor("h")
    r = math.log(ctx.minutes.price_at(H) / ctx.minutes.price_at(H - pd.Timedelta(hours=1)))
    if not np.isfinite(r) or r == 0:
        return []
    return perp_order(ctx, "D01", "short" if r > 0 else "long", 60, f"D01|{H}", r_last=r)


@hyp("D02", "D", "Long outside US hours", "long at 16:00 ET on NYSE days, exit at the next 09:30 ET open",
     "overnight drift (Lou, Polk & Skouras 2019); R2 in research.md")
def d02(ctx):
    e = et_now(ctx.now)
    if not (trading_day(e) and e.hour == 16 and e.minute < 3):
        return []
    nxt = e.normalize() + pd.Timedelta(days=1)
    while not trading_day(nxt):
        nxt += pd.Timedelta(days=1)
    exit_ = nxt + pd.Timedelta(hours=9, minutes=30)
    hold = (exit_ - e).total_seconds() / 60
    return perp_order(ctx, "D02", "long", hold, f"D02|{e:%Y-%m-%d}")


@hyp("D03", "D", "Short during US hours", "short at 09:30 ET on NYSE days, exit at 16:00 ET",
     "mirror of D02 (US-session returns were negative after the ETF launch)")
def d03(ctx):
    e = et_now(ctx.now)
    if not (trading_day(e) and e.hour == 9 and 30 <= e.minute < 33):
        return []
    return perp_order(ctx, "D03", "short", 390 - (e.minute - 30), f"D03|{e:%Y-%m-%d}")


@hyp("D04", "D", "Fade extreme funding", "30 min before each Binance funding: predicted rate > 0.015% → short, "
     "< 0.005% → long; hold 8 h", "crowded-positioning reversal (He, Manela, Ross & von Wachter 2022)")
def d04(ctx):
    nf, f = ctx.tick.get("bf_next_funding"), ctx.tick.get("bf_funding")
    if nf is None or f is None or not np.isfinite(f):
        return []
    mins = (pd.Timestamp(nf) - ctx.now).total_seconds() / 60
    if not (27 <= mins <= 31):
        return []
    side = "short" if f > 0.00015 else ("long" if f < 0.00005 else None)
    return perp_order(ctx, "D04", side, 480, f"D04|{nf}", funding=f) if side else []


@hyp("D05", "D", "Fade the perpetual premium", "HH:30: z-score of the Binance premium index (last 7 days, hourly) "
     "> 2 → short, < −2 → long; hold 60 min", "basis mean reversion")
def d05(ctx):
    if not in_window(ctx.now, 30):
        return []
    key = f"D05|{ctx.now.floor('h')}"
    if ctx.book.has_key(key + "|long") or ctx.book.has_key(key + "|short"):
        return []
    try:
        k = ctx.fapi("/fapi/v1/premiumIndexKlines", {"symbol": "BTCUSDT", "interval": "1h", "limit": 169})
        x = pd.Series([float(c[4]) for c in k])
    except Exception:
        return []
    if len(x) < 48:
        return []
    z = (x.iloc[-1] - x.iloc[:-1].mean()) / x.iloc[:-1].std()
    side = "short" if z > 2 else ("long" if z < -2 else None)
    return perp_order(ctx, "D05", side, 60, f"{key}|{side}", z=z) if side else []


@hyp("D06", "D", "Order-book imbalance", "every minute: Binance spot depth within 10 bp, (bid−ask)/(bid+ask) > 0.4 → "
     "long, < −0.4 → short; hold 5 min; one position at a time", "order-flow imbalance (Cont, Kukanov & Stoikov 2014)")
def d06(ctx):
    imb = ctx.tick.get("bn_imb10")
    if imb is None or not np.isfinite(imb) or ctx.book.open_count("D06"):
        return []
    side = "long" if imb > 0.4 else ("short" if imb < -0.4 else None)
    return perp_order(ctx, "D06", side, 5, f"D06|{ctx.now.floor('min')}", imb=imb) if side else []


@hyp("D07", "D", "Taker-flow momentum", "every 5 min: share of taker buys in the last 5 closed minutes > 0.6 → long, "
     "< 0.4 → short; hold 15 min", "trade-flow persistence")
def d07(ctx):
    if ctx.now.minute % 5 or ctx.book.open_count("D07"):
        return []
    x = ctx.tick.get("bn_taker5")
    if x is None or not np.isfinite(x):
        return []
    side = "long" if x > 0.6 else ("short" if x < 0.4 else None)
    return perp_order(ctx, "D07", side, 15, f"D07|{ctx.now.floor('5min')}", taker=x) if side else []


@hyp("D08", "D", "Coinbase premium momentum", "every 15 min: change over 60 min of Coinbase BTC-USD / (Binance BTC-USDT × "
     "USDT-USD) > +5 bp → long 60 min, < −5 bp → short", "US spot demand ('Coinbase premium')")
def d08(ctx):
    if ctx.now.minute % 15:
        return []
    t = ctx.ticks
    if t is None or len(t) < 50 or "cb_mid" not in t or "bn_mid" not in t:
        return []
    u = t["usdt_usd"].ffill() if "usdt_usd" in t else 1.0
    prem = t["cb_mid"] / (t["bn_mid"] * u) - 1
    now_p = prem.iloc[-1]
    past = prem[t["ts"] <= ctx.now - pd.Timedelta(minutes=60)]
    if past.empty or not np.isfinite(now_p):
        return []
    d = now_p - past.iloc[-1]
    side = "long" if d > 0.0005 else ("short" if d < -0.0005 else None)
    return perp_order(ctx, "D08", side, 60, f"D08|{ctx.now.floor('15min')}", dprem=d) if side else []


@hyp("D09", "D", "Coinbase → Binance lead-lag (15 s)", "every 15 s: Coinbase BTC-USD moved more than Binance BTC-USDT "
     "over the last 15 s by > 3 bp → bet Binance follows for 60 s (run by the fast loop)",
     "price discovery across venues (Makarov & Schoar 2020)")
def d09(ctx):
    return []          # handled by Lab.lead_lag in the 15-second thread


@hyp("D10", "D", "Open interest × price", "HH:01: open interest up > 1% in the last hour → follow the hour's price move; hold 60 min",
     "leverage build-up confirms trends")
def d10(ctx):
    if not in_window(ctx.now, 1):
        return []
    key = f"D10|{ctx.now.floor('h')}"
    if ctx.book.has_key(key + "|long") or ctx.book.has_key(key + "|short"):
        return []
    try:
        oi = ctx.fapi("/futures/data/openInterestHist", {"symbol": "BTCUSDT", "period": "1h", "limit": 2})
        d_oi = float(oi[-1]["sumOpenInterest"]) / float(oi[-2]["sumOpenInterest"]) - 1
    except Exception:
        return []
    H = ctx.now.floor("h")
    dp = ctx.minutes.price_at(H) / ctx.minutes.price_at(H - pd.Timedelta(hours=1)) - 1
    if d_oi <= 0.01 or not np.isfinite(dp):
        return []
    side = "long" if dp > 0 else "short"
    return perp_order(ctx, "D10", side, 60, f"{key}|{side}", d_oi=d_oi, dp=dp)


@hyp("D11", "D", "Fade retail positioning", "every 4 h: z-score (last 7 days, hourly) of Binance's global long/short "
     "account ratio > 1.5 → short, < −1.5 → long; hold 4 h", "retail crowding is contrarian")
def d11(ctx):
    if ctx.now.hour % 4 or not in_window(ctx.now, 2):
        return []
    key = f"D11|{ctx.now.floor('h')}"
    if ctx.book.has_key(key + "|long") or ctx.book.has_key(key + "|short"):
        return []
    try:
        r = ctx.fapi("/futures/data/globalLongShortAccountRatio", {"symbol": "BTCUSDT", "period": "1h", "limit": 168})
        x = pd.Series([float(v["longShortRatio"]) for v in r])
    except Exception:
        return []
    if len(x) < 48:
        return []
    z = (x.iloc[-1] - x.mean()) / x.std()
    side = "short" if z > 1.5 else ("long" if z < -1.5 else None)
    return perp_order(ctx, "D11", side, 240, f"{key}|{side}", z=z) if side else []


@hyp("D12", "D", "Implied-vol momentum", "HH:02: z-score of the last hourly change of log DVOL (last 30 days) > 1 → "
     "long 60 min, < −1 → short", "R12 in research.md (ΔDVOL led returns, t = 2.8)")
def d12(ctx):
    if not in_window(ctx.now, 2):
        return []
    key = f"D12|{ctx.now.floor('h')}"
    if ctx.book.has_key(key + "|long") or ctx.book.has_key(key + "|short"):
        return []
    dv = ctx.dvol_hourly()
    if dv is None or len(dv) < 200:
        return []
    d = np.log(dv).diff().dropna()
    z = (d.iloc[-1] - d.iloc[:-1].mean()) / d.iloc[:-1].std()
    side = "long" if z > 1 else ("short" if z < -1 else None)
    return perp_order(ctx, "D12", side, 60, f"{key}|{side}", z=z) if side else []


def _ml(ctx, hid, name):
    if not in_window(ctx.now, 0) or ctx.ml is None:
        return []
    pred = ctx.ml.predict(name, ctx.minutes, ctx.now)
    if pred is None or not np.isfinite(pred):
        return []
    side = "long" if pred > 0.0015 else ("short" if pred < -0.0015 else None)
    return perp_order(ctx, hid, side, 60, f"{hid}|{ctx.now.floor('h')}", pred=pred) if side else []


@hyp("D13", "D", "Gradient boosting, next hour", "HH:00: gradient-boosted trees on 20 price/volume/calendar features "
     "predict the next-hour return; > +15 bp → long, < −15 bp → short; retrained every 6 h on 180 days",
     "machine-learning return prediction (Gu, Kelly & Xiu 2020)")
def d13(ctx):
    return _ml(ctx, "D13", "gbm")


@hyp("D14", "D", "Ridge regression, next hour", "same features and thresholds as D13, linear ridge model",
     "linear benchmark for D13")
def d14(ctx):
    return _ml(ctx, "D14", "ridge")


@hyp("D15", "D", "Weekly time-series momentum", "00:00 UTC: sign of the last 7-day return; hold 24 h",
     "Liu & Tsyvinski (2021); R4 in research.md")
def d15(ctx):
    if not (ctx.now.hour == 0 and ctx.now.minute < 3):
        return []
    D = ctx.now.floor("D")
    r = ctx.minutes.price_at(D) / ctx.minutes.price_at(D - pd.Timedelta(days=7)) - 1
    if not np.isfinite(r):
        return []
    return perp_order(ctx, "D15", "long" if r > 0 else "short", 1440, f"D15|{D}", r7=r)


@hyp("D16", "D", "Kalshi ladder drift", "HH:06: the Kalshi hourly ladder's implied drift (median + s²/2 − log spot, "
     "in units of s) > 0.25 → long to HH+1:00, < −0.25 → short", "R9 in research.md (does the ladder know the direction?)")
def d16(ctx):
    if not in_window(ctx.now, 6):
        return []
    t = ctx.now.floor("h") + pd.Timedelta(minutes=5)
    f = config.FORECASTS_LIVE / "kx1h" / f"{t:%Y%m%dT%H%M}.json"
    if not f.exists():
        return []
    try:
        ft = json.loads(f.read_text(encoding="utf-8"))["features"]
        d = (ft["mkt_m"] + 0.5 * ft["mkt_s"] ** 2 - math.log(ft["S_t"])) / ft["mkt_s"]
    except Exception:
        return []
    side = "long" if d > 0.25 else ("short" if d < -0.25 else None)
    hold = (ctx.now.floor("h") + pd.Timedelta(hours=1) - ctx.now).total_seconds() / 60
    return perp_order(ctx, "D16", side, hold, f"D16|{t}", drift=d) if side else []


@hyp("D17", "D", "Follow the Polymarket hourly crowd", "HH:05: mid of Polymarket's hourly Up/Down market > 0.60 → long "
     "to the end of the hour, < 0.40 → short", "prediction markets as directional signals")
def d17(ctx):
    if not in_window(ctx.now, 5) or ctx.markets is None or ctx.markets.empty:
        return []
    m = ctx.markets[ctx.markets["kind"] == "pm_updown_1h"]
    if m.empty:
        return []
    r = m.sort_values("end").iloc[0]
    mid = (r["bid"] + r["ask"]) / 2
    if not np.isfinite(mid):
        return []
    side = "long" if mid > 0.6 else ("short" if mid < 0.4 else None)
    hold = (pd.Timestamp(r["end"]) - ctx.now).total_seconds() / 60
    return perp_order(ctx, "D17", side, hold, f"D17|{r['market']}", pm_mid=mid) if side and hold > 10 else []


@hyp("D18", "D", "Regime-conditioned trend", "00:05 UTC: bull trend (price > 200-day mean) and low-vol Markov state → "
     "long 24 h; bear and high-vol → short; otherwise flat", "regime timing (research.md, section 3)")
def d18(ctx):
    if not (ctx.now.hour == 0 and 5 <= ctx.now.minute < 8):
        return []
    f = config.RESULTS / "research" / "regimes_daily.csv"
    if not f.exists():
        return []
    try:
        r = pd.read_csv(f).iloc[-1]
    except Exception:
        return []
    side = "long" if (r.get("trend") == "bull" and r.get("ms") == "low-vol") else \
        ("short" if (r.get("trend") == "bear" and r.get("ms") == "high-vol") else None)
    return perp_order(ctx, "D18", side, 1440, f"D18|{ctx.now.floor('D')}", trend=r.get("trend"), ms=r.get("ms")) if side else []


@hyp("D19", "D", "Fade liquidation cascades", "every minute: forced-order notional of the last 5 min on one side > max($3M, "
     "4 × its 6-hour 95th percentile) → bet on a rebound (long after long liquidations); hold 30 min",
     "liquidation cascades overshoot (Ali 2023; folklore of crypto desks)", frozen="2026-10-29 11:00")
def d19(ctx):
    f = ctx.flow.frame(360) if getattr(ctx, "flow", None) is not None else None
    if f is None or len(f) < 40 or ctx.book.open_count("D19"):
        return []
    f = f.set_index("ts").sort_index()
    w = f[f.index > ctx.now - pd.Timedelta(minutes=5)]
    roll = f[["liq_long_usd", "liq_short_usd"]].rolling("5min").sum()
    out = []
    for col, side in (("liq_long_usd", "long"), ("liq_short_usd", "short")):
        x = w[col].sum()
        thr = max(3e6, 4 * float(roll[col].quantile(0.95)))
        if x > thr:
            out = perp_order(ctx, "D19", side, 30, f"D19|{ctx.now.floor('5min')}|{side}", liq_usd=x, thr=thr)
            break
    return out


@hyp("D20", "D", "Aggressive-flow momentum (1 min)", "every minute: (taker buys − taker sells)/(sum) of the last minute "
     "> 0.5 with notional > 3 × the 6-hour median minute → follow for 5 min (mirror for sells)",
     "order-flow imbalance predicts short-run returns (Chordia & Subrahmanyam 2004; Cont et al. 2014)",
     frozen="2026-10-29 11:00")
def d20(ctx):
    f = ctx.flow.frame(360) if getattr(ctx, "flow", None) is not None else None
    if f is None or len(f) < 40 or ctx.book.open_count("D20"):
        return []
    f = f.set_index("ts").sort_index()
    m1 = f[["buy_usd", "sell_usd"]].resample("1min").sum()
    if len(m1) < 30:
        return []
    last = m1.iloc[-2]                     # the last complete minute
    tot = last["buy_usd"] + last["sell_usd"]
    med = float((m1["buy_usd"] + m1["sell_usd"]).iloc[:-1].median())
    if tot <= 3 * med or tot <= 0:
        return []
    imb = (last["buy_usd"] - last["sell_usd"]) / tot
    side = "long" if imb > 0.5 else ("short" if imb < -0.5 else None)
    return perp_order(ctx, "D20", side, 5, f"D20|{ctx.now.floor('min')}", imb=imb, usd=tot) if side else []


@hyp("M01", "M", "Maker quotes on Kalshi 15-min", "every minute: virtual bid at fair − 4¢ and ask at fair + 4¢ (only if "
     "inside or behind the spread), valid 60 s; filled only if a real trade prints strictly through our price; held to "
     "settlement; maker fee 0.0175·P(1−P)", "market making against informed flow", frozen="2026-10-29 11:00")
def m01(ctx):
    return []          # run by Lab.maker (needs the trade tape)


@hyp("M02", "M", "Maker quotes on Kalshi hourly ladder", "as M01, strikes with fair value 0.2–0.8 (at most 4), only "
     "≥ 10 minutes before the hour", "", frozen="2026-10-29 11:00")
def m02(ctx):
    return []


@hyp("M03", "M", "Lottery seller on Kalshi (maker, queue-aware)", "15-min market and hourly-ladder strikes: when a side "
     "costs 2–20¢ and ≥ 2 min are left, join its best offer with 1 contract; filled by prints through our price or by "
     "taker volume at our price beyond the displayed queue; held to settlement; maker fee", "takers who buy longshots "
     "lose ~5¢ per contract (studies/kx15_flow_deep.py)", frozen="2026-10-29 17:40")
def m03(ctx):
    return []          # run by Lab.tape (needs the trade tape)


@hyp("M04", "M", "Lottery seller, cancels on spot moves", "as M03; the quote is cancelled at the first 15-s sample in "
     "which Binance moved ≥ 3 bp toward the cheap side winning", "avoid being picked off by fast takers",
     frozen="2026-10-29 17:40")
def m04(ctx):
    return []


@hyp("T01", "T", "Copy the favourite buyer", "Kalshi 15-min: ≥ 200 contracts bought by takers on the favourite side at "
     "65–95¢ in the last 60 s → buy the favourite at the book ask; one per market", "fast takers who buy favourites "
     "win (positive markouts)", frozen="2026-10-29 17:40")
def t01(ctx):
    return []


@hyp("T02", "T", "Fade the lottery burst", "Kalshi 15-min: ≥ 500 contracts bought by takers on the cheap side at "
     "≤ 20¢ in the last 60 s → buy the favourite at the book ask; one per market", "lottery buyers lose, the price "
     "keeps moving against them", frozen="2026-10-29 17:40")
def t02(ctx):
    return []


# ================================================================== P: fair value of short-dated binaries
FAIR_MODELS = ("ln", "t4", "kxiv", "dvol", "rv1h")


def _level(ctx, model):
    """Deseasonalised per-minute variance level for each fair-value model (cached per minute)."""
    key = ("lvl", model, ctx.now.floor("min"))
    if key in ctx.cache:
        return ctx.cache[key]
    lv = None
    if model in ("ln", "t4"):
        lv = ctx.minutes.level()
    elif model == "dvol":
        dv = ctx.tick.get("dvol") or ctx.cache.get("last_dvol")
        if dv is not None and np.isfinite(dv):
            lv = (dv / 100) ** 2 / 525600
    elif model == "rv1h":
        r = ctx.minutes.returns(1)
        r = r[r.index > r.index[-1] - pd.Timedelta(minutes=60)]
        if len(r) >= 40:
            s = ctx.minutes.seasonal()
            how = r.index.dayofweek * 24 + r.index.hour
            lv = float(((r ** 2) / s.reindex(how).values).mean())
    elif model == "kxiv":
        v = ctx.cache.get("kx_level_last")
        if v and ctx.now - v[0] < pd.Timedelta(minutes=3):
            lv = v[1]
    ctx.cache[key] = lv
    return lv


def kalshi_ladder_level(ctx):
    """Per-minute variance level implied by the current Kalshi hourly ladder: probit regression of the mid prices
    on log(S/K) gives the total sd to the hour; divided by the seasonal minutes left (needs >= 10 minutes)."""
    m = ctx.markets
    if m is None or m.empty:
        return None
    g = m[(m["kind"] == "kx_ladder_1h")].dropna(subset=["bid", "ask", "strike"])
    if g.empty:
        return None
    end = pd.Timestamp(g["end"].iloc[0])
    if (end - ctx.now).total_seconds() < 600:
        return None
    S = ctx.tick.get("usd_mid")
    if S is None or not np.isfinite(S):
        return None
    mid = (g["bid"] + g["ask"]) / 2
    ok = (mid > 0.05) & (mid < 0.95) & (g["ask"] - g["bid"] <= 0.06)
    if ok.sum() < 3:
        return None
    x = np.log(S / g.loc[ok, "strike"].values)
    z = stats.norm.ppf(mid[ok].values)
    b = float((x @ z) / (x @ x))          # z = x / sd  ->  slope = 1/sd
    if b <= 0:
        return None
    sd = 1 / b
    var1 = ctx.minutes.forecast_var(ctx.now, end - pd.Timedelta(seconds=30), level=1.0)   # sum of seasonal factors
    return sd * sd / var1 if var1 and np.isfinite(var1) else None


def fair_value(ctx, r, model: str = "ln") -> float | None:
    """Our probability that the YES side (Up / above K) pays, from the spot of the settlement source, the start
    price (up/down) or strike, and a variance forecast for the time left.
    model: ln = lab seasonal blend (the frozen P-rules); t4 = same variance, Student-t(4) tails; kxiv = level implied
    by the Kalshi hourly ladder; dvol = DVOL level; rv1h = realised variance of the last hour."""
    now, end = ctx.now, pd.Timestamp(r["end"])
    if r["source"] == "brti":
        S = ctx.tick.get("usd_mid")
    else:
        b, a = ctx.tick.get("bn_bid"), ctx.tick.get("bn_ask")
        S = (b + a) / 2 if b is not None and a is not None else None
    if S is None or not np.isfinite(S):
        return None
    # averaging at settlement: Chainlink TWAP 60 s, CF BRTI 60-s average -> the last ~30 s carry little variance
    eff_end = end - pd.Timedelta(seconds=30) if r["source"] in ("brti", "chainlink") else end
    if eff_end <= now:
        return None
    lv = None if model == "ln" else _level(ctx, model)
    if model != "ln" and (lv is None or not np.isfinite(lv) or lv <= 0):
        return None
    var = ctx.minutes.forecast_var(now, eff_end, level=lv)
    if not np.isfinite(var) or var <= 0:
        return None
    if str(r["kind"]).startswith("pm_updown"):
        K = ctx.minutes.open_at(pd.Timestamp(r["start"]))
        if r["source"] == "chainlink":
            # Chainlink BTC/USD vs Binance BTC/USDT: use the move since the start on Binance (the level basis cancels)
            S = (ctx.tick["bn_bid"] + ctx.tick["bn_ask"]) / 2
    else:
        K = r["strike"]
    if K is None or not np.isfinite(K) or K <= 0:
        return None
    x = math.log(S / K) / math.sqrt(var)
    if model == "t4":
        return float(stats.t.cdf(x * math.sqrt(2.0), 4))     # t(4) scaled to unit variance: sd = sqrt(4/2)
    return phi(x)


def binary_orders(ctx, hid, kind_filter, min_left_min=1.5, model="ln", stale_only=False):
    if ctx.markets is None or ctx.markets.empty:
        return []
    out = []
    kinds = kind_filter if isinstance(kind_filter, (tuple, list)) else (kind_filter,)
    m = ctx.markets[ctx.markets["kind"].isin(kinds)]
    for _, r in m.iterrows():
        left = (pd.Timestamp(r["end"]) - ctx.now).total_seconds() / 60
        if left < min_left_min:
            continue
        bid, ask = r["bid"], r["ask"]
        if not (np.isfinite(bid) and np.isfinite(ask)) or ask - bid > 0.06 or ask <= bid:
            continue
        p = fair_value(ctx, r, model)
        if p is None:
            continue
        if stale_only and not _stale(ctx, r):
            continue
        fee = (lambda x: kalshi_fee(x)) if r["venue"] == "kalshi" else (lambda x: pm_fee(x, r["fee_rate"] or 0.0))
        e_yes = p - ask - fee(ask)
        no_px = 1 - bid
        e_no = (1 - p) - no_px - fee(no_px)
        leg = {"venue": r["venue"], "kind": r["kind"], "market": r["market"], "event": r["event"]}
        if e_yes >= MARGIN and 0.03 <= ask <= 0.97:
            out.append(Order(hyp=hid, kind="binary", venue=r["venue"], instrument=r["market"], side="yes", price=ask,
                             fee=fee(ask), fair=p, edge=e_yes, settle_ts=str(r["end"]), unit="$/contract",
                             key=f"{hid}|{r['market']}|yes",
                             info={"legs": [{**leg, "side": "yes", "price": ask, "fee": fee(ask)}], "mid": (bid + ask) / 2,
                                   "left_min": round(left, 1), "size": r.get("ask_size")}))
        elif e_no >= MARGIN and 0.03 <= no_px <= 0.97:
            out.append(Order(hyp=hid, kind="binary", venue=r["venue"], instrument=r["market"], side="no", price=no_px,
                             fee=fee(no_px), fair=1 - p, edge=e_no, settle_ts=str(r["end"]), unit="$/contract",
                             key=f"{hid}|{r['market']}|no",
                             info={"legs": [{**leg, "side": "no", "price": no_px, "fee": fee(no_px)}], "mid": (bid + ask) / 2,
                                   "left_min": round(left, 1), "size": r.get("bid_size")}))
    return out


@hyp("P01", "P", "Polymarket 5-min Up/Down vs fair value", "buy the side whose fair value exceeds the ask + fee by ≥ 3¢; "
     "fair = Φ(log(S/S_start)/σ√τ) with the lab's seasonal variance forecast", "do short-dated crypto binaries misprice the random walk?")
def p01(ctx):
    return binary_orders(ctx, "P01", "pm_updown_5m", 1.0)


@hyp("P02", "P", "Polymarket 15-min Up/Down vs fair value", "as P01", "")
def p02(ctx):
    return binary_orders(ctx, "P02", "pm_updown_15m")


@hyp("P03", "P", "Polymarket hourly Up/Down vs fair value", "as P01; settles on the Binance 1-hour candle", "")
def p03(ctx):
    return binary_orders(ctx, "P03", "pm_updown_1h")


@hyp("P04", "P", "Polymarket 4-hour Up/Down vs fair value", "as P01", "")
def p04(ctx):
    return binary_orders(ctx, "P04", "pm_updown_4h")


@hyp("P05", "P", "Polymarket hourly ladders vs fair value", "'Bitcoin above K at h ET': fair = Φ(log(S/K)/σ√τ)", "")
def p05(ctx):
    return binary_orders(ctx, "P05", "pm_ladder_1h")


@hyp("P06", "P", "Kalshi 15-min vs fair value", "KXBTC15M: fair = Φ(log(S_usd/target)/σ√τ), S_usd = Coinbase/Kraken mid", "")
def p06(ctx):
    return binary_orders(ctx, "P06", "kx_15m")


@hyp("P07", "P", "Kalshi hourly ladder vs fair value (every minute)", "KXBTCD current hour, every strike, every minute; "
     "first signal per strike and side", "late-hour pricing of Kalshi's hourly ladder")
def p07(ctx):
    return binary_orders(ctx, "P07", "kx_ladder_1h")


# ================================================================== A: no-arbitrage in ladders and across venues
@hyp("A11", "A", "Ladder monotonicity (Polymarket hourly, Kalshi hourly)", "K1 < K2 and bid(K2) − ask(K1) > fees → buy "
     "YES K1, buy NO K2: pays ≥ 0 in every state", "A1 in alpha_live.md, now every minute")
def a11(ctx):
    if ctx.markets is None or ctx.markets.empty:
        return []
    out = []
    for kind in ("pm_ladder_1h", "kx_ladder_1h"):
        m = ctx.markets[(ctx.markets["kind"] == kind)].dropna(subset=["strike", "bid", "ask"])
        for ev, g in m.groupby("event"):
            g = g.sort_values("strike")
            ven = g["venue"].iloc[0]
            fee = kalshi_fee if ven == "kalshi" else (lambda x, fr=g["fee_rate"].iloc[0] or 0.0: pm_fee(x, fr))
            asks, bids = g["ask"].values, g["bid"].values
            for i in range(len(g)):
                for j in range(i + 1, len(g)):
                    if not (0 < bids[j] < 1 and 0.005 < asks[i] < 1):    # an empty side is not a price
                        continue
                    no_px = 1 - bids[j]
                    cost = asks[i] + no_px + fee(asks[i]) + fee(no_px)
                    if cost < 1 - 0.005:       # the pair pays at least $1 in every state
                        k1, k2 = g["strike"].iloc[i], g["strike"].iloc[j]
                        legs = [{"venue": ven, "kind": kind, "market": g["market"].iloc[i], "event": ev, "side": "yes",
                                 "price": asks[i], "fee": fee(asks[i])},
                                {"venue": ven, "kind": kind, "market": g["market"].iloc[j], "event": ev, "side": "no",
                                 "price": no_px, "fee": fee(no_px)}]
                        out.append(Order(hyp="A11", kind="binary", venue=ven, instrument=f"{ev}:{k1:.0f}<{k2:.0f}",
                                         side="pair", price=asks[i] + no_px, fee=fee(asks[i]) + fee(no_px),
                                         edge=1 - cost, settle_ts=str(g["end"].iloc[0]), unit="$/pair",
                                         key=f"A11|{ev}|{k1}|{k2}", info={"legs": legs}))
    return out


@hyp("A12", "A", "Kalshi 15-min vs Polymarket 15-min", "same quarter-hour window: Polymarket Up ask + Kalshi NO "
     "(1 − yes bid) + fees < 1 (or the mirror) → buy both: pays 1 unless the two references disagree",
     "cross-venue binary arbitrage (references: CF BRTI vs Chainlink)")
def a12(ctx):
    return _a12(ctx, "A12")


def _a12(ctx, hid):
    if ctx.markets is None or ctx.markets.empty:
        return []
    pm = ctx.markets[ctx.markets["kind"] == "pm_updown_15m"]
    kx = ctx.markets[ctx.markets["kind"] == "kx_15m"]
    out = []
    for _, a in pm.iterrows():
        for _, b in kx.iterrows():
            if abs((pd.Timestamp(a["end"]) - pd.Timestamp(b["end"])).total_seconds()) > 5:
                continue
            fr = a["fee_rate"] or 0.0
            # Up on Polymarket + No on Kalshi, and Down on Polymarket + Yes on Kalshi
            for pm_side, pm_px, kx_side, kx_px in (("yes", a["ask"], "no", 1 - b["bid"]), ("no", 1 - a["bid"], "yes", b["ask"])):
                if not (np.isfinite(pm_px) and np.isfinite(kx_px)) or not (0.02 < pm_px < 0.98 and 0.02 < kx_px < 0.98):
                    continue
                cost = pm_px + kx_px + pm_fee(pm_px, fr) + kalshi_fee(kx_px)
                if cost < 1 - 0.01:
                    legs = [{"venue": "polymarket", "kind": a["kind"], "market": a["market"], "event": a["event"],
                             "side": pm_side, "price": pm_px, "fee": pm_fee(pm_px, fr)},
                            {"venue": "kalshi", "kind": b["kind"], "market": b["market"], "event": b["event"],
                             "side": kx_side, "price": kx_px, "fee": kalshi_fee(kx_px)}]
                    out.append(Order(hyp=hid, kind="binary", venue="pm+kalshi", instrument=f"{a['market']}|{b['market']}",
                                     side=f"pm_{pm_side}+kx_{kx_side}", price=pm_px + kx_px,
                                     fee=pm_fee(pm_px, fr) + kalshi_fee(kx_px), edge=1 - cost, settle_ts=str(a["end"]),
                                     unit="$/pair", key=f"{hid}|{a['market']}|{pm_side}", info={"legs": legs,
                                     "kx_target": b["strike"], "pm_size": a.get("ask_size") if pm_side == "yes" else a.get("bid_size"),
                                     "kx_size": b.get("bid_size") if kx_side == "no" else b.get("ask_size")}))
    return out


# ================================================================== A13 and the 15-second variants (frozen 29.10, 11:00 MSK)
SHORT_KINDS = ("pm_updown_5m", "pm_updown_15m", "kx_15m")


@hyp("A13", "A", "Kalshi hourly ladder vs Polymarket hourly ladder", "same settlement hour: Polymarket 'above K' (Binance "
     "close) vs the Kalshi strike K' = K × USDT/USD within $30 (CF BRTI average); YES on the cheap venue + NO on the dear one "
     "for < 1 − fees − 1¢ → pays 1 unless the two settlement prices land on different sides", "cross-venue ladder arbitrage",
     frozen="2026-10-29 11:00")
def a13(ctx):
    m = ctx.markets
    if m is None or m.empty:
        return []
    pm = m[m["kind"] == "pm_ladder_1h"].dropna(subset=["strike", "bid", "ask"])
    kx = m[m["kind"] == "kx_ladder_1h"].dropna(subset=["strike", "bid", "ask"])
    if pm.empty or kx.empty:
        return []
    u = ctx.tick.get("usdt_usd") or 1.0
    out = []
    for _, a in pm.iterrows():
        same = kx[(pd.to_datetime(kx["end"]) - pd.Timestamp(a["end"])).abs() <= pd.Timedelta(seconds=5)]
        if same.empty:
            continue
        j = (same["strike"] - a["strike"] * u).abs().idxmin()
        b = same.loc[j]
        if abs(b["strike"] - a["strike"] * u) > 30 or (pd.Timestamp(a["end"]) - ctx.now).total_seconds() < 120:
            continue
        fr = a["fee_rate"] or 0.0
        for pm_side, pm_px, kx_side, kx_px in (("yes", a["ask"], "no", 1 - b["bid"]), ("no", 1 - a["bid"], "yes", b["ask"])):
            if not (0.02 < pm_px < 0.98 and 0.02 < kx_px < 0.98):
                continue
            cost = pm_px + kx_px + pm_fee(pm_px, fr) + kalshi_fee(kx_px)
            if cost < 1 - 0.01:
                legs = [{"venue": "polymarket", "kind": a["kind"], "market": a["market"], "event": a["event"], "side": pm_side,
                         "price": pm_px, "fee": pm_fee(pm_px, fr)},
                        {"venue": "kalshi", "kind": b["kind"], "market": b["market"], "event": b["event"], "side": kx_side,
                         "price": kx_px, "fee": kalshi_fee(kx_px)}]
                out.append(Order(hyp="A13", kind="binary", venue="pm+kalshi", instrument=f"{a['market']}|{b['market']}",
                                 side=f"pm_{pm_side}+kx_{kx_side}", price=pm_px + kx_px, fee=pm_fee(pm_px, fr) + kalshi_fee(kx_px),
                                 edge=1 - cost, settle_ts=str(a["end"]), unit="$/pair", key=f"A13|{a['market']}|{pm_side}",
                                 info={"legs": legs, "K_pm": a["strike"], "K_kx": b["strike"], "usdt": u}))
    return out


def _stale(ctx, r):
    """Spot moved >= 4 bp since the previous 15-s snapshot while this binary's mid moved <= 1¢."""
    prev = ctx.cache.get("fast_prev")
    if not prev:
        return False
    spots, mids = prev
    if r["source"] == "brti":
        s0, s1 = spots.get("usd"), ctx.tick.get("usd_mid")
    else:
        s0, s1 = spots.get("bn"), (ctx.tick.get("bn_bid") + ctx.tick.get("bn_ask")) / 2
    m0 = mids.get(r["market"])
    if s0 is None or s1 is None or m0 is None:
        return False
    return abs(math.log(s1 / s0)) >= 0.0004 and abs((r["bid"] + r["ask"]) / 2 - m0) <= 0.01


for _hid, _model, _title in (("V01", "ln", "lab blend (as P01/P02/P06)"), ("V02", "t4", "Student-t(4) tails"),
                             ("V03", "kxiv", "variance implied by the Kalshi hourly ladder"), ("V04", "dvol", "DVOL level"),
                             ("V05", "rv1h", "realised variance of the last hour")):
    def _mk(hid=_hid, model=_model):
        return lambda ctx: binary_orders(ctx, hid, SHORT_KINDS, 1.0, model)
    hyp(_hid, "V", f"15-s fair value: {_title}", "every 15 s on Polymarket 5/15-min Up/Down and Kalshi 15-min: buy the side "
        "whose fair value beats ask + fee by ≥ 3¢; one trade per market and side", "which variance forecast prices the "
        "short binaries best?", loop="fast", frozen="2026-10-29 11:00")(_mk())


@hyp("V06", "V", "Stale-quote sniping (15 s)", "as V01, but only when spot moved ≥ 4 bp in the last 15 s and the binary's "
     "mid moved ≤ 1¢", "latency arbitrage against slow quotes", loop="fast", frozen="2026-10-29 11:00")
def v06(ctx):
    return binary_orders(ctx, "V06", SHORT_KINDS, 1.0, "ln", stale_only=True)


@hyp("A12F", "A", "Kalshi 15-min vs Polymarket 15-min, every 15 s", "as A12, sampled every 15 s", "how long do "
     "cross-venue gaps live?", loop="fast", frozen="2026-10-29 11:00")
def a12f(ctx):
    return _a12(ctx, "A12F")


# ================================================================== V07: the 60-second settlement average (frozen 29.10 16:40 MSK)
PROXY_SD = 4e-4          # BRTI vs our USD proxy, basis anchored 15 min earlier: drift + noise, log units


def _avg_fair(ctx, r, samples):
    """P(60-s average of BRTI before the close > strike), with the part of the window already printed.
    Level: our USD proxy shifted by this market's own basis (strike = BRTI at the start vs proxy at the start)."""
    end, start = pd.Timestamp(r["end"]), pd.Timestamp(r["end"]) - pd.Timedelta(minutes=15)
    K = r["strike"]
    if not (np.isfinite(K) and K > 0) or not samples:
        return None
    ts = np.array([t.value for t, _ in samples])
    px = np.array([v for _, v in samples])
    i0 = np.argmin(np.abs(ts - start.value))
    if abs(ts[i0] - start.value) > 20e9:              # no proxy sample within 20 s of the window start
        return None
    b = math.log(K / px[i0])
    S = px[-1] * math.exp(b)
    w0 = end - pd.Timedelta(seconds=60)
    left = (end - ctx.now).total_seconds()
    if left <= 0:
        return None
    if left > 60:                                     # averaging not started: drift to w0, then a full average
        var = ctx.minutes.forecast_var(ctx.now, w0) + ctx.minutes.forecast_var(w0, end) / 3
        mean = S
    else:
        inw = ts >= w0.value
        known = float(np.mean(px[inw] * math.exp(b))) if inw.any() else S
        r_ = left / 60
        mean = (1 - r_) * known + r_ * S
        var = r_ ** 2 * ctx.minutes.forecast_var(ctx.now, end) / 3
    if not np.isfinite(var) or var <= 0:
        return None
    return float(stats.norm.cdf(math.log(mean / K) / math.sqrt(var + PROXY_SD ** 2)))


@hyp("V07", "V", "Kalshi 15-min: the settlement average already printed", "Kalshi settles on the average of the last "
     "60 s of BRTI; in the last 75 s part of that average is known. Fair = P(average > strike) with the printed part, "
     "the rest as a Brownian average, level = our USD proxy shifted by the market's own start basis; take the book "
     "when edge ≥ 3¢ after fee, ≥ 10 s left", "does the 15-minute book price the averaging window?", loop="fast",
     frozen="2026-10-29 16:40")
def v07(ctx):
    m = ctx.markets
    usd = (ctx.tick or {}).get("usd_mid")
    sam = ctx.cache.setdefault("v07_proxy", [])
    if usd is not None and np.isfinite(usd):
        sam.append((ctx.now, float(usd)))
    cut = ctx.now - pd.Timedelta(minutes=20)
    while sam and sam[0][0] < cut:
        sam.pop(0)
    if m is None or m.empty:
        return []
    out = []
    for _, r in m[m["kind"] == "kx_15m"].iterrows():
        left = (pd.Timestamp(r["end"]) - ctx.now).total_seconds()
        if not (10 <= left <= 75):
            continue
        bid, ask = r["bid"], r["ask"]
        if not (np.isfinite(bid) and np.isfinite(ask)):
            continue
        p = _avg_fair(ctx, r, sam)
        if p is None:
            continue
        leg = {"venue": "kalshi", "kind": r["kind"], "market": r["market"], "event": r["event"]}
        e_yes, no_px = p - ask - kalshi_fee(ask), 1 - bid
        e_no = (1 - p) - no_px - kalshi_fee(no_px)
        for side, px, e, fair in (("yes", ask, e_yes, p), ("no", no_px, e_no, 1 - p)):
            if e >= MARGIN and 0.02 <= px <= 0.98:
                out.append(Order(hyp="V07", kind="binary", venue="kalshi", instrument=r["market"], side=side, price=px,
                                 fee=kalshi_fee(px), fair=fair, edge=e, settle_ts=str(r["end"]), unit="$/contract",
                                 key=f"V07|{r['market']}|{side}",
                                 info={"legs": [{**leg, "side": side, "price": px, "fee": kalshi_fee(px)}],
                                       "mid": (bid + ask) / 2, "left_min": round(left / 60, 2),
                                       "size": r.get("ask_size") if side == "yes" else r.get("bid_size")}))
    return out


# ================================================================== W: selling the cheap wings (frozen 29.10 14:45 MSK)
def _wings(ctx, hid, model_filter):
    m = ctx.markets
    if m is None or m.empty:
        return []
    out = []
    for _, r in m[m["kind"].isin(["kx_ladder_1h", "pm_ladder_1h"])].iterrows():
        bid, ask = r["bid"], r["ask"]
        left = (pd.Timestamp(r["end"]) - ctx.now).total_seconds() / 60
        if not (np.isfinite(bid) and np.isfinite(ask)) or left < 15:
            continue
        fee = kalshi_fee if r["venue"] == "kalshi" else (lambda x, fr=(r["fee_rate"] or 0.0): pm_fee(x, fr))
        leg = {"venue": r["venue"], "kind": r["kind"], "market": r["market"], "event": r["event"]}
        # upper wing: YES is the cheap side -> sell YES at the bid (= buy NO at 1 − bid)
        # lower wing: NO is the cheap side  -> sell NO at its bid (= buy YES at the ask)
        for side, cheap_bid, px in (("no", bid, 1 - bid), ("yes", 1 - ask, ask)):
            if not (0.02 <= cheap_bid <= 0.06):
                continue
            p = fair_value(ctx, r) if model_filter else None
            if model_filter:
                if p is None:
                    continue
                p_side = p if side == "yes" else 1 - p
                if p_side - px - fee(px) < 0.01:      # only when our model also says the wing is overpriced
                    continue
            out.append(Order(hyp=hid, kind="binary", venue=r["venue"], instrument=r["market"], side=side, price=px,
                             fee=fee(px), fair=None if p is None else (p if side == "yes" else 1 - p),
                             edge=None if p is None else (p if side == "yes" else 1 - p) - px - fee(px), settle_ts=str(r["end"]),
                             unit="$/contract", key=f"{hid}|{r['market']}|{side}",
                             info={"legs": [{**leg, "side": side, "price": px, "fee": fee(px)}], "mid": (bid + ask) / 2,
                                   "left_min": round(left, 1), "wing_price": cheap_bid,
                                   "size": r.get("bid_size") if side == "no" else r.get("ask_size")}))
    return out


@hyp("W01", "W", "Sell every cheap wing (hourly ladders)", "Kalshi and Polymarket hourly ladders, ≥ 15 min before the hour: "
     "a side priced 2–6¢ at the bid is sold at that bid (once per strike); held to settlement", "favourite–longshot bias: "
     "cheap tails win less often than their price (slide 7, R10)", frozen="2026-10-29 14:45")
def w01(ctx):
    return _wings(ctx, "W01", False)


@hyp("W02", "W", "Sell cheap wings our model also calls overpriced", "as W01, only if our fair value is ≥ 1¢ + fee below "
     "the wing's price", "", frozen="2026-10-29 14:45")
def w02(ctx):
    return _wings(ctx, "W02", True)


# ================================================================== O: options (latest 10-minute snapshots)
def _options_snapshot(ctx):
    """Latest common snapshot of the four option venues, standardised to USD per 1 BTC option."""
    c = ctx.cache.get("opt")
    if c and c[0] == ctx.opt_stamp():
        return c[1]
    day = ctx.now.strftime("%Y-%m-%d")
    frames = []
    try:
        d = pd.read_csv(config.LIVE / "deribit_chain" / f"{day}.csv.gz")
        d = d[d["snap_ts"] == d["snap_ts"].max()]
        p = d["instrument"].str.split("-", expand=True)
        # premiums are paid in BTC now: value them at the spot index, not at the expiry's futures price
        spot = ctx.tick.get("db_index") or ctx.tick.get("usd_mid") or d["underlying"]
        d = d.assign(venue="deribit", expiry=pd.to_datetime(p[1], format="%d%b%y", errors="coerce").dt.tz_localize("UTC") + pd.Timedelta(hours=8),
                     strike=pd.to_numeric(p[2], errors="coerce"), cp=p[3], F=d["underlying"],
                     bid=d["bid"] * spot, ask=d["ask"] * spot)
        frames.append(d[["snap_ts", "venue", "expiry", "strike", "cp", "bid", "ask", "F"]])
    except Exception:
        pass
    try:
        d = pd.read_csv(config.LIVE / "binance_options" / f"{day}.csv.gz")
        d = d[d["snap_ts"] == d["snap_ts"].max()]
        d = d.assign(venue="binance", expiry=pd.to_datetime(d["expiry"], utc=True), bid=d["bidPrice"], ask=d["askPrice"],
                     F=d["exercisePrice"])
        frames.append(d[["snap_ts", "venue", "expiry", "strike", "cp", "bid", "ask", "F"]])
    except Exception:
        pass
    for v in ("okx", "bybit"):
        try:
            d = pd.read_csv(config.LIVE / f"{v}_options" / f"{day}.csv.gz")
            d = d[d["snap_ts"] == d["snap_ts"].max()]
            d = d[~d["symbol"].astype(str).str.contains("_UM")]    # OKX USDT-margined series: illiquid, other units
            d = d.assign(venue=v, expiry=pd.to_datetime(d["expiry"], utc=True))
            if v == "okx":   # the logger valued the BTC premium at the forward; revalue at the spot index
                spot = ctx.tick.get("db_index") or ctx.tick.get("usd_mid")
                if spot:
                    d = d.assign(bid=d["bid"] * spot / d["F"], ask=d["ask"] * spot / d["F"])
            frames.append(d[["snap_ts", "venue", "expiry", "strike", "cp", "bid", "ask", "F"]])
        except Exception:
            pass
    if not frames:
        return None
    o = pd.concat(frames, ignore_index=True)
    o["snap_ts"] = pd.to_datetime(o["snap_ts"], utc=True)
    o = o[(o["snap_ts"] > ctx.now - pd.Timedelta(minutes=15)) & (o["expiry"] > ctx.now)]
    o = o.dropna(subset=["strike", "bid", "ask"])
    o = o[(o["bid"] > 0) & (o["ask"] > o["bid"]) & (o["ask"] < o["F"]) & (o["ask"] < 3 * o["bid"] + 50)]
    # an option is worth at least its intrinsic value: a bid far below it is a stale or broken quote
    intr = np.where(o["cp"] == "C", o["F"] - o["strike"], o["strike"] - o["F"]).clip(min=0)
    o = o[o["ask"] >= 0.9 * intr]
    # option quotes are a snapshot taken minutes ago, the perpetual and spot are live: if BTC moved more than 5 bp
    # since a venue's snapshot, every parity / straddle comparison on it is a timing artifact (29.10 13:33 UTC:
    # a $200 drop between the 13:30 snapshot and 13:33 made 129 fake parity "arbitrages" worth $6 139).
    o = o[_snapshot_fresh(ctx, o)]
    ctx.cache["opt"] = (ctx.opt_stamp(), o)
    return o


MAX_SNAPSHOT_MOVE = 5e-4


def _snapshot_fresh(ctx, o):
    t = ctx.ticks
    now_px = (ctx.tick or {}).get("bn_mid")
    if t is None or len(t) == 0 or "bn_mid" not in t or not now_px:
        return pd.Series(False, index=o.index)
    s = t.set_index("ts")["bn_mid"].sort_index()
    ok = {}
    for st in o["snap_ts"].unique():
        past = s[:pd.Timestamp(st) + pd.Timedelta(seconds=30)]
        ok[st] = bool(len(past)) and abs(math.log(now_px / past.iloc[-1])) <= MAX_SNAPSHOT_MOVE
    return o["snap_ts"].map(ok).astype(bool)


# Settlement index: Deribit and OKX settle on a BTC-USD index, Binance and Bybit on a BTC-USDT index. Across the
# two groups the "same" option differs by the USDT/USD rate — that is basis, not arbitrage.
INDEX_GROUP = {"deribit": "usd", "okx": "usd", "binance": "usdt", "bybit": "usdt"}


def _every_10(ctx):
    return ctx.now.minute % 10 == 3     # three minutes after the logger's snapshot


@hyp("O01", "O", "Cross-venue option arbitrage", "same expiry/strike/type: bid on venue B − ask on venue A − both "
     "fees > 0 → buy A, sell B (identical payoff; settlement indices differ slightly)", "law of one price across venues")
def o01(ctx):
    if not _every_10(ctx):
        return []
    o = _options_snapshot(ctx)
    if o is None or o["venue"].nunique() < 2:
        return []
    S = ctx.tick.get("bn_mid") or float(o["F"].median())
    out = []
    o = o.assign(grp=o["venue"].map(INDEX_GROUP))
    for (grp, exp, k, cp), g in o.groupby(["grp", "expiry", "strike", "cp"]):
        if g["venue"].nunique() < 2:
            continue
        a = g.loc[g["ask"].idxmin()]
        b = g.loc[g["bid"].idxmax()]
        if a["venue"] == b["venue"]:
            continue
        fees = option_fee(a["ask"], S) + option_fee(b["bid"], S)
        gain = b["bid"] - a["ask"] - fees
        if gain > 0:
            out.append(Order(hyp="O01", kind="instant", venue=f"{a['venue']}>{b['venue']}",
                             instrument=f"{exp:%Y-%m-%d}-{k:.0f}-{cp}", side="arb", price=a["ask"], fee=fees,
                             edge=gain, pnl=gain, pnl_gross=b["bid"] - a["ask"], unit="$/BTC option",
                             key=f"O01|{exp}|{k}|{cp}|{a['venue']}|{b['venue']}|{ctx.now.floor('h')}",
                             info={"buy": a["venue"], "sell": b["venue"], "ask": a["ask"], "bid": b["bid"]}))
    return out


@hyp("O02", "O", "Put–call parity", "expiries ≤ 7 days, each venue: (C − P + K) vs the perpetual on the same index "
     "(Deribit perp for Deribit/OKX, Binance perp for Binance/Bybit) minus the expected funding until expiry; a gap larger "
     "than option fees + 2 perp fees → trade the synthetic against the perp", "parity (Stoll 1969); perp ≈ forward")
def o02(ctx):
    if not _every_10(ctx):
        return []
    o = _options_snapshot(ctx)
    if o is None:
        return []
    out = []
    o = o[o["expiry"] < ctx.now + pd.Timedelta(days=7)]
    for (v, exp, k), g in o.groupby(["venue", "expiry", "strike"]):
        if INDEX_GROUP.get(v) == "usd":
            pb, pa, fr = ctx.tick.get("db_bid"), ctx.tick.get("db_ask"), ctx.tick.get("db_funding8h")
        else:
            pb, pa, fr = ctx.tick.get("bf_bid"), ctx.tick.get("bf_ask"), ctx.tick.get("bf_funding")
        if pb is None or pa is None or not (np.isfinite(pb) and np.isfinite(pa)):
            continue
        S = (pb + pa) / 2
        carry = (fr if fr is not None and np.isfinite(fr) else 0.0) * (exp - ctx.now).total_seconds() / 28800 * S
        c, p = g[g["cp"] == "C"], g[g["cp"] == "P"]
        if c.empty or p.empty:
            continue
        c, p = c.iloc[0], p.iloc[0]
        fees = option_fee(c["bid"], S) + option_fee(p["ask"], S) + 2 * PERP_FEE * S
        # sell synthetic (sell C, buy P) + buy perp (a long perp pays the funding `carry` until expiry)
        g1 = (c["bid"] - p["ask"] + k) - pa - fees - carry
        # buy synthetic (buy C, sell P) + sell perp
        fees2 = option_fee(c["ask"], S) + option_fee(p["bid"], S) + 2 * PERP_FEE * S
        g2 = pb - (c["ask"] - p["bid"] + k) - fees2 + carry
        for gain, side in ((g1, "sell_synth"), (g2, "buy_synth")):
            if gain > 0:
                out.append(Order(hyp="O02", kind="instant", venue=v, instrument=f"{exp:%Y-%m-%d}-{k:.0f}", side=side,
                                 price=float(c["bid"] if side == "sell_synth" else c["ask"]), fee=fees, edge=gain,
                                 pnl=gain, pnl_gross=gain + fees, unit="$/BTC option",
                                 key=f"O02|{v}|{exp}|{k}|{side}|{ctx.now.floor('h')}",
                                 info={"perp_bid": pb, "perp_ask": pa, "C": [c["bid"], c["ask"]], "P": [p["bid"], p["ask"]]}))
    return out


@hyp("O03", "O", "Static arbitrage of the smile", "each venue and expiry: call/put prices non-monotone in the strike, "
     "or a negative-cost butterfly, after fees", "no-arbitrage bounds (Carr & Madan 2005)")
def o03(ctx):
    if not _every_10(ctx):
        return []
    o = _options_snapshot(ctx)
    if o is None:
        return []
    S = ctx.tick.get("bn_mid") or float(o["F"].median())
    out = []
    for (v, exp, cp), g in o.groupby(["venue", "expiry", "cp"]):
        g = g.sort_values("strike").drop_duplicates("strike")
        K, B, A = g["strike"].values, g["bid"].values, g["ask"].values
        for i in range(len(g) - 1):
            # calls fall with K: buy K_i (lower), sell K_{i+1} if bid(K_{i+1}) > ask(K_i); puts: the mirror
            lo, hi = (i, i + 1) if cp == "C" else (i + 1, i)
            gain = B[hi] - A[lo] - option_fee(A[lo], S) - option_fee(B[hi], S)
            if gain > 0:
                out.append(Order(hyp="O03", kind="instant", venue=v, instrument=f"{exp:%Y-%m-%d}-{cp}-{K[i]:.0f}/{K[i+1]:.0f}",
                                 side="vertical", price=A[lo], fee=0, edge=gain, pnl=gain, pnl_gross=B[hi] - A[lo],
                                 unit="$/BTC option", key=f"O03|v|{v}|{exp}|{cp}|{K[i]}|{ctx.now.floor('h')}"))
        for i in range(1, len(g) - 1):
            if abs((K[i] - K[i - 1]) - (K[i + 1] - K[i])) > 1e-6:
                continue
            cost = A[i - 1] + A[i + 1] - 2 * B[i]
            fees = option_fee(A[i - 1], S) + option_fee(A[i + 1], S) + 2 * option_fee(B[i], S)
            if -cost - fees > 0:
                out.append(Order(hyp="O03", kind="instant", venue=v, instrument=f"{exp:%Y-%m-%d}-{cp}-fly-{K[i]:.0f}",
                                 side="butterfly", price=cost, fee=fees, edge=-cost - fees, pnl=-cost - fees,
                                 pnl_gross=-cost, unit="$/BTC option", key=f"O03|b|{v}|{exp}|{cp}|{K[i]}|{ctx.now.floor('h')}"))
    return out


def _straddle_fair(F, K, sd_usd):
    mu = F - K
    if sd_usd <= 0:
        return abs(mu)
    return sd_usd * math.sqrt(2 / math.pi) * math.exp(-mu * mu / (2 * sd_usd * sd_usd)) + mu * (1 - 2 * stats.norm.cdf(-mu / sd_usd))


@hyp("O04", "O", "ATM straddle vs our volatility forecast", "each venue, nearest daily expiry 3–30 h away, ATM strike: "
     "fair = E|S_T − K| under our variance forecast; fair − ask − fees > 10% of the ask → buy; bid − fair − fees > 10% → "
     "sell; one trade per venue and expiry; settles on the 30-min average before 08:00 UTC",
     "is our volatility forecast better than the options market's?")
def o04(ctx):
    if not _every_10(ctx):
        return []
    o = _options_snapshot(ctx)
    if o is None:
        return []
    S = ctx.tick.get("bn_mid")
    if S is None or not np.isfinite(S):
        return []
    out = []
    for v, g in o.groupby("venue"):
        tau_h = (g["expiry"] - ctx.now).dt.total_seconds() / 3600
        g = g[(tau_h >= 3) & (tau_h <= 30)]
        if g.empty:
            continue
        exp = g["expiry"].min()
        g = g[g["expiry"] == exp]
        F = float(g["F"].median())
        k = g.loc[(g["strike"] - F).abs().idxmin(), "strike"]
        c, p = g[(g["strike"] == k) & (g["cp"] == "C")], g[(g["strike"] == k) & (g["cp"] == "P")]
        if c.empty or p.empty:
            continue
        c, p = c.iloc[0], p.iloc[0]
        var = ctx.minutes.forecast_var(ctx.now, exp - pd.Timedelta(minutes=15))
        if not np.isfinite(var):
            continue
        fair = _straddle_fair(F, k, S * math.sqrt(var))
        ask, bid = c["ask"] + p["ask"], c["bid"] + p["bid"]
        fa = option_fee(c["ask"], S) + option_fee(p["ask"], S)
        fb = option_fee(c["bid"], S) + option_fee(p["bid"], S)
        info = {"strike": k, "F": F, "fair": fair, "bid": bid, "ask": ask, "expiry": str(exp),
                "fc_vol_ann": math.sqrt(var / ((exp - ctx.now).total_seconds() / 60) * 525600)}
        if fair - ask - fa > 0.10 * ask:
            out.append(Order(hyp="O04", kind="option", venue=v, instrument=f"{exp:%Y-%m-%d}-{k:.0f}-straddle", side="long",
                             price=ask, fee=fa, fair=fair, edge=fair - ask - fa, settle_ts=str(exp), unit="$/BTC straddle",
                             key=f"O04|{v}|{exp}", info=info))
        elif bid - fair - fb > 0.10 * bid:
            out.append(Order(hyp="O04", kind="option", venue=v, instrument=f"{exp:%Y-%m-%d}-{k:.0f}-straddle", side="short",
                             price=bid, fee=fb, fair=fair, edge=bid - fair - fb, settle_ts=str(exp), unit="$/BTC straddle",
                             key=f"O04|{v}|{exp}", info=info))
    return out


# ================================================================== F: funding carry across venues
@hyp("F01", "F", "Cross-venue funding carry", "30 min before the 00/08/16 UTC funding: predicted rates of Binance, Bybit "
     "and OKX differ by > 2 bp → short the perp that pays most, long the one that pays least; close 10 min after "
     "the funding", "funding-rate arbitrage")
def f01(ctx):
    nf = ctx.tick.get("bf_next_funding")
    if nf is None:
        return []
    mins = (pd.Timestamp(nf) - ctx.now).total_seconds() / 60
    if not (27 <= mins <= 31):
        return []
    rates = {v: ctx.tick.get(f"{v}_funding") for v in ("bf", "by", "ok")}
    rates = {v: r for v, r in rates.items() if r is not None and np.isfinite(r)}
    if len(rates) < 2:
        return []
    hi, lo = max(rates, key=rates.get), min(rates, key=rates.get)
    spread = rates[hi] - rates[lo]
    if spread <= 0.0002:
        return []
    q = {v: (ctx.tick.get(f"{v}_bid"), ctx.tick.get(f"{v}_ask")) for v in (hi, lo)}
    if any(x is None or not np.isfinite(x) for pair in q.values() for x in pair):
        return []
    return [Order(hyp="F01", kind="perp_pair", venue=f"{hi}/{lo}", instrument="BTCUSDT", side=f"short_{hi}+long_{lo}",
                  price=q[hi][0], fee=4 * PERP_FEE, hold_min=mins + 10, unit="return", key=f"F01|{nf}",
                  info={"short": hi, "long": lo, "short_px": q[hi][0], "long_px": q[lo][1], "funding_spread": spread,
                        "short_mid": sum(q[hi]) / 2, "long_mid": sum(q[lo]) / 2})]


FAMILIES = {"D": "Directional BTC", "P": "Prediction markets: fair value", "V": "Short binaries every 15 s: fair-value models",
            "A": "No-arbitrage (binaries)", "W": "Selling cheap wings", "T": "Kalshi trade tape (taker)", "M": "Maker quotes on Kalshi (simulated fills)", "O": "Options", "F": "Funding"}
