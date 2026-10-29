"""Backtests of the trade-tape strategies on Kalshi KXBTC15M prints (data/raw/kx15_trades_agg: contracts and YES
notional per market × taker side × 1¢ × minute before the close) and Binance minutes for the spot filter.

M03  Lottery seller (maker): rest an offer on the cheap side (price 2–20¢); filled by the first taker who buys the
     cheap side in a minute with ≥ 2 minutes left; one fill per market; hold to settlement; maker fee 0.0175·P(1−P).
M04  Same, spot-aware: no fill in a minute that follows a minute in which spot moved ≥ 3 bp toward the cheap side
     winning (a maker cancels after seeing that move; only completed minutes are used — no look-ahead).
T01  Copy the favourite buyer (taker): a minute with ≥ 200 contracts bought by takers on the favourite side at
     65–95¢ → buy the same side in the next minute at that minute's favourite-side VWAP + 1¢, taker fee.
T02  Fade the lottery burst (taker): a minute with ≥ 500 contracts bought by takers on the cheap side at ≤ 20¢ →
     buy the favourite side in the next minute at the favourite-side VWAP + 1¢, taker fee.
Every strategy: one trade per market, first half / second half in time, t clustered by market (= per trade here).
Upper-bound caveats for M03/M04: queue position is ignored (live M03/M04 model the queue).
    python studies/tape_strategies.py <raw dir> [minutes.parquet]
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/raw")
SPOT = Path(sys.argv[2]) if len(sys.argv) > 2 else RAW.parent / "interim" / "minutes.parquet"


def taker_fee(p):
    return np.ceil(np.round(0.07 * p * (1 - p) * 100, 6)) / 100


def maker_fee(p):
    return 0.0175 * p * (1 - p)


def load():
    mk = pd.read_csv(RAW / "kx15" / "markets.csv", usecols=["ticker", "close_time", "result", "floor_strike"])
    mk["close_time"] = pd.to_datetime(mk["close_time"], utc=True)
    tr = pd.concat([pd.read_csv(f) for f in sorted((RAW / "kx15_trades_agg").glob("*.csv")) if f.stat().st_size > 10])
    tr = tr.merge(mk, on="ticker")
    tr = tr[tr.result.isin(["yes", "no"]) & tr.side.isin(["yes", "no"]) & (tr.contracts > 0)].copy()
    tr["p"] = tr.notional / tr.contracts                      # YES price
    tr["y"] = (tr.result == "yes").astype(float)
    tr["paid"] = np.where(tr.side == "yes", tr.p, 1 - tr.p)   # price of the side the taker bought
    tr["minute"] = tr.close_time - pd.to_timedelta(tr.left + 1, unit="min")   # start of that minute
    return tr


def spot_moves(tr):
    try:
        s = pd.read_parquet(SPOT, columns=["open_time", "close"])
    except Exception:
        return None
    s["ts"] = pd.to_datetime(s.open_time, utc=True)
    s = s.set_index("ts")["close"]
    # close(t) is the close of the minute that opens at t; the return of the minute that ENDED when minute m began
    # is close(m − 1) − close(m − 2): known to a maker at m, nothing later
    r = np.log(s).diff()
    r.index = r.index + pd.Timedelta(minutes=1)
    return r


def summarize(name, t):
    out = []
    if t.empty:
        return [{"strategy": name, "half": "all", "trades": 0}]
    split = t.minute.quantile(0.5)
    for half, g in (("first", t[t.minute < split]), ("second", t[t.minute >= split]), ("all", t)):
        x = g.pnl.values
        n = len(x)
        out.append({"strategy": name, "half": half, "trades": n, "mean ¢": 100 * x.mean() if n else np.nan,
                    "t": x.mean() / (x.std(ddof=1) / math.sqrt(n)) if n > 2 and x.std() > 0 else np.nan,
                    "hit": (x > 0).mean() if n else np.nan, "avg price ¢": 100 * g.px.mean() if n else np.nan,
                    "worst ¢": 100 * x.min() if n else np.nan, "total $": x.sum()})
    return out


def m03(tr, spot=None, bp=3e-4):
    c = tr[(tr.paid.between(0.02, 0.20)) & (tr.left >= 2)].copy()
    if spot is not None:
        c["ret"] = spot.reindex(c.minute).values
        # the cheap side wins if BTC ends above the strike (cheap YES) → adverse = spot up; cheap NO → spot down
        adverse = np.where(c.side == "yes", c.ret, -c.ret)
        c = c[~(adverse >= bp)]
        c = c[np.isfinite(c.ret)]
    c = c.sort_values(["ticker", "left"], ascending=[True, False]).groupby("ticker").head(1)
    won = np.where(c.side == "yes", c.y, 1 - c.y)            # the cheap side (that we sold) won → we pay 1
    c["px"] = c.paid
    c["pnl"] = c.paid - won - maker_fee(c.paid)
    return c


def follow(tr, side_filter, lo, hi, min_contracts, buy):
    """Signal minute: takers bought `side_filter` (favourite / cheap) at lo..hi with ≥ min_contracts; trade in the
    next minute at the VWAP of the side we buy + 1¢ (taker), one per market."""
    g = tr[(tr.paid.between(lo, hi))]
    g = g.assign(fav=np.where(g.paid >= 0.5, g.side, np.where(g.side == "yes", "no", "yes")))
    sig = g.groupby(["ticker", "left", "fav"]).contracts.sum().reset_index()
    sig = sig[sig.contracts >= min_contracts]
    # price of the favourite side in the next minute (left − 1): VWAP of prints there, side-normalised
    nxt = tr.assign(fav_px=np.where(tr.p >= 0.5, tr.p, 1 - tr.p), fav=np.where(tr.p >= 0.5, "yes", "no"))
    nv = nxt.groupby(["ticker", "left", "fav"]).apply(lambda h: np.average(h.fav_px, weights=h.contracts),
                                                      include_groups=False).rename("vwap").reset_index()
    nv["left"] = nv["left"] + 1                              # align: next minute's price on the signal row
    s = sig.merge(nv, on=["ticker", "left", "fav"]).sort_values(["ticker", "left"], ascending=[True, False])
    s = s[s.left >= 2].groupby("ticker").head(1)
    s = s.merge(tr.drop_duplicates("ticker")[["ticker", "y", "close_time"]], on="ticker")
    s["minute"] = s.close_time - pd.to_timedelta(s.left, unit="min")
    s["px"] = (s.vwap + 0.01).clip(upper=0.99)
    won = np.where(s.fav == "yes", s.y, 1 - s.y)
    s["pnl"] = won - s.px - taker_fee(s.px)
    return s


def grid(tr, spot):
    """Lottery seller by price band × minutes left, with and without the spot filter; one fill per market per cell.
    Chosen on the first half of the sample (by date), reported on the second half."""
    split = tr.close_time.quantile(0.5)
    rows = []
    for lo, hi in ((0.01, 0.05), (0.05, 0.10), (0.10, 0.20), (0.02, 0.20)):
        for l0, l1 in ((2, 4), (5, 8), (9, 14), (2, 14)):
            for sp in (False, True):
                c = tr[(tr.paid.between(lo, hi)) & (tr.left.between(l0, l1))].copy()
                if sp and spot is not None:
                    c["ret"] = spot.reindex(c.minute).values
                    adverse = np.where(c.side == "yes", c.ret, -c.ret)
                    c = c[~(adverse >= 3e-4) & np.isfinite(c.ret)]
                c = c.sort_values(["ticker", "left"], ascending=[True, False]).groupby("ticker").head(1)
                won = np.where(c.side == "yes", c.y, 1 - c.y)
                c["pnl"] = c.paid - won - maker_fee(c.paid)
                rec = {"price": f"{lo:.2f}–{hi:.2f}", "minutes left": f"{l0}–{l1}", "spot filter": sp}
                for half, g in (("first", c[c.close_time < split]), ("second", c[c.close_time >= split])):
                    x = g.pnl.values
                    rec[f"n {half}"] = len(x)
                    rec[f"¢ {half}"] = 100 * x.mean() if len(x) else np.nan
                    rec[f"t {half}"] = x.mean() / (x.std(ddof=1) / math.sqrt(len(x))) if len(x) > 2 and x.std() > 0 else np.nan
                rows.append(rec)
    g = pd.DataFrame(rows)
    best = g.sort_values("t first", ascending=False).head(1)
    return g, best


def by_month(tr, spot):
    c = m03(tr, spot)
    c["month"] = c.close_time.dt.strftime("%Y-%m")
    return c.groupby("month").pnl.agg(trades="size", mean_c=lambda x: 100 * x.mean(), worst=lambda x: 100 * x.min(),
                                      hit=lambda x: (x > 0).mean()).round(2)


def main():
    tr = load()
    spot = spot_moves(tr)
    print(f"{tr.ticker.nunique()} markets, {tr.close_time.min():%Y-%m-%d} .. {tr.close_time.max():%Y-%m-%d}\n")
    rows = []
    rows += summarize("M03 lottery seller (maker)", m03(tr))
    if spot is not None:
        rows += summarize("M04 lottery seller, spot-aware (maker)", m03(tr, spot))
    rows += summarize("T01 copy favourite buyer (taker)", follow(tr, "fav", 0.65, 0.95, 200, "fav"))
    rows += summarize("T02 fade lottery burst (taker)", follow(tr, "cheap", 0.0, 0.20, 500, "fav"))
    d = pd.DataFrame(rows)
    print(d.round(3).to_string(index=False))
    g, best = grid(tr, spot)
    print("\n== lottery seller grid (choose on the first half, read the second)\n", g.round(2).to_string(index=False))
    print("\nchosen on the first half:\n", best.round(2).to_string(index=False))
    print("\n== M04 by month\n", by_month(tr, spot).to_string())
    out = RAW.parent.parent / "results" / "studies"
    out.mkdir(parents=True, exist_ok=True)
    d.to_csv(out / "tape_strategies.csv", index=False)
    g.to_csv(out / "tape_lottery_grid.csv", index=False)


if __name__ == "__main__":
    main()
