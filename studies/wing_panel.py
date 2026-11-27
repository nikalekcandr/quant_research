"""Reusable panel for the wing studies: Kalshi hourly-ladder quotes, spot, variance model, outcomes.

Every study in this folder that asks "what if we did X on the wings" needs the same objects:

  * the 1-minute book candles of the hourly ladder (data/raw/kalshi_kxbtcd_60m, Jul-Oct 2026) and the
    November snapshots of the near-spot strikes (data/raw/wing_gap_candles, 09-25 Nov 2026);
  * the USD spot at the moment of the quote (Binance minutes) and the lagged BRTI/Binance basis;
  * a strictly causal forecast of the variance left until settlement;
  * the settled outcome of every strike and the Kalshi fee.

The panel is cached (data/interim/wing_panel.parquet). Building it from the raw files takes about a minute;
the cache makes the experiments interactive.

Variance model
--------------
`var` = level(ts) x sum(profile) over the minutes left, minus 2/3 of a minute for the 60-second BRTI
settlement average (the same convention as the frozen W03/W04 rules):
  * profile: hour-of-week multiplier of the per-minute variance, mean 1 over 168 hours, estimated as the mean
    of the same hour-of-week over the previous 13 weeks and lagged by one day;
  * level: 0.5 x EWMA(30 min) + 0.3 x mean 24 h + 0.2 x mean 7 d of deseasonalised squared 1-minute returns
    (the live lab's blend, lab/history.py), lagged by one minute.
All quantities are computed on the full minute series and only ever read at the quote's own timestamp.

The fee
-------
Kalshi rounds the taker fee of an *order* up to the next cent:
    fee(C, p) = ceil(0.07 * C * p * (1 - p) * 100) / 100          dollars, C contracts at price p
so a one-contract order pays 1 cent on an 8-cent wing while a 194-contract order pays 0.515 cents per
contract. `fee(C, p)` below is the exact order-level charge; `fee_per_contract` divides it by C.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from studies.ladders import LADDER_COLS, ladder_files, read_ladder_file   # noqa: F401 — re-exported for callers

ROOT = Path(__file__).resolve().parents[1]
LADDER_DIR = ROOT / "data/raw/kalshi_kxbtcd_60m"
NOVEMBER_DIR = ROOT / "data/raw/wing_gap_candles"
MINUTES = ROOT / "data/interim/minutes.parquet"
CACHE = ROOT / "data/interim/wing_panel.parquet"

RATE = 0.07                 # Kalshi taker fee rate
BASIS_DAYS = 2              # settlements used for the lagged BRTI basis
PROFILE_WEEKS = 13          # hour-of-week profile window
SETTLE_MINUTES = 2 / 3      # variance removed for the 60-second settlement average


# ------------------------------------------------------------------------------------- fees
def fee(contracts, price):
    """Exact Kalshi taker fee in dollars for an order of `contracts` at `price` (rounded up to a cent)."""
    c = np.maximum(np.asarray(contracts, dtype=float), 0.0)
    p = np.clip(np.asarray(price, dtype=float), 0.0, 1.0)
    return np.ceil(np.round(RATE * c * p * (1 - p) * 100, 6)) / 100


def fee_per_contract(contracts, price):
    return fee(contracts, price) / np.maximum(np.asarray(contracts, dtype=float), 1.0)


def best_block(price: float, cap: int = 1000, tolerance: float = 0.02) -> dict:
    """Smallest order size whose per-contract fee is within `tolerance` (relative) of the limiting one.

    Per-contract fee tends to 0.07 p (1-p) as C grows; rounding adds at most 1 cent per order, so the
    excess over the limit is at most 1/(C) dollars. The smallest C that gets within 2% of the limit is the
    cheapest size at which the rounding cost stops mattering.
    """
    limit = RATE * price * (1 - price)
    if limit <= 0:
        return {"contracts": cap, "fee_per_contract": 0.0, "limit": 0.0}
    for c in range(1, cap + 1):
        f = float(fee(c, price)) / c
        if f <= limit * (1 + tolerance):
            return {"contracts": c, "fee_per_contract": f, "limit": limit}
    return {"contracts": cap, "fee_per_contract": float(fee(cap, price)) / cap, "limit": limit}


# ------------------------------------------------------------------------------------- spot
def minute_close() -> pd.Series:
    """Binance BTC/USDT 1-minute closes indexed by the *close* time (the price at that minute)."""
    m = pd.read_parquet(MINUTES, columns=["open_time", "close", "volume"])
    idx = pd.DatetimeIndex(m.open_time) + pd.Timedelta(minutes=1)
    s = pd.Series(m.close.to_numpy(), index=idx, name="close")
    v = pd.Series(m.volume.to_numpy(), index=idx, name="volume")
    out = pd.concat([s, v], axis=1)
    return out[~out.index.duplicated(keep="last")].sort_index()


def returns(close: pd.Series) -> pd.Series:
    return np.log(close).diff()


def profile_table(close: pd.Series, weeks: int = PROFILE_WEEKS) -> pd.DataFrame:
    """Mean squared return by (day, hour-of-week), smoothed over the previous `weeks` weeks and lagged a day.

    Returns a DataFrame indexed by day with 168 columns (hour-of-week), each row normalised to mean 1.
    """
    r2 = returns(close) ** 2
    idx = r2.index
    how = idx.day_of_week * 24 + idx.hour
    tab = (pd.DataFrame({"r2": r2.to_numpy(), "how": how, "day": idx.normalize()})
           .groupby(["day", "how"]).r2.mean().unstack().reindex(columns=range(168)))
    days = tab.index
    full = tab.reindex(pd.date_range(days.min(), days.max(), freq="D"))
    win = weeks * 7
    smooth = full.rolling(win, min_periods=max(7, weeks // 2)).mean().shift(1)
    norm = smooth.div(smooth.mean(axis=1), axis=0)
    return norm


def variance_inputs(close: pd.Series) -> pd.DataFrame:
    """Per-minute causal variance inputs: deseasonalised level, cumulative profile sum, basis-free variance."""
    prof = profile_table(close)
    how = close.index.day_of_week * 24 + close.index.hour
    day = close.index.normalize()
    p = prof.stack().rename("sea").reset_index()
    p.columns = ["day", "how", "sea"]
    p["day"] = pd.to_datetime(p["day"], utc=True)
    sea = pd.MultiIndex.from_arrays([day, how]).map(p.set_index(["day", "how"]).sea)
    sea = pd.Series(np.asarray(sea, dtype=float), index=close.index).fillna(1.0).clip(0.05, 20)
    z = returns(close) ** 2 / sea
    level = (0.5 * z.ewm(halflife=30, min_periods=300).mean()
             + 0.3 * z.rolling(1440, min_periods=300).mean()
             + 0.2 * z.rolling(10080, min_periods=1440).mean())
    cum = (sea * level).cumsum()          # per-minute variance, cumulated: sum over any window is a difference
    out = pd.DataFrame({"sea": sea, "level": level, "cum": cum})
    return out


def remaining_variance(v: pd.DataFrame, ts: pd.Series, end: pd.Series) -> np.ndarray:
    """Variance of log spot from `ts` to settlement `end`, net of the 60-second settlement average."""
    grid = v.index
    i0 = grid.searchsorted(pd.DatetimeIndex(ts), side="left")
    i1 = grid.searchsorted(pd.DatetimeIndex(end), side="left")
    cum = v.cum.to_numpy()
    sea = v.sea.to_numpy()
    level = v.level.to_numpy()
    i0c = np.clip(i0, 0, len(cum) - 1)
    i1c = np.clip(i1, 0, len(cum) - 1)
    last = np.clip(i1c - 180, 0, len(cum) - 1)      # minute that contains the settlement average
    total = cum[i1c] - cum[i0c]
    total = total - SETTLE_MINUTES * sea[last] * level[i0c]
    return np.maximum(total, 1e-12)


# ------------------------------------------------------------------------------------- panel
def _read_ladder() -> pd.DataFrame:
    parts = []
    for path in ladder_files(LADDER_DIR):
        d = read_ladder_file(path)
        if d is None or d.empty:
            continue
        d = d[d.strike_type.eq("greater")]
        parts.append(d)
    if not parts:
        raise FileNotFoundError(f"нет ни одного файла-лестницы: '{LADDER_DIR}'")
    d = pd.concat(parts, ignore_index=True)
    d["ts"] = pd.to_datetime(d.end_ts, unit="s", utc=True)
    d["end"] = pd.to_datetime(d.close_time, utc=True)
    d = d.drop(columns=["end_ts", "close_time", "strike_type"])
    d["source"] = "candles"
    months = None
    return d


def _read_november() -> pd.DataFrame:
    parts = []
    for path in sorted(NOVEMBER_DIR.glob("*.csv.gz")):
        parts.append(pd.read_csv(path))
    d = pd.concat(parts, ignore_index=True)
    d["ts"] = pd.to_datetime(d.end_ts, unit="s", utc=True)
    d["end"] = pd.to_datetime(d.close_time, utc=True)
    d = d[d.strike_type.eq("greater")].copy()
    d["event"] = d.ticker.str.rsplit("-T", n=1).str[0]
    d = d[["ticker", "event", "ts", "end", "yes_bid", "yes_ask", "volume", "strike", "result",
           "expiration_value"]]
    d["source"] = "november"
    return d


def implied_spot(d: pd.DataFrame, min_strikes: int = 5) -> pd.Series:
    """Spot implied by a whole ladder snapshot: the one-parameter least-squares fit of Phi(log(S/K)/sigma).

    Uses the model variance (which does not depend on the spot) and every strike with a two-sided book;
    returns NaN where the ladder has fewer than `min_strikes` usable quotes. This is the fallback for hours
    without an independent spot (the November snapshots): on Jul-Oct it agrees with Binance x basis to a
    median of 3 bp, which is a quarter of the 20-minute standard deviation.
    """
    from scipy.optimize import brentq

    g = d[(d.bid > 0) & (d.ask < 1) & (d.ask - d.bid).le(0.03 + 1e-9)].copy()
    g["mid"] = (g.bid + g.ask) / 2

    def fit(x):
        x = x.sort_values("strike")
        s = x.strike.to_numpy(float)
        m = x.mid.to_numpy(float)
        if len(s) < min_strikes:
            return np.nan
        sd = math.sqrt(float(np.median(x.var_rem)))
        if not np.isfinite(sd) or sd <= 0:
            return np.nan
        f = lambda lk: float(np.mean(norm.cdf((lk - np.log(s)) / sd) - m))
        lo, hi = math.log(s.min()) - 6 * sd, math.log(s.max()) + 6 * sd
        try:
            if f(lo) * f(hi) > 0:
                return np.nan
            return float(np.exp(brentq(f, lo, hi, xtol=1e-7)))
        except Exception:
            return np.nan

    g = g[g.var_rem.notna()]
    return g.groupby(["event", "ts"])[["strike", "mid", "var_rem"]].apply(fit)


def settlement_level(d: pd.DataFrame) -> pd.Series:
    """Per-minute variance level for hours the minute series does not reach (November).

    Uses the settled BRTI values of the previous hours: the EWMA(24 h) of squared hourly log returns,
    divided by 60 minutes, lagged by one settlement. The hour-of-week profile of those hours is flat
    (the profile is estimated on the minute series, which ends in October), so this is a level-only
    estimate - coarse, but far better than a degenerate sigma.
    """
    settle = d.drop_duplicates("end")[["end", "expiration_value"]].copy()
    settle["value"] = pd.to_numeric(settle.expiration_value, errors="coerce")
    settle = settle.dropna(subset=["value"]).set_index("end").value.sort_index()
    r = np.log(settle).diff()
    level = (r ** 2).ewm(halflife=24, min_periods=6).mean().shift(1) / 60.0
    return level


def _basis(d: pd.DataFrame, close: pd.Series) -> pd.Series:
    """Lagged median of settlement / Binance close at settlement, over the previous settlements of the panel."""
    settle = d.drop_duplicates("end").set_index("end")
    settle = settle.assign(S=close.reindex(settle.index).to_numpy(),
                           value=pd.to_numeric(settle.expiration_value, errors="coerce"))
    ratio = (settle.value / settle.S).sort_index()
    ratio = ratio[(ratio > 0.9) & (ratio < 1.1)]
    lagged = ratio.shift(1).rolling(BASIS_DAYS * 12, min_periods=5).median()
    return d.end.map(lagged)


def build(force: bool = False) -> pd.DataFrame:
    """Strike-minute panel with spot, basis, variance, model probabilities and the settled outcome."""
    if CACHE.exists() and not force:
        return pd.read_parquet(CACHE)
    px = minute_close()
    v = variance_inputs(px.close)
    d = pd.concat([_read_ladder(), _read_november()], ignore_index=True)
    d = d[(d.yes_ask > 0) & (d.yes_ask <= 1) & d.strike.gt(0)].copy()
    d["left"] = (d.end - d.ts).dt.total_seconds() / 60
    d = d[d.left.between(-0.5, 60.5)]
    d["S"] = px.close.reindex(d.ts).to_numpy()
    d["basis"] = _basis(d, px.close)
    d["var_rem"] = remaining_variance(v, d.ts, d.end)
    # hours beyond the minute series: variance from the settled hourly returns, distributed over the minutes
    outside = d.ts > v.index.max()
    if outside.any():
        lvl = settlement_level(d)
        d.loc[outside, "var_rem"] = (lvl.reindex(d.loc[outside, "end"]).to_numpy()
                                    * np.maximum(d.loc[outside, "left"] - SETTLE_MINUTES, 1 / 3))
    d["bid"] = d.yes_bid.clip(lower=0)
    d["ask"] = d.yes_ask.clip(upper=1)
    d["spot_source"] = np.where(d.S.notna(), "binance", "ladder")
    missing = d.S.isna()
    if missing.any():                        # November: no independent spot in the repository
        implied = implied_spot(d[missing]).rename("S_implied")
        d = d.merge(implied.reset_index(), on=["event", "ts"], how="left")
        d["S"] = d.S.fillna(d.S_implied)
        d["basis"] = d.basis.fillna(1.0)
        d = d.drop(columns=["S_implied"])
    d["y"] = d.result.eq("yes").astype(float).where(d.result.isin(["yes", "no"]))
    d["strike_round"] = np.select(
        [np.isclose(d.strike % 1000, 0, atol=1e-6), np.isclose(d.strike % 500, 0, atol=1e-6),
         np.isclose(d.strike % 100, 0, atol=1e-6)],
        ["1000", "500", "100"], default="other")
    # distance from spot in standard deviations of the remaining move
    z = np.log(d.S * d.basis / d.strike) / np.sqrt(d["var_rem"])
    d["z"] = z
    d["p_yes"] = norm.cdf(z)
    d["p_yes_t4"] = _student_t_cdf(z)
    d = d.dropna(subset=["S", "basis", "var_rem"])
    d = d[["ticker", "event", "source", "spot_source", "ts", "end", "left", "strike", "bid", "ask",
           "volume", "strike_round", "S", "basis", "var_rem", "z", "p_yes", "p_yes_t4", "y",
           "expiration_value"]]
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    d.to_parquet(CACHE, index=False)
    return d


def _student_t_cdf(z: np.ndarray, df: float = 4.0) -> np.ndarray:
    """Student-t(df) cdf at z for a t variable scaled to unit variance (as in the frozen rules)."""
    from scipy.stats import t as student_t
    return student_t.cdf(np.asarray(z) * math.sqrt(df / (df - 2.0)), df)


# ------------------------------------------------------------------------------------- cheap sides
BANDS = {"late": (0.10, 0.20, 5, 15), "early": (0.06, 0.10, 15, 30)}


def cheap_sides(d: pd.DataFrame, lo: float, hi: float, l0: float, l1: float) -> pd.DataFrame:
    """One row per strike-minute in which one side of the strike is a cheap wing in [lo, hi].

    `wing_bid` is the bid of the cheap side (what a seller receives), `price` the price of the other side
    (what the seller pays and what the fee is charged on), `won` whether the cheap side pays out.
    """
    g = d[d.left.between(l0, l1) & d.bid.gt(0) & d.ask.lt(1) & d.ask.ge(d.bid)
          & (d.ask - d.bid).le(0.03 + 1e-9) & d.y.notna()].copy()
    cy = g.bid.between(lo, hi)
    cn = (1 - g.ask).between(lo, hi)
    a = g[cy].assign(cheap_yes=True, wing_bid=g.bid[cy], won=g.y[cy])
    b = g[cn].assign(cheap_yes=False, wing_bid=1 - g.ask[cn], won=1 - g.y[cn])
    c = pd.concat([a, b], ignore_index=True)
    c["price"] = 1 - c.wing_bid
    c["mid_wing"] = np.where(c.cheap_yes, (c.bid + c.ask) / 2, 1 - (c.bid + c.ask) / 2)
    c["fair_n"] = np.where(c.cheap_yes, c.p_yes, 1 - c.p_yes)
    c["fair_t4"] = np.where(c.cheap_yes, c.p_yes_t4, 1 - c.p_yes_t4)
    c["market"] = c.ticker
    return c


def pnl(trades: pd.DataFrame, contracts: float | str = 1, price_col: str = "price",
        pay_col: str = "won") -> pd.Series:
    """P&L of selling the cheap side at its bid and holding to settlement, at an order size.

    `contracts` may be a scalar, a column name or "block" (the fee-optimal block size for the price).
    """
    n = trades[contracts].to_numpy(dtype=float) if isinstance(contracts, str) and contracts in trades else (
        np.asarray(contracts, dtype=float))
    p = trades[price_col].to_numpy(dtype=float)
    gross = trades.wing_bid.to_numpy(dtype=float) - trades[pay_col].to_numpy(dtype=float)
    return gross * n - fee(n, p)


if __name__ == "__main__":
    d = build(force=True)
    print(f"{len(d):,} strike-minutes, {d.event.nunique():,} hours, {d.ticker.nunique():,} strikes")
    print(d.groupby("source").agg(rows=("ticker", "size"), hours=("event", "nunique"),
                                 strikes=("ticker", "nunique")))
    print(d.groupby("source").left.describe())
