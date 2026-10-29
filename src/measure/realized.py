"""Realized measures from 1-minute Binance closes, for arbitrary windows.

Conventions
-----------
* A 1-minute candle with open_time m closes at m + 59.999 s. The "price at time x" is the close of the
  candle that opened at x - 1 minute (the last completed candle before x).
* The return of candle m is log(close_m) - log(close_{m-1}).
* A window [a, b) contains the returns of candles with open_time in [a, b). Its realized variance is the
  sum of squared returns. With price_at(a) as the starting price, the window ends at price_at(b).
* Polymarket settles on the close of the candle that opens at 12:00 ET, i.e. on price_at(T + 1 min).
  Kalshi settles on the BRTI average over the minute before T, which we proxy by price_at(T).

All window sums use cumulative sums, so computing thousands of windows is instantaneous.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

MIN = pd.Timedelta(minutes=1)


class Minutes:
    """Pre-computed cumulative sums over the minute grid."""

    def __init__(self, df: pd.DataFrame):
        df = df.sort_values("open_time")
        self.t = df["open_time"].to_numpy(dtype="datetime64[ns]")
        self.t_index = pd.DatetimeIndex(df["open_time"])
        lc = np.log(df["close"].to_numpy(dtype=float))
        self.close = df["close"].to_numpy(dtype=float)
        r = np.diff(lc, prepend=lc[0])
        r5 = lc - np.concatenate([np.full(5, lc[0]), lc[:-5]])
        absr = np.abs(r)
        self.r = r
        filled = df["filled"].to_numpy(dtype=float) if "filled" in df else np.zeros(len(r))
        self._c = {
            "filled": np.concatenate([[0], np.cumsum(filled)]),
            "rv": np.concatenate([[0], np.cumsum(r ** 2)]),
            "rs_pos": np.concatenate([[0], np.cumsum(np.where(r > 0, r ** 2, 0))]),
            "rs_neg": np.concatenate([[0], np.cumsum(np.where(r < 0, r ** 2, 0))]),
            "bp": np.concatenate([[0], np.cumsum(absr * np.concatenate([[0], absr[:-1]]))]),
            "q": np.concatenate([[0], np.cumsum(r ** 4)]),
            "rv5": np.concatenate([[0], np.cumsum(r5 ** 2)]),
            "n": np.arange(len(r) + 1, dtype=float),
        }
        self.first = pd.Timestamp(self.t[0]).tz_localize("UTC")
        self.last = pd.Timestamp(self.t[-1]).tz_localize("UTC")

    def _idx(self, x) -> np.ndarray:
        x = pd.DatetimeIndex(pd.to_datetime(x, utc=True)).tz_convert("UTC").tz_localize(None)
        return np.searchsorted(self.t, x.to_numpy(dtype="datetime64[ns]"), side="left")

    def window(self, a, b) -> pd.DataFrame:
        """Realized measures for windows [a, b). a and b are arrays of UTC timestamps."""
        ia, ib = self._idx(a), self._idx(b)
        out = {k: v[ib] - v[ia] for k, v in self._c.items()}
        n = out.pop("n")
        out["bv"] = (np.pi / 2) * out.pop("bp")
        out["rq"] = n / 3 * out.pop("q")
        out["rv5"] = out["rv5"] / 5
        out["n_min"] = n
        out["filled_share"] = out.pop("filled") / np.maximum(n, 1)
        df = pd.DataFrame(out)
        # windows that are not fully covered by data are marked missing
        a_ts = pd.DatetimeIndex(pd.to_datetime(a, utc=True))
        b_ts = pd.DatetimeIndex(pd.to_datetime(b, utc=True))
        bad = (a_ts < self.first + 6 * MIN) | (b_ts > self.last + MIN)
        df.loc[np.asarray(bad), :] = np.nan
        return df

    def price_at(self, x) -> np.ndarray:
        """Close of the last candle that opened strictly before x - 1 min + 1 ns, i.e. price_at(x)."""
        i = self._idx(pd.DatetimeIndex(pd.to_datetime(x, utc=True)) - MIN)
        i = np.clip(i, 0, len(self.close) - 1)
        out = self.close[i].astype(float)
        xs = pd.DatetimeIndex(pd.to_datetime(x, utc=True))
        out[np.asarray((xs - MIN) > self.last)] = np.nan
        return out


# ------------------------------------------------------------------ intraday seasonality
def hour_of_week(ts: pd.DatetimeIndex) -> np.ndarray:
    return (ts.dayofweek * 24 + ts.hour).to_numpy()


def seasonal_profiles(m: Minutes, lookback_days: int = 90) -> dict:
    """For every calendar month, the mean squared 1-minute return in each of the 168 hours of the week,
    estimated on the `lookback_days` before the month starts (no look-ahead), normalised to mean 1."""
    ts = m.t_index
    r2 = pd.Series(m.r ** 2, index=ts)
    months = pd.date_range(ts[0].normalize().replace(day=1), ts[-1] + pd.offsets.MonthBegin(1), freq="MS")
    profiles = {}
    how_all = hour_of_week(ts)
    for mo in months:
        lo, hi = mo - pd.Timedelta(days=lookback_days), mo
        sel = (ts >= lo) & (ts < hi)
        if sel.sum() < 60 * 24 * 28:
            continue
        prof = pd.Series(r2.to_numpy()[sel]).groupby(how_all[sel]).mean().reindex(range(168))
        prof = prof.fillna(prof.mean())
        profiles[mo] = (prof / prof.mean()).to_numpy()
    return profiles


def seasonal_factor(profiles: dict, a, b) -> np.ndarray:
    """Expected variance of window [a, b) relative to a flat intraday pattern (1 = average)."""
    out = np.full(len(a), np.nan)
    keys = sorted(profiles)
    if not keys:
        return out
    for i, (x, y) in enumerate(zip(pd.to_datetime(a, utc=True), pd.to_datetime(b, utc=True))):
        mo = pd.Timestamp(year=x.year, month=x.month, day=1, tz="UTC")
        prof = profiles.get(mo)
        if prof is None:
            continue
        hrs = pd.date_range(x.floor("h"), y, freq="h", inclusive="left")
        if len(hrs) == 0:
            continue
        # minutes of the window falling in each hour
        w = np.array([(min(h + pd.Timedelta(hours=1), y) - max(h, x)).total_seconds() for h in hrs]) / 60
        w = np.clip(w, 0, None)
        out[i] = np.sum(prof[hour_of_week(hrs)] * w) / max(w.sum(), 1e-9)
    return out
