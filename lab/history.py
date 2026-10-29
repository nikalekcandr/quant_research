"""Rolling 1-minute BTC/USDT history (the project's minutes.parquet + live klines) and the lab's variance
forecaster used to price short-dated binaries and straddles.

forecast_var(now, end): expected variance of log returns over [now, end), from
    level  = 0.5 * EWMA(half-life 30 min) + 0.3 * mean(24 h) + 0.2 * mean(7 d) of deseasonalised squared returns
    shape  = hour-of-week seasonal factor (last 90 days)
so that a 5-minute window at the US open is priced with more variance than one at 03:00 UTC on a Sunday.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config
from lab.core import LAB, log, utcnow
from src.download import binance

RECENT = LAB / "klines_recent.parquet"
KEEP_DAYS = 200


class Minutes:
    def __init__(self):
        self.df = pd.DataFrame(columns=["close", "volume", "taker_buy"])
        self.season = None
        self.season_day = None
        self._load()

    def _load(self):
        parts = []
        f = config.INTERIM / "minutes.parquet"
        if f.exists():
            m = pd.read_parquet(f, columns=["open_time", "close", "volume"])
            m["open_time"] = pd.to_datetime(m["open_time"], utc=True)
            m = m[m["open_time"] > utcnow() - pd.Timedelta(days=KEEP_DAYS)]
            parts.append(m.set_index("open_time"))
        if RECENT.exists():
            r = pd.read_parquet(RECENT)
            r.index = pd.to_datetime(r.index, utc=True)
            parts.append(r)
        if parts:
            df = pd.concat(parts)
            df = df[~df.index.duplicated(keep="last")].sort_index()
            self.df = df.reindex(columns=["close", "volume", "taker_buy"])

    def update(self, http, max_calls: int = 20):
        """Fetch closed 1-minute candles after the last one we have (at most 14 days back)."""
        now = utcnow()
        start = self.df.index[-1] + pd.Timedelta(minutes=1) if len(self.df) else now - pd.Timedelta(days=14)
        start = max(start, now - pd.Timedelta(days=14))
        rows = []
        for _ in range(max_calls):
            if start > now - pd.Timedelta(minutes=1):
                break
            k = binance.api_get(http, "/api/v3/klines", {"symbol": "BTCUSDT", "interval": "1m",
                                                         "startTime": int(start.value // 10 ** 6), "limit": 1000})
            if not k:
                break
            for c in k:
                t = pd.to_datetime(int(c[0]), unit="ms", utc=True)
                if t + pd.Timedelta(minutes=1) <= now:          # closed candles only
                    rows.append((t, float(c[4]), float(c[5]), float(c[9])))
            start = pd.to_datetime(int(k[-1][0]), unit="ms", utc=True) + pd.Timedelta(minutes=1)
            if len(k) < 1000:
                break
        if rows:
            new = pd.DataFrame(rows, columns=["t", "close", "volume", "taker_buy"]).set_index("t")
            self.df = pd.concat([self.df, new])
            self.df = self.df[~self.df.index.duplicated(keep="last")].sort_index()
            self.df = self.df[self.df.index > now - pd.Timedelta(days=KEEP_DAYS)]
            if len(rows) > 50 or now.minute % 30 == 0:
                try:
                    pf = config.INTERIM / "minutes.parquet"
                    last_hist = pd.read_parquet(pf, columns=["open_time"])["open_time"].max() if pf.exists() else None
                    keep = self.df if last_hist is None else self.df[self.df.index > pd.Timestamp(last_hist)]
                    keep.to_parquet(RECENT)
                except Exception:
                    pass
        return len(rows)

    # -------------------------------------------------------------- helpers
    @property
    def close(self) -> pd.Series:
        return self.df["close"]

    def last(self) -> float:
        return float(self.df["close"].iloc[-1]) if len(self.df) else np.nan

    def price_at(self, ts: pd.Timestamp) -> float:
        """Close of the last minute that ended at or before ts (= price at ts)."""
        ts = pd.Timestamp(ts)
        idx = self.df.index.searchsorted(ts - pd.Timedelta(minutes=1), side="right") - 1
        return float(self.df["close"].iloc[idx]) if idx >= 0 else np.nan

    def open_at(self, ts: pd.Timestamp) -> float:
        """Open of the minute starting at ts (= close of the previous minute)."""
        return self.price_at(ts)

    def returns(self, days: float = 7) -> pd.Series:
        c = self.df["close"]
        c = c[c.index > c.index[-1] - pd.Timedelta(days=days)]
        return np.log(c).diff().dropna()

    def seasonal(self) -> pd.Series:
        """Hour-of-week multiplier of the per-minute variance (mean 1), last 90 days, cached for a day."""
        today = utcnow().normalize()
        if self.season is not None and self.season_day == today:
            return self.season
        r = self.returns(90)
        r2 = (r ** 2).clip(upper=(r ** 2).quantile(0.999))
        how = r2.index.dayofweek * 24 + r2.index.hour
        s = r2.groupby(how).mean()
        s = (s / s.mean()).reindex(range(168)).fillna(1.0)
        self.season = s.rolling(3, center=True, min_periods=1).mean()   # light smoothing across hours
        self.season_day = today
        return self.season

    def level(self) -> float:
        """Deseasonalised per-minute variance level (blend of 30-min EWMA, 24 h and 7 d means)."""
        key = self.df.index[-1] if len(self.df) else None
        if getattr(self, "_lv", (None, None))[0] == key and key is not None:
            return self._lv[1]
        r = self.returns(8)
        if len(r) < 600:
            return np.nan
        s = self.seasonal()
        how = r.index.dayofweek * 24 + r.index.hour
        z = (r ** 2) / s.reindex(how).values
        z = z.clip(upper=z.quantile(0.9995))
        ew = z.ewm(halflife=30).mean().iloc[-1]
        d1 = z[z.index > z.index[-1] - pd.Timedelta(hours=24)].mean()
        d7 = z.mean()
        out = 0.5 * ew + 0.3 * d1 + 0.2 * d7
        self._lv = (key, out)
        return out

    def forecast_var(self, now: pd.Timestamp, end: pd.Timestamp, level: float | None = None) -> float:
        lv = self.level() if level is None else level
        if not np.isfinite(lv) or end <= now:
            return np.nan
        s = self.seasonal()
        grid = pd.date_range(now.floor("min"), end, freq="1min", inclusive="left")
        if len(grid) == 0:
            return lv * (end - now).total_seconds() / 60
        how = grid.dayofweek * 24 + grid.hour
        total = float(s.reindex(how).values.sum()) * lv
        # the first, partial minute
        frac = 1 - (now - now.floor("min")).total_seconds() / 60
        total -= (1 - frac) * lv * float(s.iloc[int(how[0])])
        return max(total, 1e-12)
