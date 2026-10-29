"""Machine-learning return forecasts for D13 / D14: next-hour log return of BTC/USDT from 20 features of the
1-minute history. Walk-forward: retrained every 6 hours on the last 180 days (5-minute grid); every prediction
is logged with the realised return filled in later (information coefficient in the report)."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from lab.core import LAB, append_rows, log, utcnow

PRED_LOG = LAB / "ml_predictions.csv"
H = 60            # minutes ahead


def features(close: pd.Series, volume: pd.Series, at: pd.DatetimeIndex) -> pd.DataFrame:
    """Features known at each timestamp in `at` (computed from minutes strictly before it)."""
    lc = np.log(close)
    r1 = lc.diff()
    cs2 = (r1 ** 2).fillna(0).cumsum()
    cv = volume.fillna(0).cumsum()

    def at_(s, lag_min=0):
        return s.reindex(at - pd.Timedelta(minutes=1 + lag_min), method="ffill").values

    p0 = at_(lc)
    out = {}
    for m in (5, 15, 60, 240, 1440, 4320):
        out[f"r_{m}"] = p0 - at_(lc, m)
    for m in (60, 240, 1440):
        out[f"rv_{m}"] = np.log(np.maximum(at_(cs2) - at_(cs2, m), 1e-12) / m)
    out["rv_ratio"] = out["rv_60"] - out["rv_1440"]
    v60 = at_(cv) - at_(cv, 60)
    v7d = (at_(cv) - at_(cv, 10080)) / 168
    out["vol_z"] = np.log(np.maximum(v60, 1e-9) / np.maximum(v7d, 1e-9))
    hi = close.rolling(1440, min_periods=600).max()
    lo = close.rolling(1440, min_periods=600).min()
    out["dist_hi"] = p0 - np.log(at_(hi))
    out["dist_lo"] = p0 - np.log(at_(lo))
    out["z_60"] = out["r_60"] / np.sqrt(np.exp(out["rv_60"]) * 60)
    out["z_1440"] = out["r_1440"] / np.sqrt(np.exp(out["rv_1440"]) * 1440)
    hr = at.hour + at.minute / 60
    out["h_sin"], out["h_cos"] = np.sin(2 * np.pi * hr / 24), np.cos(2 * np.pi * hr / 24)
    out["dow_sin"], out["dow_cos"] = np.sin(2 * np.pi * at.dayofweek / 7), np.cos(2 * np.pi * at.dayofweek / 7)
    return pd.DataFrame(out, index=at)


class ML:
    def __init__(self):
        self.models = {}
        self.trained_at = None
        self.cols = None

    def maybe_train(self, minutes, now):
        if self.trained_at is not None and now - self.trained_at < pd.Timedelta(hours=6):
            return
        try:
            self.train(minutes, now)
        except Exception as e:
            log(f"ml: training failed ({str(e)[:160]})")
            self.trained_at = now - pd.Timedelta(hours=5)     # retry in an hour

    def train(self, minutes, now):
        from sklearn.ensemble import HistGradientBoostingRegressor
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        df = minutes.df
        df = df[df.index > now - pd.Timedelta(days=185)]
        close, vol = df["close"], df["volume"]
        grid = pd.date_range(close.index[0] + pd.Timedelta(days=4), now - pd.Timedelta(minutes=H + 2), freq="5min")
        X = features(close, vol, grid)
        lc = np.log(close)
        y = lc.reindex(grid + pd.Timedelta(minutes=H - 1), method="ffill").values - lc.reindex(grid - pd.Timedelta(minutes=1), method="ffill").values
        ok = np.isfinite(X.values).all(1) & np.isfinite(y)
        X, y = X[ok], y[ok]
        y = np.clip(y, -0.05, 0.05)
        if len(X) < 5000:
            log(f"ml: only {len(X)} training rows, skipped")
            self.trained_at = now
            return
        gbm = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.03, max_leaf_nodes=15, min_samples_leaf=300,
                                            l2_regularization=1.0, random_state=0)
        gbm.fit(X.values, y)
        ridge = make_pipeline(StandardScaler(), Ridge(alpha=30.0)).fit(X.values, y)
        self.models = {"gbm": gbm, "ridge": ridge}
        self.cols = list(X.columns)
        self.trained_at = now
        log(f"ml: trained on {len(X)} rows ({X.index[0]:%Y-%m-%d} .. {X.index[-1]:%Y-%m-%d %H:%M})")

    def predict(self, name, minutes, now):
        m = self.models.get(name)
        if m is None:
            return None
        df = minutes.df[minutes.df.index > now - pd.Timedelta(days=8)]
        X = features(df["close"], df["volume"], pd.DatetimeIndex([now.floor("min")]))
        if not np.isfinite(X.values).all():
            return None
        p = float(m.predict(X.values)[0])
        append_rows(PRED_LOG, [{"ts": now.floor("min"), "model": name, "pred": p}])
        return p

    @staticmethod
    def information_coefficient(minutes) -> pd.DataFrame:
        if not PRED_LOG.exists():
            return pd.DataFrame()
        d = pd.read_csv(PRED_LOG)
        d["ts"] = pd.to_datetime(d["ts"], utc=True)
        d = d.drop_duplicates(["ts", "model"])
        lc = np.log(minutes.close)
        a = lc.reindex(d["ts"] - pd.Timedelta(minutes=1), method="ffill").values
        b = lc.reindex(d["ts"] + pd.Timedelta(minutes=H - 1)).values
        d["realised"] = b - a
        d = d[d["ts"] + pd.Timedelta(minutes=H) <= minutes.close.index[-1]]
        rows = []
        for m, g in d.dropna(subset=["realised"]).groupby("model"):
            if len(g) >= 5:
                ic = g["pred"].corr(g["realised"], method="spearman")
                hit = (np.sign(g["pred"]) == np.sign(g["realised"])).mean()
                rows.append({"model": m, "n": len(g), "rank IC": ic, "t(IC)": ic * math.sqrt(max(len(g) - 2, 1)) / math.sqrt(max(1e-9, 1 - ic ** 2)),
                             "hit rate (sign)": hit})
        return pd.DataFrame(rows)
