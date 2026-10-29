"""Volatility model zoo, run every hour at HH:05 for two windows:
    1h   [HH:05, HH+1:00)  — the window of the project's Kalshi forecasts (kx1h), so the frozen models compete too
    24h  [HH:05, HH:05 + 24 h)
Members: the lab's seasonal blend, RiskMetrics EWMA, hourly log-HAR with hour-of-week seasonality, GARCH(1,1),
GJR-GARCH and EGARCH on deseasonalised hourly returns, DVOL-implied, gradient-boosted trees on log RV, an equal-
weight combination, and every model of the frozen project (read from forecasts_live/kx1h, 1h window only).
All forecasts are variances of log returns over the window; scored with QLIKE once the window has passed.
"""
from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd

import config
from lab.core import ZOO, append_rows, log

FIT_EVERY = pd.Timedelta(hours=24)


class VolZoo:
    def __init__(self):
        self.fits = {}
        self.fit_at = None

    # -------------------------------------------------------------- hourly data
    @staticmethod
    def hourly(minutes, days=120):
        r = minutes.returns(days)
        rv = (r ** 2).groupby(r.index.floor("h")).agg(["sum", "count"])
        rv = rv[rv["count"] >= 50]["sum"]
        ret = np.log(minutes.close).groupby(minutes.close.index.floor("h")).last().diff()
        return rv, ret.reindex(rv.index)

    def refit(self, minutes, now):
        rv, ret = self.hourly(minutes, 120)
        s = minutes.seasonal()
        how = rv.index.dayofweek * 24 + rv.index.hour
        sea = s.reindex(how).values
        out = {}
        # log-HAR with seasonality: log(rv/s) on its lags (1 h, 24 h mean, 168 h mean)
        z = np.log(rv / sea)
        X = pd.DataFrame({"l1": z.shift(1), "l24": z.rolling(24).mean().shift(1), "l168": z.rolling(168).mean().shift(1)})
        d = pd.concat([z.rename("y"), X], axis=1).dropna()
        if len(d) > 500:
            A = np.c_[np.ones(len(d)), d[["l1", "l24", "l168"]].values]
            b, *_ = np.linalg.lstsq(A, d["y"].values, rcond=None)
            res = d["y"].values - A @ b
            out["har"] = (b, float(np.mean(np.exp(res))))
        # GARCH family on deseasonalised hourly returns (in percent)
        try:
            from arch import arch_model
            x = (ret / np.sqrt(sea)).dropna().iloc[-2000:] * 100
            for name, kw in (("garch", {"vol": "GARCH", "p": 1, "q": 1}), ("gjr", {"vol": "GARCH", "p": 1, "o": 1, "q": 1}),
                             ("egarch", {"vol": "EGARCH", "p": 1, "o": 1, "q": 1})):
                try:
                    out[name] = arch_model(x, mean="Zero", dist="t", **kw).fit(disp="off")
                except Exception as e:
                    log(f"volzoo: {name} fit failed ({str(e)[:80]})")
        except ImportError:
            pass
        # gradient-boosted trees on log RV
        try:
            from sklearn.ensemble import HistGradientBoostingRegressor
            F = pd.DataFrame({"l1": np.log(rv).shift(1), "l3": np.log(rv.rolling(3).mean()).shift(1),
                              "l24": np.log(rv.rolling(24).mean()).shift(1), "l168": np.log(rv.rolling(168).mean()).shift(1),
                              "sea": np.log(sea), "hour": rv.index.hour, "dow": rv.index.dayofweek})
            dd = pd.concat([np.log(rv).rename("y"), F], axis=1).dropna()
            if len(dd) > 500:
                g = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, max_leaf_nodes=15,
                                                  min_samples_leaf=50, random_state=0).fit(dd.drop(columns="y").values, dd["y"].values)
                res = dd["y"].values - g.predict(dd.drop(columns="y").values)
                out["gbm"] = (g, float(np.mean(np.exp(res))))
        except Exception as e:
            log(f"volzoo: gbm fit failed ({str(e)[:80]})")
        self.fits = out
        self.fit_at = now
        log(f"volzoo: refitted {', '.join(out)}")

    # -------------------------------------------------------------- forecasts
    def forecast(self, minutes, now, dvol: float | None, windows=None, kx_level: float | None = None):
        """windows: list of (horizon label, start, end); default = the hourly pair (1h to the hour, 24h).
        The 5-minute calls add a rolling 60-minute window and the window to the next full hour (where the Kalshi
        hourly ladder's own variance competes: model 'kx_ladder')."""
        if self.fit_at is None or now - self.fit_at > FIT_EVERY:
            self.refit(minutes, now)
        t = now.floor("h") + pd.Timedelta(minutes=5)
        if windows is None:
            windows = [("1h", t, now.floor("h") + pd.Timedelta(hours=1)), ("24h", t, t + pd.Timedelta(hours=24))]
        rows = []
        s = minutes.seasonal()
        rv, ret = self.hourly(minutes, 10)
        for hz, t, end in windows:
            grid = pd.date_range(t, end, freq="1min", inclusive="left")
            how = grid.dayofweek * 24 + grid.hour
            shape_min = s.reindex(how).values          # per-minute seasonal factors over the window
            n = len(grid)
            fc = {}
            fc["lab_blend"] = minutes.forecast_var(t, end)
            # RiskMetrics on the last 10 days of hourly RV (deseasonalised), lambda = 0.94 per hour
            z = rv / s.reindex(rv.index.dayofweek * 24 + rv.index.hour).values
            lvl = z.ewm(alpha=0.06).mean().iloc[-1] / 60          # per-minute deseasonalised variance
            fc["ewma_rm"] = float(lvl * shape_min.sum())
            if "har" in self.fits:
                b, smear = self.fits["har"]
                zz = np.log(z)
                x = np.r_[1, zz.iloc[-1], zz.iloc[-24:].mean(), zz.iloc[-168:].mean() if len(zz) >= 168 else zz.mean()]
                lvl_h = math.exp(x @ b) * smear / 60
                fc["har_hourly"] = float(lvl_h * shape_min.sum())
            for name in ("garch", "gjr", "egarch"):
                m = self.fits.get(name)
                if m is None:
                    continue
                try:
                    steps = max(1, math.ceil(n / 60))
                    f = m.forecast(horizon=steps, reindex=False).variance.values[-1] / 1e4 / 60   # per-minute, deseasonalised
                    per_min = np.repeat(f, 60)[:n]
                    fc[name] = float((per_min * shape_min).sum())
                except Exception:
                    pass
            if dvol is not None and np.isfinite(dvol):
                fc["dvol"] = float((dvol / 100) ** 2 / 525600 * shape_min.sum())
            if "gbm" in self.fits:
                g, smear = self.fits["gbm"]
                ok = len(rv) >= 168
                if ok:
                    hrs = pd.date_range(t.floor("h"), end, freq="h", inclusive="left")
                    tot = 0.0
                    for hh in hrs:
                        x = np.array([[math.log(rv.iloc[-1]), math.log(rv.iloc[-3:].mean()), math.log(rv.iloc[-24:].mean()),
                                       math.log(rv.iloc[-168:].mean()), math.log(s.iloc[hh.dayofweek * 24 + hh.hour]),
                                       hh.hour, hh.dayofweek]])
                        tot += math.exp(g.predict(x)[0]) * smear
                    frac = n / (len(hrs) * 60)
                    fc["gbm_vol"] = float(tot * frac)
            vals = [v for v in fc.values() if v is not None and np.isfinite(v) and v > 0]
            if vals:
                fc["combo"] = float(math.exp(np.mean(np.log(vals))))
            if kx_level is not None and np.isfinite(kx_level) and end == end.floor("h") and hz in ("1h", "to_hour"):
                fc["kx_ladder"] = float(kx_level * shape_min[: max(1, n - 1)].sum())   # BRTI 60-s average at the end
            if hz == "1h":       # the frozen project's models, if its live forecast for this hour exists
                f = config.FORECASTS_LIVE / "kx1h" / f"{t:%Y%m%dT%H%M}.json"
                if f.exists():
                    try:
                        for mname, v in json.loads(f.read_text(encoding="utf-8"))["forecasts"].items():
                            if v.get("level") is not None:
                                fc[f"proj_{mname}"] = float(v["level"])
                    except Exception:
                        pass
            for mname, v in fc.items():
                if v is not None and np.isfinite(v) and v > 0:
                    rows.append({"t": t, "end": end, "horizon": hz, "model": mname, "var": v})
        append_rows(ZOO, rows)
        return rows

    @staticmethod
    def score(minutes) -> tuple[pd.DataFrame, pd.DataFrame]:
        if not ZOO.exists():
            return pd.DataFrame(), pd.DataFrame()
        z = pd.read_csv(ZOO)
        z["t"], z["end"] = pd.to_datetime(z["t"], utc=True), pd.to_datetime(z["end"], utc=True)
        z = z.drop_duplicates(["t", "horizon", "model"], keep="last")
        last = minutes.close.index[-1] + pd.Timedelta(minutes=1)
        z = z[z["end"] <= last]
        if z.empty:
            return pd.DataFrame(), pd.DataFrame()
        r2 = np.log(minutes.close).diff() ** 2
        cs = r2.fillna(0).cumsum()
        a = cs.reindex(z["t"] - pd.Timedelta(minutes=1), method="ffill").values
        b = cs.reindex(z["end"] - pd.Timedelta(minutes=1), method="ffill").values
        z["rv"] = b - a
        z = z[z["rv"] > 0]
        z["qlike"] = z["rv"] / z["var"] - np.log(z["rv"] / z["var"]) - 1
        z["mse_log"] = (np.log(z["rv"]) - np.log(z["var"])) ** 2
        rows = []
        for (hz, m), g in z.groupby(["horizon", "model"]):
            rows.append({"horizon": hz, "model": m, "n": len(g), "QLIKE": g["qlike"].mean(), "MSE log": g["mse_log"].mean(),
                         "bias log": (np.log(g["var"]) - np.log(g["rv"])).mean()})
        tab = pd.DataFrame(rows)
        # Diebold–Mariano vs the lab blend on the common sample
        dm = []
        for hz, g in z.groupby("horizon"):
            piv = g.pivot_table(index="t", columns="model", values="qlike")
            if "lab_blend" not in piv:
                continue
            for m in piv.columns:
                d = (piv["lab_blend"] - piv[m]).dropna()
                if len(d) >= 5 and m != "lab_blend":
                    se = d.std() / math.sqrt(len(d)) if d.std() > 0 else np.nan
                    dm.append({"horizon": hz, "model": m, "n common": len(d), "QLIKE gain vs lab_blend": d.mean(),
                               "t": d.mean() / se if se else np.nan})
        return tab.sort_values(["horizon", "QLIKE"]), pd.DataFrame(dm)
