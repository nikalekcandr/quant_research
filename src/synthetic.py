"""Synthetic raw data in exactly the formats the downloaders save. Used only to test the pipeline end to end
(`python run_all.py --synthetic`). The numbers it produces mean nothing about real markets.

Truth: 1-minute log returns with a two-factor stochastic volatility (slow + fast AR(1) in log variance),
an intraday/weekly pattern, jumps at US macro releases and random jumps, Student-t(5) innovations.
"Markets" observe a noisy version of the current volatility state; Deribit adds a variance premium, the
ladders compress probabilities toward 0.5 in the tails (favourite-longshot bias).
"""
from __future__ import annotations

import gzip
import io
import json
import zipfile
from datetime import timedelta

import numpy as np
import pandas as pd
from scipy import stats

import config
from src.download.kalshi import event_ticker
from src.download.polymarket import slug_candidates
from src.utils import ET, ensure, noon_et_utc, save_json

RNG = np.random.default_rng(7)
MIN_PER_YEAR = 365 * 24 * 60


def simulate_prices(start: str, end: str) -> pd.DataFrame:
    t = pd.date_range(start, end, freq="1min", tz="UTC", inclusive="left")
    n_h = int(np.ceil(len(t) / 60)) + 1
    slow, fast = np.zeros(n_h), np.zeros(n_h)
    for i in range(1, n_h):
        slow[i] = 0.9995 * slow[i - 1] + 0.035 * RNG.standard_normal()
        fast[i] = 0.96 * fast[i - 1] + 0.12 * RNG.standard_normal()
    base = np.log(0.45 ** 2 / MIN_PER_YEAR)
    logv_h = base + slow + fast
    hr = t.hour.to_numpy()
    diurnal = np.where((hr >= 13) & (hr < 21), 1.5, np.where((hr >= 0) & (hr < 6), 0.75, 1.0))
    weekend = np.where(t.dayofweek.to_numpy() >= 5, 0.6, 1.0)
    var_m = np.exp(np.repeat(logv_h, 60)[: len(t)]) * diurnal * weekend
    z = stats.t.rvs(5, size=len(t), random_state=RNG) / np.sqrt(5 / 3)
    r = np.sqrt(var_m) * z
    # jumps: macro releases and Poisson
    cal = pd.read_csv(config.MANUAL / "us_macro_calendar.csv")
    rel = pd.to_datetime(cal["date"] + " " + cal["time_et"]).dt.tz_localize("America/New_York").dt.tz_convert("UTC")
    idx = t.get_indexer(pd.DatetimeIndex(rel))
    idx = idx[idx >= 0]
    r[idx] += 0.006 * RNG.standard_normal(len(idx))
    pj = RNG.random(len(t)) < 1 / (2 * 24 * 60)
    r[pj] += 0.008 * RNG.standard_normal(pj.sum())
    price = 60000 * np.exp(np.cumsum(r))
    return pd.DataFrame({"open_time": t, "close": price, "var_m": var_m})


def write_binance(px: pd.DataFrame):
    d = ensure(config.RAW / "binance")
    for month, g in px.groupby(px["open_time"].dt.strftime("%Y-%m")):
        ot = (g["open_time"] - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta(microseconds=1)  # microseconds
        if month < "2025-01":
            ot = ot // 1000                           # milliseconds before 2025, as in the real archives
        c = g["close"].round(2)
        df = pd.DataFrame({0: ot, 1: c, 2: c, 3: c, 4: c, 5: 1.0, 6: ot + 59_999, 7: 1.0, 8: 10, 9: 0.5, 10: 0.5, 11: 0})
        buf = io.StringIO()
        df.to_csv(buf, header=False, index=False)
        with zipfile.ZipFile(d / f"BTCUSDT-1m-{month}.zip", "w", zipfile.ZIP_DEFLATED) as z:
            z.writestr(f"BTCUSDT-1m-{month}.csv", buf.getvalue())


def _state_var(px: pd.DataFrame) -> pd.Series:
    """What markets 'see': trailing 24h realized variance per minute (smoothed), indexed by minute."""
    r = np.log(px["close"]).diff().fillna(0)
    rv24 = (r ** 2).rolling(24 * 60, min_periods=60).mean()
    return pd.Series(0.5 * rv24.to_numpy() + 0.5 * px["var_m"].rolling(60, min_periods=1).mean().to_numpy(),
                     index=px["open_time"])


def write_polymarket(px: pd.DataFrame, since: str):
    ev_dir, px_dir = ensure(config.RAW / "polymarket" / "events"), ensure(config.RAW / "polymarket" / "prices")
    sv = _state_var(px)
    close = pd.Series(px["close"].to_numpy(), index=px["open_time"])
    days = pd.date_range(since, px["open_time"].iloc[-1].tz_localize(None).normalize() + timedelta(days=6), freq="D")
    for d in days:
        T = noon_et_utc(d)
        open_ts = T - pd.Timedelta(days=7)
        if open_ts < px["open_time"].iloc[0] + pd.Timedelta(days=2):
            continue
        S0 = close.asof(open_ts)
        center = round(S0 / 2000) * 2000
        strikes = center + 2000 * np.arange(-5, 6)
        grid = pd.date_range(open_ts + pd.Timedelta(minutes=5), min(T, px["open_time"].iloc[-1]), freq="10min")
        S = close.reindex(grid, method="ffill").to_numpy()
        v_rate = sv.reindex(grid, method="ffill").to_numpy() * 1.05
        noise = np.exp(0.15 * RNG.standard_normal())
        tau_min = (T - grid).total_seconds().to_numpy() / 60
        settled = T <= px["open_time"].iloc[-1]
        S_T = close.asof(T) if settled else np.nan
        markets = []
        for k in strikes:
            var = np.maximum(v_rate * noise * tau_min, 1e-10)
            p = stats.norm.cdf((np.log(S / k) - var / 2) / np.sqrt(var))
            lg = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
            p = 1 / (1 + np.exp(-0.92 * lg))              # longshot bias
            p = np.clip(np.round(p + 0.003 * RNG.standard_normal(len(p)), 3), 0.0005, 0.9995)
            tok = f"syn{d:%Y%m%d}{int(k)}"
            save_json({"history": [{"t": int(g.timestamp()), "p": float(q)} for g, q in zip(grid, p)]}, px_dir / f"{tok}.json")
            out = '["1", "0"]' if settled and S_T > k else '["0", "1"]'
            markets.append({"question": f"Will the price of Bitcoin be above ${int(k):,} on {d:%B} {d.day}?",
                            "groupItemTitle": f"{int(k):,}", "clobTokenIds": json.dumps([tok, tok + "n"]),
                            "startDate": open_ts.isoformat().replace("+00:00", "Z"), "slug": f"btc-{int(k)}",
                            "conditionId": "0x0", "volumeNum": 1000.0, "closed": bool(settled),
                            "outcomePrices": out if settled else '["0.5", "0.5"]'})
        ev = {"slug": slug_candidates(d.date())[0], "title": f"Bitcoin above ___ on {d:%B} {d.day}?",
              "endDate": T.isoformat().replace("+00:00", "Z"), "startDate": open_ts.isoformat(), "markets": markets}
        save_json(ev, ev_dir / f"{d:%Y-%m-%d}.json")


def write_kalshi(px: pd.DataFrame, since: str):
    out_dir = ensure(config.RAW / "kalshi")
    sv = _state_var(px)
    close = pd.Series(px["close"].to_numpy(), index=px["open_time"])
    last = px["open_time"].iloc[-1]
    day = pd.Timestamp(since).date()
    while pd.Timestamp(day).tz_localize("UTC") < last:
        rows = []
        for hour in range(24):
            close_et = pd.Timestamp(day.year, day.month, day.day, hour).tz_localize(ET).to_pydatetime()
            T = pd.Timestamp(close_et).tz_convert("UTC")
            O = T - pd.Timedelta(hours=1)
            if T > last:
                break
            ev = event_ticker(close_et)
            S0 = close.asof(O)
            S_T = close.asof(T) * (1 + 0.0003 + 0.0001 * RNG.standard_normal())
            strikes = np.round(S0 / 100) * 100 + 100 * np.arange(-20, 21) - 0.01
            for snap_min in (4, 5):
                ts = O + pd.Timedelta(minutes=snap_min)
                S = close.asof(ts)
                var = max(sv.asof(ts) * 1.05 * np.exp(0.25 * RNG.standard_normal()) * (60 - snap_min), 1e-10)
                p = stats.norm.cdf((np.log(S / strikes) - var / 2) / np.sqrt(var))
                lg = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
                p = 1 / (1 + np.exp(-0.95 * lg))
                bid = np.clip(np.floor((p - 0.004) * 100) / 100, 0.0, 0.99)
                ask = np.clip(bid + np.where(RNG.random(len(p)) < 0.7, 0.01, 0.02), 0.01, 1.0)
                for k, b, a in zip(strikes, bid, ask):
                    rows.append({"ticker": f"{ev}-T{k:.2f}", "end_ts": int(ts.timestamp()), "yes_bid": b, "yes_ask": a,
                                 "last": round((a + b) / 2, 2), "volume": 10.0, "open_interest": 100.0, "strike": k,
                                 "result": "yes" if S_T > k else "no", "expiration_value": round(S_T, 2),
                                 "open_time": O.isoformat().replace("+00:00", "Z"),
                                 "close_time": T.isoformat().replace("+00:00", "Z"), "volume_total": 100.0, "event": ev})
        if rows:
            pd.DataFrame(rows).to_csv(out_dir / f"{day.isoformat()}.csv.gz", index=False)
        day += timedelta(days=1)


def write_deribit(px: pd.DataFrame, since: str):
    tr_dir = ensure(config.RAW / "deribit" / "trades")
    sv = _state_var(px)
    close = pd.Series(px["close"].to_numpy(), index=px["open_time"])
    last = px["open_time"].iloc[-1]
    for d in pd.date_range(since, last.tz_localize(None).normalize(), freq="D"):
        snap = noon_et_utc(d)
        if snap > last:
            continue
        S = close.asof(snap)
        atm = np.sqrt(sv.asof(snap) * MIN_PER_YEAR * 1.2) * 100 * np.exp(0.05 * RNG.standard_normal())
        exps = [(snap.normalize() + pd.Timedelta(days=i, hours=8)) for i in range(1, 4)]
        fri = snap.normalize() + pd.Timedelta(days=(4 - snap.dayofweek) % 7 or 7, hours=8)
        exps += [fri + pd.Timedelta(days=7 * j) for j in range(3)]
        trades = []
        for e in sorted(set(exps)):
            if e <= snap:
                continue
            for _ in range(int(RNG.integers(3, 12))):
                k = S * np.exp(0.06 * RNG.standard_normal())
                kk = round(k / 1000) * 1000
                mny = np.log(kk / S)
                iv = atm * (1 + 0.02 * RNG.standard_normal()) + 8 * mny ** 2 * 100 - 3 * mny * 10
                ts = snap - pd.Timedelta(minutes=float(RNG.uniform(1, 59)))
                trades.append({"timestamp": int(ts.timestamp() * 1000), "instrument_name": f"BTC-{e:%d%b%y}".upper() + f"-{int(kk)}-{'C' if kk >= S else 'P'}",
                               "iv": round(float(iv), 2), "price": 0.01, "index_price": round(float(S), 2), "amount": 1.0})
        save_json(sorted(trades, key=lambda x: x["timestamp"]), tr_dir / f"{snap:%Y%m%d%H%M}.json")
    # DVOL: hourly
    hourly = sv.resample("1h").last().dropna()
    dv = np.sqrt(hourly.rolling(24 * 7, min_periods=1).mean() * MIN_PER_YEAR * 1.2) * 100
    df = pd.DataFrame({"ts": dv.index, "open": dv.values, "high": dv.values, "low": dv.values, "close": dv.values})
    df.to_csv(config.RAW / "deribit" / "dvol.csv.gz", index=False)


def generate(start="2024-09-01", end=None):
    end = end or pd.Timestamp.now(tz="UTC").floor("min").tz_localize(None).isoformat()
    px = simulate_prices(start, end)
    write_binance(px)
    write_polymarket(px, config.POLYMARKET_START)
    write_kalshi(px, config.KALSHI_START)
    write_deribit(px, config.DERIBIT_START)
    return px
