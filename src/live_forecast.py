"""Live (true out-of-sample) forecasts, written and committed to git before the outcome is known.

Called by live.py every 5 minutes. A forecast is due:
  pm24h, pm6d  at 12:00 ET each day (t), for the Polymarket ladders settling 24 h and 144 h later
  kx1h         at HH:05 each hour (t), for the Kalshi ladder settling at HH+1:00

For each due occasion the script
  1. fetches the latest Binance minutes, DVOL, the Deribit chain and the ladder's order book at t,
  2. builds the feature row with exactly the same code as the historical panel,
  3. estimates every model on the historical panel (occasions whose window ended before t) and predicts,
  4. converts the forecasts into probabilities for every strike of the ladder,
  5. writes forecasts_live/<horizon>/<t>.json and commits it to git (pushes too if a remote exists).
The git commit time, and the push to GitHub, prove the forecast existed before settlement.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

import config
from src import alpha, build, evaluate, models
from src.download import binance, deribit, kalshi, polymarket
from src.measure import deribit_iv, ladder
from src.measure.realized import Minutes, seasonal_profiles
from src.utils import Http, ensure, log, ms, noon_et_utc, read_table, utcnow

_cache: dict = {}

# A forecast made this many minutes after its t is still written (for the record) but flagged late and left out of
# the live evaluation: part of its target window, and the market's reaction to it, were already known when it was made.
MAX_LATE_MIN = {"kx1h": 2.0, "pm24h": 10.0, "pm6d": 10.0}

# Strategies fixed on 15 October 2026, before any live outcome was known (alpha search, results/tables/alpha_*.csv).
# They are the only ones whose live profit counts as a test; anything else run on live data is exploration.
# Both are the volatility-only bet on Kalshi's hourly ladder (the only family that came close: t = 2.4-2.5,
# SPA p = 0.11); no Polymarket variant survived once prices observed after t were used.
PREREGISTERED = [("kx1h", "hyb_harx_d", 0.05), ("kx1h", "hyb_harx_all", 0.05)]


def due_occasions(now: pd.Timestamp) -> list[tuple[str, pd.Timestamp, pd.Timestamp]]:
    out = []
    noon = noon_et_utc(now.tz_convert("America/New_York").date())
    if noon <= now < noon + pd.Timedelta(minutes=30):
        out.append(("pm24h", noon, noon + pd.Timedelta(hours=24)))
        out.append(("pm6d", noon, noon + pd.Timedelta(hours=144)))
    t_kx = now.floor("h") + pd.Timedelta(minutes=5)
    if t_kx <= now < t_kx + pd.Timedelta(minutes=5):
        out.append(("kx1h", t_kx, now.floor("h") + pd.Timedelta(hours=1)))
    return out


def _path(h, t):
    return config.FORECASTS_LIVE / h / f"{t:%Y%m%dT%H%M}.json"


def run_due(now):
    now = pd.Timestamp(now)
    for h, t, T in due_occasions(now):
        if _path(h, t).exists():
            continue
        try:
            make_forecast(h, t, T)
        except FileNotFoundError as e:
            log(f"live forecast {h} skipped: run `python run_all.py` first ({e})")
        except Exception as e:
            log(f"live forecast {h} t={t:%H:%M} failed: {e}")


# ------------------------------------------------------------------ data at t
def _minutes(http: Http) -> Minutes:
    hist = _cache.get("minutes_hist")
    if hist is None:
        hist = read_table(config.INTERIM / "minutes")[["open_time", "close"]]
        _cache["minutes_hist"] = hist
    last = hist["open_time"].max()
    rows, start = [], ms(last + pd.Timedelta(minutes=1))
    while True:
        res = binance.api_get(http, "/api/v3/klines", {"symbol": "BTCUSDT", "interval": "1m", "startTime": start, "limit": 1000})
        if not res:
            break
        rows.extend(res)
        if len(res) < 1000:
            break
        start = res[-1][0] + 60_000
    if rows:
        new = pd.DataFrame(rows, columns=binance.COLS)
        new["open_time"] = binance._to_ts(new["open_time"])
        new["close"] = new["close"].astype(float)
        # drop the candle that is still open
        new = new[new["open_time"] + pd.Timedelta(minutes=1) <= pd.Timestamp(utcnow())]
        hist = pd.concat([hist, new[["open_time", "close"]]]).drop_duplicates("open_time").sort_values("open_time")
        _cache["minutes_hist"] = hist
    full = pd.date_range(hist["open_time"].iloc[0], hist["open_time"].iloc[-1], freq="1min", tz="UTC")
    df = hist.set_index("open_time").reindex(full).ffill().rename_axis("open_time").reset_index()
    return Minutes(df)


def _dvol(http: Http) -> pd.DataFrame:
    now = pd.Timestamp(utcnow())
    res = http.get(f"{deribit.LIVE}/get_volatility_index_data",
                   {"currency": "BTC", "start_timestamp": ms(now - pd.Timedelta(hours=12)), "end_timestamp": ms(now),
                    "resolution": 3600})
    df = pd.DataFrame(((res or {}).get("result") or {}).get("data", []), columns=["ts", "open", "high", "low", "close"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True).dt.as_unit("ns")
    return df


def _ladder_now(http: Http, h: str, T: pd.Timestamp) -> pd.DataFrame:
    if h == "kx1h":
        ev = kalshi.event_ticker(T.tz_convert("America/New_York").to_pydatetime())
        res = http.get(f"{kalshi.BASE}/markets", {"event_ticker": ev, "limit": 1000}) or {}
        rows = [{"strike": m.get("floor_strike"), "bid": kalshi._num(m, "yes_bid"), "ask": kalshi._num(m, "yes_ask"),
                 "ticker": m.get("ticker")} for m in res.get("markets", [])]
    else:
        ev = polymarket.fetch_event(http, T.tz_convert("America/New_York").date())
        if ev is None:
            return pd.DataFrame()
        lad = polymarket.parse_event(ev)
        bk = polymarket.books(http, list(lad["yes_token"]))
        rows = []
        for _, mrow in lad.iterrows():
            b = bk.get(mrow["yes_token"], {})
            bid = polymarket._top(b.get("bids"), "max")[0]
            ask = polymarket._top(b.get("asks"), "min")[0]
            rows.append({"strike": mrow["strike"], "bid": bid, "ask": ask, "ticker": mrow["yes_token"]})
    df = pd.DataFrame(rows).dropna(subset=["strike"])
    if df.empty:
        return df
    df = df[(df["bid"].fillna(0) > 0) & (df["ask"].fillna(1) < 1) & (df["ask"] > df["bid"])].copy()
    df["p"] = (df["bid"] + df["ask"]) / 2
    df["spread"] = df["ask"] - df["bid"]
    if h == "kx1h" and len(df):
        df = build.economical(df)   # same strike design as the history the models were trained on
    return df.sort_values("strike").reset_index(drop=True)


# ------------------------------------------------------------------ forecast
def make_forecast(h: str, t: pd.Timestamp, T: pd.Timestamp):
    http = Http(min_interval=0.05)
    created = pd.Timestamp(utcnow())
    m = _minutes(http)
    if "profiles" not in _cache or _cache.get("profiles_month") != t.month:
        _cache["profiles"] = seasonal_profiles(m)
        _cache["profiles_month"] = t.month
    occ = pd.DataFrame({"T": [T], "t": [t]})
    occ["e"] = occ["T"] + (pd.Timedelta(minutes=1) if h != "kx1h" else pd.Timedelta(0))
    occ["horizon"] = h
    occ["tau_h"] = (occ["e"] - occ["t"]).dt.total_seconds() / 3600
    occ = build.norm_dt(occ)
    hz = {h: occ["tau_h"].iloc[0]}
    try:
        div = pd.DataFrame([{"snap_ts": t, **deribit_iv.from_chain(deribit.snapshot_chain(http, t), t, hz)}]) if h != "kx1h" else None
    except Exception:
        div = None
    lad = _ladder_now(http, h, T)
    snaps = pd.DataFrame()
    if len(lad) >= config.MIN_STRIKES:
        snaps = pd.DataFrame([{"horizon": h, "T": T, "t": t, "n_strikes": len(lad), **ladder.fit_snapshot(lad)}])
    row = pd.concat([occ, build.realized_features(occ, m, _cache["profiles"]), build.calendar_features(occ),
                     build.implied_features(occ, _dvol(http), div), build.market_features(occ, snaps)], axis=1)
    row["y"] = np.nan
    row["rv"] = np.nan
    panel = read_table(config.INTERIM / f"panel_{h}")
    panel = panel[panel["t"] < t]
    p2 = pd.concat([panel, row], ignore_index=True)
    # gradient-boosted trees are refitted on the whole history at every call: too slow for an hourly forecast
    names = [k for k in models.specs(h) if k != "gbm_all"]
    fc, _ = models.run_all_models(p2, m if h != "kx1h" else None, models=names)
    fc = fc[fc["t"] == t]
    late_min = (created - t).total_seconds() / 60
    out = {"horizon": h, "t": str(t), "T": str(T), "created_utc": str(created),
           "late_min": round(late_min, 2), "late": bool(late_min > MAX_LATE_MIN[h]),
           "features": {k: (None if pd.isna(v) else float(v)) for k, v in row.iloc[0].items()
                        if isinstance(v, (int, float, np.floating)) and k not in ("y", "rv")},
           "forecasts": {}, "ladder": []}
    for _, r in fc.iterrows():
        out["forecasts"][r["model"]] = {k: (None if pd.isna(r[k]) else float(r[k]))
                                        for k in ["m1", "s1", "w1", "m2", "s2", "w2", "m3", "s3", "w3", "mean_log", "level"]}
    if len(lad):
        S = float(row["S_t"].iloc[0])
        if h == "kx1h":
            S *= _kalshi_basis()
        for mdl in ("harx", "harx_d", "harx_all"):
            f = fc[fc["model"] == mdl]
            if f.empty:
                continue
            mm = np.array([[f[f"m{j}"].iloc[0] if pd.notna(f[f"m{j}"].iloc[0]) else 0 for j in (1, 2, 3)]])
            ss = np.array([[f[f"s{j}"].iloc[0] if pd.notna(f[f"s{j}"].iloc[0]) else 1 for j in (1, 2, 3)]])
            ww = np.array([[f[f"w{j}"].iloc[0] if pd.notna(f[f"w{j}"].iloc[0]) else 0 for j in (1, 2, 3)]])
            lad[f"p_{mdl}"] = evaluate.digital_prob(np.full(len(lad), S), lad["strike"].to_numpy(float),
                                                    np.repeat(mm, len(lad), 0), np.repeat(ss, len(lad), 0),
                                                    np.repeat(ww, len(lad), 0))
            # the 'hybrid' vol-only probability: the ladder's own median as the centre, the model's width
            center = float(snaps["mkt_m"].iloc[0]) if len(snaps) and pd.notna(snaps["mkt_m"].iloc[0]) else np.nan
            if mdl in ("harx_d", "harx_all") and np.isfinite(center):
                lad[f"p_hyb_{mdl}"] = alpha.mix_digital(np.full(len(lad), center), lad["strike"].to_numpy(float),
                                                        np.repeat(mm, len(lad), 0), np.repeat(ss, len(lad), 0),
                                                        np.repeat(ww, len(lad), 0), False)
        out["ladder"] = json.loads(lad.to_json(orient="records"))
        out["S_ref"] = S
    path = _path(h, t)
    ensure(path.parent)
    path.write_text(json.dumps(out, indent=1), encoding="utf-8")
    _ledger_append(path, h, t, T, created)
    best = out["forecasts"].get("harx_all") or out["forecasts"].get("harx_d") or out["forecasts"].get("harx") or {}
    vol = np.sqrt(best.get("level", np.nan) * 365 * 24 / out["features"].get("tau_h", 24)) * 100 if best else np.nan
    log(f"live forecast {h} t={t:%Y-%m-%d %H:%M}: {len(out['forecasts'])} models, {len(out['ladder'])} strikes, "
        f"expected vol {vol:.0f}% annualized" + (f" — LATE by {late_min:.1f} min, excluded from evaluation"
                                                 if out["late"] else ""))
    _git_commit([path, LEDGER()], f"live forecast {h} t={t:%Y-%m-%d %H:%M} UTC")


def _kalshi_basis() -> float:
    try:
        kx = read_table(config.INTERIM / "kalshi")
        mins = _cache["minutes_hist"].set_index("open_time")["close"]
        ev = kx.groupby("settle_ts")["expiration_value"].first().dropna().tail(48)
        px = mins.reindex(ev.index - pd.Timedelta(minutes=1))
        return float(np.nanmedian(ev.to_numpy(float) / px.to_numpy(float)))
    except Exception:
        return 1.0


# ------------------------------------------------------------------ tamper-evident ledger
def LEDGER():
    return config.FORECASTS_LIVE / "ledger.csv"


def _sha256(path) -> str:
    import hashlib
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _ledger_append(path, h, t, T, created):
    """Append the forecast file's SHA-256 to a hash chain: chain_n = sha256(chain_{n-1} + file_hash_n).
    Changing or back-dating any earlier forecast changes every later chain value, so one chain value sent to
    someone (or committed to GitHub) on a given day fixes all forecasts made up to that day."""
    import hashlib
    led = LEDGER()
    ensure(led.parent)
    prev = "0" * 64
    if led.exists():
        last = led.read_text(encoding="utf-8").strip().splitlines()
        if len(last) > 1:
            prev = last[-1].split(",")[-1]
    fh = _sha256(path)
    chain = hashlib.sha256((prev + fh).encode()).hexdigest()
    new = not led.exists()
    with open(led, "a", encoding="utf-8") as f:
        if new:
            f.write("written_utc,horizon,t,T,file,file_sha256,chain\n")
        rel = Path(path).relative_to(config.FORECASTS_LIVE).as_posix()
        f.write(f"{pd.Timestamp(utcnow()):%Y-%m-%d %H:%M:%S},{h},{t},{T},{rel},{fh},{chain}\n")


def ledger_head() -> dict | None:
    led = LEDGER()
    if not led.exists():
        return None
    lines = led.read_text(encoding="utf-8").strip().splitlines()
    if len(lines) < 2:
        return None
    return {"n": len(lines) - 1, "chain": lines[-1].split(",")[-1], "last": lines[-1].split(",")[0]}


def verify_ledger() -> tuple[bool, str]:
    """Recompute every file hash and the chain."""
    import hashlib
    led = LEDGER()
    if not led.exists():
        return True, "no ledger yet"
    df = pd.read_csv(led)
    prev = "0" * 64
    for r in df.itertuples():
        f = config.FORECASTS_LIVE / r.file
        if not f.exists():
            return False, f"missing {r.file}"
        fh = _sha256(f)
        if fh != r.file_sha256:
            return False, f"{r.file} changed after it was written"
        prev = hashlib.sha256((prev + fh).encode()).hexdigest()
        if prev != r.chain:
            return False, f"chain broken at {r.file}"
    return True, f"{len(df)} forecasts, chain intact"


def _git_commit(paths, msg: str):
    root = config.ROOT
    if not (root / ".git").exists():
        return
    paths = paths if isinstance(paths, (list, tuple)) else [paths]
    try:
        subprocess.run(["git", "add", *[str(p) for p in paths]], cwd=root, check=True, capture_output=True)
        subprocess.run(["git", "commit", "-m", msg], cwd=root, check=True, capture_output=True)
        remotes = subprocess.run(["git", "remote"], cwd=root, capture_output=True, text=True).stdout.strip()
        if remotes:
            subprocess.run(["git", "push", "--quiet"], cwd=root, capture_output=True, timeout=60)
    except Exception as e:
        log(f"git commit failed: {e!r}")


# ------------------------------------------------------------------ evaluation of live forecasts
def evaluate_live() -> pd.DataFrame:
    """Score every committed live forecast whose window has ended. Uses the latest data/interim/minutes."""
    files = sorted(config.FORECASTS_LIVE.glob("*/*.json"))
    if not files:
        return pd.DataFrame()
    m = Minutes(read_table(config.INTERIM / "minutes"))
    rows, lad_rows = [], []
    n_late = 0
    for f in files:
        d = json.loads(f.read_text(encoding="utf-8"))
        t, T = pd.Timestamp(d["t"]), pd.Timestamp(d["T"])
        # files written before the late flag existed: recompute it from the creation time
        late = d.get("late")
        if late is None:
            late = (pd.Timestamp(d["created_utc"]) - t).total_seconds() / 60 > MAX_LATE_MIN[d["horizon"]]
        if late:
            n_late += 1
            continue
        e = T + (pd.Timedelta(minutes=1) if d["horizon"] != "kx1h" else pd.Timedelta(0))
        if e > m.last:
            continue
        w = m.window(pd.DatetimeIndex([t]), pd.DatetimeIndex([e]))
        rv = float(w["rv"].iloc[0])
        S_e = float(m.price_at(pd.DatetimeIndex([e]))[0])
        for mdl, fcd in d["forecasts"].items():
            rows.append({"horizon": d["horizon"], "t": t, "T": T, "e": e, "model": mdl, "rv": rv, "y": np.log(rv), **fcd})
        for s in d.get("ladder", []):
            lad_rows.append({"horizon": d["horizon"], "t": t, "T": T, **s, "S_e": S_e,
                             "outcome": float(S_e > s["strike"]) if d["horizon"] != "kx1h" else np.nan})
    if n_late:
        log(f"live evaluation: {n_late} late forecasts left out (see MAX_LATE_MIN)")
    fc = pd.DataFrame(rows)
    if fc.empty:
        return fc
    if lad_rows:
        lr = pd.DataFrame(lad_rows)
        try:  # Kalshi outcomes come from the Kalshi download (BRTI settlement), not from Binance
            kx = read_table(config.INTERIM / "kalshi")[["settle_ts", "strike", "outcome"]].drop_duplicates(["settle_ts", "strike"])
            kx = kx.rename(columns={"settle_ts": "T", "outcome": "outcome_kx"})
            lr["T"] = pd.to_datetime(lr["T"], utc=True).dt.as_unit("ns")
            lr = lr.merge(kx, on=["T", "strike"], how="left")
            lr["outcome"] = lr["outcome"].fillna(lr["outcome_kx"])
        except FileNotFoundError:
            pass
        lad_rows = lr.to_dict("records")
    if lad_rows:
        _live_trading(pd.DataFrame(lad_rows))
    fc["t"] = pd.to_datetime(fc["t"], utc=True)
    sc = evaluate.scores(fc)
    tab = evaluate.summary_table(sc)
    ensure(config.TABLES)
    sc.to_csv(config.TABLES / "live_scores_rows.csv", index=False)
    fc.to_csv(config.TABLES / "live_forecasts_rows.csv", index=False)
    tab.to_csv(config.TABLES / "live_forecasts_scores.csv", index=False)
    if lad_rows:
        pd.DataFrame(lad_rows).to_csv(config.TABLES / "live_ladder_records.csv", index=False)
    return tab


def _live_trading(lr: pd.DataFrame):
    """Profit of the pre-registered strategies on live forecasts, at the real bid and ask of the live book."""
    rows = []
    for h, src, th in PREREGISTERED:
        d = lr[(lr["horizon"] == h)].copy()
        col = f"p_{src}"
        if d.empty or col not in d:
            continue
        d = d.dropna(subset=[col, "bid", "ask", "outcome"])
        fee = config.KALSHI_FEE_RATE if h == "kx1h" else 0.0
        buy = d[col] - d["ask"] > th
        sell = (d["bid"] - d[col] > th) & ~buy
        pnl = np.where(buy, d["outcome"] - d["ask"] - fee * d["ask"] * (1 - d["ask"]), 0.0) + \
            np.where(sell, d["bid"] - d["outcome"] - fee * d["bid"] * (1 - d["bid"]), 0.0)
        d["pnl"] = pnl
        tr = d[buy | sell]
        per_snap = tr.groupby("T")["pnl"].sum()
        mean, se, n = evaluate.hac_mean(per_snap.to_numpy(float), evaluate.nw_lag(len(per_snap), evaluate.overlap(h)))
        rows.append({"horizon": h, "source": src, "threshold": th, "n_snapshots_live": d["T"].nunique(),
                     "n_trades": len(tr), "n_snapshots_traded": len(per_snap),
                     "pnl_per_contract": tr["pnl"].mean() if len(tr) else np.nan, "pnl_total": tr["pnl"].sum(),
                     "pnl_per_snapshot": mean, "t_stat": mean / se if se and np.isfinite(se) else np.nan,
                     "median_spread": float((d["ask"] - d["bid"]).median()) if len(d) else np.nan})
    if rows:
        pd.DataFrame(rows).to_csv(config.TABLES / "live_trading_preregistered.csv", index=False)
