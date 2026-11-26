"""How close is our spot to CF BRTI, and is the wings variance calibrated? (04.11.2026)

Kalshi KXBTCD settles on the 60-s average of CF BRTI before the hour. BRTI = exponentially depth-weighted mid of the
consolidated BTC-USD order book of constituent exchanges (Coinbase, Kraken, Bitstamp, Gemini, LMAX, ...), every
200 ms (CME CF Real Time Indices Methodology). We cannot buy BRTI history; the hourly settlement values are free.

A. Replication of the settlement value from 1-minute candles of the settlement minute (close, typical price), per
   venue and averaged, raw and with the lagged 48-hour median basis (no look-ahead).
B. Proxy quality at decision time: sd of log(E_h / (S_{h-τ} · basis)). The true BRTI path is common to all proxies,
   so a lower sd = a proxy closer to BRTI.
C. Variance calibration of the wings fair value: z = log(E/(S·basis)) / sd_model, model = EWMA level of
   deseasonalised 1-minute returns × hour-of-week profile × (τ − 2/3) (as W03/W04). var(z), tails vs normal and t4.
D. The W03/W04 backtest (studies/wings_ladder.py data) with a different spot / basis and a variance scale k fitted on
   Jul–Aug, read on Sep–Oct.
    python studies/brti_proxy.py [usd_minutes dir] [lab ticks dir]
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm, t as student_t

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from studies import wings_ladder as wl  # noqa: E402
wl.LAD, wl.MIN = Path("data/raw/kalshi_kxbtcd_60m"), Path("data/interim/minutes.parquet")   # wl reads sys.argv

USD = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/raw/usd_minutes")
TICKS = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("data/lab/ticks")
OUT = Path("results/studies")
SPLIT = wl.SPLIT
TAUS = [1, 2, 5, 10, 15, 20, 30]


def settlements():
    k = pd.read_parquet("data/interim/kalshi.parquet", columns=["event", "settle_ts", "expiration_value"])
    e = k.drop_duplicates("event").set_index("settle_ts")["expiration_value"].astype(float)
    f = USD / "kx_settle.parquet"
    if f.exists():
        r = pd.read_parquet(f).dropna(subset=["expiration_value"])
        r = r.set_index(pd.to_datetime(r.close_time, utc=True))["expiration_value"].astype(float)
        e = pd.concat([e, r[~r.index.isin(e.index)]])
    e = e[e > 1000].sort_index()
    return e[~e.index.duplicated()]


def candles(name):
    """close-time-indexed close and typical price ((o+h+l+c)/4) of each 1-minute candle."""
    f = USD / f"{name}.parquet"
    if not f.exists():
        return None
    d = pd.read_parquet(f)
    d["ts"] = pd.to_datetime(d.ts, utc=True) + pd.Timedelta(minutes=1)
    d = d.drop_duplicates("ts").set_index("ts").sort_index()
    return pd.DataFrame({"c": d.c, "typ": (d.o + d.h + d.l + d.c) / 4, "v": d.v})


def binance():
    m = pd.read_parquet("data/interim/minutes.parquet", columns=["open_time", "close"])
    m = m.set_index(pd.to_datetime(m.open_time, utc=True) + pd.Timedelta(minutes=1))["close"]
    return m[~m.index.duplicated()].sort_index()


def lagged_basis(E, S, n=48):
    """median of E/S over the previous n settlements (strictly before this hour)."""
    r = (E / S.reindex(E.index)).dropna()
    return r.shift(1).rolling(n, min_periods=5).median().reindex(E.index)


def bp(x):
    return 1e4 * x


def summary(err):
    err = err.dropna()
    return {"n": len(err), "bias bp": bp(err.mean()), "sd bp": bp(err.std()), "MAE bp": bp(err.abs().mean()),
            "p95 |e| bp": bp(err.abs().quantile(0.95))}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    E = settlements()
    bn = binance()
    venues = {"binance (USDT)": pd.DataFrame({"c": bn, "typ": bn})}
    for nm, lab in (("cb", "coinbase"), ("bs", "bitstamp"), ("kr", "kraken")):
        d = candles(nm)
        if d is not None and len(d) > 10000:      # Kraken's API keeps 720 minutes only: too short to use here
            venues[lab] = d
    if "coinbase" in venues and "bitstamp" in venues:
        cb, bs = venues["coinbase"], venues["bitstamp"]
        j = cb.join(bs, lsuffix="_cb", rsuffix="_bs", how="inner")
        venues["mean cb+bs"] = pd.DataFrame({"c": (j.c_cb + j.c_bs) / 2, "typ": (j.typ_cb + j.typ_bs) / 2})
        w = j.v_cb / (j.v_cb + j.v_bs)
        venues["vol-wtd cb+bs"] = pd.DataFrame({"c": w * j.c_cb + (1 - w) * j.c_bs,
                                                "typ": w * j.typ_cb + (1 - w) * j.typ_bs})
    lines = ["# Our spot vs CF BRTI and the wings variance (studies/brti_proxy.py)", "",
             f"Settlements: {len(E)} hours, {E.index.min():%Y-%m-%d} .. {E.index.max():%Y-%m-%d}. "
             f"Venues with minutes: {', '.join(venues)}.", ""]

    # ---------------------------------------------------------------- A: the settlement minute
    rows = []
    for nm, d in venues.items():
        for col in ("c", "typ"):
            S = d[col]
            common = E.index[E.index.isin(S.index)]
            if len(common) < 50:
                continue
            raw = np.log(E[common] / S[common])
            b = lagged_basis(E[common], S)
            adj = np.log(E[common] / (S[common] * b))
            rows.append({"proxy": nm, "price": "close" if col == "c" else "typical (o+h+l+c)/4",
                         **{f"raw {k}": v for k, v in summary(raw).items() if k != "n"},
                         **{f"basis-adj {k}": v for k, v in summary(adj).items()}})
    A = pd.DataFrame(rows)
    A.to_csv(OUT / "brti_proxy_A.csv", index=False)
    lines += ["## A. Settlement value from the settlement minute's candle", "",
              A.round(2).to_markdown(index=False), ""]

    # ---------------------------------------------------------------- B: decision time
    rows = []
    for nm, d in venues.items():
        S = d["c"]
        b = lagged_basis(E, S)
        rec = {"proxy": nm}
        for tau in TAUS:
            St = S.reindex(E.index - pd.Timedelta(minutes=tau)).values
            e = pd.Series(np.log(E.values / (St * b.values)), index=E.index)
            rec[f"sd τ={tau}"] = bp(e.std())
            rec[f"n τ={tau}"] = e.notna().sum()
        rows.append(rec)
    B = pd.DataFrame(rows)
    B.to_csv(OUT / "brti_proxy_B.csv", index=False)
    lines += ["## B. sd of log(E / (S_{h−τ} · basis)), bp — lower = closer to BRTI (the BRTI path is common)", "",
              B[["proxy"] + [f"sd τ={t}" for t in TAUS]].round(2).to_markdown(index=False), ""]
    # same sample for all proxies, to be fair
    if len(venues) > 1:
        idx = E.index
        for nm, d in venues.items():
            idx = idx[idx.isin(d.index)]
        rows = []
        for nm, d in venues.items():
            S, b = d["c"], lagged_basis(E, d["c"])
            rec = {"proxy": nm, "hours": len(idx)}
            for tau in TAUS:
                St = S.reindex(idx - pd.Timedelta(minutes=tau)).values
                rec[f"sd τ={tau}"] = bp(np.nanstd(np.log(E[idx].values / (St * b[idx].values)), ddof=1))
            rows.append(rec)
        lines += ["Same hours for every proxy:", "", pd.DataFrame(rows).round(2).to_markdown(index=False), ""]

    # ---------------------------------------------------------------- C: variance calibration
    rows = []
    zs = {}
    for nm, d in venues.items():
        S = d["c"]
        vm = wl.variance_level(S[S.index >= S.index.min()])
        b = lagged_basis(E, S)
        for tau in (5, 10, 15, 20, 30):
            t0 = E.index - pd.Timedelta(minutes=tau)
            var = vm["lvl"].reindex(t0).values * vm["sea"].reindex(t0).values * max(tau - 2 / 3, 1 / 3)
            z = np.log(E.values / (S.reindex(t0).values * b.values)) / np.sqrt(var)
            z = pd.Series(z, index=E.index).dropna()
            z = z[np.isfinite(z)]
            zs[(nm, tau)] = z
            for half, zz in (("Dec–Aug", z[z.index < SPLIT]), ("Sep–Nov", z[z.index >= SPLIT])):
                rows.append({"variance from": nm, "τ min": tau, "half": half, "n": len(zz),
                             "var(z)": zz.var(), "robust var(z)": (zz.abs().median() / 0.6745) ** 2,
                             "P|z|>2 %": 100 * (zz.abs() > 2).mean(), "P|z|>3 %": 100 * (zz.abs() > 3).mean()})
    C = pd.DataFrame(rows)
    C.to_csv(OUT / "brti_proxy_C.csv", index=False)
    lines += ["## C. Calibration of the wings variance: z = log(E/(S·basis)) / sd_model", "",
              "Normal: P|z|>2 = 4.55%, P|z|>3 = 0.27%. Student t4 with unit variance: 3.7% and 1.2%. var(z) > 1 = the "
              "model's variance too low.", "", C.round(3).to_markdown(index=False), ""]

    # ---------------------------------------------------------------- D: wings backtest
    lines += reliability(venues, E)
    lines += wings(venues, E)
    # ---------------------------------------------------------------- live proxy of the lab
    lines += live_check(E)
    (OUT / "brti_proxy.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


def wings_frame(S, E, k_var=1.0, lvl_from=None):
    """wings_ladder.load() with spot S (close-time minutes), basis = lagged median vs E, variance × k."""
    d = pd.concat([pd.read_csv(f) for f in sorted(wl.LAD.glob("*.csv*"))], ignore_index=True)
    d = d[(d.strike_type == "greater") & (d.yes_bid > 0) & (d.yes_ask < 1) & (d.yes_ask - d.yes_bid <= 0.03)]
    d["ts"] = pd.to_datetime(d.end_ts, unit="s", utc=True)
    d["T"] = pd.to_datetime(d.close_time, utc=True)
    d["left"] = (d["T"] - d["ts"]).dt.total_seconds() / 60
    d["y"] = (d.result == "yes").astype(float)
    src = lvl_from if lvl_from is not None else S
    vm = wl.variance_level(src)
    d["S"] = S.reindex(d.ts).values
    d["lvl"] = vm["lvl"].reindex(d.ts).values
    d["sea"] = vm["sea"].reindex(d.ts).values
    b = lagged_basis(E, S)
    d["basis"] = d["T"].map(b)
    d = d.dropna(subset=["S", "lvl", "basis"])
    var = k_var * d.lvl * d.sea * np.maximum(d.left - 2 / 3, 1 / 3)
    x = np.log(d.S * d.basis / d.strike) / np.sqrt(var)
    d["p_n"], d["p_t"] = norm.cdf(x), student_t.cdf(x * math.sqrt(2), 4)
    cy, cn = d.yes_bid.between(0.005, 0.20), (1 - d.yes_ask).between(0.005, 0.20)
    a = d[cy].assign(cheap="yes", c_bid=d.yes_bid, c_fair_n=d.p_n, c_fair_t=d.p_t, c_won=d.y)
    bb = d[cn].assign(cheap="no", c_bid=1 - d.yes_ask, c_fair_n=1 - d.p_n, c_fair_t=1 - d.p_t, c_won=1 - d.y)
    c = pd.concat([a, bb], ignore_index=True)
    c["px"] = 1 - c.c_bid
    c["pnl"] = c.c_bid - c.c_won - wl.fee(c.px)
    return c


RULES = {"W03": (0.10, 0.20, 5, 15, "normal", 0.01, 1), "W04": (0.06, 0.10, 15, 30, "t4", 0.0, 1)}


def book(c):
    out = {}
    gs = {h: wl.run(c, *p) for h, p in RULES.items()}
    for h, g in gs.items():
        for half, gg in (("Jul–Aug", g[g["T"] < SPLIT]), ("Sep–Oct", g[g["T"] >= SPLIT])):
            s = wl.stats(gg)
            out[f"{h} {half} ¢/tr"] = s.get("¢/trade", np.nan)
            out[f"{h} {half} n"] = s.get("trades", 0)
    both = pd.concat(gs.values())
    for half, gg in (("Jul–Aug", both[both["T"] < SPLIT]), ("Sep–Oct", both[both["T"] >= SPLIT])):
        s = wl.stats(gg)
        out[f"book {half} $/h"] = s.get("$/hour", np.nan)
        out[f"book {half} t"] = s.get("t (hours)", np.nan)
    return out


def wings(venues, E):
    rows = []
    cands = {"binance (USDT)": venues["binance (USDT)"]["c"]}
    for nm in ("coinbase", "mean cb+bs"):
        if nm in venues:
            cands[nm] = venues[nm]["c"]
    for nm, S in cands.items():
        for k in (0.8, 0.9, 1.0, 1.1, 1.25, 1.5):
            c = wings_frame(S, E, k)
            rows.append({"spot": nm, "var × k": k, **book(c)})
    D = pd.DataFrame(rows)
    D.to_csv(OUT / "brti_proxy_D.csv", index=False)
    cols = ["spot", "var × k", "W03 Jul–Aug ¢/tr", "W03 Sep–Oct ¢/tr", "W04 Jul–Aug ¢/tr", "W04 Sep–Oct ¢/tr",
            "book Jul–Aug $/h", "book Jul–Aug t", "book Sep–Oct $/h", "book Sep–Oct t", "W03 Sep–Oct n", "W04 Sep–Oct n"]
    return ["## D. W03/W04 backtest (1 contract, taker) with another spot and variance × k",
            "k is chosen on Jul–Aug only; Sep–Oct is the test. Binance × basis, k = 1 is the frozen rule.", "",
            D[cols].round(3).to_markdown(index=False), ""]


def reliability(venues, E):
    """Does the model's probability of the cheap side match how often it pays? One row per strike and hour (first
    minute in the window), so a strike is not counted 10 times. Cluster: hours."""
    out = ["## C2. Reliability of the cheap-side fair value (the number the wings rules trade on)",
           "First minute of each strike in the window; buckets of model fair; won = share that paid. "
           "fair > won = the model overprices the wing (we sell less than we should); fair < won = underprices "
           "(we sell wings that pay — the dangerous side).", ""]
    cands = {"binance (USDT)": venues["binance (USDT)"]["c"]}
    for nm in ("coinbase", "mean cb+bs"):
        if nm in venues:
            cands[nm] = venues[nm]["c"]
    rows = []
    for nm, S in cands.items():
        c = wings_frame(S, E)
        for (l0, l1) in ((5, 15), (15, 30)):
            g = c[c.left.between(l0, l1)].sort_values("ts").drop_duplicates(["T", "ticker"])
            for col, model in (("c_fair_n", "normal"), ("c_fair_t", "t4")):
                g = g.assign(bk=pd.cut(g[col], [0, 0.03, 0.06, 0.10, 0.15, 0.20, 0.30]))
                for bk, h in g.groupby("bk", observed=True):
                    if len(h) < 30:
                        continue
                    hr = h.groupby("T").agg(w=("c_won", "sum"), f=(col, "sum"), n=(col, "size"))
                    d = (hr.w - hr.f)
                    se = d.std(ddof=1) * math.sqrt(len(hr)) / hr.n.sum() if len(hr) > 2 else np.nan
                    rows.append({"spot": nm, "min left": f"{l0}–{l1}", "model": model, "fair bucket": str(bk),
                                 "n": len(h), "hours": len(hr), "mean fair %": 100 * h[col].mean(),
                                 "won %": 100 * h.c_won.mean(), "won−fair pp": 100 * (h.c_won.mean() - h[col].mean()),
                                 "se pp": 100 * se, "mean bid %": 100 * h.c_bid.mean()})
    R = pd.DataFrame(rows)
    R.to_csv(OUT / "brti_proxy_C2.csv", index=False)
    return out + [R.round(2).to_markdown(index=False), ""]


def live_check(E):
    """The live lab prices wings with usd_mid = (Coinbase mid + Kraken mid)/2 and NO basis (lab/feeds.py); the
    backtest used Binance × the 48-hour basis. How far is the live proxy from the settlement?"""
    fs = sorted(TICKS.glob("*.csv*")) if TICKS.exists() else []
    if not fs:
        return ["## E. Live lab proxy: no lab ticks found", ""]
    t = pd.concat([pd.read_csv(f, usecols=lambda c: c in ("ts", "usd_mid", "cb_mid", "kr_mid", "bn_mid"))
                   for f in fs], ignore_index=True)
    t["ts"] = pd.to_datetime(t.ts, utc=True, format="mixed")
    t = t.set_index("ts").sort_index()
    t = t[~t.index.duplicated()]
    rows = []
    for col in [c for c in ("usd_mid", "cb_mid", "kr_mid", "bn_mid") if c in t]:
        s = t[col].dropna()
        for tau in (0.5, 5, 15, 30):
            tt = E.index - pd.Timedelta(minutes=tau)
            pos = s.index.searchsorted(tt, side="right") - 1
            ok = pos >= 0
            St = pd.Series(np.nan, index=E.index)
            St[ok] = s.values[pos[ok]]
            age = pd.Series(np.nan, index=E.index)
            age[ok] = (tt[ok] - s.index[pos[ok]]).total_seconds()
            St[age > 120] = np.nan
            e = np.log(E / St)
            b = lagged_basis(E, St)
            ea = np.log(E / (St * b))
            rows.append({"lab column": col, "τ min": tau, **{f"raw {k}": v for k, v in summary(e).items()},
                         **{f"basis-adj {k}": v for k, v in summary(ea).items() if k != "n"}})
    L = pd.DataFrame(rows)
    return ["## E. Live lab spot vs the settlement (lab ticks)",
            "usd_mid = (Coinbase mid + Kraken mid)/2 is what W03/W04 use live, without a basis correction.", "",
            L.round(2).to_markdown(index=False), ""]


if __name__ == "__main__":
    main()
