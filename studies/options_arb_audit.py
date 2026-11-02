"""Audit and strict replay of the lab's option "arbitrages" O01 (same option across venues) and O02 (put–call parity
against the perpetual) on the logger's option snapshots, 28.10–02.11.2026.

Why an audit: an arbitrage that shows up in a paper book is first of all a hypothesis about the data. Three earlier
"alphas" in this project were timing artifacts (AGENTS.md rule 7). Here every candidate is re-priced under
increasingly strict assumptions and decomposed into its sources:

  A. as the lab did it       option snapshot (taken at :x0) vs the perpetual and the spot index 3 minutes later,
                              guard = BTC moved ≤ 5 bp since the snapshot
  B. synchronous             perp quote and spot index from the tick 1 s after the snapshot (no 3-minute gap)
  C. hedged with the future  the same parity trade hedged with the venue's own future of the SAME expiry (Deribit
                              lists daily and weekly futures): this is the riskless part; whatever B earns on top is
                              the future–perp basis, a carry bet whose value depends on funding until expiry
  O01: both venues' snapshots within 30 s, the older quote moved by delta × ΔS to the newer moment; persistence of the
       same opportunity across consecutive snapshots (a real arbitrage on a liquid venue lives seconds, not hours).

Parity: linear (USDT) options C − P = F − K. Coin-margined (Deribit, OKX; premium in BTC, payoff (S−K)⁺/S):
C_btc − P_btc = 1 − K/F, so the option-implied forward is F_syn = K / (1 − (C_btc − P_btc)).
    python studies/options_arb_audit.py <dir with live/ and lab/ticks/>
"""
from __future__ import annotations

import glob
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import config  # noqa: E402

D = Path(sys.argv[1]) if len(sys.argv) > 1 else config.DATA
OUT = config.RESULTS / "studies"
FEE_RATE, FEE_CAP, PERP_FEE = 0.0003, 0.125, 0.0005
FUT_FEE = 0.0005                      # Deribit dated future, taker
GROUP = {"deribit": "usd", "okx": "usd", "binance": "usdt", "bybit": "usdt"}
INVERSE = {"deribit", "okx"}


def ofee(px, S):
    return np.minimum(FEE_RATE * S, FEE_CAP * np.maximum(px, 0.0))


# ------------------------------------------------------------------ data
def ticks() -> pd.DataFrame:
    t = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(D / "lab" / "ticks" / "*.csv")))], ignore_index=True)
    t["ts"] = pd.to_datetime(t["ts"], utc=True)
    return t.drop_duplicates("ts").sort_values("ts").set_index("ts")


def snaps(t: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for f in sorted(glob.glob(str(D / "live" / "deribit_chain" / "*.csv.gz"))):
        d = pd.read_csv(f)
        p = d["instrument"].str.split("-", expand=True)
        d = d.assign(venue="deribit", expiry=pd.to_datetime(p[1], format="%d%b%y", errors="coerce").dt.tz_localize("UTC")
                     + pd.Timedelta(hours=8), strike=pd.to_numeric(p[2], errors="coerce"), cp=p[3],
                     bid_btc=d["bid"], ask_btc=d["ask"], F=d["underlying"], fut=d["underlying_index"])
        rows.append(d[["snap_ts", "venue", "expiry", "strike", "cp", "bid_btc", "ask_btc", "F", "fut"]])
    for f in sorted(glob.glob(str(D / "live" / "okx_options" / "*.csv.gz"))):
        d = pd.read_csv(f)
        d = d[~d["symbol"].astype(str).str.contains("_UM")]
        d = d.assign(venue="okx", bid_btc=d["bid"] / d["F"], ask_btc=d["ask"] / d["F"], fut=None)
        rows.append(d[["snap_ts", "venue", "expiry", "strike", "cp", "bid_btc", "ask_btc", "F", "fut"]])
    for f in sorted(glob.glob(str(D / "live" / "binance_options" / "*.csv.gz"))):
        d = pd.read_csv(f)
        d = d.assign(venue="binance", bid=d["bidPrice"], ask=d["askPrice"], F=d["exercisePrice"], fut=None)
        rows.append(d[["snap_ts", "venue", "expiry", "strike", "cp", "bid", "ask", "F", "fut"]])
    for f in sorted(glob.glob(str(D / "live" / "bybit_options" / "*.csv.gz"))):
        d = pd.read_csv(f)
        d = d.assign(venue="bybit", fut=None)
        rows.append(d[["snap_ts", "venue", "expiry", "strike", "cp", "bid", "ask", "F", "fut"]])
    o = pd.concat(rows, ignore_index=True)
    o["snap_ts"] = pd.to_datetime(o["snap_ts"], utc=True)
    o["expiry"] = pd.to_datetime(o["expiry"], utc=True)
    # spot index at the snapshot (tick ≤ 30 s after it) and at the lab's decision time (the next minute ending in 3)
    idx = t["db_index"].dropna()
    o["S_snap"] = idx.reindex(o["snap_ts"].drop_duplicates().sort_values(), method="bfill", tolerance=pd.Timedelta(seconds=45)) \
        .reindex(o["snap_ts"]).values
    dec = o["snap_ts"].dt.floor("10min") + pd.Timedelta(minutes=3, seconds=3)
    o["t_dec"] = dec
    o["S_dec"] = idx.reindex(dec.drop_duplicates().sort_values(), method="nearest", tolerance=pd.Timedelta(seconds=45)) \
        .reindex(dec).values
    inv = o["venue"].isin(INVERSE)
    for when in ("snap", "dec"):
        o[f"bid_{when}"] = np.where(inv, o["bid_btc"] * o[f"S_{when}"], o["bid"])
        o[f"ask_{when}"] = np.where(inv, o["ask_btc"] * o[f"S_{when}"], o["ask"])
    o = o.dropna(subset=["strike", "S_snap"])
    ok = (o["bid_snap"] > 0) & (o["ask_snap"] > o["bid_snap"]) & (o["ask_snap"] < o["F"]) & (o["ask_snap"] < 3 * o["bid_snap"] + 50)
    intr = np.where(o["cp"] == "C", o["F"] - o["strike"], o["strike"] - o["F"]).clip(min=0)
    return o[ok & (o["ask_snap"] >= 0.9 * intr) & (o["expiry"] > o["snap_ts"])].reset_index(drop=True)


# ------------------------------------------------------------------ O02
def o02(o: pd.DataFrame, t: pd.DataFrame) -> pd.DataFrame:
    o = o[o["expiry"] < o["snap_ts"] + pd.Timedelta(days=7)]
    pairs = o.pivot_table(index=["venue", "snap_ts", "expiry", "strike"], columns="cp",
                          values=["bid_snap", "ask_snap", "bid_dec", "ask_dec", "bid_btc", "ask_btc", "F", "S_snap", "S_dec"],
                          aggfunc="first").dropna(subset=[("bid_snap", "C"), ("bid_snap", "P")])
    pairs.columns = [f"{a}_{b}" for a, b in pairs.columns]
    pairs = pairs.reset_index()
    fut = o[o["venue"] == "deribit"].groupby(["snap_ts", "expiry"])["fut"].first()
    rows = []
    tk = t[["db_bid", "db_ask", "db_funding8h", "bf_bid", "bf_ask", "bf_funding", "db_index"]]
    for r in pairs.itertuples(index=False):
        usd = GROUP[r.venue] == "usd"
        tau8 = (r.expiry - r.snap_ts).total_seconds() / 28800
        rec = {"venue": r.venue, "snap_ts": r.snap_ts, "expiry": r.expiry, "strike": r.strike, "tau_h": tau8 * 8,
               "F_ven": r.F_C, "guard_ok": abs(math.log(r.S_dec_C / r.S_snap_C)) <= 5e-4 if r.S_dec_C == r.S_dec_C else False}
        for mode, at, sfx in (("A_lab", r.snap_ts.floor("10min") + pd.Timedelta(minutes=3, seconds=3), "dec"),
                              ("B_sync", r.snap_ts, "snap")):
            i = tk.index.searchsorted(at)
            i = min(max(i, 0), len(tk) - 1)
            row = tk.iloc[i]
            if abs((tk.index[i] - at).total_seconds()) > 45:
                continue
            pb, pa, fr = (row.db_bid, row.db_ask, row.db_funding8h) if usd else (row.bf_bid, row.bf_ask, row.bf_funding)
            if not (np.isfinite(pb) and np.isfinite(pa)):
                continue
            S = (pb + pa) / 2
            carry = (fr if np.isfinite(fr) else 0.0) * tau8 * S
            cb, ca = getattr(r, f"bid_{sfx}_C"), getattr(r, f"ask_{sfx}_C")
            pbid, pask = getattr(r, f"bid_{sfx}_P"), getattr(r, f"ask_{sfx}_P")
            f1 = ofee(cb, S) + ofee(pask, S) + 2 * PERP_FEE * S
            f2 = ofee(ca, S) + ofee(pbid, S) + 2 * PERP_FEE * S
            rec[f"{mode}_sell"] = (cb - pask + r.strike) - pa - f1 - carry
            rec[f"{mode}_buy"] = pb - (ca - pbid + r.strike) - f2 + carry
            rec[f"{mode}_perp_mid"] = S
            rec[f"{mode}_carry"] = carry
        # synthetic forward from mid quotes (exact for coin-margined) and the venue's own forward
        if r.venue in INVERSE:
            cpm = (r.bid_btc_C + r.ask_btc_C) / 2 - (r.bid_btc_P + r.ask_btc_P) / 2
            rec["F_syn"] = r.strike / (1 - cpm) if cpm < 1 else np.nan
        else:
            rec["F_syn"] = r.strike + (r.bid_snap_C + r.ask_snap_C) / 2 - (r.bid_snap_P + r.ask_snap_P) / 2
        # C: hedge with the Deribit future of the same expiry (riskless parity) — Deribit only, where it exists
        if r.venue == "deribit":
            Ff = r.F_C
            S = r.S_snap_C
            fb, fa = Ff * (1 - 0.00005), Ff * (1 + 0.00005)          # 1 tick ≈ 0.5 bp half-spread on the future
            f1 = ofee(r.bid_snap_C, S) + ofee(r.ask_snap_P, S) + 2 * FUT_FEE * S
            f2 = ofee(r.ask_snap_C, S) + ofee(r.bid_snap_P, S) + 2 * FUT_FEE * S
            # inverse parity in USD at expiry: buying the synthetic costs (C−P)_btc BTC = (C−P)_btc·S now
            rec["C_fut_sell"] = (r.bid_snap_C - r.ask_snap_P) - (fa - r.strike) * S / fa - f1
            rec["C_fut_buy"] = (fb - r.strike) * S / fb - (r.ask_snap_C - r.bid_snap_P) - f2
        rows.append(rec)
    return pd.DataFrame(rows)


def o02_tables(p: pd.DataFrame) -> dict:
    out = {}
    stages = [("A_lab", "A. as the lab: snapshot vs perp 3 min later"), ("B_sync", "B. synchronous: perp at the snapshot"),
              ("C_fut", "C. hedged with the same-expiry future (Deribit)")]
    rows = []
    for key, label in stages:
        for side in ("buy", "sell"):
            c = f"{key}_{side}"
            if c not in p:
                continue
            x = p[[c, "snap_ts", "venue", "guard_ok"]].dropna()
            if key == "A_lab":
                x = x[x["guard_ok"].astype(bool)]                     # the lab's 5-bp snapshot guard
            hit = x[x[c] > 0]
            rows.append({"stage": label, "side": f"{side} synthetic", "candidates": len(x), "signals": len(hit),
                         "snapshots with a signal": hit["snap_ts"].nunique(), "sum $ (1 BTC each)": hit[c].sum(),
                         "median edge $": hit[c].median() if len(hit) else np.nan,
                         "by venue": hit.groupby("venue")[c].size().to_dict()})
    out["stages"] = pd.DataFrame(rows)
    # decomposition: synthetic forward − perp − carry = (synthetic − venue forward) + (venue forward − perp − carry)
    q = p.dropna(subset=["F_syn", "B_sync_perp_mid"]).copy()
    q["syn_minus_fwd_bp"] = 1e4 * (q["F_syn"] / q["F_ven"] - 1)
    q["fwd_minus_perp_carry_bp"] = 1e4 * ((q["F_ven"] - q["B_sync_carry"]) / q["B_sync_perp_mid"] - 1)
    agg = q.groupby(["venue"]).agg(n=("strike", "size"),
                                   syn_vs_own_forward_bp=("syn_minus_fwd_bp", "median"),
                                   syn_vs_own_forward_iqr=("syn_minus_fwd_bp", lambda s: s.quantile(.75) - s.quantile(.25)),
                                   own_forward_vs_perp_plus_carry_bp=("fwd_minus_perp_carry_bp", "median"))
    out["decomp"] = agg.reset_index()
    # per expiry and snapshot, how much of the A-stage edge is common to all strikes (basis) vs strike-specific
    e = p.dropna(subset=["A_lab_buy"]).copy()
    e["edge"] = e[["A_lab_buy", "A_lab_sell"]].max(axis=1)
    g = e.groupby(["venue", "snap_ts", "expiry"])["edge"]
    e["common"] = g.transform("median")
    share = 1 - ((e["edge"] - e["common"]) ** 2).sum() / ((e["edge"] - e["edge"].mean()) ** 2).sum()
    out["common_share"] = share
    return out


# ------------------------------------------------------------------ O01
def bs_delta(F, K, sig, T, cp):
    if not (sig > 0 and T > 0):
        return 1.0 if (cp == "C" and F > K) else (-1.0 if cp == "P" and F < K else 0.0)
    d1 = (math.log(F / K) + 0.5 * sig * sig * T) / (sig * math.sqrt(T))
    return norm.cdf(d1) if cp == "C" else norm.cdf(d1) - 1


def o01(o: pd.DataFrame, t: pd.DataFrame) -> pd.DataFrame:
    """Every venue pair's snapshots matched to the nearest snapshot of the other venue; quotes moved to a common
    moment by delta × ΔS (delta from a 40% vol, close enough for the check); bid on B − ask on A − fees."""
    idx = t["db_index"].dropna()
    o = o.assign(grp=o["venue"].map(GROUP))
    rows = []
    for grp, g in o.groupby("grp"):
        venues = sorted(g["venue"].unique())
        for i, va in enumerate(venues):
            for vb in venues[i + 1:]:
                A, B = g[g["venue"] == va], g[g["venue"] == vb]
                tb = pd.DatetimeIndex(B["snap_ts"].unique()).sort_values()
                for ta, ga in A.groupby("snap_ts"):
                    j = tb.searchsorted(ta)
                    cands = [tb[k] for k in (j - 1, j) if 0 <= k < len(tb)]
                    if not cands:
                        continue
                    tbb = min(cands, key=lambda x: abs(x - ta))
                    lag = abs((tbb - ta).total_seconds())
                    gb = B[B["snap_ts"] == tbb]
                    m = ga.merge(gb, on=["expiry", "strike", "cp"], suffixes=("_a", "_b"))
                    if m.empty:
                        continue
                    Sa = idx.asof(ta + pd.Timedelta(seconds=30))
                    Sb = idx.asof(tbb + pd.Timedelta(seconds=30))
                    dS = Sb - Sa
                    T = (m["expiry"] - ta).dt.total_seconds() / (365 * 86400)
                    dl = np.array([bs_delta(f, k, 0.4, tt, c) for f, k, tt, c in zip(m["F_a"], m["strike"], T, m["cp"])])
                    S = Sa
                    for buy, sell in (("a", "b"), ("b", "a")):
                        # move venue a's quotes to venue b's moment (a is the older or newer, symmetric)
                        shift = dl * dS
                        ask = m[f"ask_snap_{buy}"] + (shift if buy == "a" else 0)
                        bid = m[f"bid_snap_{sell}"] + (shift if sell == "a" else 0)
                        raw = m[f"bid_snap_{sell}"] - m[f"ask_snap_{buy}"] - ofee(m[f"ask_snap_{buy}"], S) - ofee(m[f"bid_snap_{sell}"], S)
                        adj = bid - ask - ofee(ask, S) - ofee(bid, S)
                        for k in np.where(raw > 0)[0]:
                            rows.append({"group": grp, "buy": va if buy == "a" else vb, "sell": vb if buy == "a" else va,
                                         "t": max(ta, tbb), "lag_s": lag, "dS": dS, "expiry": m["expiry"].iloc[k],
                                         "strike": m["strike"].iloc[k], "cp": m["cp"].iloc[k],
                                         "ask": m[f"ask_snap_{buy}"].iloc[k], "bid": m[f"bid_snap_{sell}"].iloc[k],
                                         "edge_raw": raw.iloc[k], "edge_adj": adj.iloc[k]})
    return pd.DataFrame(rows)


def md(df, digits=1):
    d = df.copy()
    for c in d.columns:
        if pd.api.types.is_float_dtype(d[c]):
            d[c] = d[c].map(lambda v: "" if pd.isna(v) else f"{v:,.{digits}f}")
    return d.to_markdown(index=False)


def main():
    t = ticks()
    o = snaps(t)
    parts = [f"Snapshots: {o.groupby('venue')['snap_ts'].nunique().to_dict()} from {o['snap_ts'].min():%d.%m %H:%M} "
             f"to {o['snap_ts'].max():%d.%m %H:%M} UTC; {len(o):,} quotes after the lab's filters"]
    lag = o.groupby(["venue", "snap_ts"]).first().reset_index()
    lag["move_bp_to_decision"] = 1e4 * (lag["S_dec"] / lag["S_snap"] - 1)
    parts.append("Spot move between the snapshot and the lab's decision 3 minutes later (bp): "
                 + md(lag.groupby("venue")["move_bp_to_decision"].describe(percentiles=[.5, .9]).reset_index()[
                     ["venue", "count", "50%", "90%", "max"]]))
    p = o02(o, t)
    p.to_csv(OUT / "o02_audit_rows.csv", index=False)
    tb = o02_tables(p)
    parts.append("## O02 stages\n\n" + md(tb["stages"]))
    parts.append("## O02 decomposition (median over strikes, bp of the forward)\n\n" + md(tb["decomp"], 2))
    parts.append(f"Share of the lab-stage edge variance common to all strikes of one expiry and snapshot: "
                 f"{100 * tb['common_share']:.0f}%")
    a = o01(o, t)
    a.to_csv(OUT / "o01_audit_rows.csv", index=False)
    if len(a):
        a["key"] = a["buy"] + ">" + a["sell"] + "|" + a["expiry"].astype(str) + "|" + a["strike"].astype(str) + a["cp"]
        a = a.sort_values("t")
        a["persist_snaps"] = a.groupby("key")["t"].transform("size")
        rows = []
        for lab_, x in (("all raw signals", a), ("snapshots ≤ 30 s apart", a[a.lag_s <= 30]),
                        ("≤ 30 s and still > 0 after the delta shift", a[(a.lag_s <= 30) & (a.edge_adj > 0)])):
            rows.append({"filter": lab_, "signals": len(x), "distinct options": x["key"].nunique(),
                         "sum $": x["edge_raw"].sum() if "raw" in lab_ else x["edge_adj"].sum(),
                         "median $": x["edge_adj"].median(), "pairs": x.groupby(["buy", "sell"]).size().to_dict()})
        parts.append("## O01 cross-venue\n\n" + md(pd.DataFrame(rows)))
        top = a[(a.lag_s <= 30) & (a.edge_adj > 0)].groupby("key").agg(first=("t", "min"), last=("t", "max"),
                                                                          snaps=("t", "size"), edge=("edge_adj", "median"),
                                                                          ask=("ask", "median"), bid=("bid", "median"))
        top["hours alive"] = (top["last"] - top["first"]).dt.total_seconds() / 3600
        parts.append("### Longest-lived O01 opportunities\n\n" + md(top.sort_values("snaps", ascending=False).head(12).reset_index()))
    text = "\n\n".join(parts)
    (OUT / "options_arb_audit_tables.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
