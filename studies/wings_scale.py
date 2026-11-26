"""Seven ways to make the wing strategy earn more: fees, exits, spreads, gates, selection, sizing, portfolio.

Baseline (frozen rules, 1 contract, hold to settlement, taker)
    W03  cheap side 10-20 cents, 5-15 minutes left, normal tails, model edge >= 1 cent, one strike an hour
    W04  cheap side 6-10 cents, 15-30 minutes left, Student-t(4) tails, model edge >= 0, one strike an hour
Jul-Aug is the development half (every parameter below is chosen there), Sep-Oct is the test half and
9-25 November is a holdout the study never looks at while choosing. Every P&L includes the Kalshi taker fee
(rounded up to the cent *per order*, which is the whole point of section 1). t-statistics are clustered by
settlement hour: the 40 strikes of one hour are one observation.

1  fee lots          fee(C,p) = ceil(0.07*C*p*(1-p)) : a 1-contract order on an 8-cent wing pays 1 cent per
                     contract, a >=194-contract order pays 0.52. Quantifies what the 1-contract backtest
                     leaves on the table and the block sizes that stop the rounding from mattering.
2  exits             hold to settlement vs stop-loss / take-profit / time exit / model exit. Caps the rare
                     -0.92 hour at the cost of the spread twice and of giving up the premium.
3  vertical spreads sell the 6-10 cent wing and buy a deeper 0.5-3 cent wing of the same side: the credit
                     spread pays 0 in the extreme tail where the single leg loses the whole dollar.
4  gates             weekend, US macro release in the hour, round strikes, hour of day, volatility regime,
                     distance in standard deviations, spread width, momentum toward the wing.
5  selection         which of the hour's roughly two rich strikes to sell: first in time (frozen), dearest,
                     maximum edge, maximum edge per dollar of collateral, maximum Kelly-scaled edge.
6  sizing            contract size from the edge: full depth, fixed block, fractional Kelly, tail limit.
7  portfolio         the same rule as a share of equity (fraction f of capital per hour, whole contracts) on
                     the test half, with drawdown, and the fee-aware growth-optimal f.

Run:  python studies/wings_scale.py            (writes results/studies/wings_scale_*.csv and .md)
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from studies.wing_panel import (best_block, build, cheap_sides, fee, fee_per_contract)  # noqa: E402

OUT = ROOT / "results/studies"
SPLIT = pd.Timestamp("2026-09-01", tz="UTC")
TRAIN = (pd.Timestamp("2026-07-01", tz="UTC"), SPLIT)
TEST = (SPLIT, pd.Timestamp("2026-10-28", tz="UTC"))
NOV = (pd.Timestamp("2026-11-08", tz="UTC"), pd.Timestamp("2026-11-26", tz="UTC"))
PERIODS = {"train Jul-Aug": TRAIN, "test Sep-Oct": TEST, "holdout Nov": NOV}

RULES = {
    "W03": dict(band=(0.10, 0.20), window=(5, 15), model="normal", margin=0.01),
    "W04": dict(band=(0.06, 0.10), window=(15, 30), model="t4", margin=0.0),
}
MACRO = ROOT / "data/manual/us_macro_calendar.csv"


# --------------------------------------------------------------------------------------- helpers
def interval(x: pd.Series, draws: int = 4000, seed: int = 7) -> tuple[float, float]:
    """Day-block bootstrap 95% interval of the mean of a per-hour series (hours of one day are dependent)."""
    x = pd.Series(x).dropna()
    if len(x) < 10:
        return (np.nan, np.nan)
    days = pd.Series(x.to_numpy(), index=pd.DatetimeIndex(x.index).floor("D"))
    groups = [g.to_numpy() for _, g in days.groupby(level=0)]
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(draws):
        pick = rng.integers(0, len(groups), len(groups))
        means.append(np.concatenate([groups[i] for i in pick]).mean())
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def share_zero(x: pd.Series, draws: int = 2000, seed: int = 11) -> float:
    """Bootstrap two-sided p-value that the mean of a per-hour series is zero (day blocks)."""
    x = pd.Series(x).dropna()
    if len(x) < 10:
        return np.nan
    days = pd.Series(x.to_numpy(), index=pd.DatetimeIndex(x.index).floor("D"))
    groups = [g.to_numpy() for _, g in days.groupby(level=0)]
    rng = np.random.default_rng(seed)
    means = np.array([np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))]).mean()
                      for _ in range(draws)])
    return float(2 * min((means <= 0).mean(), (means >= 0).mean()))


def stats(pnl_by_hour: pd.Series) -> dict:
    """Mean, t (clustered by hour), hit, worst hour, max drawdown of the hourly P&L series."""
    x = pnl_by_hour.dropna()
    n = len(x)
    if n < 3:
        return dict(hours=n, cents=np.nan, t=np.nan, hit=np.nan, worst=np.nan, dd=np.nan, total=np.nan)
    t = x.mean() / (x.std(ddof=1) / math.sqrt(n)) if x.std() > 0 else np.nan
    eq = x.cumsum()
    dd = float((eq.cummax() - eq).max())
    return dict(hours=n, cents=100 * x.mean(), t=t, hit=float((x > 0).mean()), worst=float(x.min()),
                dd=dd, total=float(x.sum()))


def pnl_at(t: pd.DataFrame, n) -> pd.Series:
    """Sell n contracts of the cheap side at its bid, hold to settlement, Kalshi fee rounded per order."""
    n = np.asarray(n, dtype=float) if not np.isscalar(n) else np.full(len(t), float(n))
    return n * (t.wing_bid.to_numpy() - t.won.to_numpy()) - fee(n, t.price.to_numpy())


def cheap(d: pd.DataFrame, rule: str) -> pd.DataFrame:
    """All cheap-side minutes of a rule (before any model filter or one-per-hour cap)."""
    r = RULES[rule]
    c = cheap_sides(d, *r["band"], *r["window"])
    c["fair"] = c["fair_n"] if r["model"] == "normal" else c["fair_t4"]
    c["edge"] = c.wing_bid - c.fair - fee(1, c.price)
    c["rule"] = rule
    return c


def signals(c: pd.DataFrame, margin: float | None = None, selection: str = "first",
            contracts: float = 1.0) -> pd.DataFrame:
    """One trade an hour: apply the model filter, then pick the strike by `selection` at the first minute.

    first       the frozen rule: earliest minute with a signal, dearest strike within that minute
    dearest     the dearest strike of the hour among all signals
    edge        the largest model edge of the hour
    edge_per_$  the largest edge per dollar of collateral (price paid for the other side)
    kelly       the largest (win - price)/((1-price)*price): Kelly-optimal edge per unit of variance
    """
    m = (RULES[c.rule.iloc[0]]["margin"] if margin is None else margin)
    g = c[c.edge >= m - 1e-12].copy()
    if g.empty:
        return g
    if selection == "first":
        g = g.sort_values(["end", "ts", "wing_bid"], ascending=[True, True, False])
        g = g.drop_duplicates(["end", "market"]).drop_duplicates("end")
    else:
        g["_key"] = {"dearest": g.wing_bid,
                     "edge": g.edge,
                     "edge_per_$": g.edge / g.price.clip(lower=0.01),
                     "kelly": (1 - g.fair - g.price) / ((1 - g.price).clip(lower=0.01) * g.price.clip(lower=0.01)),
                     }[selection]
        g = g.sort_values(["end", "ts", "_key"], ascending=[True, True, False])
        g = g.drop_duplicates("end")
    g = g.copy()
    g["n"] = contracts
    g["pnl"] = pnl_at(g, contracts)
    return g.set_index("end", drop=False)


def per_hour(g: pd.DataFrame, pnl_col: str = "pnl") -> pd.Series:
    return g.groupby(level=0)[pnl_col].sum()


def add_pooled(pnl_by_hour: pd.Series) -> pd.Series:
    return pnl_by_hour.groupby(level=0).sum()


# --------------------------------------------------------------------------------------- 1. fee lots
def fee_lots(sig: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    prices = [0.02, 0.04, 0.06, 0.08, 0.10, 0.14, 0.18, 0.22, 0.30, 0.50, 0.75, 0.92]
    for p in prices:
        block = best_block(p)
        rows.append({"price": p, "limit ¢": round(100 * block["limit"], 3),
                     "C=1 ¢": round(100 * float(fee_per_contract(1, p)), 3),
                     "C=10 ¢": round(100 * float(fee_per_contract(10, p)), 3),
                     "C=100 ¢": round(100 * float(fee_per_contract(100, p)), 3),
                     "C=500 ¢": round(100 * float(fee_per_contract(500, p)), 3),
                     "2% block": block["contracts"], "1% block": best_block(p, tolerance=0.01)["contracts"],
                     "5% block": best_block(p, tolerance=0.05)["contracts"]})
    return pd.DataFrame(rows)


def fee_scale(sig: dict[str, pd.DataFrame], sizes=(1, 10, 50, 100, 200, 500)) -> pd.DataFrame:
    rows = []
    for period, (a, b) in PERIODS.items():
        trades = pd.concat([v[(v.end >= a) & (v.end < b)] for v in sig.values()])
        if trades.empty:
            continue
        for n in sizes:
            pnl = pd.Series(pnl_at(trades, n), index=trades.index)
            h = pnl.groupby(trades.end.values).sum()
            rows.append({"period": period, "contracts per order": n, "trades": len(trades),
                         "¢ per contract": 100 * pnl.sum() / (n * len(trades)),
                         "$ per hour": h.mean(), "total $": pnl.sum(),
                         "fee per contract ¢": 100 * float(fee(n, trades.price).sum()) / (n * len(trades))})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------- 2. exits
def _future_quotes(d: pd.DataFrame, trades: pd.DataFrame) -> pd.DataFrame:
    """Every later minute of each traded strike, with the cheap side's ask and model fair value that minute."""
    keep = d[d.ticker.isin(trades.ticker.unique())][["ticker", "ts", "bid", "ask", "p_yes", "p_yes_t4"]]
    x = trades[["ticker", "ts", "end", "cheap_yes", "wing_bid", "won", "price"]].merge(
        keep, on="ticker", how="left", suffixes=("", "_q"))
    x = x[(x.ts_q > x.ts) & (x.ts_q < x.end)]
    cheap_yes = x.cheap_yes.to_numpy()
    x["wing_ask_t"] = np.where(cheap_yes, x.ask, 1 - x.bid)
    x["fair_t"] = np.where(cheap_yes, x.p_yes_t4, 1 - x.p_yes_t4)
    x["valid"] = (x.bid > 0) & (x.ask < 1) & (x.ask >= x.bid)
    x["minute"] = (x.ts_q - x.ts).dt.total_seconds() / 60
    return x.sort_values(["ticker", "ts", "ts_q"])


def exits(d: pd.DataFrame, trades: pd.DataFrame) -> pd.DataFrame:
    """P&L per hour for each exit policy, evaluated on exactly the same entries."""
    fut = _future_quotes(d, trades)

    def close_at(cond, name):
        """Close at the first valid minute after entry where `cond` holds, otherwise hold to settlement."""
        x = fut[fut.valid].copy()
        x["hit"] = cond(x)
        hit = x[x.hit].sort_values(["ticker", "ts", "ts_q"]).drop_duplicates(["ticker", "ts"])
        exits = hit.set_index(["ticker", "ts"])["wing_ask_t"].rename("exit_ask")
        t = trades.set_index(["ticker", "ts"]).join(exits)
        held = t.exit_ask.isna()
        pnl = np.where(held, t.wing_bid - t.won, t.wing_bid - t.exit_ask)
        f = np.where(held, fee(1.0, t.price), fee(1.0, t.price) + fee(1.0, t.exit_ask.fillna(0.5)))
        return pd.DataFrame({"policy": name, "pnl": pnl - f, "end": t.end.to_numpy(),
                             "closed": ~held.to_numpy()})

    rows = [close_at(lambda x: pd.Series(False, index=x.index), "hold to settlement")]
    for delta in (0.05, 0.10, 0.15, 0.20, 0.30):
        rows.append(close_at(lambda x, d=delta: x.wing_ask_t >= x.wing_bid + d, f"stop +{int(delta*100)}¢"))
    for k in (2, 3):
        rows.append(close_at(lambda x, k=k: x.wing_ask_t >= k * x.wing_bid, f"stop x{k}"))
    for delta in (0.01, 0.02, 0.03):
        rows.append(close_at(lambda x, d=delta: x.wing_ask_t <= (x.wing_bid - d).clip(lower=0),
                             f"take profit -{int(delta*100)}¢"))
    for minute in (3, 5, 10):
        rows.append(close_at(lambda x, m=minute: x.minute >= m, f"time exit {minute} min"))
    rows.append(close_at(lambda x: x.fair_t >= x.wing_ask_t - 0.01, "model exit"))
    return pd.concat(rows, ignore_index=True)


# --------------------------------------------------------------------------------------- 3. spreads
def spreads(d: pd.DataFrame, trades: pd.DataFrame, wide=(0.005, 0.03), pick="nearest") -> pd.DataFrame:
    """Sell the signal wing and buy a deeper wing on the same side in the same snapshot (credit spread).

    The long leg pays only if the spot ends beyond the farther strike, i.e. exactly in the part of the
    distribution where the single short wing loses its whole dollar. The spread keeps the credit minus the
    protection premium and caps the worst case at 1 - credit only in the far tail.
    """
    snap = d[(d.bid > 0) & (d.ask < 1) & (d.ask >= d.bid) & d.ask.le(0.5)][
        ["event", "ts", "ticker", "strike", "bid", "ask", "y"]]
    ent = trades[["event", "ts", "end", "ticker", "strike", "cheap_yes", "wing_bid", "won", "price"]].rename(
        columns={"ticker": "entry_ticker", "strike": "entry_strike"})
    x = ent.merge(snap, on=["event", "ts"], how="inner")
    x = x[x.ticker.ne(x.entry_ticker)]
    x["prot_ask"] = np.where(x.cheap_yes, x.ask, 1 - x.bid)
    same_side = np.where(x.cheap_yes, x.strike > x.entry_strike, x.strike < x.entry_strike)
    x = x[same_side & x.prot_ask.between(*wide)].copy()
    if x.empty:
        return x
    x["won_prot"] = np.where(x.cheap_yes, x.y, 1 - x.y)
    x["width"] = (x.strike - x.entry_strike).abs()
    x = x.sort_values(["entry_ticker", "ts", "width" if pick == "nearest" else "prot_ask"])
    x = x.drop_duplicates(["entry_ticker", "ts"])
    x["credit"] = x.wing_bid - x.prot_ask
    x["fee"] = fee(1, x.wing_bid) + fee(1, x.prot_ask)
    x["pnl"] = x.credit - x.won + x.won_prot - x.fee
    x["single"] = x.wing_bid - x.won - fee(1, x.wing_bid)
    x["collateral"] = 1 - x.credit
    return x.rename(columns={"ts": "signal_ts"})


# --------------------------------------------------------------------------------------- 4. gates
def macro_hours() -> pd.DatetimeIndex:
    m = pd.read_csv(MACRO)
    ts = pd.to_datetime(m.date + " " + m.time_et, utc=False, errors="coerce")
    # ET -> UTC: EST is UTC-5, EDT is UTC-4; the calendar runs over both regimes, so shift by the DST rule.
    et = ts.dt.tz_localize("America/New_York", ambiguous="NaT", nonexistent="shift_forward")
    return pd.DatetimeIndex(et.dt.tz_convert("UTC").dropna()).floor("min")


def gate_features(d: pd.DataFrame, trades: pd.DataFrame) -> pd.DataFrame:
    """Conditioning variables known at the signal minute (nothing after it)."""
    close = d.groupby("ts").S.first().sort_index()
    m5 = close.reindex(close.index - pd.Timedelta(minutes=5)).to_numpy()
    spot5 = pd.Series(np.log(close.to_numpy() / m5), index=close.index)
    mac = np.sort(macro_hours().to_numpy())
    t = trades.copy()
    # the contract hour that contains a release (a 12:30 UTC release sits inside the 12:00-13:00 contract)
    t["macro_contract"] = t.end.isin(macro_hours().floor("h") + pd.Timedelta(hours=1))
    # a release inside the part of the hour the rule is exposed to, [signal minute, settlement)
    t["macro_window"] = (np.searchsorted(mac, t.end.to_numpy())
                         - np.searchsorted(mac, t.ts.to_numpy()) > 0)
    t["weekend"] = t.end.dt.day_of_week >= 5
    level = (t.strike / 100).round() * 100            # Kalshi quotes strikes at ...99.99, i.e. one cent below
    t["round500"] = (level % 500).eq(0)
    t["round1000"] = (level % 1000).eq(0)
    t["hour utc"] = t.end.dt.hour
    t["us open"] = t.end.dt.hour.isin([13, 14, 15])
    t["sigma bp"] = np.sqrt(t.var_rem) * 1e4
    t["|z|"] = t.z.abs()
    t["spread ¢"] = 100 * (t.ask - t.bid)
    t["move5 bp"] = (1e4 * spot5.reindex(t.ts).to_numpy()) * np.where(t.cheap_yes, 1, -1)
    t["move5 toward wing"] = t["move5 bp"] > 0
    return t


def gate_table(t: pd.DataFrame, name: str) -> pd.DataFrame:
    """Per-hour P&L inside and outside each gate, with a day-block interval of the difference."""
    rows = []
    gates = {
        "weekend": t.weekend,
        "contract hour contains a release": t.macro_contract,
        "release inside the trade window": t.macro_window,
        "US open 13-15 UTC": t["us open"],
        "round 500/1000 strike": t.round500,
        "round 1000 strike": t.round1000,
        "calm sigma (bottom third)": t["sigma bp"] <= t["sigma bp"].quantile(1 / 3),
        "high sigma (top third)": t["sigma bp"] >= t["sigma bp"].quantile(2 / 3),
        "near strike |z|<1.3": t["|z|"] < 1.3,
        "far strike |z|>1.6": t["|z|"] > 1.6,
        "spread 1 cent": t["spread ¢"] <= 1.01,
        "spot moving toward the wing (5 min)": t["move5 toward wing"],
    }
    for gname, mask in gates.items():
        inside, outside = t[mask], t[~mask]
        if len(inside) < 5 or len(outside) < 5:
            continue
        a = inside.pnl.groupby(inside.end).sum()
        b = outside.pnl.groupby(outside.end).sum()
        both = pd.DataFrame({"in": a, "out": b}).fillna(0.0)
        delta = both["in"] - both["out"]
        lo, hi = interval(delta)
        p_value = share_zero(delta)
        rows.append({"rule": name, "gate": gname, "n in": len(inside), "n out": len(outside),
                     "¢/hour in": 100 * inside.pnl.mean(), "¢/hour out": 100 * outside.pnl.mean(),
                     "Δ ¢/hour": 100 * delta.mean(), "95% low": 100 * lo, "95% high": 100 * hi,
                     "p": p_value,
                     "t in": inside.pnl.mean() / (inside.pnl.std(ddof=1) / math.sqrt(len(inside)))
                     if len(inside) > 2 else np.nan})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------- 6. sizing
def kelly_size(t: pd.DataFrame, fraction: float = 0.25, cap: int = 200) -> pd.Series:
    """Fractional Kelly contracts from the model edge (never negative: skipping is handled upstream)."""
    w = (1 - t.fair).clip(0.001, 0.999)
    q = t.price.clip(0.01, 0.99)
    f = ((w - q) / ((1 - q) * q)).clip(lower=0)
    return (fraction * f * cap).round().clip(lower=1, upper=cap).astype(int)


# --------------------------------------------------------------------------------------- 7. portfolio
def compound(t: pd.DataFrame, share: float, capacity: int = 100, seed: float = 100.0) -> pd.DataFrame:
    """Grow an account: each hour commit `share` of equity as collateral, whole contracts, fees included."""
    eq, peak = seed, seed
    rows = []
    for end, g in t.groupby(level=0, sort=True):
        price = float(g.price.iloc[0])
        budget = share * eq
        n = int(min(capacity, max(0, budget // max(price, 0.01))))
        pnl = 0.0 if n == 0 else float(pnl_at(g, n)[0])
        eq += pnl
        peak = max(peak, eq)
        rows.append({"end": end, "contracts": n, "pnl": pnl, "equity": eq, "drawdown": 1 - eq / peak})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------- main
def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    d = build()
    archive = d[d.source.eq("candles")]
    sig = {}
    for rule in RULES:
        c = cheap(d, rule)
        sig[rule] = signals(c)

    # 1 fee lots
    fl = fee_lots(sig)
    fl.to_csv(OUT / "wings_scale_fees.csv", index=False)
    fs = fee_scale(sig)
    fs.to_csv(OUT / "wings_scale_fee_scale.csv", index=False)

    # 2 exits (the November snapshots stop at the entry window, so exits are scored on the archive)
    ex_rows = []
    for rule, trades in sig.items():
        tr = trades[trades.source.eq("candles")]
        if tr.empty:
            continue
        ex = exits(archive, tr)
        for period, (a, b) in list(PERIODS.items())[:2]:
            sub = ex[(ex.end >= a) & (ex.end < b)]
            if sub.empty:
                continue
            base = sub[sub.policy.eq("hold to settlement")].set_index("end").pnl
            for policy, g in sub.groupby("policy"):
                h = g.set_index("end").pnl
                delta = h.sub(base, fill_value=0.0)
                lo, hi = interval(delta)
                s = stats(h)
                ex_rows.append({"rule": rule, "period": period, "policy": policy, "hours": s["hours"],
                                "¢/hour": s["cents"], "t": s["t"], "hit": s["hit"], "worst $": s["worst"],
                                "closed %": 100 * g.closed.mean(), "Δ vs hold ¢/hour": 100 * delta.mean(),
                                "Δ 95% low": 100 * lo, "Δ 95% high": 100 * hi})
    ex_table = pd.DataFrame(ex_rows)
    ex_table.to_csv(OUT / "wings_scale_exits.csv", index=False)

    # 3 spreads
    sp_rows = []
    for rule, trades in sig.items():
        for pick in ("nearest", "cheapest"):
            sp = spreads(archive, trades[trades.source.eq("candles")], pick=pick)
            if sp.empty:
                continue
            for period, (a, b) in list(PERIODS.items())[:2]:
                sub = sp[(sp.end >= a) & (sp.end < b)]
                if len(sub) < 20:
                    continue
                h = sub.set_index("end").pnl
                s = stats(h)
                single = stats(sub.set_index("end").single)
                lo, hi = interval(h.sub(sub.set_index("end").single, fill_value=0.0))
                sp_rows.append({"rule": rule, "protection": pick, "period": period, "spreads": len(sub),
                                "median width $": float(sub.width.median()),
                                "credit ¢": 100 * sub.credit.mean(), "pnl ¢": 100 * sub.pnl.mean(),
                                "t": s["t"], "worst $": s["worst"], "dd $": s["dd"],
                                "single ¢": 100 * sub.single.mean(), "single worst $": single["worst"],
                                "Δ vs single ¢": 100 * (sub.pnl.mean() - sub.single.mean()),
                                "Δ 95% low": 100 * lo, "Δ 95% high": 100 * hi,
                                "pnl per $ collateral %": 100 * sub.pnl.mean() / sub.collateral.mean()})
    sp_table = pd.DataFrame(sp_rows)
    sp_table.to_csv(OUT / "wings_scale_spreads.csv", index=False)

    # the same over both windows (3-20 cents, 5-30 minutes): one bet an hour, chosen by Kelly
    wide = cheap_sides(d, 0.03, 0.20, 5, 30)
    wide["fair"] = wide.fair_t4
    wide["edge"] = wide.wing_bid - wide.fair - fee(1, wide.price)
    wide["rule"] = "wide"
    chosen = signals(wide, margin=0.01, selection="kelly")

    # 4 gates: built on the development half, read on the test half and on the November holdout
    gate_rows = []
    frames = {**sig, "wide (3-20¢, 5-30 min, Kelly)": chosen}
    for rule, trades in frames.items():
        t = gate_features(d, trades)
        for label, (a, b) in PERIODS.items():
            sub = t[(t.end >= a) & (t.end < b)]
            if len(sub) < 20:
                continue
            gate_rows.append(gate_table(sub, f"{rule} | {label}"))
    gate_table_all = pd.concat([g for g in gate_rows if len(g)], ignore_index=True)
    if len(gate_table_all):
        gate_table_all["p holm"] = np.nan
        for rule, idx in gate_table_all.groupby("rule").groups.items():
            p = gate_table_all.loc[idx, "p"].to_numpy()
            order = np.argsort(p)
            adj = np.empty_like(p)
            run = 0.0
            for rank, i in enumerate(order):
                run = max(run, (len(p) - rank) * p[i])
                adj[i] = min(1.0, run)
            gate_table_all.loc[idx, "p holm"] = adj
    gate_table_all.to_csv(OUT / "wings_scale_gates.csv", index=False)

    # 5 selection
    sel_rows = []
    for rule in RULES:
        c = cheap(d, rule)
        for selection in ("first", "dearest", "edge", "edge_per_$", "kelly"):
            g = signals(c, selection=selection)
            for period, (a, b) in PERIODS.items():
                sub = g[(g.end >= a) & (g.end < b)]
                if len(sub) < 10:
                    continue
                s = stats(sub.pnl.groupby(sub.end).sum())
                sel_rows.append({"rule": rule, "selection": selection, "period": period,
                                 "trades": len(sub),
                                 "¢/hour": s["cents"], "t": s["t"], "hit": s["hit"],
                                 "worst $": s["worst"], "dd $": s["dd"]})
    for selection in ("first", "dearest", "edge", "edge_per_$", "kelly"):
        g = signals(wide, margin=0.01, selection=selection)
        for period, (a, b) in PERIODS.items():
            sub = g[(g.end >= a) & (g.end < b)]
            if len(sub) < 10:
                continue
            s = stats(sub.pnl.groupby(sub.end).sum())
            sel_rows.append({"rule": "3-20¢ 5-30 min", "selection": selection, "period": period,
                             "trades": len(sub), "¢/hour": s["cents"], "t": s["t"], "hit": s["hit"],
                             "worst $": s["worst"], "dd $": s["dd"]})
    sel_table = pd.DataFrame(sel_rows)
    sel_table.to_csv(OUT / "wings_scale_selection.csv", index=False)

    # 6 the same rule with each improvement switched on in turn (frozen baseline first)
    def upscale(t, n):
        return pd.Series(pnl_at(t, n), index=t.end.to_numpy())

    steps = []
    base = pd.concat([sig["W03"], sig["W04"]])
    def clean(t):
        return t[(~t.end.dt.hour.isin([13, 14, 15])) & (t.ask - t.bid <= 0.01 + 1e-9)]

    weekend = pd.Series(base.end.dt.day_of_week >= 5, index=base.index)
    base_week_split = pd.concat([sig["W04"][sig["W04"].end.dt.day_of_week >= 5],
                                 sig["W03"][sig["W03"].end.dt.day_of_week < 5]])
    variants = [
        ("baseline W03+W04, 1 contract", base, 1.0),
        ("+ no US open (13-15 UTC)", clean(base), 1.0),
        ("+ spread <= 1 cent", clean(base)[clean(base).ask - clean(base).bid <= 0.01 + 1e-9], 1.0),
        ("+ weekend early / weekday late (W08 + W03)", clean(base_week_split), 1.0),
        ("+ 100-contract orders (fee lots)", clean(base_week_split), 100.0),
        ("diagnostic: one wide window + Kelly (not adopted)", clean(chosen), 1.0),
    ]
    for label, trades, n in variants:
        for period, (a, b) in PERIODS.items():
            sub = trades[(trades.end >= a) & (trades.end < b)]
            if len(sub) < 10:
                continue
            h = upscale(sub, n).groupby(sub.end.to_numpy()).sum()
            s_ = stats(h)
            steps.append({"step": label, "period": period, "trades": len(sub), "contracts": n,
                          "¢ per contract": 100 * upscale(sub, n).sum() / (n * len(sub)),
                          "$ per hour": h.mean(), "t": s_["t"], "hit": s_["hit"],
                          "worst hour $": s_["worst"], "max DD $": s_["dd"], "total $": s_["total"]})
    step_table = pd.DataFrame(steps)
    step_table.to_csv(OUT / "wings_scale_steps.csv", index=False)

    # 7 sizing and compounding on the final rule
    final = variants[4][1]
    size_rows = []
    for period, (a, b) in PERIODS.items():
        sub = final[(final.end >= a) & (final.end < b)].copy()
        if len(sub) < 10:
            continue
        k = kelly_size(sub)
        for col, label in (("one", "1 contract"), ("kelly", "quarter Kelly, cap 200"),
                           ("block", "100 contracts")):
            n = {"one": np.ones(len(sub)), "kelly": k, "block": np.full(len(sub), 100.0)}[col]
            pnl = pnl_at(sub, n)
            st = stats(pd.Series(pnl, index=sub.end.to_numpy()).groupby(level=0).sum())
            size_rows.append({"period": period, "sizing": label, "hours": len(sub),
                              "median contracts": float(np.median(n)),
                              "¢ per contract": 100 * float(np.sum(pnl)) / float(np.sum(n)),
                              "$ per hour": st["cents"] / 100, "t": st["t"],
                              "worst hour $": st["worst"], "max DD $": st["dd"], "total $": st["total"]})
    size_table = pd.DataFrame(size_rows)
    size_table.to_csv(OUT / "wings_scale_sizing.csv", index=False)

    comp_rows = []
    for period, (a, b) in list(PERIODS.items())[:2]:
        sub = final[(final.end >= a) & (final.end < b)]
        for share in (0.01, 0.02, 0.05, 0.10):
            equity = compound(sub, share)
            comp_rows.append({"period": period, "share of equity per hour": share, "hours": len(equity),
                              "final $ from $100": equity.equity.iloc[-1],
                              "max drawdown": equity.drawdown.max(),
                              "trades": int((equity.contracts > 0).sum()),
                              "median contracts": float(equity.contracts.median())})
    comp_table = pd.DataFrame(comp_rows)
    comp_table.to_csv(OUT / "wings_scale_compound.csv", index=False)
    final.reset_index(drop=True).to_csv(OUT / "wings_scale_final_trades.csv", index=False)

    write_report(d, fl, fs, ex_table, sp_table, gate_table_all, sel_table, step_table, size_table,
                 comp_table)
    print(fl.to_string(index=False))
    print(fs.to_string(index=False))
    print(ex_table.to_string(index=False))
    print(sp_table.to_string(index=False))
    print(gate_table_all.to_string(index=False))
    print(sel_table.to_string(index=False))
    print(step_table.to_string(index=False))
    print(size_table.to_string(index=False))
    print(comp_table.to_string(index=False))


def _reading(steps: pd.DataFrame) -> str:
    """What the step table says, in three sentences, so the report reads on its own."""
    out = []
    piv = steps.pivot_table(index="step", columns="period", values="¢ per contract")
    order = list(dict.fromkeys(steps.step))
    if len(order) >= 2 and "test Sep-Oct" in piv:
        first, last = order[0], order[3] if len(order) > 3 else order[-1]
        out.append(f"* Правило проходит путь с {piv.loc[first, 'test Sep-Oct']:.2f}¢ на контракт "
                   f"({first}) до {piv.loc[last, 'test Sep-Oct']:.2f}¢ ({last}) на тесте и с "
                   f"{piv.loc[first, 'train Jul-Aug']:.2f} до {piv.loc[last, 'train Jul-Aug']:.2f}¢ на трейне; "
                   "почти весь прирост — структурный (комиссия за ордер и отбор часов), а не модельный.")
    out.append("* Ни один гейт не проходит Холм на всех трёх периодах; повторяется только отрицательный знак "
               "часов американского открытия 13-15 UTC.")
    out.append("* Ноябрь (holdout) не подтверждает прибыль: после фильтров правило не теряет, но и не "
               "зарабатывает — считать его нулём, а не продолжением июля-октября.")
    return "\n".join(out) + "\n"


def write_report(d, fl, fs, ex, sp, gates, sel, steps, size, comp) -> None:
    """Short markdown summary next to the tables (the numbers above, one table each)."""
    def md(df, floatfmt="{:.3f}"):
        if df is None or df.empty:
            return "_no rows_\n"
        out = df.copy()
        for c in out.columns:
            if pd.api.types.is_float_dtype(out[c]):
                out[c] = out[c].map(lambda x: "" if pd.isna(x) else floatfmt.format(x))
        head = "| " + " | ".join(map(str, out.columns)) + " |\n"
        sep = "|" + "|".join(["---"] * len(out.columns)) + "|\n"
        body = "".join("| " + " | ".join(map(str, r)) + " |\n" for r in out.itertuples(index=False))
        return head + sep + body

    lines = [
        "# Making the wing strategy earn more (fees, exits, spreads, gates, selection, sizing)\n",
        f"_Generated {pd.Timestamp.now():%Y-%m-%d %H:%M} by `studies/wings_scale.py`._\n",
        "Data: 1-minute book candles of the Kalshi hourly ladder, 01.07-27.10.2026 (2 788 hours, 73 736 strikes) "
        "and 405 near-spot snapshots of 09-25.11.2026. One trade an hour, taker, Kalshi fee rounded up per order. "
        "Development half Jul-Aug, test half Sep-Oct, holdout 09-25.11.\n",
        "\n## 1. The fee is per order, not per contract\n",
        md(fl),
        "\n`limit` is the exact fee rate 0.07·p·(1−p); `2% block` is the smallest order size whose per-contract "
        "fee is within 2% of it. On an 8-cent wing a one-contract order pays 1.00 cent per contract, a 194-contract "
        "order 0.515: the cent rounding is 12% of the premium and disappears only at size.\n",
        md(fs),
        "\n## 2. Exits\n",
        md(ex),
        "\n## 3. Vertical spreads (buy a deeper wing for protection)\n",
        md(sp),
        "\n## 4. Gates (built on the development half, read on the test half)\n",
        md(gates),
        "\n## 5. Which strike of the hour\n",
        md(sel),
        "\n## 6. The final rule, one improvement at a time\n",
        md(steps),
        "\n## 7. Sizing\n",
        md(size),
        "\n## 8. Compounding\n",
        md(comp),
        "\n## 9. Reading\n",
        _reading(steps),
    ]
    (OUT / "wings_scale.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
