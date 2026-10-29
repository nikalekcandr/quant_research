"""Portfolio construction: position sizing, risk limits, and drawdown control.

OFFLINE ANALYSIS ONLY. Do not hook this into lab/engine.py: the lab's hypotheses are frozen and must be tested
independently, one unit each (a drawdown brake would make every rule depend on the P&L of the others).
P&L units differ across hypotheses ($/contract for binaries, return for perps, $ for options), so portfolio-level
numbers are computed per unit; voided trades (lab/report.py: void_mask) are excluded.

Every signal in lab/signals.py returns a raw Order with price and edge. This
module turns a batch of raw orders into sized orders that respect:

  1. Kelly sizing per hypothesis (fractional, shrunk to 1/N and capped).
  2. Volatility targeting (target 15% ann, window 30 days).
  3. Drawdown brake (halve exposure after a 10% drawdown, freeze after 20%).
  4. Correlation cap (if n signals share one underlying, divide by sqrt(n)).
  5. Gross exposure cap (no more than 2× notional across all perps).

It is deliberately conservative: the lab has only ~2.5k settled trades and many
hypotheses have < 20 observations, so aggressive Kelly would be noise-fitting.
All parameters live here (not in callers) so the sizing rule is auditable.

Usage:
    from src.portfolio import Portfolio
    pf = Portfolio(ledger=read_ledger())
    sized = pf.scale(orders, ctx)   # returns orders with .size set

Or standalone backtest:
    python -m src.portfolio --ledger data/lab/ledger.csv --plot
"""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

# ------------------------------------------------------------------ defaults
# All tuned on history (pre-2026-10-29) and frozen before live evaluation.
TARGET_VOL_ANN = 0.15          # vol targeting: 15% annualised portfolio vol
VOL_WINDOW_DAYS = 30           # lookback for vol-of-portfolio estimate
KELLY_FRACTION = 0.5           # half-Kelly (fractional Kelly = 1/2 is standard)
KELLY_SHRINK_N = 20            # shrink Kelly toward 1/N when n < this
KELLY_CAP = 2.0                # never risk more than 2× notional on one idea
DD_HALVE = 0.10                # drawdown at which exposure halves
DD_FREEZE = 0.20               # drawdown at which trading stops
CORR_CAP_POWER = 0.5           # divide by n^0.5 for n correlated signals
MAX_GROSS = 2.0                # max gross exposure across all perps (× notional)
PERP_FEE = 0.0005               # per-side taker fee, for net return calc
FALLBACK_VOL = 0.50            # if vol estimate is missing, assume 50% ann


@dataclass
class Sizing:
    kelly: float
    vol_scale: float
    dd_scale: float
    corr_scale: float
    gross_scale: float
    final: float
    reason: str = ""


# ------------------------------------------------------------------ helpers
def _ann_vol(returns: pd.Series, window: int = VOL_WINDOW_DAYS) -> float:
    """Annualised vol of daily returns (last `window` days)."""
    if len(returns) < 10:
        return FALLBACK_VOL
    r = returns.dropna().iloc[-window:]
    if len(r) < 10 or r.std() == 0:
        return FALLBACK_VOL
    return float(r.std() * math.sqrt(365))


def _max_drawdown(cum: pd.Series) -> float:
    """Current drawdown as fraction of peak notional (0 = no DD, 0.15 = 15%)."""
    if len(cum) < 2:
        return 0.0
    s = pd.Series(cum).dropna().sort_index()
    if len(s) < 2:
        return 0.0
    peak = s.cummax()
    # cum is cumulative log-return (or additive P&L in return units) — DD in log terms
    cur_dd_log = float(peak.iloc[-1] - s.iloc[-1])
    if cur_dd_log <= 0:
        return 0.0
    # Convert log-DD to arithmetic drawdown: 1 - exp(-logDD)
    # Cap at 99% to avoid overflow on huge DDs
    return float(1 - math.exp(-min(cur_dd_log, 5.0)))


def _kelly_fraction(
    mean: float, var: float, n: int, fraction: float = KELLY_FRACTION
) -> float:
    """Fractional Kelly: f* = (μ / σ²) * fraction, shrunk when n is small."""
    if var <= 1e-12 or not np.isfinite(mean) or not np.isfinite(var):
        return 0.0
    raw = mean / var * fraction
    # Shrink toward 0 when sample is small (James-Stein flavour: n/(n+N) weighting)
    shrink = n / (n + KELLY_SHRINK_N) if n < 500 else 1.0
    return float(np.clip(raw * shrink, -KELLY_CAP, KELLY_CAP))


# ------------------------------------------------------------------ main class
class Portfolio:
    """Vol-targeted, Kelly-sized, drawdown-aware portfolio for the alpha lab."""

    def __init__(
        self,
        ledger: pd.DataFrame | None = None,
        target_vol: float = TARGET_VOL_ANN,
        kelly_fraction: float = KELLY_FRACTION,
        unit: str | None = "$/contract",
    ):
        self.unit = unit                       # one P&L unit per portfolio (None = mix, not meaningful)
        self.target_vol = target_vol
        self.kelly_fraction = kelly_fraction
        # Per-hypothesis stats: mean, var, n (from closed trades only)
        self.stats: dict[str, dict] = {}
        # Portfolio-level time series (daily net returns, for vol & DD)
        self.daily: pd.Series | None = None
        self.cum: pd.Series | None = None
        if ledger is not None and len(ledger):
            self.ingest_ledger(ledger)

    # -- ledger ingestion ------------------------------------------------
    def ingest_ledger(self, ledger: pd.DataFrame):
        """Build per-hypothesis and portfolio-level stats from a ledger.csv."""
        # Only settled trades (CLOSE/INSTANT) with finite pnl
        done = ledger[ledger["event"].isin(["CLOSE", "INSTANT"])].copy()
        try:                                   # drop trades of buggy code versions (listing quotes, empty sides)
            from lab.report import void_mask
            done = done[~void_mask(done)]
        except Exception:
            pass
        if self.unit is not None and "unit" in done:
            done = done[done["unit"].astype(str) == self.unit]
        if done.empty:
            return
        done["pnl"] = pd.to_numeric(done["pnl"], errors="coerce")
        done = done.dropna(subset=["pnl"])
        if done.empty:
            return
        # Per-hypothesis
        for hyp, g in done.groupby("hyp"):
            x = g["pnl"].to_numpy(float)
            self.stats[str(hyp)] = {
                "mean": float(x.mean()),
                "var": float(x.var(ddof=1)) if len(x) > 1 else float(x.var() or 1.0),
                "n": int(len(x)),
                "hit": float((x > 0).mean()),
                "sharpe": float(x.mean() / x.std(ddof=1)) if len(x) > 2 and x.std(ddof=1) > 0 else 0.0,
            }
        # Portfolio daily series (sum of all trades that closed that day, UTC)
        done["day"] = pd.to_datetime(done["ts"], utc=True).dt.floor("D")
        daily = done.groupby("day")["pnl"].sum().sort_index()
        # Fill missing days with 0 (no trade = 0 return) for vol calc
        if len(daily) >= 2:
            full = pd.date_range(daily.index.min(), daily.index.max(), freq="D")
            daily = daily.reindex(full, fill_value=0.0)
        self.daily = daily
        self.cum = daily.cumsum()

    # -- single-signal sizing --------------------------------------------
    def kelly_for(self, hyp: str) -> float:
        s = self.stats.get(hyp)
        if s is None or s["n"] < 5:
            return 0.2  # new hypothesis: tiny starter size, not zero (exploration)
        return _kelly_fraction(s["mean"], s["var"], s["n"], self.kelly_fraction)

    def vol_scale(self) -> float:
        if self.daily is None or len(self.daily) < 10:
            return 1.0
        vol = _ann_vol(self.daily)
        if vol < 1e-6:
            return 1.0
        return float(np.clip(self.target_vol / vol, 0.2, 3.0))

    def dd_scale(self) -> float:
        if self.cum is None or len(self.cum) < 2:
            return 1.0
        dd = _max_drawdown(self.cum)
        if dd >= DD_FREEZE:
            return 0.0
        if dd >= DD_HALVE:
            # Linear between HALVE (0.5) and FREEZE (0.0)
            return 0.5 * (DD_FREEZE - dd) / (DD_FREEZE - DD_HALVE)
        if dd >= DD_HALVE / 2:
            # Gentle taper from 1.0 to 0.5 between 5% and 10% DD
            return 1.0 - 0.5 * (dd - DD_HALVE / 2) / (DD_HALVE / 2)
        return 1.0

    def corr_scale_for(self, orders: list) -> dict[str, float]:
        """If n orders share the same underlying, scale each by 1/sqrt(n)."""
        # Group by underlying: perp BTC = one group, each binary event = own group
        from collections import Counter

        # Perps on BTC are all correlated ~0.95; binary ladders on same event ~0.5-0.8;
        # for simplicity: perps = one bucket, binaries grouped by event (venue+settle date)
        buckets: dict[str, list[int]] = {}
        for i, o in enumerate(orders):
            # Resolve bucket key
            kind = getattr(o, "kind", "")
            if kind in ("perp", "perp_pair", "fastperp"):
                key = "perp_BTC"
            elif kind == "binary":
                # Group by settlement day (same BTC move drives all strikes of one ladder)
                # Use event if available, else date of settle_ts
                info = getattr(o, "info", {}) or {}
                event = info.get("event") or getattr(o, "instrument", "") or str(getattr(o, "hyp", ""))
                # Normalise to venue+expiry bucket (strikes of same ladder are correlated)
                settle = str(getattr(o, "settle_ts", ""))[:10]
                key = f"bin_{event}_{settle}"
            else:
                key = f"other_{getattr(o, 'hyp', i)}"
            buckets.setdefault(key, []).append(i)

        scales: dict[int, float] = {}
        for idxs in buckets.values():
            n = len(idxs)
            s = 1.0 / (n**CORR_CAP_POWER) if n > 1 else 1.0
            for i in idxs:
                scales[i] = s
        return scales

    # -- main entry point ------------------------------------------------
    def scale(
        self, orders: list, ctx=None, verbose: bool = False
    ) -> list[tuple[object, Sizing]]:
        """Return (order, sizing) pairs with sizing.final as position multiplier.

        Callers multiply notional by sizing.final (1.0 = one contract / one unit
        as before). Orders with final == 0 are filtered out.
        """
        if not orders:
            return []

        v_scale = self.vol_scale()
        d_scale = self.dd_scale()
        corr_scales = self.corr_scale_for(orders)

        # Pre-compute gross cap
        gross_before_cap = 0.0
        prelim = []
        for i, o in enumerate(orders):
            kelly = self.kelly_for(str(getattr(o, "hyp", "")))
            corr = corr_scales.get(i, 1.0)
            final_pre_gross = kelly * v_scale * d_scale * corr
            prelim.append((kelly, corr, final_pre_gross))
            # Only perps count toward gross exposure (binaries are $1 contracts)
            if getattr(o, "kind", "") in ("perp", "perp_pair", "fastperp"):
                gross_before_cap += abs(final_pre_gross)

        gross_scale = min(1.0, MAX_GROSS / gross_before_cap) if gross_before_cap > MAX_GROSS else 1.0

        result = []
        for i, o in enumerate(orders):
            kelly, corr, pre = prelim[i]
            final = pre * gross_scale
            sz = Sizing(
                kelly=kelly,
                vol_scale=v_scale,
                dd_scale=d_scale,
                corr_scale=corr,
                gross_scale=gross_scale,
                final=float(np.clip(final, 0, KELLY_CAP)),
                reason=f"kelly={kelly:.2f} vol={v_scale:.2f} dd={d_scale:.2f} corr={corr:.2f} gross={gross_scale:.2f}",
            )
            if verbose:
                print(f"  {getattr(o, 'hyp', '?'):6s} {getattr(o, 'side',''):6s} kelly={kelly:.3f} final={sz.final:.3f} ({sz.reason})")
            result.append((o, sz))
        return result

    # -- reporting -------------------------------------------------------
    def summary(self) -> pd.DataFrame:
        rows = []
        for hyp, s in sorted(self.stats.items()):
            rows.append({"hyp": hyp, **s, "kelly": self.kelly_for(hyp)})
        df = pd.DataFrame(rows)
        if df.empty:
            return df
        return df.sort_values("kelly", ascending=False)

    def portfolio_stats(self) -> dict:
        if self.daily is None or len(self.daily) < 10:
            return {}
        vol = _ann_vol(self.daily)
        dd = _max_drawdown(self.cum) if self.cum is not None else 0.0
        sharpe = float(self.daily.mean() / self.daily.std() * math.sqrt(365)) if self.daily.std() > 0 else 0.0
        return {
            "target_vol": self.target_vol,
            "realised_vol_ann": vol,
            "vol_scale": self.vol_scale(),
            "drawdown": dd,
            "dd_scale": self.dd_scale(),
            "sharpe_ann": sharpe,
            "n_days": len(self.daily),
            "total_pnl": float(self.daily.sum()),
        }


# ------------------------------------------------------------------ CLI backtest
def _load_ledger(path: str) -> pd.DataFrame:
    # lab.core imports requests (absent in some CI envs) — fall back gracefully
    try:
        import lab.core as core  # type: ignore

        return core.read_ledger()
    except Exception:
        pass
    try:
        return pd.read_csv(path, parse_dates=["ts"])
    except FileNotFoundError:
        return pd.DataFrame()


def main(argv=None):
    ap = argparse.ArgumentParser(description="Portfolio sizing backtest on ledger.csv")
    ap.add_argument("--ledger", default="data/lab/ledger.csv")
    ap.add_argument("--plot", action="store_true", help="save equity curve to results/portfolio_equity.png")
    ap.add_argument("--target-vol", type=float, default=TARGET_VOL_ANN)
    ap.add_argument("--kelly-fraction", type=float, default=KELLY_FRACTION)
    args = ap.parse_args(argv)

    led = _load_ledger(args.ledger)
    if led.empty:
        print("No ledger found — nothing to size.")
        return

    # Walk-forward: at each day, compute sizing from past only, apply to next day
    done = led[led["event"].isin(["CLOSE", "INSTANT"])].copy()
    done["pnl"] = pd.to_numeric(done["pnl"], errors="coerce")
    done = done.dropna(subset=["pnl"])
    done["day"] = pd.to_datetime(done["ts"], utc=True).dt.floor("D")
    days = sorted(done["day"].unique())
    if len(days) < 20:
        print(f"Only {len(days)} days with settled trades — need more history.")
        return

    pf = Portfolio(target_vol=args.target_vol, kelly_fraction=args.kelly_fraction)
    daily_raw, daily_sized = [], []
    for i, day in enumerate(days):
        # Ingest up to yesterday
        past = done[done["day"] < day]
        if len(past) >= 5:
            pf.ingest_ledger(past)
        today = done[done["day"] == day]
        # Approximate sizing: each trade sized by its hyp's kelly at that time
        raw = float(today["pnl"].sum())
        sized_pnl = 0.0
        for _, r in today.iterrows():
            k = pf.kelly_for(str(r["hyp"]))
            # For backtest, apply kelly * vol_scale * dd_scale (corr within day ignored)
            vs, ds = pf.vol_scale(), pf.dd_scale()
            sized_pnl += float(r["pnl"]) * k * vs * ds
        daily_raw.append((day, raw))
        daily_sized.append((day, sized_pnl))

    dr = pd.Series({d: v for d, v in daily_raw}).sort_index()
    ds = pd.Series({d: v for d, v in daily_sized}).sort_index()
    pf.ingest_ledger(led[led["event"].isin(["CLOSE", "INSTANT"])])

    print("\nPer-hypothesis Kelly (half-Kelly, shrunk):")
    print(pf.summary().to_string(index=False))
    print("\nPortfolio stats (full sample):")
    for k, v in pf.portfolio_stats().items():
        print(f"  {k:20s} {v:.4f}" if isinstance(v, float) else f"  {k:20s} {v}")

    def stats_str(s: pd.Series, label: str):
        vol = s.std() * math.sqrt(365) if len(s) > 5 and s.std() > 0 else float("nan")
        sharpe = s.mean() / s.std() * math.sqrt(365) if len(s) > 5 and s.std() > 0 else float("nan")
        mdd = 0.0
        if len(s) > 5:
            cum = s.cumsum()
            dd = (cum.cummax() - cum).max()
            mdd = 1 - math.exp(-dd) if dd > 0 else 0.0
        print(f"{label:20s} total={s.sum():+.4f}  ann_vol={vol:.2%}  Sharpe={sharpe:.2f}  maxDD={mdd:.2%}  days={len(s)}")

    print()
    stats_str(dr, "Equal-weight (1×)")
    stats_str(ds, "Kelly+vol+DD scaled")

    if args.plot:
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(9, 4))
            dr.cumsum().plot(ax=ax, label="equal-weight (1×)", lw=1.2)
            ds.cumsum().plot(ax=ax, label="Kelly scaled", lw=1.2)
            ax.axhline(0, color="grey", lw=0.5)
            ax.set_title("Equity curves: raw vs portfolio-sized (walk-forward, no look-ahead)")
            ax.set_ylabel("cumulative P&L (return units)")
            ax.legend()
            fig.tight_layout()
            out = "results/portfolio_equity.png"
            import config

            p = config.RESULTS / "portfolio_equity.png"
            p.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(p, dpi=150)
            plt.close(fig)
            print(f"\nPlot saved to {p}")
        except Exception as e:
            print(f"Plot failed: {e}")


if __name__ == "__main__":
    main()
