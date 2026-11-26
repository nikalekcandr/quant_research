"""Funding carry as a second sleeve, and how it combines with the wing book (Nov 2026).

The wing book sells cheap binaries and holds to settlement. Its monthly P&L swings sign and it is capacity
bound: even at 100 contracts an hour it earns a few hundred dollars a month. This study asks the two questions
that follow: (1) can funding carry be run as a systematic sleeve with a rule that is decided in advance and
(2) does it pay to hold it next to the wing book, i.e. is the correlation low enough that the combined Sharpe
beats either sleeve alone?

Carry sleeve, decided in advance
    position = long spot / short perpetual, held only when the trailing 9 funding prints (3 days) average more
    than the threshold; the rate is collected at every 8 h print while in position. Entry and exit cost a
    round trip of 12 bp on notional (4 bp spot taker + 2 bp perp taker each side); the sensitivity at 20 bp and
    30 bp is reported. No basis P&L is assumed: the perpetual is assumed to track spot, which is exactly the
    assumption that makes this a funding sleeve and not a basis trade. Basis risk is stated, not modelled.

Wing sleeve
    the final rule of `wings_scale.py` (weekend early / weekday late, no US open, hold to settlement),
    quarter Kelly with a 200-contract cap, at 1 contract = 1 dollar of payoff.

Outputs: results/studies/carry_portfolio.md and _funding.csv, _sleeves.csv, _portfolio.csv, _predict.csv.
Run: python -m studies.carry_portfolio
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OUT = ROOT / "results/studies"
FUNDING = ROOT / "data/raw/funding/btc_funding.csv.gz"
THRESHOLDS = [0.0, 0.00005, 0.00010, 0.00020]      # per 8 h print
COSTS = [0.0006, 0.0010, 0.0015]                   # round trip on notional
CAPITAL = 10_000.0


def load_funding() -> pd.DataFrame:
    f = pd.read_csv(FUNDING)
    f["ts"] = pd.to_datetime(f.ts, unit="ms", utc=True)
    f = f[f.venue == "binance"].sort_values("ts").reset_index(drop=True)
    f["trail"] = f.rate.rolling(9, min_periods=5).mean().shift(1)   # decided before the print it is paid at
    f["month"] = f.ts.dt.to_period("M").astype(str)
    return f


def carry(f: pd.DataFrame, threshold: float, cost: float) -> pd.DataFrame:
    """Daily P&L of the carry sleeve; `cost` is charged on every position change."""
    x = f.copy()
    x["on"] = (x.trail > threshold).fillna(False)
    x["switch"] = x.on.ne(x.on.shift(1)).astype(float)
    x["pnl"] = np.where(x.on, x.rate, 0.0) - x.switch * cost / 2.0
    x["day"] = x.ts.dt.floor("D")
    daily = x.groupby("day").agg(pnl=("pnl", "sum"), rate=("rate", "mean"), on=("on", "mean"))
    return daily


def metrics(daily: pd.DataFrame) -> dict:
    p = daily.pnl
    ann = p.mean() * 365
    sd = p.std(ddof=1) * math.sqrt(365)
    eq = p.cumsum()
    dd = float((eq.cummax() - eq).max())
    return dict(annual_return=ann, annual_vol=sd, sharpe=ann / sd if sd > 0 else np.nan,
                max_drawdown=dd, days=len(p), share_in_market=float(daily.on.mean()),
                total=p.sum())


def wings_daily() -> pd.DataFrame:
    """The final wing rule of `wings_scale.py`, quarter Kelly, aggregated to daily dollar P&L."""
    path = OUT / "wings_scale_final_trades.csv"
    if not path.exists():
        return pd.DataFrame()
    from studies.wing_panel import fee
    from studies.wings_scale import kelly_size
    t = pd.read_csv(path)
    t["price"] = t.price if "price" in t else t.wing_bid
    n = kelly_size(t).to_numpy()
    # fee() returns the fee for the whole order, so keep the price term per contract and the fee per order
    pnl = (t.wing_bid.to_numpy() - t.won.to_numpy()) * n - fee(n, t.price.to_numpy())
    t["day"] = pd.to_datetime(t.end, utc=True).dt.floor("D")
    return pd.DataFrame({"pnl": pnl}, index=t.day.to_numpy()).groupby(level=0).sum()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    f = load_funding()

    # 1 the funding level itself, by month
    hist = (f.groupby("month").agg(mean_bp=("rate", lambda s: 1e4 * s.mean()),
                                   median_bp=("rate", lambda s: 1e4 * s.median()),
                                   share_above_05bp=("rate", lambda s: float((s > 0.00005).mean())),
                                   share_above_10bp=("rate", lambda s: float((s > 0.00010).mean())),
                                   share_negative=("rate", lambda s: float((s < 0).mean())),
                                   prints=("rate", "size")))
    hist["annualised_at_median"] = hist.median_bp * 1e-4 * 3 * 365
    hist.to_csv(OUT / "carry_portfolio_funding.csv")

    # 2 is the trailing average informative about the next print?
    f["next_rate"] = f.rate
    predict = []
    for lag, label in ((1, "trailing 3 d"), (3, "trailing 7 d"), (9, "trailing 3 d, 1 d gap")):
        x = f.dropna(subset=["trail"])
        col = f.rate.rolling(3 * 8 if lag <= 3 else 9, min_periods=5).mean().shift(lag)
        ok = col.notna()
        corr = float(np.corrcoef(col[ok], f.rate[ok])[0, 1])
        sign = float((np.sign(col[ok]) == np.sign(f.rate[ok])).mean())
        predict.append({"predictor": label, "corr with next print": corr,
                        "same sign share": sign, "prints": int(ok.sum())})
    predict_t = pd.DataFrame(predict)
    predict_t.to_csv(OUT / "carry_portfolio_predict.csv", index=False)

    # 3 the sleeve at several thresholds and cost assumptions
    sleeves = []
    for thr in THRESHOLDS:
        for cost in COSTS:
            daily = carry(f, thr, cost)
            for window, (a, b) in {"2023-2024": ("2023-01-01", "2025-01-01"),
                                   "2025": ("2025-01-01", "2026-01-01"),
                                   "2026 to Oct": ("2026-01-01", "2026-11-01"),
                                   "full history": ("2023-01-01", "2026-11-01")}.items():
                d = daily.loc[a:b]
                if len(d) < 30:
                    continue
                m = metrics(d)
                sleeves.append({"threshold bp per 8h": thr * 1e4, "round trip bp": cost * 1e4,
                                "window": window, **m})
    sleeve_t = pd.DataFrame(sleeves)
    sleeve_t.to_csv(OUT / "carry_portfolio_sleeves.csv", index=False)

    # 4 the two sleeves side by side on the months they share
    chosen = carry(f, 0.00005, 0.0010)
    w = wings_daily()
    port_rows = []
    if not w.empty:
        both = pd.concat([chosen.pnl.rename("carry"), w.pnl.rename("wings")], axis=1).fillna(0.0)
        both["month"] = both.index.to_period("M").astype(str)
        mon = both.groupby("month").sum()
        mon["corr"] = np.nan
        for m in mon.index:
            d = both[both.month == m]
            if d.carry.std() > 0 and d.wings.std() > 0:
                mon.loc[m, "corr"] = float(np.corrcoef(d.carry, d.wings)[0, 1])
        mon.to_csv(OUT / "carry_portfolio_months.csv")
        # combinations over the shared window, scaled to a $10k account
        shared = both.loc["2026-07-01":"2026-10-31"]
        shared = shared[(shared.wings != 0) | (shared.carry != 0)]
        if len(shared) > 30:
            for name, wt in (("carry only", 1.0), ("wings only", 0.0), ("50/50", 0.5),
                             ("30/70 carry/wings", 0.3)):
                # carry is a fraction of notional, the wing book is already in dollars (capacity sizing)
                p = wt * shared.carry + (1 - wt) * shared.wings / CAPITAL
                port_rows.append({"combination": name, "carry weight": wt,
                                  "annual return on $10k": p.mean() * 365 * CAPITAL,
                                  "annual vol on $10k": p.std(ddof=1) * math.sqrt(365) * CAPITAL,
                                  "sharpe": (p.mean() * 365) / (p.std(ddof=1) * math.sqrt(365))
                                  if p.std(ddof=1) > 0 else np.nan,
                                  "correlation": float(np.corrcoef(shared.carry, shared.wings)[0, 1])
                                  if shared.carry.std() > 0 else np.nan,
                                  "t of daily mean": float(p.mean() / (p.std(ddof=1) / math.sqrt(len(p))))
                                  if p.std(ddof=1) > 0 else np.nan,
                                  "days": len(p),
                                  "total $ on $10k": p.sum() * CAPITAL})
    port_t = pd.DataFrame(port_rows)
    port_t.to_csv(OUT / "carry_portfolio_portfolio.csv", index=False)

    write_report(hist, predict_t, sleeve_t, port_t)
    print(hist.tail(12).to_string())
    print(predict_t.to_string(index=False))
    print(sleeve_t[sleeve_t.window == "full history"].to_string(index=False))
    print(port_t.to_string(index=False))


def write_report(hist, predict_t, sleeve_t, port_t) -> None:
    def md(df, fmt="{:.4f}"):
        if df is None or df.empty:
            return "_no rows_\n"
        out = df.copy()
        for c in out.columns:
            if pd.api.types.is_float_dtype(out[c]):
                out[c] = out[c].map(lambda x: "" if pd.isna(x) else fmt.format(x))
        head = "| " + " | ".join(map(str, out.columns)) + " |\n"
        sep = "|" + "|".join(["---"] * len(out.columns)) + "|\n"
        body = "".join("| " + " | ".join(map(str, r)) + " |\n" for r in out.itertuples(index=False))
        return head + sep + body

    lines = [
        "# Кэрри на фандинге как второе плечо и его сочетание с крылом\n",
        f"_Сгенерировано {pd.Timestamp.now():%Y-%m-%d %H:%M}, `studies/carry_portfolio.py`._\n",
        "Фандинг: `data/raw/funding/btc_funding.csv.gz`, Binance BTCUSDT, только фактические печати "
        "(каждые 8 часов), история 2023-01-01…2026-10-28. Правило входа решено заранее: держим "
        "спот-лонг/перп-шорт, когда среднее предыдущих 9 печатей выше порога, и платим 12 б.п. на разворот "
        "позиции; ниже — чувствительность к 20 и 30 б.п. Базис не моделируется (перп считается идущим за "
        "спотом) — это и есть риск этого плеча, и он назван, а не спрятан.\n",
        "\n## Уровень фандинга по месяцам\n",
        md(hist, "{:.2f}"),
        "\n## Предсказуемость: говорит ли прошлое среднее о следующей печати\n",
        md(predict_t, "{:.3f}"),
        "\n## Плечо кэрри: пороги и издержки\n",
        md(sleeve_t, "{:.4f}"),
        "\n## Два плеча вместе (общие месяцы, счёт $10 000)\n",
        md(port_t, "{:.2f}"),
    ]
    (OUT / "carry_portfolio.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
