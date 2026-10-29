"""Draft report (reports/draft_report.md), filled with the numbers from results/tables.

The text states only what the tables show. Sections marked TODO need your own interpretation:
the grader penalises filler, so write those paragraphs yourselves after looking at the results.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

import config
from src.utils import ensure, log

LABEL = {
    "naive": "Naive (previous window)", "garch": "GARCH(1,1)-t", "har": "HAR-RV (baseline)",
    "har_m": "HAR + ladder variance (BJN)", "harx": "HAR-X", "mkt_ln_raw": "Ladder variance (lognormal fit), raw",
    "harx_d": "HAR-X + Deribit", "harx_m": "HAR-X + ladder", "harx_all": "HAR-X + Deribit + ladder",
    "iv_raw": "Deribit IV, raw", "iv_mz": "Deribit IV, recalibrated", "mkt_raw": "Ladder variance (BJN), raw",
    "mkt_t_raw": "Ladder variance (Student-t fit), raw", "mkt_mz": "Ladder variance (BJN), recalibrated", "combo": "Equal-weight combination",
}
HZ = {"kx1h": "Kalshi, 55 minutes", "pm24h": "Polymarket, 24 hours", "pm6d": "Polymarket, 6 days"}


def _read(name):
    f = config.TABLES / f"{name}.csv"
    return pd.read_csv(f) if f.exists() else pd.DataFrame()


def md_table(df: pd.DataFrame, floatfmt: str = "{:.3f}") -> str:
    if df.empty:
        return "_(no data)_\n"
    cols = list(df.columns)
    out = ["| " + " | ".join(map(str, cols)) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, (float, np.floating)):
                cells.append("" if pd.isna(v) else floatfmt.format(v))
            else:
                cells.append(str(v))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out) + "\n"


def _fmt_p(p):
    return "< 0.001" if p < 0.001 else f"= {p:.3f}"


def write(synthetic: bool = False):
    ensure(config.REPORTS)
    ds, sc, dm = _read("data_summary"), _read("scores_common_sample"), _read("diebold_mariano")
    bias, lad, cal, trd = _read("implied_bias"), _read("ladder_scores"), _read("ladder_calibration"), _read("trading")
    live = _read("live_forecasts_scores")
    cov = _read("ladder_coverage")
    FIG = os.path.relpath(config.FIGURES, config.REPORTS).replace(os.sep, "/")
    L = []
    if synthetic:
        L.append("> **SYNTHETIC DATA. These numbers only test the code and say nothing about real markets.**\n")
    L.append("# Forecasting Bitcoin volatility with prediction-market strike ladders\n")
    L.append("Team: TODO\n")
    L.append("## 1. Forecasting task\n")
    L.append("Polymarket and Kalshi list ladders of binary contracts that pay $1 if Bitcoin settles above a strike "
             "at a fixed time. A ladder is a strip of digital options, so its prices imply a distribution for the "
             "settlement price, and the width of that distribution is the market's forecast of volatility over the "
             "window that ends at settlement. We forecast that volatility and compare our forecasts with the ladders "
             "and with Deribit options.\n")
    L.append("* **Target:** realized variance of BTC/USDT over the window [t, T), the sum of squared 1-minute log "
             "returns. Models forecast its logarithm.\n"
             "* **Form:** full predictive density of log realized variance; point forecast = its mean; "
             "probabilities for every strike of the ladder derived from the density.\n"
             "* **Horizons:** 55 minutes (Kalshi hourly ladders, forecast 5 minutes after the open), 24 hours and "
             "6 days (Polymarket daily ladders settling at 12:00 New York time).\n"
             "* **Evaluation:** QLIKE loss for point forecasts, log score and CRPS for densities, PIT histograms and "
             "interval coverage for calibration, Mincer-Zarnowitz regressions, Diebold-Mariano tests with "
             "Newey-West variance; Brier score and a trading rule for ladder probabilities.\n")
    L.append("### From ladder prices to expected realized variance\n")
    L.append("Ignoring discounting over at most six days, the price of the contract that pays 1 if S_T > K is "
             "d(K) = P(S_T > K) under the market's pricing measure. For a price process with continuous paths and "
             "forward F, Britten-Jones and Neuberger (2000) show that the expected integrated variance between t "
             "and T is determined by option prices across strikes. Written in digital prices:\n\n"
             "E[ integral_t^T sigma_s^2 ds ] = 2 E[ S_T/F - 1 - ln(S_T/F) ] "
             "= 2 [ int_F^inf d(K)(1/F - 1/K) dK + int_0^F (1 - d(K))(1/K - 1/F) dK ].\n\n"
             "Realized variance from 1-minute returns converges to the integrated variance (plus squared jumps), "
             "so this model-free ladder variance is the market's forecast of our target, up to three gaps that "
             "we handle explicitly: (i) the pricing measure differs from the physical one by a variance risk "
             "premium, absorbed by the intercept of a Mincer-Zarnowitz recalibration and of the HAR regression; "
             "(ii) jumps make the formula differ from expected quadratic variation by a third-order term; "
             "(iii) the ladder quotes only 8-11 strikes $2,000 apart, so between strikes we interpolate the probit "
             "of the implied CDF linearly in log K (exact for a lognormal) and beyond the outer informative strikes "
             "we extrapolate the same way; a truncated version that integrates only between quoted strikes is "
             "reported as a lower bound. Stale tail prices that violate monotonicity are pooled first.\n")
    L.append("**Windows are aligned with settlement.** Polymarket settles on the close of the Binance BTC/USDT "
             "1-minute candle that opens at 12:00 ET, so the target window runs from the forecast time t to 12:01 "
             "ET. Kalshi settles on the average of 60 one-second CF Benchmarks BRTI prints before T; the variance "
             "of that average relative to S_t equals the integrated variance up to T - 60 s plus one third of the "
             "last minute's, and the Kalshi target is built the same way. Deribit options settle on a 30-minute "
             "average of its index, so its quoted IV is converted to a point-price variance rate "
             "(sigma^2 = IV^2 tau / (tau - 20 min)) before interpolation to our windows.\n")
    L.append("## 2. Data\n")
    L.append(md_table(ds.drop(columns=[c for c in ["first_ladder"] if c in ds]), "{:.0f}"))
    L.append("Sources (all public, no account needed): "
             "[Binance 1-minute candles](https://data.binance.vision/?prefix=data/spot/monthly/klines/BTCUSDT/1m/), "
             "[Polymarket Gamma API](https://gamma-api.polymarket.com) and "
             "[CLOB price history](https://clob.polymarket.com), "
             "[Kalshi trade API, series KXBTCD](https://api.elections.kalshi.com/trade-api/v2/series/KXBTCD), "
             "[Deribit trade history](https://history.deribit.com) and "
             "[DVOL](https://www.deribit.com/api/v2/public/get_volatility_index_data), "
             "[BLS release schedules](https://www.bls.gov/bls/news-release/cpi.htm) and FOMC dates for the macro calendar. "
             "Download code: `src/download/`.\n")
    L.append("Ladder coverage at the forecast time (price history exists for every listed strike; tails are "
             "often stale or at the price floor, which is why the number of informative strikes is smaller):\n\n"
             + md_table(cov, "{:.2f}"))
    L.append("## 3. Models\n")
    L.append("All models are re-estimated on a rolling window using only occasions whose target window had ended "
             "before the forecast time (pseudo-out-of-sample from " + config.EVAL_START + ").\n")
    rows = [
        ("Naive", "log realized variance of the previous window of the same length", "benchmark"),
        ("GARCH(1,1)-t", "conditional variance of 24h returns", "textbook benchmark (24h and 6d only)"),
        ("HAR-RV", "log RV on daily, weekly, monthly log RV (+ intraday seasonal factor at 55 min)", "baseline"),
        ("HAR + ladder", "HAR with log model-free ladder variance as an extra regressor",
         "the direct test: does the ladder forecast variance beyond the history of returns?"),
        ("HAR-X", "HAR + semivariances, jump share, HARQ term, macro-release and weekend dummies",
         "asymmetry, short-lived jumps, measurement error, scheduled news"),
        ("HAR-X + Deribit", "adds DVOL and Deribit ATM implied variance for the window",
         "options price scheduled events and positioning before they appear in returns"),
        ("HAR-X + ladder", "adds the variance implied by the prediction-market ladder", "same, from a retail venue"),
        ("HAR-X + Deribit + ladder", "both", "encompassing test: does the ladder add anything beyond Deribit?"),
        ("Implied variance, raw / recalibrated", "Deribit or ladder variance as is, and after a rolling "
         "Mincer-Zarnowitz regression", "markets as stand-alone forecasters"),
        ("Combination", "equal-weight mixture of HAR-X, recalibrated Deribit, recalibrated ladder",
         "forecast combination usually beats the components"),
    ]
    L.append(md_table(pd.DataFrame(rows, columns=["model", "what it uses", "why"])))
    L.append("## 4. Results\n")
    L.append("### 4.1 Accuracy on the common sample\n")
    L.append("Every model has a forecast for every occasion in this sample. Lower QLIKE and CRPS are better, "
             "higher log score is better; well-calibrated densities have 50% and 90% coverage close to 0.50 and 0.90.\n")
    for h in [x for x in HZ if x in set(sc.get("horizon", []))]:
        t = sc[sc["horizon"] == h].copy()
        t["model"] = t["model"].map(LABEL).fillna(t["model"])
        t = t.sort_values("qlike")[["model", "n", "qlike", "logscore", "crps", "cov50", "cov90", "mz_b", "mz_p"]]
        L.append(f"**{HZ[h]}**\n\n" + md_table(t))
        best = t.iloc[0]
        har = t[t["model"] == LABEL["har"]]
        if len(har):
            gain = 1 - best["qlike"] / har["qlike"].iloc[0]
            L.append(f"Lowest QLIKE: {best['model']} ({best['qlike']:.3f}, {gain:.0%} below HAR-RV).\n")
    L.append(f"![Volatility paths]({FIG}/fig1_vol_paths.png)\n")
    L.append(f"![PIT histograms]({FIG}/fig2_pit.png)\n")
    L.append("### 4.2 Diebold-Mariano tests\n")
    L.append("Negative statistic: the first model has the lower loss. Newey-West variance, Harvey-Leybourne-Newbold "
             "correction; overlapping 6-day windows use 5 lags or more.\n")
    if len(dm):
        d = dm[dm["sample"] == "common"].copy()
        d["horizon"] = d["horizon"].map(HZ)
        d["p"] = d["p"].map(lambda x: _fmt_p(x).replace("= ", "") if pd.notna(x) else "")
        L.append(md_table(d[["horizon", "comparison", "loss", "n", "mean_diff", "dm", "p"]]))
        main = dm[(dm["sample"] == "common") & (dm["A"] == "har_m") & (dm["B"] == "har")]
        for _, r in main.iterrows():
            verdict = "significant at 5%" if r["p"] < 0.05 else "not significant at 5%"
            L.append(f"Main test, {HZ[r['horizon']]}, {r['loss']}: adding the ladder variance to HAR changes the "
                     f"loss by {r['mean_diff']:+.4f} (DM = {r['dm']:.2f}, p {_fmt_p(r['p'])}), {verdict}.\n")
        key = dm[(dm["sample"] == "common") & (dm["A"] == "harx_all") & (dm["B"] == "harx_d") & (dm["loss"] == "qlike")]
        for _, r in key.iterrows():
            verdict = "significant at 5%" if r["p"] < 0.05 else "not significant at 5%"
            L.append(f"Key test, {HZ[r['horizon']]}: adding the ladder to HAR-X + Deribit changes QLIKE by "
                     f"{r['mean_diff']:+.4f} (DM = {r['dm']:.2f}, p {_fmt_p(r['p'])}), {verdict}.\n")
    L.append("TODO: why does (or doesn't) the ladder add information beyond Deribit?\n")
    L.append("### 4.3 Are implied variances biased?\n")
    L.append("Mean of log realized variance minus log implied variance. Negative values mean the implied variance "
             "was too high on average (a variance risk premium).\n")
    L.append(md_table(bias))
    L.append("TODO: compare the premium in the ladder (retail) with Deribit (professional market makers).\n")
    L.append("### 4.4 Ladder probabilities\n")
    L.append("Model probabilities for each strike come from the predictive density of variance: log S_T is normal "
             "given the variance, and the variance itself is uncertain, which fattens the tails without extra "
             "parameters.\n")
    L.append(md_table(lad))
    L.append("Calibration by bins of the market price (realized = share of contracts that paid out):\n\n" + md_table(cal))
    L.append(f"![Ladder calibration]({FIG}/fig3_ladder_calibration.png)\n")
    L.append("TODO: is there a favourite-longshot bias in the tails, and is it bigger than the model's own error?\n")
    L.append("### 4.5 Trading rule\n")
    L.append("Buy at the ask when the model probability exceeds it by the threshold, sell at the bid in the opposite "
             "case; hold to settlement. Kalshi uses recorded bid/ask and its taker fee; Polymarket history has no "
             f"quotes, so a half-spread of {config.PM_HALF_SPREAD} is assumed there (the live logger measures it).\n")
    L.append(md_table(trd))
    L.append(f"![Trading]({FIG}/fig4_pnl.png)\n")
    L.append("### 4.6 True out-of-sample forecasts\n")
    if len(live):
        L.append(f"Forecasts committed to git before settlement since {config.LIVE_START}:\n\n" + md_table(live))
    else:
        L.append(f"Forecasts are committed to git before settlement from {config.LIVE_START}; "
                 "scores appear here once their windows have ended and `python run_all.py` is re-run.\n")
    L.append("## 5. Interpretation\n")
    L.append("TODO: which model wins, by how much, is it significant, and why. Discuss limits: Binance as a proxy for "
             "the CF Benchmarks index that settles Kalshi; Polymarket history without quotes; the short live window.\n")
    path = config.REPORTS / "draft_report.md"
    path.write_text("\n".join(L), encoding="utf-8")
    log(f"report: {path}")
