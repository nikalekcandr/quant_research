# Kalshi 15-minute BTC markets on book candles — clean study (29.10.2026)

`studies/kx15_study.py data/raw results/studies --binance` · data: `data/raw/kx15` (Kalshi 1-minute candles of the
book: best YES bid/ask; settled results), spot = Binance BTC/USDT minutes with the BRTI basis estimated from past
strikes (median −5.4 bp, IQR 8.7 bp).

**Sample:** 20 317 markets, 148 366 decision points (14, 12, 10, 8, 6, 4, 2, 1 min before the close),
16 Jan – 22 Aug 2026, median spread 1¢. Split in time at 9 May 2026.

| Forecast | Brier | log loss | gain vs market | t (clustered by market) |
|---|---|---|---|---|
| **Market book mid** | **0.1463** | **0.4399** | — | — |
| Random-walk model (normal) | 0.1522 | 0.4576 | −0.0059 | −19.3 |
| Same, Student-t(4) | 0.1513 | 0.4545 | −0.0050 | −15.3 |
| Model with spot 1 min old | 0.1689 | 0.5048 | −0.0226 | −62.7 |
| Last trade | 0.1464 | 0.4403 | −0.0001 | −4.6 |

* The market beats the model at every horizon (t −9 at 14 min, −17 at 1 min), in both halves, at every hour,
  on weekends, at every spread. Our live "model beats market" came from the lagged `/markets` listing.
* Encompassing (fit on the first half): logit(mid) 0.97 (t 22.9), logit(model) 0.10 (t 2.2); out of sample the
  combination gains 0.0001 Brier (t 1.2). The model adds nothing the book does not know.
* Calibration: the book is slightly compressed toward 50¢ (outcome − price ≈ −0.6 pp below 50¢, +0.6…+1 pp above),
  smaller than the taker fee plus half the spread. Buying either side at the book loses 1–5¢ per contract in almost
  every price × time-left cell; the few positive cells (e.g. YES at 65–80¢ with 1 min left, +3¢, t 1.4) do not
  survive multiple testing.
* Taker trading on the model, thresholds 0–8¢, train and test: −1.9 … −3.1¢ per trade, t −3.6 … −6.6.
* Under-reaction test (last-minute spot return, last-2-minute change of the mid, model − mid on top of the mid):
  no out-of-sample gain (t 1.5, −0.3, −1.2).

**Verdict:** the Kalshi 15-minute book is efficient at the 1-minute scale for a taker. What remains: the maker side
(`studies/kx15_makers.py` on every print, `data/raw/kx15_trades`), the settlement average inside the last minute
(live rule V07, 15-s loop), and cross-venue Kalshi vs Polymarket 15-minute on books (live A12/A12F).
