# Volatility models — where gradient boosting wins (29.10.2026)

## 1. Horse race on Binance minutes, 2024-01 → 2026-10 out of sample (`studies/vol_horserace.py`)
Walk-forward, refit monthly, QLIKE (lower is better), DM vs HAR with Newey–West.

| Horizon | HAR | HAR deseasonalised | HAR + activity | **GBM** (HAR + activity + calendar) | GARCH(1,1) | EWMA |
|---|---|---|---|---|---|---|
| 15 min | 0.459 | unstable | 0.430 | **0.335** (−27%, t 11.7) | — | 0.466 |
| 1 hour | 0.330 | 0.265 | 0.331 | **0.253** (−23%, t 13.2; −4% vs deseasonalised HAR) | 0.440 | 0.457 |
| 1 day | 0.188 | **0.165** (t 5.5) | 0.198 | 0.230 | 0.355 | 0.322 |

Intraday the trees win every year (2024, 2025, 2026): they learn how the level interacts with the hour of the
week. At one day, a HAR with the weekday profile removed is best; trees overfit 1 000 days.

## 2. The course project's own target: variance from HH:05 to the hour (`studies/gbm_kx1h.py`)
GBM trained on three years of minutes, rescored with the project's CRPS on one common sample (hours with a
ladder). DM < 0 = better than the reference.

| Forecast | CRPS 15.01–12.10 (5 744 h) | DM vs ladder centre | CRPS 13.10–28.10 (342 h) |
|---|---|---|---|
| **GBM + ladder centre (average of logs)** | **0.3255** | **−5.3** | 0.3374 |
| GBM + HAR-X + ladder (`harx_all`) | 0.3278 | −2.3 | 0.3369 |
| Ladder centre (market) | 0.3335 | — | 0.3385 |
| GBM alone (3 years of minutes) | 0.3342 | 0.2 | 0.3519 |
| HAR-X + Deribit + ladder (`harx_all`, the project's best) | 0.3392 | 2.2 | **0.3347** |
| `gbm_all` (project's GBM, short panel) | 0.3488 | 5.1 | 0.3516 |

Over nine months the long-history GBM is as good as the ladder on its own, and averaging it with the ladder is
the best forecast (−2.4% CRPS vs the ladder, t −5.3). On the last two weeks (342 hours) `harx_all` is best and
the combinations are within noise.

## 3. Does the better variance forecast give better strike probabilities? No.
Brier per strike of the Kalshi hourly ladder (`src/evaluate.py: ladder_eval`, same snapshots), DM vs the market
price per ladder (positive = worse than the market):

| Source | Brier 15.01–12.10 (38 051 strikes, 5 744 ladders) | DM | Brier 13–28.10 (2 366 / 342) | DM |
|---|---|---|---|---|
| **Market price** | **0.10254** | — | **0.09158** | — |
| GBM + ladder centre | 0.10263 | +0.8 | 0.09223 | +1.7 |
| `harx_all` | 0.10272 | +1.4 | 0.09197 | +0.8 |
| GBM alone | 0.10276 | +1.8 | 0.09280 | +2.8 |

Strike probabilities depend first on where spot is relative to the strike and only second on the variance; the
book sees BRTI and the order flow in real time. A 2–3% better variance forecast does not beat the price.
