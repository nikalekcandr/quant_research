# Alpha lab — live paper trading

_Updated 2026-10-29 12:00 UTC · lab running 0.0 h · rules frozen 2026-10-29 (version 90d9edec9bf4) · 2396 settled trades, 71 open · last loop — s_

Every rule was written down before its first live trade. Entry and exit at the real bid/ask, real fees, binaries held to the venue's own settlement. Holm controls the family-wise error over all hypotheses with ≥ 10 trades; DSR is the probability that the per-trade Sharpe beats the best of as many zero-skill rules.

## Leaderboard

| id | family | hypothesis | unit | trades | clusters | open | mean net | t | Holm p | hit rate | total net | mean gross | DSR | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| D01 | D | Hourly reversal | return | 10 | 10 | 1 | -0.0018 | -1.0626 | 1.0000 | 0.4000 | -0.0181 | -0.0008 | 0.0003 | nothing |
| D02 | D | Long outside US hours |  | 0 | 0 | 0 | — | — | — | — | 0.0000 | — | — | no signal yet |
| D03 | D | Short during US hours |  | 0 | 0 | 0 | — | — | — | — | 0.0000 | — | — | no signal yet |
| D04 | D | Fade extreme funding |  | 0 | 0 | 0 | — | — | — | — | 0.0000 | — | — | no signal yet |
| D05 | D | Fade the perpetual premium | return | 1 | 1 | 0 | -0.0012 | — | — | 0.0000 | -0.0012 | -0.0002 | — | collecting |
| D06 | D | Order-book imbalance | return | 113 | 13 | 1 | -0.0010 | -8.6402 | 6.74e-17 | 0.0885 | -0.1139 | -6.93e-06 | 2.38e-19 | loses money |
| D07 | D | Taker-flow momentum | return | 31 | 13 | 1 | -0.0009 | -3.0714 | 0.0234 | 0.2581 | -0.0268 | 0.0001 | 0.0020 | loses money |
| D08 | D | Coinbase premium momentum |  | 0 | 0 | 0 | — | — | — | — | 0.0000 | — | — | no signal yet |
| D09 | D | Coinbase → Binance lead-lag (15 s) | return | 3 | 3 | 0 | -0.0037 | -1.3309 | — | 0.0000 | -0.0110 | -0.0027 | — | collecting |
| D10 | D | Open interest × price |  | 0 | 0 | 0 | — | — | — | — | 0.0000 | — | — | no signal yet |
| D11 | D | Fade retail positioning |  | 0 | 0 | 0 | — | — | — | — | 0.0000 | — | — | no signal yet |
| D12 | D | Implied-vol momentum |  | 0 | 0 | 0 | — | — | — | — | 0.0000 | — | — | no signal yet |
| D13 | D | Gradient boosting, next hour |  | 0 | 0 | 0 | — | — | — | — | 0.0000 | — | — | no signal yet |
| D14 | D | Ridge regression, next hour |  | 0 | 0 | 0 | — | — | — | — | 0.0000 | — | — | no signal yet |
| D15 | D | Weekly time-series momentum |  | 0 | 0 | 1 | — | — | — | — | 0.0000 | — | — | collecting |
| D16 | D | Kalshi ladder drift | return | 6 | 6 | 1 | 0.0008 | 0.3105 | — | 0.5000 | 0.0049 | 0.0018 | — | collecting |
| D17 | D | Follow the Polymarket hourly crowd |  | 0 | 0 | 1 | — | — | — | — | 0.0000 | — | — | collecting |
| D18 | D | Regime-conditioned trend |  | 0 | 0 | 0 | — | — | — | — | 0.0000 | — | — | no signal yet |
| D19 | D | Fade liquidation cascades |  | 0 | 0 | 0 | — | — | — | — | 0.0000 | — | — | no signal yet |
| D20 | D | Aggressive-flow momentum (1 min) | return | 3 | 1 | 0 | -0.0006 | — | — | 0.0000 | -0.0019 | 0.0004 | — | collecting |
| M01 | M | Maker quotes on Kalshi 15-min | $/contract | 5 | 1 | 2 | -0.0940 | — | — | 0.4000 | -0.4700 | -0.0840 | — | collecting |
| M02 | M | Maker quotes on Kalshi hourly ladder |  | 0 | 0 | 1 | — | — | — | — | 0.0000 | — | — | collecting |
| P01 | P | Polymarket 5-min Up/Down vs fair value | $/contract | 163 | 13 | 2 | 0.0042 | 0.1424 | 1.0000 | 0.3620 | 0.6915 | 0.0240 | 0.0607 | nothing |
| P02 | P | Polymarket 15-min Up/Down vs fair value | $/contract | 60 | 13 | 2 | 0.0597 | 1.4846 | 1.0000 | 0.4500 | 3.5839 | 0.0798 | 0.2461 | nothing |
| P03 | P | Polymarket hourly Up/Down vs fair value | $/contract | 16 | 11 | 1 | 0.0026 | 0.0355 | 1.0000 | 0.3125 | 0.0410 | 0.0216 | 0.0498 | nothing |
| P04 | P | Polymarket 4-hour Up/Down vs fair value | $/contract | 3 | 3 | 1 | -0.1386 | -1.4753 | — | 0.3333 | -0.4159 | -0.1250 | — | collecting |
| P05 | P | Polymarket hourly ladders vs fair value | $/contract | 20 | 10 | 2 | 0.0296 | 0.4022 | 1.0000 | 0.2500 | 0.5921 | 0.0582 | 0.0775 | nothing |
| P06 | P | Kalshi 15-min vs fair value | $/contract | 5 | 1 | 2 | 0.0620 | — | — | 0.4000 | 0.3100 | 0.0870 | — | collecting |
| P07 | P | Kalshi hourly ladder vs fair value (every minute) |  | 0 | 0 | 1 | — | — | — | — | 0.0000 | — | — | collecting |
| A11 | A | Ladder monotonicity (Polymarket hourly, Kalshi hourly) |  | 0 | 0 | 0 | — | — | — | — | 0.0000 | — | — | no signal yet |
| A12 | A | Kalshi 15-min vs Polymarket 15-min | $/pair | 1 | 1 | 0 | 0.0337 | — | — | 1.0000 | 0.0337 | 0.0700 | — | collecting |
| A13 | A | Kalshi hourly ladder vs Polymarket hourly ladder |  | 0 | 0 | 0 | — | — | — | — | 0.0000 | — | — | no signal yet |
| V01 | V | 15-s fair value: lab blend (as P01/P02/P06) | $/contract | 114 | 5 | 6 | 0.0033 | 0.1132 | 1.0000 | 0.4211 | 0.3724 | 0.0242 | 0.0553 | nothing |
| V02 | V | 15-s fair value: Student-t(4) tails | $/contract | 114 | 5 | 6 | -0.0013 | -0.0479 | 1.0000 | 0.4298 | -0.1528 | 0.0195 | 0.0452 | nothing |
| V03 | V | 15-s fair value: variance implied by the Kalshi hourly ladder | $/contract | 77 | 3 | 3 | -0.0116 | -0.3543 | 1.0000 | 0.4416 | -0.8917 | 0.0090 | 0.0311 | nothing |
| V04 | V | 15-s fair value: DVOL level | $/contract | 111 | 5 | 6 | 0.0144 | 0.3573 | 1.0000 | 0.4054 | 1.5990 | 0.0353 | 0.0868 | nothing |
| V05 | V | 15-s fair value: realised variance of the last hour | $/contract | 119 | 5 | 6 | -0.0036 | -0.2442 | 1.0000 | 0.4454 | -0.4325 | 0.0170 | 0.0405 | nothing |
| V06 | V | Stale-quote sniping (15 s) | $/contract | 1 | 1 | 0 | -0.0852 | — | — | 0.0000 | -0.0852 | -0.0750 | — | collecting |
| A12F | A | Kalshi 15-min vs Polymarket 15-min, every 15 s | $/pair | 4 | 1 | 0 | 0.0433 | — | — | 1.0000 | 0.1731 | 0.0800 | — | collecting |
| W01 | W | Sell every cheap wing (hourly ladders) |  | 0 | 0 | 7 | — | — | — | — | 0.0000 | — | — | collecting |
| W02 | W | Sell cheap wings our model also calls overpriced |  | 0 | 0 | 1 | — | — | — | — | 0.0000 | — | — | collecting |
| O01 | O | Cross-venue option arbitrage |  | 0 | 0 | 0 | — | — | — | — | 0.0000 | — | — | no signal yet |
| O02 | O | Put–call parity |  | 0 | 0 | 0 | — | — | — | — | 0.0000 | — | — | no signal yet |
| O03 | O | Static arbitrage of the smile |  | 0 | 0 | 0 | — | — | — | — | 0.0000 | — | — | no signal yet |
| O04 | O | ATM straddle vs our volatility forecast | $/BTC straddle | 1 | 1 | 3 | -468.7898 | — | — | 0.0000 | -468.7898 | -404.8124 | — | collecting |
| F01 | F | Cross-venue funding carry |  | 0 | 0 | 0 | — | — | — | — | 0.0000 | — | — | no signal yet |


## Families

| family | hypotheses | trades | open |
|---|---|---|---|
| A | 4 | 5 | 0 |
| D | 20 | 167 | 6 |
| F | 1 | 0 | 0 |
| M | 2 | 5 | 3 |
| O | 4 | 1 | 3 |
| P | 7 | 267 | 11 |
| V | 6 | 536 | 27 |
| W | 2 | 0 | 8 |


## Model vs market: who forecasts short-dated binaries better?

Brier score (lower is better) of our fair value and of the market mid over every quoted minute, averaged per market; t > 2 means our model is significantly better calibrated than the market.

| market kind | markets settled | quotes | Brier ours | Brier market mid | market − ours | t (by market) | better |
|---|---|---|---|---|---|---|---|
| kx_15m | 41 | 563 | 0.1517 | 0.1644 | 0.0126 | 2.5469 | ours |
| kx_ladder_1h | 640 | 30556 | 0.0168 | 0.0168 | -4.82e-05 | -0.1898 | market |
| pm_ladder_1h | 58 | 1802 | 0.0733 | 0.0729 | -0.0004 | -0.2140 | market |
| pm_updown_15m | 49 | 657 | 0.1742 | 0.1810 | 0.0067 | 0.8924 | ours |
| pm_updown_1h | 12 | 629 | 0.1503 | 0.1469 | -0.0034 | -0.6524 | market |
| pm_updown_4h | 3 | 455 | 0.0515 | 0.0424 | -0.0091 | -1.3426 | market |
| pm_updown_5m | 147 | 615 | 0.1825 | 0.1741 | -0.0084 | -1.0255 | market |


## Volatility zoo (QLIKE, lower is better)

| index | horizon | model | n | QLIKE | MSE log | bias log |
|---|---|---|---|---|---|---|
| 5 | 1h | gbm_vol | 10 | 0.2200 | 0.4635 | 0.3626 |
| 0 | 1h | combo | 10 | 0.2558 | 0.5590 | 0.3618 |
| 3 | 1h | ewma_rm | 10 | 0.2615 | 0.6017 | 0.4004 |
| 2 | 1h | egarch | 10 | 0.2619 | 0.5576 | 0.2565 |
| 6 | 1h | gjr | 10 | 0.2679 | 0.5375 | 0.1902 |
| 4 | 1h | garch | 10 | 0.2757 | 0.5493 | 0.1756 |
| 8 | 1h | lab_blend | 10 | 0.2857 | 0.5323 | 0.3408 |
| 7 | 1h | har_hourly | 10 | 0.3120 | 0.6342 | 0.3945 |
| 13 | 1h | proj_harx_all | 7 | 0.3173 | 0.4784 | 0.0533 |
| 14 | 1h | proj_harx_d | 7 | 0.3174 | 0.5052 | 0.1068 |
| 9 | 1h | proj_combo | 7 | 0.3183 | 0.5611 | 0.2136 |
| 16 | 1h | proj_iv_mz | 7 | 0.3216 | 0.7458 | 0.4163 |
| 15 | 1h | proj_harx_m | 7 | 0.3239 | 0.4813 | 0.0424 |
| 12 | 1h | proj_harx | 7 | 0.3277 | 0.5127 | 0.0925 |
| 21 | 1h | proj_mkt_t_raw | 7 | 0.3303 | 0.5535 | 0.1780 |
| 10 | 1h | proj_har | 7 | 0.3500 | 0.5249 | 0.0748 |
| 11 | 1h | proj_har_m | 7 | 0.3540 | 0.4988 | 0.0232 |
| 1 | 1h | dvol | 10 | 0.3556 | 1.0186 | 0.7740 |
| 20 | 1h | proj_mkt_raw | 7 | 0.3573 | 0.5353 | 0.1216 |
| 17 | 1h | proj_iv_raw | 7 | 0.3657 | 0.9884 | 0.6056 |
| 19 | 1h | proj_mkt_mz | 7 | 0.4057 | 0.5555 | 0.0448 |
| 22 | 1h | proj_naive | 7 | 0.4078 | 0.4883 | -0.0696 |
| 18 | 1h | proj_mkt_ln_raw | 7 | 0.4209 | 0.4725 | -0.2204 |


Diebold–Mariano vs the lab blend:


| horizon | model | n common | QLIKE gain vs lab_blend | t |
|---|---|---|---|---|
| 1h | combo | 10 | 0.0299 | 0.4862 |
| 1h | dvol | 10 | -0.0699 | -0.4427 |
| 1h | egarch | 10 | 0.0238 | 0.3886 |
| 1h | ewma_rm | 10 | 0.0242 | 0.3092 |
| 1h | garch | 10 | 0.0100 | 0.2021 |
| 1h | gbm_vol | 10 | 0.0656 | 0.9614 |
| 1h | gjr | 10 | 0.0178 | 0.3601 |
| 1h | har_hourly | 10 | -0.0263 | -1.2888 |
| 1h | proj_combo | 7 | -0.0034 | -0.0919 |
| 1h | proj_har | 7 | -0.0350 | -0.6335 |
| 1h | proj_har_m | 7 | -0.0391 | -0.5691 |
| 1h | proj_harx | 7 | -0.0128 | -0.3970 |
| 1h | proj_harx_all | 7 | -0.0024 | -0.0708 |
| 1h | proj_harx_d | 7 | -0.0024 | -0.1023 |
| 1h | proj_harx_m | 7 | -0.0089 | -0.2301 |
| 1h | proj_iv_mz | 7 | -0.0066 | -0.0481 |
| 1h | proj_iv_raw | 7 | -0.0508 | -0.2552 |
| 1h | proj_mkt_ln_raw | 7 | -0.1060 | -0.5636 |
| 1h | proj_mkt_mz | 7 | -0.0907 | -0.8125 |
| 1h | proj_mkt_raw | 7 | -0.0424 | -0.7874 |
| 1h | proj_mkt_t_raw | 7 | -0.0153 | -0.3654 |
| 1h | proj_naive | 7 | -0.0928 | -0.5740 |


## ML forecasts: information coefficient

| model | n | rank IC | t(IC) | hit rate (sign) |
|---|---|---|---|---|
| gbm | 21 | 0.6081 | 3.3393 | 0.1905 |
| ridge | 21 | 0.5909 | 3.1927 | 0.5714 |


## Data health (last hour)

| source | minutes in last hour | filled % |
|---|---|---|
| Binance spot | 59 | 100.0 |
| Binance perp | 59 | 100.0 |
| Bybit | 59 | 100.0 |
| OKX | 59 | 100.0 |
| Coinbase | 59 | 100.0 |
| Kraken | 59 | 100.0 |
| Deribit | 59 | 100.0 |
| DVOL | 59 | 100.0 |


## Rules

- **D01** Hourly reversal. at HH:00 take the opposite side of the last hour's return; hold 60 min. _short-term reversal (Jegadeesh 1990; for crypto hourly: R4c in research.md)_
- **D02** Long outside US hours. long at 16:00 ET on NYSE days, exit at the next 09:30 ET open. _overnight drift (Lou, Polk & Skouras 2019); R2 in research.md_
- **D03** Short during US hours. short at 09:30 ET on NYSE days, exit at 16:00 ET. _mirror of D02 (US-session returns were negative after the ETF launch)_
- **D04** Fade extreme funding. 30 min before each Binance funding: predicted rate > 0.015% → short, < 0.005% → long; hold 8 h. _crowded-positioning reversal (He, Manela, Ross & von Wachter 2022)_
- **D05** Fade the perpetual premium. HH:30: z-score of the Binance premium index (last 7 days, hourly) > 2 → short, < −2 → long; hold 60 min. _basis mean reversion_
- **D06** Order-book imbalance. every minute: Binance spot depth within 10 bp, (bid−ask)/(bid+ask) > 0.4 → long, < −0.4 → short; hold 5 min; one position at a time. _order-flow imbalance (Cont, Kukanov & Stoikov 2014)_
- **D07** Taker-flow momentum. every 5 min: share of taker buys in the last 5 closed minutes > 0.6 → long, < 0.4 → short; hold 15 min. _trade-flow persistence_
- **D08** Coinbase premium momentum. every 15 min: change over 60 min of Coinbase BTC-USD / (Binance BTC-USDT × USDT-USD) > +5 bp → long 60 min, < −5 bp → short. _US spot demand ('Coinbase premium')_
- **D09** Coinbase → Binance lead-lag (15 s). every 15 s: Coinbase BTC-USD moved more than Binance BTC-USDT over the last 15 s by > 3 bp → bet Binance follows for 60 s (run by the fast loop). _price discovery across venues (Makarov & Schoar 2020)_
- **D10** Open interest × price. HH:01: open interest up > 1% in the last hour → follow the hour's price move; hold 60 min. _leverage build-up confirms trends_
- **D11** Fade retail positioning. every 4 h: z-score (last 7 days, hourly) of Binance's global long/short account ratio > 1.5 → short, < −1.5 → long; hold 4 h. _retail crowding is contrarian_
- **D12** Implied-vol momentum. HH:02: z-score of the last hourly change of log DVOL (last 30 days) > 1 → long 60 min, < −1 → short. _R12 in research.md (ΔDVOL led returns, t = 2.8)_
- **D13** Gradient boosting, next hour. HH:00: gradient-boosted trees on 20 price/volume/calendar features predict the next-hour return; > +15 bp → long, < −15 bp → short; retrained every 6 h on 180 days. _machine-learning return prediction (Gu, Kelly & Xiu 2020)_
- **D14** Ridge regression, next hour. same features and thresholds as D13, linear ridge model. _linear benchmark for D13_
- **D15** Weekly time-series momentum. 00:00 UTC: sign of the last 7-day return; hold 24 h. _Liu & Tsyvinski (2021); R4 in research.md_
- **D16** Kalshi ladder drift. HH:06: the Kalshi hourly ladder's implied drift (median + s²/2 − log spot, in units of s) > 0.25 → long to HH+1:00, < −0.25 → short. _R9 in research.md (does the ladder know the direction?)_
- **D17** Follow the Polymarket hourly crowd. HH:05: mid of Polymarket's hourly Up/Down market > 0.60 → long to the end of the hour, < 0.40 → short. _prediction markets as directional signals_
- **D18** Regime-conditioned trend. 00:05 UTC: bull trend (price > 200-day mean) and low-vol Markov state → long 24 h; bear and high-vol → short; otherwise flat. _regime timing (research.md, section 3)_
- **D19** Fade liquidation cascades. every minute: forced-order notional of the last 5 min on one side > max($3M, 4 × its 6-hour 95th percentile) → bet on a rebound (long after long liquidations); hold 30 min. _liquidation cascades overshoot (Ali 2023; folklore of crypto desks)_
- **D20** Aggressive-flow momentum (1 min). every minute: (taker buys − taker sells)/(sum) of the last minute > 0.5 with notional > 3 × the 6-hour median minute → follow for 5 min (mirror for sells). _order-flow imbalance predicts short-run returns (Chordia & Subrahmanyam 2004; Cont et al. 2014)_
- **M01** Maker quotes on Kalshi 15-min. every minute: virtual bid at fair − 4¢ and ask at fair + 4¢ (only if inside or behind the spread), valid 60 s; filled only if a real trade prints strictly through our price; held to settlement; maker fee 0.0175·P(1−P). _market making against informed flow_
- **M02** Maker quotes on Kalshi hourly ladder. as M01, strikes with fair value 0.2–0.8 (at most 4), only ≥ 10 minutes before the hour. __
- **P01** Polymarket 5-min Up/Down vs fair value. buy the side whose fair value exceeds the ask + fee by ≥ 3¢; fair = Φ(log(S/S_start)/σ√τ) with the lab's seasonal variance forecast. _do short-dated crypto binaries misprice the random walk?_
- **P02** Polymarket 15-min Up/Down vs fair value. as P01. __
- **P03** Polymarket hourly Up/Down vs fair value. as P01; settles on the Binance 1-hour candle. __
- **P04** Polymarket 4-hour Up/Down vs fair value. as P01. __
- **P05** Polymarket hourly ladders vs fair value. 'Bitcoin above K at h ET': fair = Φ(log(S/K)/σ√τ). __
- **P06** Kalshi 15-min vs fair value. KXBTC15M: fair = Φ(log(S_usd/target)/σ√τ), S_usd = Coinbase/Kraken mid. __
- **P07** Kalshi hourly ladder vs fair value (every minute). KXBTCD current hour, every strike, every minute; first signal per strike and side. _late-hour pricing of Kalshi's hourly ladder_
- **A11** Ladder monotonicity (Polymarket hourly, Kalshi hourly). K1 < K2 and bid(K2) − ask(K1) > fees → buy YES K1, buy NO K2: pays ≥ 0 in every state. _A1 in alpha_live.md, now every minute_
- **A12** Kalshi 15-min vs Polymarket 15-min. same quarter-hour window: Polymarket Up ask + Kalshi NO (1 − yes bid) + fees < 1 (or the mirror) → buy both: pays 1 unless the two references disagree. _cross-venue binary arbitrage (references: CF BRTI vs Chainlink)_
- **A13** Kalshi hourly ladder vs Polymarket hourly ladder. same settlement hour: Polymarket 'above K' (Binance close) vs the Kalshi strike K' = K × USDT/USD within $30 (CF BRTI average); YES on the cheap venue + NO on the dear one for < 1 − fees − 1¢ → pays 1 unless the two settlement prices land on different sides. _cross-venue ladder arbitrage_
- **V01** 15-s fair value: lab blend (as P01/P02/P06). every 15 s on Polymarket 5/15-min Up/Down and Kalshi 15-min: buy the side whose fair value beats ask + fee by ≥ 3¢; one trade per market and side. _which variance forecast prices the short binaries best?_
- **V02** 15-s fair value: Student-t(4) tails. every 15 s on Polymarket 5/15-min Up/Down and Kalshi 15-min: buy the side whose fair value beats ask + fee by ≥ 3¢; one trade per market and side. _which variance forecast prices the short binaries best?_
- **V03** 15-s fair value: variance implied by the Kalshi hourly ladder. every 15 s on Polymarket 5/15-min Up/Down and Kalshi 15-min: buy the side whose fair value beats ask + fee by ≥ 3¢; one trade per market and side. _which variance forecast prices the short binaries best?_
- **V04** 15-s fair value: DVOL level. every 15 s on Polymarket 5/15-min Up/Down and Kalshi 15-min: buy the side whose fair value beats ask + fee by ≥ 3¢; one trade per market and side. _which variance forecast prices the short binaries best?_
- **V05** 15-s fair value: realised variance of the last hour. every 15 s on Polymarket 5/15-min Up/Down and Kalshi 15-min: buy the side whose fair value beats ask + fee by ≥ 3¢; one trade per market and side. _which variance forecast prices the short binaries best?_
- **V06** Stale-quote sniping (15 s). as V01, but only when spot moved ≥ 4 bp in the last 15 s and the binary's mid moved ≤ 1¢. _latency arbitrage against slow quotes_
- **A12F** Kalshi 15-min vs Polymarket 15-min, every 15 s. as A12, sampled every 15 s. _how long do cross-venue gaps live?_
- **W01** Sell every cheap wing (hourly ladders). Kalshi and Polymarket hourly ladders, ≥ 15 min before the hour: a side priced 2–6¢ at the bid is sold at that bid (once per strike); held to settlement. _favourite–longshot bias: cheap tails win less often than their price (slide 7, R10)_
- **W02** Sell cheap wings our model also calls overpriced. as W01, only if our fair value is ≥ 1¢ + fee below the wing's price. __
- **O01** Cross-venue option arbitrage. same expiry/strike/type: bid on venue B − ask on venue A − both fees > 0 → buy A, sell B (identical payoff; settlement indices differ slightly). _law of one price across venues_
- **O02** Put–call parity. expiries ≤ 7 days, each venue: (C − P + K) vs the perpetual on the same index (Deribit perp for Deribit/OKX, Binance perp for Binance/Bybit) minus the expected funding until expiry; a gap larger than option fees + 2 perp fees → trade the synthetic against the perp. _parity (Stoll 1969); perp ≈ forward_
- **O03** Static arbitrage of the smile. each venue and expiry: call/put prices non-monotone in the strike, or a negative-cost butterfly, after fees. _no-arbitrage bounds (Carr & Madan 2005)_
- **O04** ATM straddle vs our volatility forecast. each venue, nearest daily expiry 3–30 h away, ATM strike: fair = E|S_T − K| under our variance forecast; fair − ask − fees > 10% of the ask → buy; bid − fair − fees > 10% → sell; one trade per venue and expiry; settles on the 30-min average before 08:00 UTC. _is our volatility forecast better than the options market's?_
- **F01** Cross-venue funding carry. 30 min before the 00/08/16 UTC funding: predicted rates of Binance, Bybit and OKX differ by > 2 bp → short the perp that pays most, long the one that pays least; close 10 min after the funding. _funding-rate arbitrage_