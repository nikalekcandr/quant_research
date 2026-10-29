# Strategy catalogue — alpha lab

_29.10.2026 17:55 MSK. Live: settled trades / clusters (entry hours, the unit of independence), mean per trade, t over clusters; voided trades excluded. Rules are frozen; a changed rule gets a new ID. Retired rules are no longer run; their trades stay in the ledger._

Backtests: `studies/kx15_study.py`, `studies/kx15_flow_deep.py`, `studies/tape_strategies.py`, `studies/kx_15m_vs_hourly.py`, `studies/invariance_vol.py`, `studies/vol_horserace.py`, `studies/gbm_kx1h.py`, `research.py` (R1–R21).


## Active (38)


### A — No-arbitrage (binaries)

| ID | Strategy | Rule | Frozen | Live | History / backtest |
|---|---|---|---|---|---|
| A11 | Ladder monotonicity (Polymarket hourly, Kalshi hourly) | K1 < K2 and bid(K2) − ask(K1) > fees → buy YES K1, buy NO K2: pays ≥ 0 in every state | 2026-10-29 | 1 / 1 cl., 0.01349 $/pair |  |
| A12 | Kalshi 15-min vs Polymarket 15-min | same quarter-hour window: Polymarket Up ask + Kalshi NO (1 − yes bid) + fees < 1 (or the mirror) → buy both: pays 1 unless the two references disagree | 2026-10-29 | 5 / 4 cl., 0.06878 $/pair, t 1.5 | history test pending (Polymarket 15m) |
| A12F | Kalshi 15-min vs Polymarket 15-min, every 15 s | as A12, sampled every 15 s | 2026-10-29 11:00 | 9 / 4 cl., 0.02977 $/pair, t 1.8 |  |
| A13 | Kalshi hourly ladder vs Polymarket hourly ladder | same settlement hour: Polymarket 'above K' (Binance close) vs the Kalshi strike K' = K × USDT/USD within $30 (CF BRTI average); YES on the cheap venue + NO on the dear one for < 1 − fees − 1¢ → pays 1 unless the two settlement prices land on different sides | 2026-10-29 11:00 | 4 / 1 cl., 0.01873 $/pair |  |

### D — Directional BTC

| ID | Strategy | Rule | Frozen | Live | History / backtest |
|---|---|---|---|---|---|
| D01 | Hourly reversal | at HH:00 take the opposite side of the last hour's return; hold 60 min | 2026-10-29 | 13 / 13 cl., -0.001198 return, t -0.9 |  |
| D02 | Long outside US hours | long at 16:00 ET on NYSE days, exit at the next 09:30 ET open | 2026-10-29 | — |  |
| D03 | Short during US hours | short at 09:30 ET on NYSE days, exit at 16:00 ET | 2026-10-29 | — |  |
| D04 | Fade extreme funding | 30 min before each Binance funding: predicted rate > 0.015% → short, < 0.005% → long; hold 8 h | 2026-10-29 | — |  |
| D05 | Fade the perpetual premium | HH:30: z-score of the Binance premium index (last 7 days, hourly) > 2 → short, < −2 → long; hold 60 min | 2026-10-29 | 3 / 3 cl., -0.003478 return, t -2.3 |  |
| D08 | Coinbase premium momentum | every 15 min: change over 60 min of Coinbase BTC-USD / (Binance BTC-USDT × USDT-USD) > +5 bp → long 60 min, < −5 bp → short | 2026-10-29 | — |  |
| D09 | Coinbase → Binance lead-lag (15 s) | every 15 s: Coinbase BTC-USD moved more than Binance BTC-USDT over the last 15 s by > 3 bp → bet Binance follows for 60 s (run by the fast loop) | 2026-10-29 | 5 / 5 cl., -0.002739 return, t -1.7 |  |
| D10 | Open interest × price | HH:01: open interest up > 1% in the last hour → follow the hour's price move; hold 60 min | 2026-10-29 | — |  |
| D11 | Fade retail positioning | every 4 h: z-score (last 7 days, hourly) of Binance's global long/short account ratio > 1.5 → short, < −1.5 → long; hold 4 h | 2026-10-29 | — |  |
| D12 | Implied-vol momentum | HH:02: z-score of the last hourly change of log DVOL (last 30 days) > 1 → long 60 min, < −1 → short | 2026-10-29 | — |  |
| D13 | Gradient boosting, next hour | HH:00: gradient-boosted trees on 20 price/volume/calendar features predict the next-hour return; > +15 bp → long, < −15 bp → short; retrained every 6 h on 180 days | 2026-10-29 | — |  |
| D14 | Ridge regression, next hour | same features and thresholds as D13, linear ridge model | 2026-10-29 | — |  |
| D15 | Weekly time-series momentum | 00:00 UTC: sign of the last 7-day return; hold 24 h | 2026-10-29 | — |  |
| D16 | Kalshi ladder drift | HH:06: the Kalshi hourly ladder's implied drift (median + s²/2 − log spot, in units of s) > 0.25 → long to HH+1:00, < −0.25 → short | 2026-10-29 | 9 / 9 cl., -1.47e-05 return, t -0.0 |  |
| D17 | Follow the Polymarket hourly crowd | HH:05: mid of Polymarket's hourly Up/Down market > 0.60 → long to the end of the hour, < 0.40 → short | 2026-10-29 | 3 / 3 cl., -5.243e-05 return, t -0.1 |  |
| D18 | Regime-conditioned trend | 00:05 UTC: bull trend (price > 200-day mean) and low-vol Markov state → long 24 h; bear and high-vol → short; otherwise flat | 2026-10-29 | — |  |
| D19 | Fade liquidation cascades | every minute: forced-order notional of the last 5 min on one side > max($3M, 4 × its 6-hour 95th percentile) → bet on a rebound (long after long liquidations); hold 30 min | 2026-10-29 11:00 | — |  |
| D20 | Aggressive-flow momentum (1 min) | every minute: (taker buys − taker sells)/(sum) of the last minute > 0.5 with notional > 3 × the 6-hour median minute → follow for 5 min (mirror for sells) | 2026-10-29 11:00 | 4 / 2 cl., -0.0009215 return |  |

### F — Funding

| ID | Strategy | Rule | Frozen | Live | History / backtest |
|---|---|---|---|---|---|
| F01 | Cross-venue funding carry | 30 min before the 00/08/16 UTC funding: predicted rates of Binance, Bybit and OKX differ by > 2 bp → short the perp that pays most, long the one that pays least; close 10 min after the funding | 2026-10-29 | — |  |

### M — Maker quotes on Kalshi (simulated fills)

| ID | Strategy | Rule | Frozen | Live | History / backtest |
|---|---|---|---|---|---|
| M03 | Lottery seller on Kalshi (maker, queue-aware) | 15-min market and hourly-ladder strikes: when a side costs 2–20¢ and ≥ 2 min are left, join its best offer with 1 contract; filled by prints through our price or by taker volume at our price beyond the displayed queue; held to settlement; maker fee | 2026-10-29 17:40 | — | backtest Oct 10–29: +5.3¢ (t 1.9), halves +9.1 / +1.5¢; queue ignored |
| M04 | Lottery seller, cancels on spot moves | as M03; the quote is cancelled at the first 15-s sample in which Binance moved ≥ 3 bp toward the cheap side winning | 2026-10-29 17:40 | — | backtest: +8.7¢ (t 3.4), halves +9.6 / +7.8¢; queue ignored |

### O — Options

| ID | Strategy | Rule | Frozen | Live | History / backtest |
|---|---|---|---|---|---|
| O01 | Cross-venue option arbitrage | same expiry/strike/type: bid on venue B − ask on venue A − both fees > 0 → buy A, sell B (identical payoff; settlement indices differ slightly) | 2026-10-29 | — |  |
| O02 | Put–call parity | expiries ≤ 7 days, each venue: (C − P + K) vs the perpetual on the same index (Deribit perp for Deribit/OKX, Binance perp for Binance/Bybit) minus the expected funding until expiry; a gap larger than option fees + 2 perp fees → trade the synthetic against the perp | 2026-10-29 | — | voided before the snapshot-age fix |
| O03 | Static arbitrage of the smile | each venue and expiry: call/put prices non-monotone in the strike, or a negative-cost butterfly, after fees | 2026-10-29 | — |  |
| O04 | ATM straddle vs our volatility forecast | each venue, nearest daily expiry 3–30 h away, ATM strike: fair = E/S_T − K/ under our variance forecast; fair − ask − fees > 10% of the ask → buy; bid − fair − fees > 10% → sell; one trade per venue and expiry; settles on the 30-min average before 08:00 UTC | 2026-10-29 | — |  |

### P — Prediction markets: fair value

| ID | Strategy | Rule | Frozen | Live | History / backtest |
|---|---|---|---|---|---|
| P02 | Polymarket 15-min Up/Down vs fair value | as P01 | 2026-10-29 | 75 / 16 cl., 4.09¢, t 1.1 |  |
| P03 | Polymarket hourly Up/Down vs fair value | as P01; settles on the Binance 1-hour candle | 2026-10-29 | 20 / 14 cl., 1.27¢, t 0.2 |  |
| P05 | Polymarket hourly ladders vs fair value | 'Bitcoin above K at h ET': fair = Φ(log(S/K)/σ√τ) | 2026-10-29 | 28 / 12 cl., -0.13¢, t -0.0 |  |

### T — Kalshi trade tape (taker)

| ID | Strategy | Rule | Frozen | Live | History / backtest |
|---|---|---|---|---|---|
| T01 | Copy the favourite buyer | Kalshi 15-min: ≥ 200 contracts bought by takers on the favourite side at 65–95¢ in the last 60 s → buy the favourite at the book ask; one per market | 2026-10-29 17:40 | 1 / 1 cl., 27.00¢ | backtest: +1.5¢ (t 0.4) — noise |
| T02 | Fade the lottery burst | Kalshi 15-min: ≥ 500 contracts bought by takers on the cheap side at ≤ 20¢ in the last 60 s → buy the favourite at the book ask; one per market | 2026-10-29 17:40 | — | backtest: +3.3¢ (t 1.4); halves +6.3 / +0.3¢ |

### V — Short binaries every 15 s: fair-value models

| ID | Strategy | Rule | Frozen | Live | History / backtest |
|---|---|---|---|---|---|
| V06 | Stale-quote sniping (15 s) | as V01, but only when spot moved ≥ 4 bp in the last 15 s and the binary's mid moved ≤ 1¢ | 2026-10-29 11:00 | 4 / 3 cl., -13.88¢, t -0.5 |  |
| V07 | Kalshi 15-min: the settlement average already printed | Kalshi settles on the average of the last 60 s of BRTI; in the last 75 s part of that average is known. Fair = P(average > strike) with the printed part, the rest as a Brownian average, level = our USD proxy shifted by the market's own start basis; take the book when edge ≥ 3¢ after fee, ≥ 10 s left | 2026-10-29 16:40 | — | live only (needs sub-minute data) |

### W — Selling cheap wings

| ID | Strategy | Rule | Frozen | Live | History / backtest |
|---|---|---|---|---|---|
| W01 | Sell every cheap wing (hourly ladders) | Kalshi and Polymarket hourly ladders, ≥ 15 min before the hour: a side priced 2–6¢ at the bid is sold at that bid (once per strike); held to settlement | 2026-10-29 14:45 | 51 / 3 cl., -4.94¢, t -1.1 | history: selling ≤ 5¢ as taker ≈ 0 |
| W02 | Sell cheap wings our model also calls overpriced | as W01, only if our fair value is ≥ 1¢ + fee below the wing's price | 2026-10-29 14:45 | 27 / 3 cl., 3.57¢, t 2.2 | see W01 |


## Retired 29.10 (13) — losing or disproved

| ID | Strategy | Why retired | Live at retirement |
|---|---|---|---|
| D06 | Order-book imbalance | loses: −10 bp per trade, t −8.7, hit 8% | 141 trades, -0.001032 return |
| D07 | Taker-flow momentum | loses: t −2.8 | 37 trades, -0.0007725 return |
| M01 | Maker quotes on Kalshi 15-min | symmetric maker at fair ± 4¢ gets picked off: −6.6¢, t −6.1; replaced by M03/M04 | 26 trades, -6.62¢ |
| M02 | Maker quotes on Kalshi hourly ladder | as M01 on the hourly ladder: −3.1¢; replaced by M03/M04 | 26 trades, -3.12¢ |
| P01 | Polymarket 5-min Up/Down vs fair value | taker against our fair value: disproved on history (kx15_study) | 198 trades, -0.97¢ |
| P04 | Polymarket 4-hour Up/Down vs fair value | loses: −15¢, t −2.3 | 4 trades, -15.42¢ |
| P06 | Kalshi 15-min vs fair value | taker against our fair value on Kalshi 15m: −2…−3¢ on 20 317 markets | 19 trades, -2.29¢ |
| P07 | Kalshi hourly ladder vs fair value (every minute) | loses: −3.9¢, t −2.1 | 37 trades, -3.86¢ |
| V01 | 15-s fair value: lab blend (as P01/P02/P06) | 15-s fair value, ≈ 0 on 203 trades; the book is better than the model | 203 trades, 0.08¢ |
| V02 | 15-s fair value: Student-t(4) tails | −0.9¢ on 205 trades | 205 trades, -0.87¢ |
| V03 | 15-s fair value: variance implied by the Kalshi hourly ladder | −1.8¢ on 157 trades | 157 trades, -1.77¢ |
| V04 | 15-s fair value: DVOL level | ≈ 0 on 197 trades | 197 trades, -0.19¢ |
| V05 | 15-s fair value: realised variance of the last hour | −0.6¢ on 211 trades | 211 trades, -0.63¢ |
