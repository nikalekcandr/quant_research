# Strategy catalogue — every hypothesis of the alpha lab

_Generated 29.10.2026 17:25 MSK from `lab/signals.py` and the live ledger (voided trades excluded). Live numbers: settled trades, clusters = entry hours (the unit of independence), mean per trade, t over clusters. Rules are frozen; a changed rule gets a new ID._

Backtests: `studies/kx15_study.py` (taker vs model, 20 317 Kalshi 15m markets), `studies/kx15_flow_deep.py` and `studies/tape_strategies.py` (prints), `studies/kx_15m_vs_hourly.py`, `studies/invariance_vol.py`, `research.py` (R1–R21), `src/alpha.py` (ladders).


## A — No-arbitrage (binaries)

| ID | Strategy | Rule | Frozen | Live (lab) | History / backtest |
|---|---|---|---|---|---|
| A11 | Ladder monotonicity (Polymarket hourly, Kalshi hourly) | K1 < K2 and bid(K2) − ask(K1) > fees → buy YES K1, buy NO K2: pays ≥ 0 in every state | 2026-10-29 | 1 closed / 1 clusters, mean 0.01349 $/pair |  |
| A12 | Kalshi 15-min vs Polymarket 15-min | same quarter-hour window: Polymarket Up ask + Kalshi NO (1 − yes bid) + fees < 1 (or the mirror) → buy both: pays 1 unless the two references disagree | 2026-10-29 | 5 closed / 4 clusters, mean 0.06878 $/pair, t 1.5 | history test pending (Polymarket 15m download) |
| A12F | Kalshi 15-min vs Polymarket 15-min, every 15 s | as A12, sampled every 15 s | 2026-10-29 11:00 | 9 closed / 4 clusters, mean 0.02977 $/pair, t 1.8 |  |
| A13 | Kalshi hourly ladder vs Polymarket hourly ladder | same settlement hour: Polymarket 'above K' (Binance close) vs the Kalshi strike K' = K × USDT/USD within $30 (CF BRTI average); YES on the cheap venue + NO on the dear one for < 1 − fees − 1¢ → pays 1 unless the two settlement prices land on different sides | 2026-10-29 11:00 | 4 closed / 1 clusters, mean 0.01873 $/pair |  |

## D — Directional BTC

| ID | Strategy | Rule | Frozen | Live (lab) | History / backtest |
|---|---|---|---|---|---|
| D01 | Hourly reversal | at HH:00 take the opposite side of the last hour's return; hold 60 min | 2026-10-29 | 13 closed / 13 clusters, mean -0.001198 return, t -0.9 |  |
| D02 | Long outside US hours | long at 16:00 ET on NYSE days, exit at the next 09:30 ET open | 2026-10-29 | no settled trades |  |
| D03 | Short during US hours | short at 09:30 ET on NYSE days, exit at 16:00 ET | 2026-10-29 | no settled trades |  |
| D04 | Fade extreme funding | 30 min before each Binance funding: predicted rate > 0.015% → short, < 0.005% → long; hold 8 h | 2026-10-29 | no settled trades |  |
| D05 | Fade the perpetual premium | HH:30: z-score of the Binance premium index (last 7 days, hourly) > 2 → short, < −2 → long; hold 60 min | 2026-10-29 | 2 closed / 2 clusters, mean -0.002064 return |  |
| D06 | Order-book imbalance | every minute: Binance spot depth within 10 bp, (bid−ask)/(bid+ask) > 0.4 → long, < −0.4 → short; hold 5 min; one position at a time | 2026-10-29 | 138 closed / 16 clusters, mean -0.00107 return, t -9.3 |  |
| D07 | Taker-flow momentum | every 5 min: share of taker buys in the last 5 closed minutes > 0.6 → long, < 0.4 → short; hold 15 min | 2026-10-29 | 36 closed / 15 clusters, mean -0.0008025 return, t -2.9 |  |
| D08 | Coinbase premium momentum | every 15 min: change over 60 min of Coinbase BTC-USD / (Binance BTC-USDT × USDT-USD) > +5 bp → long 60 min, < −5 bp → short | 2026-10-29 | no settled trades |  |
| D09 | Coinbase → Binance lead-lag (15 s) | every 15 s: Coinbase BTC-USD moved more than Binance BTC-USDT over the last 15 s by > 3 bp → bet Binance follows for 60 s (run by the fast loop) | 2026-10-29 | 5 closed / 5 clusters, mean -0.002739 return, t -1.7 |  |
| D10 | Open interest × price | HH:01: open interest up > 1% in the last hour → follow the hour's price move; hold 60 min | 2026-10-29 | no settled trades |  |
| D11 | Fade retail positioning | every 4 h: z-score (last 7 days, hourly) of Binance's global long/short account ratio > 1.5 → short, < −1.5 → long; hold 4 h | 2026-10-29 | no settled trades |  |
| D12 | Implied-vol momentum | HH:02: z-score of the last hourly change of log DVOL (last 30 days) > 1 → long 60 min, < −1 → short | 2026-10-29 | no settled trades |  |
| D13 | Gradient boosting, next hour | HH:00: gradient-boosted trees on 20 price/volume/calendar features predict the next-hour return; > +15 bp → long, < −15 bp → short; retrained every 6 h on 180 days | 2026-10-29 | no settled trades |  |
| D14 | Ridge regression, next hour | same features and thresholds as D13, linear ridge model | 2026-10-29 | no settled trades |  |
| D15 | Weekly time-series momentum | 00:00 UTC: sign of the last 7-day return; hold 24 h | 2026-10-29 | no settled trades |  |
| D16 | Kalshi ladder drift | HH:06: the Kalshi hourly ladder's implied drift (median + s²/2 − log spot, in units of s) > 0.25 → long to HH+1:00, < −0.25 → short | 2026-10-29 | 9 closed / 9 clusters, mean -1.47e-05 return, t -0.0 |  |
| D17 | Follow the Polymarket hourly crowd | HH:05: mid of Polymarket's hourly Up/Down market > 0.60 → long to the end of the hour, < 0.40 → short | 2026-10-29 | 3 closed / 3 clusters, mean -5.243e-05 return, t -0.1 |  |
| D18 | Regime-conditioned trend | 00:05 UTC: bull trend (price > 200-day mean) and low-vol Markov state → long 24 h; bear and high-vol → short; otherwise flat | 2026-10-29 | no settled trades |  |
| D19 | Fade liquidation cascades | every minute: forced-order notional of the last 5 min on one side > max($3M, 4 × its 6-hour 95th percentile) → bet on a rebound (long after long liquidations); hold 30 min | 2026-10-29 11:00 | no settled trades |  |
| D20 | Aggressive-flow momentum (1 min) | every minute: (taker buys − taker sells)/(sum) of the last minute > 0.5 with notional > 3 × the 6-hour median minute → follow for 5 min (mirror for sells) | 2026-10-29 11:00 | 4 closed / 2 clusters, mean -0.0009215 return |  |

## F — Funding

| ID | Strategy | Rule | Frozen | Live (lab) | History / backtest |
|---|---|---|---|---|---|
| F01 | Cross-venue funding carry | 30 min before the 00/08/16 UTC funding: predicted rates of Binance, Bybit and OKX differ by > 2 bp → short the perp that pays most, long the one that pays least; close 10 min after the funding | 2026-10-29 | no settled trades |  |

## M — Maker quotes on Kalshi (simulated fills)

| ID | Strategy | Rule | Frozen | Live (lab) | History / backtest |
|---|---|---|---|---|---|
| M01 | Maker quotes on Kalshi 15-min | every minute: virtual bid at fair − 4¢ and ask at fair + 4¢ (only if inside or behind the spread), valid 60 s; filled only if a real trade prints strictly through our price; held to settlement; maker fee 0.0175·P(1−P) | 2026-10-29 11:00 | 24 closed / 4 clusters, mean -6.63¢, t -3.9 |  |
| M02 | Maker quotes on Kalshi hourly ladder | as M01, strikes with fair value 0.2–0.8 (at most 4), only ≥ 10 minutes before the hour | 2026-10-29 11:00 | 26 closed / 3 clusters, mean -3.12¢, t -1.3 |  |
| M03 | Lottery seller on Kalshi (maker, queue-aware) | 15-min market and hourly-ladder strikes: when a side costs 2–20¢ and ≥ 2 min are left, join its best offer with 1 contract; filled by prints through our price or by taker volume at our price beyond the displayed queue; held to settlement; maker fee | 2026-10-29 17:40 | no settled trades | backtest Oct 10–29, 107 markets: +5.3¢/trade (t 1.9), hit 0.91; halves +9.1 / +1.5¢; queue ignored |
| M04 | Lottery seller, cancels on spot moves | as M03; the quote is cancelled at the first 15-s sample in which Binance moved ≥ 3 bp toward the cheap side winning | 2026-10-29 17:40 | no settled trades | backtest: +8.7¢/trade (t 3.4), hit 0.95; halves +9.6 / +7.8¢; queue ignored |

## O — Options

| ID | Strategy | Rule | Frozen | Live (lab) | History / backtest |
|---|---|---|---|---|---|
| O01 | Cross-venue option arbitrage | same expiry/strike/type: bid on venue B − ask on venue A − both fees > 0 → buy A, sell B (identical payoff; settlement indices differ slightly) | 2026-10-29 | no settled trades |  |
| O02 | Put–call parity | expiries ≤ 7 days, each venue: (C − P + K) vs the perpetual on the same index (Deribit perp for Deribit/OKX, Binance perp for Binance/Bybit) minus the expected funding until expiry; a gap larger than option fees + 2 perp fees → trade the synthetic against the perp | 2026-10-29 | no settled trades | live: voided before the snapshot-age fix (one $200 move = 129 fake trades) |
| O03 | Static arbitrage of the smile | each venue and expiry: call/put prices non-monotone in the strike, or a negative-cost butterfly, after fees | 2026-10-29 | no settled trades |  |
| O04 | ATM straddle vs our volatility forecast | each venue, nearest daily expiry 3–30 h away, ATM strike: fair = E/S_T − K/ under our variance forecast; fair − ask − fees > 10% of the ask → buy; bid − fair − fees > 10% → sell; one trade per venue and expiry; settles on the 30-min average before 08:00 UTC | 2026-10-29 | no settled trades |  |

## P — Prediction markets: fair value

| ID | Strategy | Rule | Frozen | Live (lab) | History / backtest |
|---|---|---|---|---|---|
| P01 | Polymarket 5-min Up/Down vs fair value | buy the side whose fair value exceeds the ask + fee by ≥ 3¢; fair = Φ(log(S/S_start)/σ√τ) with the lab's seasonal variance forecast | 2026-10-29 | 193 closed / 16 clusters, mean -0.59¢, t -0.2 | history: model vs book loses (Kalshi 15m study, −2…−3¢) |
| P02 | Polymarket 15-min Up/Down vs fair value | as P01 | 2026-10-29 | 74 closed / 16 clusters, mean 4.61¢, t 1.2 |  |
| P03 | Polymarket hourly Up/Down vs fair value | as P01; settles on the Binance 1-hour candle | 2026-10-29 | 20 closed / 14 clusters, mean 1.27¢, t 0.2 |  |
| P04 | Polymarket 4-hour Up/Down vs fair value | as P01 | 2026-10-29 | 4 closed / 4 clusters, mean -15.42¢, t -2.3 |  |
| P05 | Polymarket hourly ladders vs fair value | 'Bitcoin above K at h ET': fair = Φ(log(S/K)/σ√τ) | 2026-10-29 | 28 closed / 12 clusters, mean -0.13¢, t -0.0 |  |
| P06 | Kalshi 15-min vs fair value | KXBTC15M: fair = Φ(log(S_usd/target)/σ√τ), S_usd = Coinbase/Kraken mid | 2026-10-29 | 18 closed / 4 clusters, mean -0.64¢, t -0.1 | history (20k Kalshi 15m markets): taker on model −2…−3¢, t −4…−6 |
| P07 | Kalshi hourly ladder vs fair value (every minute) | KXBTCD current hour, every strike, every minute; first signal per strike and side | 2026-10-29 | 37 closed / 3 clusters, mean -3.86¢, t -2.1 |  |

## T — Kalshi trade tape (taker)

| ID | Strategy | Rule | Frozen | Live (lab) | History / backtest |
|---|---|---|---|---|---|
| T01 | Copy the favourite buyer | Kalshi 15-min: ≥ 200 contracts bought by takers on the favourite side at 65–95¢ in the last 60 s → buy the favourite at the book ask; one per market | 2026-10-29 17:40 | no settled trades | backtest: +1.5¢ (t 0.4) — noise |
| T02 | Fade the lottery burst | Kalshi 15-min: ≥ 500 contracts bought by takers on the cheap side at ≤ 20¢ in the last 60 s → buy the favourite at the book ask; one per market | 2026-10-29 17:40 | no settled trades | backtest: +3.3¢ (t 1.4); halves +6.3 / +0.3¢ |

## V — Short binaries every 15 s: fair-value models

| ID | Strategy | Rule | Frozen | Live (lab) | History / backtest |
|---|---|---|---|---|---|
| V01 | 15-s fair value: lab blend (as P01/P02/P06) | every 15 s on Polymarket 5/15-min Up/Down and Kalshi 15-min: buy the side whose fair value beats ask + fee by ≥ 3¢; one trade per market and side | 2026-10-29 11:00 | 192 closed / 8 clusters, mean 0.30¢, t 0.1 |  |
| V02 | 15-s fair value: Student-t(4) tails | every 15 s on Polymarket 5/15-min Up/Down and Kalshi 15-min: buy the side whose fair value beats ask + fee by ≥ 3¢; one trade per market and side | 2026-10-29 11:00 | 194 closed / 8 clusters, mean -0.64¢, t -0.4 |  |
| V03 | 15-s fair value: variance implied by the Kalshi hourly ladder | every 15 s on Polymarket 5/15-min Up/Down and Kalshi 15-min: buy the side whose fair value beats ask + fee by ≥ 3¢; one trade per market and side | 2026-10-29 11:00 | 146 closed / 6 clusters, mean -1.58¢, t -0.9 |  |
| V04 | 15-s fair value: DVOL level | every 15 s on Polymarket 5/15-min Up/Down and Kalshi 15-min: buy the side whose fair value beats ask + fee by ≥ 3¢; one trade per market and side | 2026-10-29 11:00 | 186 closed / 8 clusters, mean -0.16¢, t -0.1 |  |
| V05 | 15-s fair value: realised variance of the last hour | every 15 s on Polymarket 5/15-min Up/Down and Kalshi 15-min: buy the side whose fair value beats ask + fee by ≥ 3¢; one trade per market and side | 2026-10-29 11:00 | 200 closed / 8 clusters, mean -0.46¢, t -0.4 |  |
| V06 | Stale-quote sniping (15 s) | as V01, but only when spot moved ≥ 4 bp in the last 15 s and the binary's mid moved ≤ 1¢ | 2026-10-29 11:00 | 4 closed / 3 clusters, mean -13.88¢, t -0.5 |  |
| V07 | Kalshi 15-min: the settlement average already printed | Kalshi settles on the average of the last 60 s of BRTI; in the last 75 s part of that average is known. Fair = P(average > strike) with the printed part, the rest as a Brownian average, level = our USD proxy shifted by the market's own start basis; take the book when edge ≥ 3¢ after fee, ≥ 10 s left | 2026-10-29 16:40 | no settled trades | live only (needs sub-minute data) |

## W — Selling cheap wings

| ID | Strategy | Rule | Frozen | Live (lab) | History / backtest |
|---|---|---|---|---|---|
| W01 | Sell every cheap wing (hourly ladders) | Kalshi and Polymarket hourly ladders, ≥ 15 min before the hour: a side priced 2–6¢ at the bid is sold at that bid (once per strike); held to settlement | 2026-10-29 14:45 | 51 closed / 3 clusters, mean -4.94¢, t -1.1 | history: selling ≤5¢ as taker ≈ 0 (Kalshi 1h −0.18¢, Polymarket 6d −1.1¢) |
| W02 | Sell cheap wings our model also calls overpriced | as W01, only if our fair value is ≥ 1¢ + fee below the wing's price | 2026-10-29 14:45 | 27 closed / 3 clusters, mean 3.57¢, t 2.2 | see W01 |
