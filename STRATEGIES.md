# Strategy catalogue — alpha lab

_29.10.2026 20:00 MSK. Live: settled trades / clusters (entry hours), mean per trade, t over clusters; voided trades excluded. Rules are frozen; a changed rule gets a new ID. Retired rules no longer run (their data feeds are switched off where nothing else needs them); their trades stay in the ledger._

**Bottom line:** no rule has survived the full history after fees and multiple testing. The lab now runs only rules with live or historical promise.


## Active (19)


### A — No-arbitrage (binaries)

| ID | Strategy | Rule | Frozen | Live | History / backtest |
|---|---|---|---|---|---|
| A11 | Ladder monotonicity (Polymarket hourly, Kalshi hourly) | K1 < K2 and bid(K2) − ask(K1) > fees → buy YES K1, buy NO K2: pays ≥ 0 in every state | 2026-10-29 | 1 / 1 cl., 0.01349 $/pair |  |
| A12 | Kalshi 15-min vs Polymarket 15-min | same quarter-hour window: Polymarket Up ask + Kalshi NO (1 − yes bid) + fees < 1 (or the mirror) → buy both: pays 1 unless the two references disagree | 2026-10-29 | 5 / 4 cl., 0.06878 $/pair, t 1.5 | history not measurable (20–40 s timestamp offset); simultaneous books: pair < $1 in 3.7% of samples |
| A12F | Kalshi 15-min vs Polymarket 15-min, every 15 s | as A12, sampled every 15 s | 2026-10-29 11:00 | 10 / 4 cl., 0.02805 $/pair, t 1.9 | see A12 |
| A13 | Kalshi hourly ladder vs Polymarket hourly ladder | same settlement hour: Polymarket 'above K' (Binance close) vs the Kalshi strike K' = K × USDT/USD within $30 (CF BRTI average); YES on the cheap venue + NO on the dear one for < 1 − fees − 1¢ → pays 1 unless the two settlement prices land on different sides | 2026-10-29 11:00 | 4 / 1 cl., 0.01873 $/pair |  |

### D — Directional BTC

| ID | Strategy | Rule | Frozen | Live | History / backtest |
|---|---|---|---|---|---|
| D02 | Long outside US hours | long at 16:00 ET on NYSE days, exit at the next 09:30 ET open | 2026-10-29 | — | R2 likely (+21 bp/day outside US hours) |
| D03 | Short during US hours | short at 09:30 ET on NYSE days, exit at 16:00 ET | 2026-10-29 | — | R2 |
| D12 | Implied-vol momentum | HH:02: z-score of the last hourly change of log DVOL (last 30 days) > 1 → long 60 min, < −1 → short | 2026-10-29 | — | R12 likely (IV changes lead returns) |

### F — Funding

| ID | Strategy | Rule | Frozen | Live | History / backtest |
|---|---|---|---|---|---|
| F01 | Cross-venue funding carry | 30 min before the 00/08/16 UTC funding: predicted rates of Binance, Bybit and OKX differ by > 2 bp → short the perp that pays most, long the one that pays least; close 10 min after the funding | 2026-10-29 | — |  |

### O — Options

| ID | Strategy | Rule | Frozen | Live | History / backtest |
|---|---|---|---|---|---|
| O01 | Cross-venue option arbitrage | same expiry/strike/type: bid on venue B − ask on venue A − both fees > 0 → buy A, sell B (identical payoff; settlement indices differ slightly) | 2026-10-29 | — |  |
| O02 | Put–call parity | expiries ≤ 7 days, each venue: (C − P + K) vs the perpetual on the same index (Deribit perp for Deribit/OKX, Binance perp for Binance/Bybit) minus the expected funding until expiry; a gap larger than option fees + 2 perp fees → trade the synthetic against the perp | 2026-10-29 | — |  |
| O03 | Static arbitrage of the smile | each venue and expiry: call/put prices non-monotone in the strike, or a negative-cost butterfly, after fees | 2026-10-29 | — |  |
| O04 | ATM straddle vs our volatility forecast | each venue, nearest daily expiry 3–30 h away, ATM strike: fair = E/S_T − K/ under our variance forecast; fair − ask − fees > 10% of the ask → buy; bid − fair − fees > 10% → sell; one trade per venue and expiry; settles on the 30-min average before 08:00 UTC | 2026-10-29 | — |  |

### P — Prediction markets: fair value

| ID | Strategy | Rule | Frozen | Live | History / backtest |
|---|---|---|---|---|---|
| P02 | Polymarket 15-min Up/Down vs fair value | as P01 | 2026-10-29 | 77 / 16 cl., 2.69¢, t 0.7 |  |
| P03 | Polymarket hourly Up/Down vs fair value | as P01; settles on the Binance 1-hour candle | 2026-10-29 | 20 / 14 cl., 1.27¢, t 0.2 |  |
| P05 | Polymarket hourly ladders vs fair value | 'Bitcoin above K at h ET': fair = Φ(log(S/K)/σ√τ) | 2026-10-29 | 34 / 13 cl., 2.02¢, t 0.4 |  |

### V — Short binaries every 15 s: fair-value models

| ID | Strategy | Rule | Frozen | Live | History / backtest |
|---|---|---|---|---|---|
| V07 | Kalshi 15-min: the settlement average already printed | Kalshi settles on the average of the last 60 s of BRTI; in the last 75 s part of that average is known. Fair = P(average > strike) with the printed part, the rest as a Brownian average, level = our USD proxy shifted by the market's own start basis; take the book when edge ≥ 3¢ after fee, ≥ 10 s left | 2026-10-29 16:40 | — | live only |

### W — Selling cheap wings

| ID | Strategy | Rule | Frozen | Live | History / backtest |
|---|---|---|---|---|---|
| W02 | Sell cheap wings our model also calls overpriced | as W01, only if our fair value is ≥ 1¢ + fee below the wing's price | 2026-10-29 14:45 | 30 / 4 cl., 3.47¢, t 2.1 | history (Kalshi ladder Jul–Oct): 2–6¢ wings with a model filter — in the grid, near 0 |
| W03 | Sell the most overpriced 10–20¢ wing, last 5–15 min (Kalshi hourly) | Kalshi hourly ladder, 5–15 minutes before the hour: among strikes whose cheap side bids 10–20¢, sell the one whose bid exceeds our fair value + fee by the most, if by ≥ 1¢; one per hour; held to settlement | 2026-10-29 20:00 | — | chosen on Jul–Aug: +3.0¢ (t 1.6); Sep–Oct test +3.6¢/hour (t 2.7) |
| W04 | Sell the most overpriced 6–10¢ wing, 15–30 min left (Kalshi hourly) | as W03 with cheap bids 6–10¢, 15–30 minutes left, Student-t(4) fair value, edge ≥ 0 | 2026-10-29 20:00 | — | Jul–Aug +1.2¢ (t 1.3); Sep–Oct test +1.8¢ (t 2.5) |


## Retired (34)

| ID | Strategy | Why retired | Live at retirement |
|---|---|---|---|
| D01 | Hourly reversal | −0.20% per trade on 14 trades; R3/R4 not supported | 14 trades, -0.00201 return |
| D04 | Fade extreme funding | R11 not supported (funding does not predict returns) | — |
| D05 | Fade the perpetual premium | −0.35% per trade on 3 trades | 3 trades, -0.003478 return |
| D06 | Order-book imbalance | loses: −10 bp per trade, t −8.7, hit 8% | 142 trades, -0.001009 return |
| D07 | Taker-flow momentum | loses: t −2.8 | 38 trades, -0.0007532 return |
| D08 | Coinbase premium momentum | no signal in 30 h, no historical support | — |
| D09 | Coinbase → Binance lead-lag (15 s) | −0.20% per trade on 9 trades (Coinbase→Binance lead is contemporaneous) | 9 trades, -0.002025 return |
| D10 | Open interest × price | no signal, no historical support | — |
| D11 | Fade retail positioning | no signal, no historical support | — |
| D13 | Gradient boosting, next hour | R20: ML does not predict the next-hour return | — |
| D14 | Ridge regression, next hour | R20: ML does not predict the next-hour return | — |
| D15 | Weekly time-series momentum | R4 not supported (weekly momentum) | — |
| D16 | Kalshi ladder drift | −0.13% per trade on 10 trades | 10 trades, -0.001323 return |
| D17 | Follow the Polymarket hourly crowd | ≈ 0 on 3 trades; R9 not supported (ladders do not know the direction) | 3 trades, -5.243e-05 return |
| D18 | Regime-conditioned trend | no signal, R19 negative | — |
| D19 | Fade liquidation cascades | no signal in 30 h | — |
| D20 | Aggressive-flow momentum (1 min) | −0.09% per trade; Binance flow reverses by 0.1 bp — far below fees | 4 trades, -0.0009215 return |
| M01 | Maker quotes on Kalshi 15-min | symmetric maker at fair ± 4¢ gets picked off: −6.6¢, t −6.1; replaced by M03/M04 | 27 trades, -7.96¢ |
| M02 | Maker quotes on Kalshi hourly ladder | as M01 on the hourly ladder: −3.1¢; replaced by M03/M04 | 39 trades, -1.03¢ |
| M03 | Lottery seller on Kalshi (maker, queue-aware) | 220-day backtest −1.9¢ (t −1.7), October +8.5¢ was the outlier month; ladder strikes uncapped (5 correlated losses in one hour live) | 21 trades, -16.15¢ |
| M04 | Lottery seller, cancels on spot moves | as M03: −2.0¢ (t −1.8) on 1 192 markets | 20 trades, -17.21¢ |
| P01 | Polymarket 5-min Up/Down vs fair value | taker against our fair value: disproved on history (kx15_study) | 200 trades, -1.35¢ |
| P04 | Polymarket 4-hour Up/Down vs fair value | loses: −15¢, t −2.3 | 4 trades, -15.42¢ |
| P06 | Kalshi 15-min vs fair value | taker against our fair value on Kalshi 15m: −2…−3¢ on 20 317 markets | 20 trades, -3.42¢ |
| P07 | Kalshi hourly ladder vs fair value (every minute) | loses: −3.9¢, t −2.1 | 47 trades, 1.21¢ |
| T01 | Copy the favourite buyer | 220-day backtest −3.7¢ (t −2.9) | 3 trades, 23.00¢ |
| T02 | Fade the lottery burst | 220-day backtest −2.4¢ (t −2.3) | 1 trades, 11.00¢ |
| V01 | 15-s fair value: lab blend (as P01/P02/P06) | 15-s fair value, ≈ 0 on 203 trades; the book is better than the model | 207 trades, -0.68¢ |
| V02 | 15-s fair value: Student-t(4) tails | −0.9¢ on 205 trades | 209 trades, -1.60¢ |
| V03 | 15-s fair value: variance implied by the Kalshi hourly ladder | −1.8¢ on 157 trades | 161 trades, -2.55¢ |
| V04 | 15-s fair value: DVOL level | ≈ 0 on 197 trades | 201 trades, -0.99¢ |
| V05 | 15-s fair value: realised variance of the last hour | −0.6¢ on 211 trades | 215 trades, -1.35¢ |
| V06 | Stale-quote sniping (15 s) | −12.3¢ per trade on 7 trades; stale quotes are caught within 15 s | 7 trades, -12.33¢ |
| W01 | Sell every cheap wing (hourly ladders) | −9.9¢ live (12 losses of 94), ≈ 0 on history; 4 months of Kalshi ladder candles: −0.9¢, t −4.4 | 94 trades, -9.87¢ |
