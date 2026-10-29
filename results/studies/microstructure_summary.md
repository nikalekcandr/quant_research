# Microstructure tests (29.10.2026, 17:00 MSK)

## 1. Kyle–Obizhaeva / trading activity → volatility (`studies/invariance_vol.py`, Binance 1m 2023–2026)
Out of sample 2024-01 → 2026-10, expanding window refit monthly, QLIKE, DM with Newey–West vs seasonal HAR.

| Horizon | HAR | + trades N | + $ volume | + Kyle λ | + abs(flow imbalance) | + all | invariance (var/trade × trades) |
|---|---|---|---|---|---|---|---|
| 15 min | 0.459 | 0.435 (t 3.2) | 0.438 (t 3.5) | 0.452 (t 4.0) | 0.450 (t 3.9) | **0.430 (t 3.7)** | 0.449 (t 2.0) |
| 1 hour | 0.330 | 0.331 (t −2.4) | 0.329 (t 3.5) | 0.329 (t 3.6) | 0.330 (t 2.9) | 0.331 | 0.334 (t −4.1) |

Activity improves the 15-minute variance forecast by ~6% QLIKE, every year (2024, 2025, 2026); at 1 hour the gain
is ~0.5%. Use for short binaries' fair value and in the course project as a microstructure extension.

## 2. Signed taker flow → future returns (Binance, 2 M minutes)
Impact 3.1 bp per sd of flow in the same minute (t 152). Next minute: 0.003 bp (t 0.2); 5 min: −0.09 bp
(t −3.6); after the 0.1% largest sell flows +3.5 bp in 5 min. Real but far below a 10 bp round trip: no trade.

## 3. Who earns on Kalshi 15-minute markets (`studies/kx15_makers.py`, 105 markets, 2.25 M prints, Oct 2026)
Favourite–longshot bias in executed trades: takers who buy the cheap side lose −1.9¢ (≤5¢, t −19.7), −6.0¢
(5–10¢, t −6.5), −9.8¢ (10–20¢, t −4.3) per contract; takers who buy the favourite at 90–100¢ win +1.9…+5.3¢
(t 3.9…11.8). The resting side of lottery-ticket buys earns +1.9¢ (t 19.5) after the maker fee.
Only October 2026 so far; the 220-day sample is downloading (`jobs/hist_kx15_trades_sample_now.py`).

## 4. Where the taker losses come from (`studies/kx15_flow_deep.py`, 146 markets, 3.1 M prints, 10–29 Oct)
* Longshot buyers (paid ≤ 20¢) lose −4.8¢ per contract (t −6.2); without the 5 worst markets −4.4¢; leave-one-day-out
  −4.7 … −5.2¢. At ≤ 5¢: 98.6% of markets negative. Not a few whales: per print, per contract and per market
  (equal weight) give the same picture.
* Paid 10–20¢: average price 15.4¢, win rate 5.4% → −10.8¢. Markout: the price keeps moving against these buyers,
  −2…−3¢ within 1–3 minutes (t −3): they buy the cheap side while it is falling.
* Favourite buyers at 65–95¢ win +4.5…+7.7¢ and their markouts are positive (+1.6…+2.3¢ in 1 minute, t 2–3.3):
  fast takers who react to spot before the resting quotes move.
* Fixed-time book calibration (candles, Jan–Aug) shows only a ±0.5–1 pp bias: the mispricing is in *when* trades
  happen, not in the average price. A taker selling wings at a random moment (W01) earns ≈ 0; the resting side of
  the people who come to buy them earns. Market making must cancel when spot moves (the favourite side gets picked off).

## 5. 15-minute market vs hourly ladder on the same settlement (`studies/kx_15m_vs_hourly.py`)
The 15-minute market closing on the hour and the hourly ladder settle on the identical BRTI value (2 788 of 2 788).
Pure no-arbitrage violations after fees: 13 minutes in 10 of 1 206 markets (~2¢). Relative value: each book prices
its own contract better (15m mid Brier 0.186 vs ladder-interpolated 0.193, t −2.4; ladder 0.191 vs 15m-implied
0.203, t −2.8); trading either toward the other loses or is noise. The two Kalshi products are consistent.

## 6. Full sample reverses the October picture (29.10 18:20; 1 303 markets, 23.03–28.10, 18 M prints)
* Longshot buyers (≤ 20¢): −0.5¢ (t −0.7) instead of −4.8¢; ≤ 5¢: −0.8¢ (t −2.0). Makers net ≈ 0 in every month
  (−0.4 … +0.3¢); takers lose roughly the fee (−0.9 … −1.6¢ per contract, gross ≈ −0.3¢).
* Lottery seller backtest (`studies/tape_strategies.py`): M03 −1.9¢ (t −1.7), M04 −2.0¢ (t −1.8); by month
  −9.2 … +8.5¢, October the best month. T01 −3.7¢ (t −2.9), T02 −2.4¢ (t −2.3). Grid of price × time left chosen
  on the first half: nothing survives in the second.
* Lesson: one month (and one day of 40 markets) is not a sample. M03/M04/T01/T02 retired; live they lost $3.4 each
  in one hour: five adjacent ladder strikes sold in the same hour = one bet on BTC direction.

## 7. Kalshi 15m vs Polymarket 15m (`studies/kx_pm_15m_arb.py`, 4 272 windows 14.09–28.10)
Outcomes agree in 98.6% of windows (strikes differ by a median $2.5: BRTI vs Chainlink). History shows a fake
"arbitrage" (+2…+8¢, t 10–19): Polymarket history points are stamped at :20–:24 s, Kalshi candles at :00, and a
30-second offset alone gives a 4¢ gap sd (13¢ at 1 minute). Simultaneous books from the lab (same second, 669
samples, 16 windows): gap sd 1.8¢, cost of the pair < $1 in 3.7% of samples, best 98.2¢ — below the 1.4%
disagreement risk. No cross-venue arbitrage after fees.
