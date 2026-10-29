# Alpha on real order books

Rules fixed on 2026-10-28; thresholds {'B1': 0.03, 'B2': 0.03, 'B3': 0.03, 'C1': 0.05, 'D1': 0.05, 'E1': 0.02, 'F1': 0.03}; Kalshi fee 0.07·P(1−P) rounded up to the cent; first signal per (event, strike, side), held to settlement; t over settlement events; Holm across rows.

Books: Polymarket ladders 67 snapshots, Kalshi 58, Kalshi ranges 28, Deribit 68, Binance options 28, other markets 28, fast logger 0.

## Trading hypotheses
| hyp | venue | signals | settled | events | pnl_per_contract | hit_rate | pnl_total | pnl_per_event | t | p_one_sided | p_holm |
|---|---|---|---|---|---|---|---|---|---|---|---|
| B1 | polymarket | 9 | 8 | 3 | 0.2950 | 0.7500 | 2.3600 |  |  |  |  |
| B2 | polymarket | 23 | 20 | 10 | 0.1066 | 0.8500 | 2.1330 | 0.2133 | 1.9312 | 0.0267 | 0.1069 |
| B3 bybit | polymarket | 7 | 0 | 0 |  |  | 0.0000 |  |  |  |  |
| B3 okx | polymarket | 7 | 0 | 0 |  |  | 0.0000 |  |  |  |  |
| C1 | kalshi | 158 | 145 | 12 | -0.0822 | 0.7379 | -11.9200 | -0.9933 | -1.7117 | 0.9565 | 1.0000 |
| C1 | polymarket | 198 | 125 | 14 | -0.0536 | 0.9360 | -6.7020 | -0.4787 | -1.0063 | 0.8429 | 1.0000 |
| D1 | kalshi | 6 | 5 | 5 | 0.2588 | 0.6000 | 1.2940 |  |  |  |  |
| D1 | polymarket | 16 | 0 | 0 |  |  | 0.0000 |  |  |  |  |
| F1 | limitless | 16 | 14 | 13 | -0.1348 | 0.2143 | -1.8870 | -0.1452 | -1.1872 | 0.8824 | 1.0000 |

## Risk-free arbitrage found (A1 ladder monotonicity, A2 Kalshi above vs ranges, A3 Limitless vs Polymarket)
| hyp | venue | cases | snapshots | median_edge | max_edge | max_profit_usd |
|---|---|---|---|---|---|---|
| A1 | kalshi | 26 | 1 | 0.3900 | 0.9500 | 0.0000 |
| A1 | polymarket | 97 | 34 | 0.0040 | 0.0060 | 0.0500 |
| A2 | kalshi | 78 | 3 | 0.1100 | 0.4000 | 0.4000 |
| A3 | limitless+polymarket | 29 | 18 | 0.0300 | 0.3200 | 0.3200 |
| A4 | polymarket | 398 | 27 | 0.0050 | 0.0500 | 0.0500 |

## Who is more accurate on settled quotes: the market or the fair price?
| comparison | quotes | events | brier_market_mid | brier_fair | mean_gap_mid_minus_fair |
|---|---|---|---|---|---|
| Polymarket ladder vs Deribit smile | 1468 | 10 | 0.0399 | 0.0391 | 0.0002 |
| other Bitcoin markets vs Deribit smile | 18 | 16 | 0.1475 | 0.1448 | -0.0072 |
