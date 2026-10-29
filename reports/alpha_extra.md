# Historical follow-ups

## X1. Kalshi fee rounded up to the cent
| fee | model | threshold | trades | pnl_per_contract | t |
|---|---|---|---|---|---|
| exact fee (as in the project) | harx | 0.0200 | 12552 | -0.0093 | -2.5354 |
| exact fee (as in the project) | harx | 0.0500 | 2867 | 0.0035 | 0.4309 |
| exact fee (as in the project) | harx_d | 0.0200 | 12474 | -0.0070 | -1.8627 |
| exact fee (as in the project) | harx_d | 0.0500 | 2895 | -0.0005 | -0.0602 |
| rounded up, 1 contract per order | harx | 0.0200 | 12552 | -0.0141 | -3.8674 |
| rounded up, 1 contract per order | harx | 0.0500 | 2867 | -0.0013 | -0.1585 |
| rounded up, 1 contract per order | harx_d | 0.0200 | 12474 | -0.0118 | -3.1662 |
| rounded up, 1 contract per order | harx_d | 0.0500 | 2895 | -0.0053 | -0.6473 |
| rounded up, 10 contracts per order | harx | 0.0200 | 12552 | -0.0097 | -2.6681 |
| rounded up, 10 contracts per order | harx | 0.0500 | 2867 | 0.0030 | 0.3718 |
| rounded up, 10 contracts per order | harx_d | 0.0200 | 12474 | -0.0074 | -1.9928 |
| rounded up, 10 contracts per order | harx_d | 0.0500 | 2895 | -0.0010 | -0.1198 |

## X2. Edge by minutes before settlement (Kalshi, model = + Options, hybrid rule 5c)
| minutes_before | snapshots | strikes | brier_market | brier_model | brier_hybrid | dm_hybrid_vs_market | p | median_spread_c | hyb_trades | hyb_pnl_per_contract | hyb_t |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 55 | 2557 | 27551 | 0.0914 | 0.0915 | 0.0914 | -0.2924 | 0.7700 | 1.0000 | 937 | 0.0310 | 1.8459 |
| 40 | 2399 | 23094 | 0.0913 | 0.0919 | 0.0916 | 1.4591 | 0.1447 | 1.0000 | 1430 | 0.0078 | 0.6514 |
| 25 | 2096 | 17689 | 0.0857 | 0.0868 | 0.0865 | 3.0787 | 0.0021 | 1.0000 | 1960 | -0.0041 | -0.4357 |
| 10 | 1393 | 8373 | 0.0877 | 0.0908 | 0.0900 | 5.4396 | 0.0000 | 1.0000 | 1569 | -0.0301 | -3.1538 |

## X3. Box arbitrage KXBTCD vs KXBTC, minute by minute
| how | cases | minutes | median_edge | max_edge |
|---|---|---|---|---|
| long box | 58 | 47 | 0.0100 | 0.8900 |
| short box | 31 | 25 | 0.0200 | 0.1000 |

167263 event-minutes checked
