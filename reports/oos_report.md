# Out-of-sample report

Specification frozen 2026-10-15; history = 2025-03-01 .. 2026-10-13; new = from 2026-10-13; live = forecasts in the hash-chained ledger.

**Model code identical to the frozen version:** yes  
**Live ledger:** 5 forecasts, chain intact

## 1 hour

| forecast | CRPS history (n=5478) | CRPS live (n=6) | CRPS new (n=344) |
|---|---|---|---|
| Last window | 0.428 | 0.386 | 0.437 |
| Past volatility (HAR) | 0.358 | 0.439 | 0.366 |
| + Calendar & jumps | 0.352 | 0.462 | 0.360 |
| + Options | 0.351 | 0.451 | 0.357 |
| + Ladder | 0.338 | 0.441 | 0.333 |
| Equal-weight blend | 0.372 | 0.509 | 0.370 |
| Market: full ladder | 0.390 | 0.463 | 0.392 |
| Market: ladder centre | 0.329 | 0.381 | 0.336 |
| Market: Deribit | 0.609 | 0.686 | 0.699 |

| source | Brier history (6200 ladders) | Brier live (2 ladders) | Brier new (355 ladders) |
|---|---|---|---|
| Market price | 0.1020 | 0.0729 | 0.0918 |
| Last window | 0.1038 |  | 0.0943 |
| Past volatility (HAR) | 0.1026 |  | 0.0930 |
| + Calendar & jumps | 0.1025 | 0.0763 | 0.0927 |
| + Options | 0.1024 | 0.0765 | 0.0927 |
| + Ladder | 0.1022 | 0.0764 | 0.0922 |
| Equal-weight blend | 0.1025 |  | 0.0924 |

Diebold–Mariano on CRPS (negative = A better):

| sample | A | B | n | mean_diff | dm | p |
|---|---|---|---|---|---|---|
| history | Past volatility (HAR) | Last window | 5478 | -0.070 | -25.181 | 0.000 |
| history | + Calendar & jumps | Past volatility (HAR) | 5478 | -0.006 | -5.934 | 0.000 |
| history | + Options | + Calendar & jumps | 5478 | -0.001 | -2.167 | 0.030 |
| history | + Ladder | + Options | 5478 | -0.013 | -11.475 | 0.000 |
| history | Equal-weight blend | + Ladder | 5478 | 0.034 | 16.873 | 0.000 |
| history | + Ladder | Market: ladder centre | 5478 | 0.009 | 3.754 | 0.000 |
| history | + Options | Market: ladder centre | 5478 | 0.022 | 8.219 | 0.000 |
| new | Past volatility (HAR) | Last window | 344 | -0.070 | -6.451 | 0.000 |
| new | + Calendar & jumps | Past volatility (HAR) | 344 | -0.006 | -0.861 | 0.390 |
| new | + Options | + Calendar & jumps | 344 | -0.002 | -1.864 | 0.063 |
| new | + Ladder | + Options | 344 | -0.024 | -4.370 | 0.000 |
| new | Equal-weight blend | + Ladder | 344 | 0.037 | 4.628 | 0.000 |
| new | + Ladder | Market: ladder centre | 344 | -0.002 | -0.306 | 0.760 |
| new | + Options | Market: ladder centre | 344 | 0.022 | 2.451 | 0.015 |

## 24 hours

| forecast | CRPS history (n=280) | CRPS new (n=11) |
|---|---|---|
| Last window | 0.409 | 0.335 |
| Past volatility (HAR) | 0.332 | 0.248 |
| + Calendar & jumps | 0.288 | 0.216 |
| + Options | 0.229 | 0.191 |
| + Ladder | 0.232 | 0.195 |
| Equal-weight blend | 0.235 | 0.200 |
| Market: full ladder | 0.307 | 0.258 |
| Market: ladder centre | 0.225 | 0.201 |
| Market: Deribit | 0.225 | 0.209 |

| source | Brier history (328 ladders) | Brier new (14 ladders) |
|---|---|---|
| Market price | 0.0387 | 0.0375 |
| Last window | 0.0396 | 0.0373 |
| Past volatility (HAR) | 0.0393 | 0.0381 |
| + Calendar & jumps | 0.0389 | 0.0378 |
| + Options | 0.0387 | 0.0389 |
| + Ladder | 0.0387 | 0.0390 |
| Equal-weight blend | 0.0387 | 0.0386 |

Diebold–Mariano on CRPS (negative = A better):

| sample | A | B | n | mean_diff | dm | p |
|---|---|---|---|---|---|---|
| history | Past volatility (HAR) | Last window | 280 | -0.077 | -6.327 | 0.000 |
| history | + Calendar & jumps | Past volatility (HAR) | 280 | -0.044 | -3.999 | 0.000 |
| history | + Options | + Calendar & jumps | 280 | -0.059 | -6.261 | 0.000 |
| history | + Ladder | + Options | 280 | 0.003 | 0.943 | 0.347 |
| history | Equal-weight blend | + Ladder | 280 | 0.003 | 0.702 | 0.483 |
| history | + Ladder | Market: ladder centre | 280 | 0.007 | 1.049 | 0.295 |
| history | + Options | Market: ladder centre | 280 | 0.004 | 0.737 | 0.462 |
| new | Past volatility (HAR) | Last window | 11 | -0.087 | -2.017 | 0.071 |
| new | + Calendar & jumps | Past volatility (HAR) | 11 | -0.033 | -0.933 | 0.373 |
| new | + Options | + Calendar & jumps | 11 | -0.025 | -0.461 | 0.654 |
| new | + Ladder | + Options | 11 | 0.004 | 0.458 | 0.657 |
| new | Equal-weight blend | + Ladder | 11 | 0.005 | 0.331 | 0.748 |
| new | + Ladder | Market: ladder centre | 11 | -0.006 | -0.570 | 0.582 |
| new | + Options | Market: ladder centre | 11 | -0.010 | -1.090 | 0.301 |

## 6 days

| forecast | CRPS history (n=325) | CRPS new (n=10) |
|---|---|---|
| Last window | 0.273 | 0.159 |
| Past volatility (HAR) | 0.260 | 0.167 |
| + Calendar & jumps | 0.250 | 0.163 |
| + Options | 0.263 | 0.174 |
| + Ladder | 0.285 | 0.153 |
| Equal-weight blend | 0.251 | 0.151 |
| Market: full ladder | 0.320 | 0.192 |
| Market: ladder centre | 0.268 | 0.142 |
| Market: Deribit | 0.246 | 0.123 |

| source | Brier history (325 ladders) | Brier new (10 ladders) |
|---|---|---|
| Market price | 0.0988 | 0.0515 |
| Last window | 0.0985 | 0.0537 |
| Past volatility (HAR) | 0.0978 | 0.0548 |
| + Calendar & jumps | 0.0976 | 0.0545 |
| + Options | 0.0980 | 0.0543 |
| + Ladder | 0.0978 | 0.0536 |
| Equal-weight blend | 0.0977 | 0.0540 |

Diebold–Mariano on CRPS (negative = A better):

| sample | A | B | n | mean_diff | dm | p |
|---|---|---|---|---|---|---|
| history | Past volatility (HAR) | Last window | 325 | -0.013 | -0.864 | 0.388 |
| history | + Calendar & jumps | Past volatility (HAR) | 325 | -0.011 | -2.302 | 0.022 |
| history | + Options | + Calendar & jumps | 325 | 0.013 | 1.336 | 0.182 |
| history | + Ladder | + Options | 325 | 0.022 | 1.333 | 0.183 |
| history | Equal-weight blend | + Ladder | 325 | -0.033 | -1.728 | 0.085 |
| history | + Ladder | Market: ladder centre | 325 | 0.017 | 0.921 | 0.358 |
| history | + Options | Market: ladder centre | 325 | -0.005 | -0.405 | 0.686 |
| new | Past volatility (HAR) | Last window | 10 | 0.008 | 0.176 | 0.864 |
| new | + Calendar & jumps | Past volatility (HAR) | 10 | -0.004 | -0.154 | 0.881 |
| new | + Options | + Calendar & jumps | 10 | 0.012 | 0.420 | 0.684 |
| new | + Ladder | + Options | 10 | -0.022 | -1.384 | 0.200 |
| new | Equal-weight blend | + Ladder | 10 | -0.002 | -0.274 | 0.790 |
| new | + Ladder | Market: ladder centre | 10 | 0.011 | 0.267 | 0.795 |
| new | + Options | Market: ladder centre | 10 | 0.032 | 0.585 | 0.573 |

## Spreads measured by the live logger

| venue_horizon | quotes | snapshots | median_spread_c | mean_spread_c |
|---|---|---|---|---|
| Polymarket ≤1.5 d | 330 | 67 | 1.00 | 0.92 |
| Polymarket 1.5–3 d | 443 | 67 | 1.00 | 1.10 |
| Polymarket 3–5.5 d | 1141 | 67 | 2.00 | 2.28 |
| Polymarket 5.5–8 d | 839 | 67 | 2.00 | 2.23 |
| Kalshi 1 h | 574 | 53 | 1.00 | 5.24 |

## Live forecasts vs the nightly replay of the same t

| horizon | model | n | corr | mean_abs_diff |
|---|---|---|---|---|
| kx1h | combo | 2 | 1.000 | 0.113 |
| kx1h | har | 6 | 1.000 | 0.008 |
| kx1h | har_m | 2 | 1.000 | 0.120 |
| kx1h | harx | 6 | 1.000 | 0.006 |
| kx1h | harx_all | 2 | 1.000 | 0.132 |
| kx1h | harx_d | 6 | 1.000 | 0.006 |
| kx1h | harx_m | 2 | 1.000 | 0.148 |
| kx1h | iv_mz | 6 | 1.000 | 0.003 |
| kx1h | iv_raw | 6 | 1.000 | 0.000 |
| kx1h | mkt_ln_raw | 2 | 1.000 | 0.039 |
| kx1h | mkt_mz | 2 | 1.000 | 0.333 |
| kx1h | mkt_raw | 2 | 1.000 | 0.123 |
| kx1h | mkt_t_raw | 2 | 1.000 | 0.082 |
| kx1h | naive | 6 | 1.000 | 0.000 |

## Code check

| file | frozen_sha256 | current_sha256 | identical |
|---|---|---|---|
| src/models.py | b2032433df42cf7c | b2032433df42cf7c | True |
| src/build.py | fdb5a6c94dd9f425 | fdb5a6c94dd9f425 | True |
| src/evaluate.py | 9d3099ef10815d5f | 9d3099ef10815d5f | True |
| src/measure/ladder.py | 6988dc20d8588819 | 6988dc20d8588819 | True |
| src/measure/realized.py | e224086ce0e0e28b | e224086ce0e0e28b | True |
| src/measure/deribit_iv.py | 97c02b077650330f | 97c02b077650330f | True |
