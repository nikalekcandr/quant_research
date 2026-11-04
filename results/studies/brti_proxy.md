# Our spot vs CF BRTI and the wings variance (studies/brti_proxy.py)

Settlements: 7074 hours, 2025-12-01 .. 2026-10-28. Venues with minutes: binance (USDT).

## A. Settlement value from the settlement minute's candle

| proxy          | price               |   raw bias bp |   raw sd bp |   raw MAE bp |   raw p95 |e| bp |   basis-adj n |   basis-adj bias bp |   basis-adj sd bp |   basis-adj MAE bp |   basis-adj p95 |e| bp |
|:---------------|:--------------------|--------------:|------------:|-------------:|-----------------:|--------------:|--------------------:|------------------:|-------------------:|-----------------------:|
| binance (USDT) | close               |         -5.33 |        6.13 |         6.52 |            15.09 |          7069 |               -0.09 |              3.58 |               2.48 |                   7.31 |
| binance (USDT) | typical (o+h+l+c)/4 |         -5.33 |        6.13 |         6.52 |            15.09 |          7069 |               -0.09 |              3.58 |               2.48 |                   7.31 |

## B. sd of log(E / (S_{h−τ} · basis)), bp — lower = closer to BRTI (the BRTI path is common)

| proxy          |   sd τ=1 |   sd τ=2 |   sd τ=5 |   sd τ=10 |   sd τ=15 |   sd τ=20 |   sd τ=30 |
|:---------------|---------:|---------:|---------:|----------:|----------:|----------:|----------:|
| binance (USDT) |     3.43 |      5.7 |    10.65 |     16.61 |     21.32 |     25.18 |     32.65 |

## C. Calibration of the wings variance: z = log(E/(S·basis)) / sd_model

Normal: P|z|>2 = 4.55%, P|z|>3 = 0.27%. Student t4 with unit variance: 3.7% and 1.2%. var(z) > 1 = the model's variance too low.

| variance from   |   τ min | half    |    n |   var(z) |   robust var(z) |   P|z|>2 % |   P|z|>3 % |
|:----------------|--------:|:--------|-----:|---------:|----------------:|-----------:|-----------:|
| binance (USDT)  |       5 | Dec–Aug | 5729 |    0.726 |           0.432 |      3.124 |      0.733 |
| binance (USDT)  |       5 | Sep–Nov | 1340 |    0.835 |           0.48  |      3.134 |      0.821 |
| binance (USDT)  |      10 | Dec–Aug | 5729 |    0.805 |           0.502 |      3.508 |      1.012 |
| binance (USDT)  |      10 | Sep–Nov | 1340 |    0.858 |           0.501 |      3.731 |      1.045 |
| binance (USDT)  |      15 | Dec–Aug | 5729 |    0.945 |           0.551 |      4.172 |      1.152 |
| binance (USDT)  |      15 | Sep–Nov | 1340 |    0.875 |           0.533 |      3.507 |      1.194 |
| binance (USDT)  |      20 | Dec–Aug | 5729 |    0.964 |           0.543 |      4.591 |      1.257 |
| binance (USDT)  |      20 | Sep–Nov | 1340 |    0.887 |           0.556 |      3.358 |      1.045 |
| binance (USDT)  |      30 | Dec–Aug | 5729 |    1.136 |           0.613 |      5.446 |      1.78  |
| binance (USDT)  |      30 | Sep–Nov | 1340 |    1.427 |           0.623 |      5.075 |      1.866 |

## C2. Reliability of the cheap-side fair value (the number the wings rules trade on)
First minute of each strike in the window; buckets of model fair; won = share that paid. fair > won = the model overprices the wing (we sell less than we should); fair < won = underprices (we sell wings that pay — the dangerous side).

| spot           | min left   | model   | fair bucket   |    n |   hours |   mean fair % |   won % |   won−fair pp |   se pp |   mean bid % |
|:---------------|:-----------|:--------|:--------------|-----:|--------:|--------------:|--------:|--------------:|--------:|-------------:|
| binance (USDT) | 5–15       | normal  | (0.0, 0.03]   | 5400 |    2273 |          0.94 |    1.5  |          0.56 |    0.21 |         1.69 |
| binance (USDT) | 5–15       | normal  | (0.03, 0.06]  | 1819 |    1358 |          4.37 |    3.24 |         -1.12 |    0.46 |         3.75 |
| binance (USDT) | 5–15       | normal  | (0.06, 0.1]   | 1694 |    1258 |          7.92 |    6.08 |         -1.84 |    0.59 |         6.15 |
| binance (USDT) | 5–15       | normal  | (0.1, 0.15]   | 1660 |    1245 |         12.43 |    8.67 |         -3.76 |    0.69 |         9.63 |
| binance (USDT) | 5–15       | normal  | (0.15, 0.2]   | 1567 |    1173 |         17.44 |   10.53 |         -6.91 |    0.77 |        12.8  |
| binance (USDT) | 5–15       | normal  | (0.2, 0.3]    | 2228 |    1497 |         24.27 |   15.75 |         -8.52 |    0.77 |        15.82 |
| binance (USDT) | 5–15       | t4      | (0.0, 0.03]   | 5633 |    2317 |          1.38 |    1.53 |          0.14 |    0.21 |         1.74 |
| binance (USDT) | 5–15       | t4      | (0.03, 0.06]  | 2574 |    1673 |          4.34 |    3.96 |         -0.38 |    0.43 |         4.49 |
| binance (USDT) | 5–15       | t4      | (0.06, 0.1]   | 2044 |    1433 |          7.89 |    7.93 |          0.03 |    0.59 |         8.5  |
| binance (USDT) | 5–15       | t4      | (0.1, 0.15]   | 1896 |    1336 |         12.4  |   10.65 |         -1.75 |    0.7  |        12.52 |
| binance (USDT) | 5–15       | t4      | (0.15, 0.2]   | 1422 |    1068 |         17.31 |   15.89 |         -1.42 |    0.96 |        15.35 |
| binance (USDT) | 5–15       | t4      | (0.2, 0.3]    | 1154 |     939 |         23.62 |   16.46 |         -7.16 |    1.11 |        17.02 |
| binance (USDT) | 15–30      | normal  | (0.0, 0.03]   | 9070 |    2561 |          0.84 |    2.28 |          1.44 |    0.34 |         1.78 |
| binance (USDT) | 15–30      | normal  | (0.03, 0.06]  | 2598 |    1733 |          4.35 |    3.85 |         -0.5  |    0.38 |         4.07 |
| binance (USDT) | 15–30      | normal  | (0.06, 0.1]   | 2265 |    1575 |          7.89 |    6.75 |         -1.13 |    0.53 |         6.71 |
| binance (USDT) | 15–30      | normal  | (0.1, 0.15]   | 2261 |    1547 |         12.5  |    9.86 |         -2.63 |    0.63 |        10.19 |
| binance (USDT) | 15–30      | normal  | (0.15, 0.2]   | 2163 |    1453 |         17.47 |   12.48 |         -4.99 |    0.68 |        13.9  |
| binance (USDT) | 15–30      | normal  | (0.2, 0.3]    | 3225 |    1743 |         23.99 |   16.37 |         -7.62 |    0.66 |        16.98 |
| binance (USDT) | 15–30      | t4      | (0.0, 0.03]   | 9410 |    2585 |          1.29 |    2.33 |          1.04 |    0.33 |         1.83 |
| binance (USDT) | 15–30      | t4      | (0.03, 0.06]  | 3577 |    2007 |          4.31 |    4.7  |          0.39 |    0.37 |         4.9  |
| binance (USDT) | 15–30      | t4      | (0.06, 0.1]   | 2734 |    1727 |          7.89 |    8.6  |          0.71 |    0.54 |         9.03 |
| binance (USDT) | 15–30      | t4      | (0.1, 0.15]   | 2645 |    1646 |         12.4  |   12.51 |          0.11 |    0.62 |        13.5  |
| binance (USDT) | 15–30      | t4      | (0.15, 0.2]   | 2198 |    1408 |         17.29 |   15.01 |         -2.28 |    0.76 |        16.54 |
| binance (USDT) | 15–30      | t4      | (0.2, 0.3]    | 1322 |     930 |         23.1  |   18.08 |         -5.02 |    1.1  |        18.08 |

## D. W03/W04 backtest (1 contract, taker) with another spot and variance × k
k is chosen on Jul–Aug only; Sep–Oct is the test. Binance × basis, k = 1 is the frozen rule.

| spot           |   var × k |   W03 Jul–Aug ¢/tr |   W03 Sep–Oct ¢/tr |   W04 Jul–Aug ¢/tr |   W04 Sep–Oct ¢/tr |   book Jul–Aug $/h |   book Jul–Aug t |   book Sep–Oct $/h |   book Sep–Oct t |   W03 Sep–Oct n |   W04 Sep–Oct n |
|:---------------|----------:|-------------------:|-------------------:|-------------------:|-------------------:|-------------------:|-----------------:|-------------------:|-----------------:|----------------:|----------------:|
| binance (USDT) |      0.8  |              1.989 |              3.637 |             -1.208 |              1.739 |             -0.004 |           -0.359 |              0.037 |            3.855 |             658 |            1188 |
| binance (USDT) |      0.9  |              3.378 |              3.549 |             -0.061 |              2.102 |              0.01  |            1.032 |              0.038 |            3.931 |             541 |            1093 |
| binance (USDT) |      1    |              2.286 |              3.651 |              1.234 |              1.771 |              0.018 |            1.782 |              0.033 |            3.306 |             427 |             980 |
| binance (USDT) |      1.1  |              5.079 |              2.424 |              1.036 |              1.539 |              0.023 |            2.149 |              0.024 |            2.245 |             344 |             872 |
| binance (USDT) |      1.25 |              3.821 |             -0.047 |              0.075 |              1.826 |              0.01  |            0.793 |              0.017 |            1.485 |             253 |             700 |
| binance (USDT) |      1.5  |              3.321 |              0.777 |             -1.276 |              2.238 |             -0.003 |           -0.18  |              0.024 |            1.818 |             148 |             471 |

## E. Live lab spot vs the settlement (lab ticks)
usd_mid = (Coinbase mid + Kraken mid)/2 is what W03/W04 use live, without a basis correction.

| lab column   |   τ min |   raw n |   raw bias bp |   raw sd bp |   raw MAE bp |   raw p95 |e| bp |   basis-adj bias bp |   basis-adj sd bp |   basis-adj MAE bp |   basis-adj p95 |e| bp |
|:-------------|--------:|--------:|--------------:|------------:|-------------:|-----------------:|--------------------:|------------------:|-------------------:|-----------------------:|
| usd_mid      |     0.5 |       0 |           nan |         nan |          nan |              nan |                 nan |               nan |                nan |                    nan |
| usd_mid      |     5   |       0 |           nan |         nan |          nan |              nan |                 nan |               nan |                nan |                    nan |
| usd_mid      |    15   |       0 |           nan |         nan |          nan |              nan |                 nan |               nan |                nan |                    nan |
| usd_mid      |    30   |       0 |           nan |         nan |          nan |              nan |                 nan |               nan |                nan |                    nan |
| cb_mid       |     0.5 |       0 |           nan |         nan |          nan |              nan |                 nan |               nan |                nan |                    nan |
| cb_mid       |     5   |       0 |           nan |         nan |          nan |              nan |                 nan |               nan |                nan |                    nan |
| cb_mid       |    15   |       0 |           nan |         nan |          nan |              nan |                 nan |               nan |                nan |                    nan |
| cb_mid       |    30   |       0 |           nan |         nan |          nan |              nan |                 nan |               nan |                nan |                    nan |
| kr_mid       |     0.5 |       0 |           nan |         nan |          nan |              nan |                 nan |               nan |                nan |                    nan |
| kr_mid       |     5   |       0 |           nan |         nan |          nan |              nan |                 nan |               nan |                nan |                    nan |
| kr_mid       |    15   |       0 |           nan |         nan |          nan |              nan |                 nan |               nan |                nan |                    nan |
| kr_mid       |    30   |       0 |           nan |         nan |          nan |              nan |                 nan |               nan |                nan |                    nan |
| bn_mid       |     0.5 |       0 |           nan |         nan |          nan |              nan |                 nan |               nan |                nan |                    nan |
| bn_mid       |     5   |       0 |           nan |         nan |          nan |              nan |                 nan |               nan |                nan |                    nan |
| bn_mid       |    15   |       0 |           nan |         nan |          nan |              nan |                 nan |               nan |                nan |                    nan |
| bn_mid       |    30   |       0 |           nan |         nan |          nan |              nan |                 nan |               nan |                nan |                    nan |
