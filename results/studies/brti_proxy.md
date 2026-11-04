# BRTI and the wings models: what the CF methodology gives (04.11.2026)

**Short answer: the index methodology gives little accuracy, but the test exposed one real risk: W03 depends on which spot it is priced on.**
Data: 7 243 Kalshi hourly settlements (Dec 2025 – Nov 2026), 1-minute candles of Coinbase and Bitstamp (BRTI
constituents, ~496k minutes each), Binance BTC/USDT, the lab's own ticks 28.10–04.11. Code: `studies/brti_proxy.py`,
`studies/brti_leadlag.py`, `studies/brti_wings_robust.py`; tables below in this file.

## 1. How the index is made (CME CF Real Time Indices Methodology)
Constituent BTC-USD order books (Coinbase, Kraken, Bitstamp, Gemini, LMAX, ...) every 200 ms; stale books (> 10 s),
crossed or empty books and books > 5% from the median are dropped; sizes capped at trimmed mean + 5σ; consolidated
book uncrossed; index = mid price-volume curve weighted by λe^(−λv), λ = 1/(0.3 × utilized depth), depth where the
spread ≤ 5%. Kalshi KXBTCD settles on the average of the 60 BRTI seconds before the hour.

## 2. Findings
| # | Question | Result | So |
|---|---|---|---|
| 1 | Settlement from the settlement minute's candle | Coinbase/Bitstamp typical price (o+h+l+c)/4: sd 1.2–1.3 bp (~$10); close: 2.6–2.9 bp; Binance × basis: 3.6 bp | a USD-venue average replicates the 60-s average to ~$10 |
| 2 | Spot quality at decision time (sd of log(E/(S·basis))) | τ = 1 min: 2.7 bp (USD) vs 3.4 bp (Binance); τ = 5–30 min: 10.5 vs 10.7 … 32.5 vs 32.7 bp | the proxy error is 2–4% of the variance 5–30 min out: **not the bottleneck** |
| 3 | Live spot of the lab (usd_mid = (Coinbase + Kraken)/2, no basis) | bias +0.8 bp, sd 3.0 bp 30 s before the hour (80 hours); Binance raw −16 bp (USDT premium, 3× the year average) | the live spot is fine; Binance must never be used without the basis |
| 4 | Variance calibration, z = log(E/(S·b)) / sd_model | var(z) 0.64–0.95 at 5–20 min, 1.1–1.5 at 30 min; robust var 0.4–0.6; P(abs z > 3) ≈ 1% (normal 0.27%) | body narrower than the model, tails much fatter: a t4 shape, not a scale problem |
| 5 | Reliability of the cheap-side fair (the number the rules trade on) | normal, 5–15 min: fair 12.4% → paid 8.7%, fair 17.4% → 10.5%; t4: within ±2 pp from 3% to 20% | the normal model (W03) overprices wings; t4 (W04) is calibrated |
| 6 | Variance × k (k fitted on Jul–Aug) | best train k = 1.1 → Sep–Oct worse (book t 3.3 → 2.2) | keep k = 1 |
| 7 | Binance leads USD venues? | gap BN/USD predicts the next minute's USD move with β 0.06 (t 7) but nothing at 5–30 min | no edge for the hour |
| 8 | **W03/W04 on a USD spot instead of Binance × basis** | **W03 Sep–Oct: +3.65¢ (t 2.5) on Binance → +0.8¢ (t 0.4) on Coinbase, +1.2¢ (t 0.7) on Coinbase+Bitstamp.** W04: +1.8…+2.0¢ (t 2.4–2.7) on every spot | **W03 is fragile; W04 is robust** |

## 3. What this means
* The backtest of W03/W04 priced the spot as Binance × the 48-hour basis; the live lab prices on (Coinbase + Kraken)/2.
  A ±2 bp difference in spot changes which W03 trades pass the 1¢ filter, and the W03 test edge drops from 3.7¢ to
  about 1¢. Live expectation for W03: **≈ +1¢ per trade, not +3.7¢**; for the book about +2.3¢ per hour instead of
  +3.3¢. W04 carries the strategy.
* No new rule is frozen: W03 with t4 is better on Sep–Oct (+1.7…+2.7¢) but weak on Jul–Aug (−0.1…+1.4¢), which is
  the same kind of selection the frozen rules were chosen to avoid. Candidate for later: "W03 on t4" once more
  history exists.
* The BRTI copy from candles is accurate enough for anything near the strike (V07, settlement averaging): sd ≈ $10
  using the typical price of the last minute. A full order-book replica would need book data we do not store.
* What would move accuracy: the shock regime (z tails ≈ 4× normal). That is the storm filter in `docs/WINGS.md` §11,
  not the index.

---

# Our spot vs CF BRTI and the wings variance (studies/brti_proxy.py)

Settlements: 7243 hours, 2025-12-01 .. 2026-11-04. Venues with minutes: binance (USDT), coinbase, bitstamp, mean cb+bs, vol-wtd cb+bs.

## A. Settlement value from the settlement minute's candle

| proxy          | price               |   raw bias bp |   raw sd bp |   raw MAE bp |   raw p95 |e| bp |   basis-adj n |   basis-adj bias bp |   basis-adj sd bp |   basis-adj MAE bp |   basis-adj p95 |e| bp |
|:---------------|:--------------------|--------------:|------------:|-------------:|-----------------:|--------------:|--------------------:|------------------:|-------------------:|-----------------------:|
| binance (USDT) | close               |         -5.35 |        6.13 |         6.54 |            15.1  |          7092 |               -0.08 |              3.58 |               2.48 |                   7.31 |
| binance (USDT) | typical (o+h+l+c)/4 |         -5.35 |        6.13 |         6.54 |            15.1  |          7092 |               -0.08 |              3.58 |               2.48 |                   7.31 |
| coinbase       | close               |          0.61 |        2.91 |         2.04 |             6.03 |          7232 |               -0.01 |              2.89 |               1.92 |                   5.95 |
| coinbase       | typical (o+h+l+c)/4 |          0.63 |        1.36 |         1.1  |             2.91 |          7232 |                0.01 |              1.27 |               0.87 |                   2.5  |
| bitstamp       | close               |          0.45 |        2.61 |         1.7  |             5.44 |          7238 |               -0.02 |              2.6  |               1.62 |                   5.47 |
| bitstamp       | typical (o+h+l+c)/4 |          0.49 |        1.29 |         1    |             2.66 |          7238 |                0    |              1.25 |               0.88 |                   2.43 |
| mean cb+bs     | close               |          0.53 |        2.68 |         1.8  |             5.58 |          7232 |               -0.03 |              2.67 |               1.69 |                   5.58 |
| mean cb+bs     | typical (o+h+l+c)/4 |          0.56 |        1.25 |         0.99 |             2.62 |          7232 |                0.01 |              1.19 |               0.8  |                   2.35 |
| vol-wtd cb+bs  | close               |          0.59 |        2.83 |         1.96 |             5.87 |          7232 |               -0.02 |              2.81 |               1.85 |                   5.78 |
| vol-wtd cb+bs  | typical (o+h+l+c)/4 |          0.62 |        1.32 |         1.06 |             2.83 |          7232 |                0    |              1.23 |               0.84 |                   2.43 |

## B. sd of log(E / (S_{h−τ} · basis)), bp — lower = closer to BRTI (the BRTI path is common)

| proxy          |   sd τ=1 |   sd τ=2 |   sd τ=5 |   sd τ=10 |   sd τ=15 |   sd τ=20 |   sd τ=30 |
|:---------------|---------:|---------:|---------:|----------:|----------:|----------:|----------:|
| binance (USDT) |     3.44 |     5.7  |    10.65 |     16.6  |     21.31 |     25.17 |     32.64 |
| coinbase       |     2.71 |     5.27 |    10.46 |     16.39 |     21.08 |     24.96 |     32.48 |
| bitstamp       |     2.79 |     5.37 |    10.5  |     16.41 |     21.09 |     24.99 |     32.49 |
| mean cb+bs     |     2.68 |     5.29 |    10.46 |     16.39 |     21.08 |     24.97 |     32.48 |
| vol-wtd cb+bs  |     2.69 |     5.28 |    10.46 |     16.39 |     21.07 |     24.96 |     32.48 |

Same hours for every proxy:

| proxy          |   hours |   sd τ=1 |   sd τ=2 |   sd τ=5 |   sd τ=10 |   sd τ=15 |   sd τ=20 |   sd τ=30 |
|:---------------|--------:|---------:|---------:|---------:|----------:|----------:|----------:|----------:|
| binance (USDT) |    7091 |     3.44 |     5.7  |    10.66 |     16.6  |     21.32 |     25.18 |     32.65 |
| coinbase       |    7091 |     2.7  |     5.26 |    10.48 |     16.45 |     21.15 |     25.01 |     32.54 |
| bitstamp       |    7091 |     2.79 |     5.36 |    10.53 |     16.47 |     21.17 |     25.05 |     32.57 |
| mean cb+bs     |    7091 |     2.67 |     5.28 |    10.48 |     16.45 |     21.15 |     25.02 |     32.54 |
| vol-wtd cb+bs  |    7091 |     2.68 |     5.26 |    10.49 |     16.45 |     21.15 |     25.01 |     32.54 |

## C. Calibration of the wings variance: z = log(E/(S·basis)) / sd_model

Normal: P|z|>2 = 4.55%, P|z|>3 = 0.27%. Student t4 with unit variance: 3.7% and 1.2%. var(z) > 1 = the model's variance too low.

| variance from   |   τ min | half    |    n |   var(z) |   robust var(z) |   P|z|>2 % |   P|z|>3 % |
|:----------------|--------:|:--------|-----:|---------:|----------------:|-----------:|-----------:|
| binance (USDT)  |       5 | Dec–Aug | 5729 |    0.726 |           0.432 |      3.124 |      0.733 |
| binance (USDT)  |       5 | Sep–Nov | 1363 |    0.837 |           0.492 |      3.081 |      0.807 |
| binance (USDT)  |      10 | Dec–Aug | 5729 |    0.805 |           0.502 |      3.508 |      1.012 |
| binance (USDT)  |      10 | Sep–Nov | 1363 |    0.861 |           0.5   |      3.815 |      1.027 |
| binance (USDT)  |      15 | Dec–Aug | 5729 |    0.945 |           0.551 |      4.172 |      1.152 |
| binance (USDT)  |      15 | Sep–Nov | 1363 |    0.887 |           0.535 |      3.522 |      1.247 |
| binance (USDT)  |      20 | Dec–Aug | 5729 |    0.964 |           0.543 |      4.591 |      1.257 |
| binance (USDT)  |      20 | Sep–Nov | 1363 |    0.897 |           0.556 |      3.448 |      1.101 |
| binance (USDT)  |      30 | Dec–Aug | 5729 |    1.136 |           0.613 |      5.446 |      1.78  |
| binance (USDT)  |      30 | Sep–Nov | 1363 |    1.438 |           0.624 |      5.062 |      1.908 |
| coinbase        |       5 | Dec–Aug | 5721 |    0.638 |           0.376 |      2.622 |      0.577 |
| coinbase        |       5 | Sep–Nov | 1509 |    0.758 |           0.448 |      2.916 |      0.596 |
| coinbase        |      10 | Dec–Aug | 5721 |    0.757 |           0.467 |      3.146 |      0.926 |
| coinbase        |      10 | Sep–Nov | 1509 |    0.786 |           0.475 |      3.048 |      0.795 |
| coinbase        |      15 | Dec–Aug | 5721 |    0.898 |           0.524 |      3.933 |      1.066 |
| coinbase        |      15 | Sep–Nov | 1509 |    0.809 |           0.511 |      3.446 |      0.928 |
| coinbase        |      20 | Dec–Aug | 5722 |    0.925 |           0.526 |      4.422 |      1.171 |
| coinbase        |      20 | Sep–Nov | 1509 |    0.836 |           0.524 |      3.313 |      1.06  |
| coinbase        |      30 | Dec–Aug | 5721 |    1.101 |           0.605 |      5.244 |      1.8   |
| coinbase        |      30 | Sep–Nov | 1509 |    1.316 |           0.57  |      4.771 |      1.856 |
| bitstamp        |       5 | Dec–Aug | 5729 |    0.709 |           0.43  |      2.898 |      0.716 |
| bitstamp        |       5 | Sep–Nov | 1509 |    0.863 |           0.497 |      3.38  |      0.928 |
| bitstamp        |      10 | Dec–Aug | 5729 |    0.827 |           0.512 |      3.613 |      1.047 |
| bitstamp        |      10 | Sep–Nov | 1509 |    0.911 |           0.551 |      3.777 |      0.994 |
| bitstamp        |      15 | Dec–Aug | 5729 |    1.006 |           0.562 |      4.311 |      1.309 |
| bitstamp        |      15 | Sep–Nov | 1509 |    0.942 |           0.59  |      4.042 |      1.458 |
| bitstamp        |      20 | Dec–Aug | 5729 |    1.022 |           0.55  |      4.957 |      1.309 |
| bitstamp        |      20 | Sep–Nov | 1509 |    0.985 |           0.612 |      4.573 |      1.325 |
| bitstamp        |      30 | Dec–Aug | 5729 |    1.203 |           0.652 |      5.655 |      1.972 |
| bitstamp        |      30 | Sep–Nov | 1509 |    1.543 |           0.652 |      6.03  |      2.121 |
| mean cb+bs      |       5 | Dec–Aug | 5721 |    0.69  |           0.407 |      2.884 |      0.647 |
| mean cb+bs      |       5 | Sep–Nov | 1509 |    0.823 |           0.482 |      3.313 |      0.729 |
| mean cb+bs      |      10 | Dec–Aug | 5721 |    0.817 |           0.505 |      3.426 |      1.049 |
| mean cb+bs      |      10 | Sep–Nov | 1509 |    0.866 |           0.528 |      3.711 |      0.928 |
| mean cb+bs      |      15 | Dec–Aug | 5721 |    0.988 |           0.56  |      4.213 |      1.241 |
| mean cb+bs      |      15 | Sep–Nov | 1509 |    0.895 |           0.563 |      3.777 |      1.06  |
| mean cb+bs      |      20 | Dec–Aug | 5722 |    1.011 |           0.556 |      4.963 |      1.293 |
| mean cb+bs      |      20 | Sep–Nov | 1509 |    0.931 |           0.568 |      4.241 |      1.193 |
| mean cb+bs      |      30 | Dec–Aug | 5721 |    1.197 |           0.66  |      5.646 |      1.905 |
| mean cb+bs      |      30 | Sep–Nov | 1509 |    1.471 |           0.629 |      5.434 |      1.988 |
| vol-wtd cb+bs   |       5 | Dec–Aug | 5721 |    0.655 |           0.385 |      2.814 |      0.629 |
| vol-wtd cb+bs   |       5 | Sep–Nov | 1509 |    0.786 |           0.464 |      3.115 |      0.596 |
| vol-wtd cb+bs   |      10 | Dec–Aug | 5721 |    0.777 |           0.48  |      3.164 |      0.961 |
| vol-wtd cb+bs   |      10 | Sep–Nov | 1509 |    0.819 |           0.492 |      3.446 |      0.795 |
| vol-wtd cb+bs   |      15 | Dec–Aug | 5721 |    0.928 |           0.536 |      4.038 |      1.154 |
| vol-wtd cb+bs   |      15 | Sep–Nov | 1509 |    0.844 |           0.537 |      3.645 |      0.994 |
| vol-wtd cb+bs   |      20 | Dec–Aug | 5722 |    0.953 |           0.537 |      4.631 |      1.206 |
| vol-wtd cb+bs   |      20 | Sep–Nov | 1509 |    0.874 |           0.543 |      3.844 |      1.06  |
| vol-wtd cb+bs   |      30 | Dec–Aug | 5721 |    1.132 |           0.619 |      5.419 |      1.835 |
| vol-wtd cb+bs   |      30 | Sep–Nov | 1509 |    1.375 |           0.603 |      4.97  |      1.922 |

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
| coinbase       | 5–15       | normal  | (0.0, 0.03]   | 5196 |    2208 |          0.96 |    1.33 |          0.37 |    0.2  |         1.62 |
| coinbase       | 5–15       | normal  | (0.03, 0.06]  | 1886 |    1378 |          4.39 |    3.61 |         -0.78 |    0.47 |         3.47 |
| coinbase       | 5–15       | normal  | (0.06, 0.1]   | 1688 |    1269 |          7.92 |    5.33 |         -2.58 |    0.55 |         5.81 |
| coinbase       | 5–15       | normal  | (0.1, 0.15]   | 1658 |    1241 |         12.44 |    8.32 |         -4.12 |    0.67 |         9    |
| coinbase       | 5–15       | normal  | (0.15, 0.2]   | 1596 |    1152 |         17.44 |   10.59 |         -6.85 |    0.75 |        12.62 |
| coinbase       | 5–15       | normal  | (0.2, 0.3]    | 2586 |    1682 |         24.18 |   15.7  |         -8.48 |    0.71 |        16.22 |
| coinbase       | 5–15       | t4      | (0.0, 0.03]   | 5434 |    2258 |          1.41 |    1.44 |          0.03 |    0.21 |         1.67 |
| coinbase       | 5–15       | t4      | (0.03, 0.06]  | 2619 |    1714 |          4.33 |    4.05 |         -0.29 |    0.42 |         4.13 |
| coinbase       | 5–15       | t4      | (0.06, 0.1]   | 2036 |    1433 |          7.85 |    7.51 |         -0.34 |    0.59 |         7.98 |
| coinbase       | 5–15       | t4      | (0.1, 0.15]   | 1941 |    1326 |         12.39 |   10.15 |         -2.24 |    0.66 |        12.22 |
| coinbase       | 5–15       | t4      | (0.15, 0.2]   | 1727 |    1288 |         17.41 |   15.06 |         -2.36 |    0.85 |        15.61 |
| coinbase       | 5–15       | t4      | (0.2, 0.3]    | 1080 |     844 |         23.07 |   16.85 |         -6.22 |    1.17 |        17.71 |
| coinbase       | 15–30      | normal  | (0.0, 0.03]   | 8880 |    2534 |          0.88 |    2.26 |          1.38 |    0.34 |         1.75 |
| coinbase       | 15–30      | normal  | (0.03, 0.06]  | 2586 |    1712 |          4.35 |    3.79 |         -0.56 |    0.39 |         3.87 |
| coinbase       | 15–30      | normal  | (0.06, 0.1]   | 2300 |    1570 |          7.88 |    6.48 |         -1.4  |    0.52 |         6.43 |
| coinbase       | 15–30      | normal  | (0.1, 0.15]   | 2198 |    1503 |         12.47 |    9.6  |         -2.87 |    0.62 |         9.7  |
| coinbase       | 15–30      | normal  | (0.15, 0.2]   | 2266 |    1492 |         17.55 |   11.74 |         -5.81 |    0.66 |        13.65 |
| coinbase       | 15–30      | normal  | (0.2, 0.3]    | 3516 |    1828 |         24.03 |   16.38 |         -7.65 |    0.63 |        17.14 |
| coinbase       | 15–30      | t4      | (0.0, 0.03]   | 9222 |    2559 |          1.33 |    2.27 |          0.94 |    0.33 |         1.8  |
| coinbase       | 15–30      | t4      | (0.03, 0.06]  | 3594 |    2010 |          4.32 |    4.67 |          0.35 |    0.38 |         4.63 |
| coinbase       | 15–30      | t4      | (0.06, 0.1]   | 2673 |    1716 |          7.85 |    8.68 |          0.83 |    0.55 |         8.63 |
| coinbase       | 15–30      | t4      | (0.1, 0.15]   | 2751 |    1667 |         12.47 |   11.49 |         -0.98 |    0.6  |        13.21 |
| coinbase       | 15–30      | t4      | (0.15, 0.2]   | 2407 |    1513 |         17.39 |   15.91 |         -1.48 |    0.74 |        16.67 |
| coinbase       | 15–30      | t4      | (0.2, 0.3]    | 1280 |     866 |         22.61 |   17.19 |         -5.42 |    1.07 |        18.29 |
| mean cb+bs     | 5–15       | normal  | (0.0, 0.03]   | 5500 |    2276 |          0.9  |    1.45 |          0.55 |    0.21 |         1.69 |
| mean cb+bs     | 5–15       | normal  | (0.03, 0.06]  | 1832 |    1354 |          4.35 |    3.55 |         -0.8  |    0.48 |         3.72 |
| mean cb+bs     | 5–15       | normal  | (0.06, 0.1]   | 1668 |    1247 |          7.9  |    6.06 |         -1.84 |    0.59 |         6.21 |
| mean cb+bs     | 5–15       | normal  | (0.1, 0.15]   | 1628 |    1232 |         12.47 |    8.91 |         -3.57 |    0.7  |         9.51 |
| mean cb+bs     | 5–15       | normal  | (0.15, 0.2]   | 1591 |    1184 |         17.55 |   10.94 |         -6.61 |    0.77 |        13.03 |
| mean cb+bs     | 5–15       | normal  | (0.2, 0.3]    | 2414 |    1597 |         24.15 |   15.74 |         -8.41 |    0.74 |        16.39 |
| mean cb+bs     | 5–15       | t4      | (0.0, 0.03]   | 5743 |    2318 |          1.35 |    1.53 |          0.18 |    0.21 |         1.74 |
| mean cb+bs     | 5–15       | t4      | (0.03, 0.06]  | 2552 |    1679 |          4.32 |    3.96 |         -0.37 |    0.44 |         4.45 |
| mean cb+bs     | 5–15       | t4      | (0.06, 0.1]   | 1993 |    1392 |          7.85 |    8.28 |          0.43 |    0.61 |         8.45 |
| mean cb+bs     | 5–15       | t4      | (0.1, 0.15]   | 1938 |    1357 |         12.46 |   10.89 |         -1.58 |    0.69 |        12.66 |
| mean cb+bs     | 5–15       | t4      | (0.15, 0.2]   | 1596 |    1205 |         17.36 |   15.48 |         -1.88 |    0.89 |        15.82 |
| mean cb+bs     | 5–15       | t4      | (0.2, 0.3]    | 1017 |     794 |         22.97 |   16.22 |         -6.74 |    1.18 |        17.71 |
| mean cb+bs     | 15–30      | normal  | (0.0, 0.03]   | 9352 |    2574 |          0.83 |    2.3  |          1.47 |    0.33 |         1.82 |
| mean cb+bs     | 15–30      | normal  | (0.03, 0.06]  | 2498 |    1685 |          4.36 |    4.16 |         -0.2  |    0.41 |         4.2  |
| mean cb+bs     | 15–30      | normal  | (0.06, 0.1]   | 2231 |    1534 |          7.88 |    6.99 |         -0.89 |    0.54 |         6.85 |
| mean cb+bs     | 15–30      | normal  | (0.1, 0.15]   | 2180 |    1511 |         12.47 |    9.63 |         -2.84 |    0.63 |        10.25 |
| mean cb+bs     | 15–30      | normal  | (0.15, 0.2]   | 2241 |    1485 |         17.55 |   12.45 |         -5.1  |    0.69 |        14.14 |
| mean cb+bs     | 15–30      | normal  | (0.2, 0.3]    | 3260 |    1724 |         23.92 |   16.56 |         -7.35 |    0.66 |        17.26 |
| mean cb+bs     | 15–30      | t4      | (0.0, 0.03]   | 9688 |    2591 |          1.27 |    2.35 |          1.08 |    0.32 |         1.87 |
| mean cb+bs     | 15–30      | t4      | (0.03, 0.06]  | 3476 |    1968 |          4.33 |    4.86 |          0.53 |    0.39 |         5.02 |
| mean cb+bs     | 15–30      | t4      | (0.06, 0.1]   | 2649 |    1703 |          7.88 |    9.02 |          1.14 |    0.57 |         9.14 |
| mean cb+bs     | 15–30      | t4      | (0.1, 0.15]   | 2703 |    1655 |         12.49 |   12.21 |         -0.28 |    0.62 |        13.74 |
| mean cb+bs     | 15–30      | t4      | (0.15, 0.2]   | 2260 |    1438 |         17.32 |   15.88 |         -1.43 |    0.77 |        16.81 |
| mean cb+bs     | 15–30      | t4      | (0.2, 0.3]    | 1152 |     774 |         22.56 |   17.53 |         -5.03 |    1.13 |        18.4  |

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
| coinbase       |      0.8  |             -0.242 |              0.43  |             -0.768 |              1.748 |             -0.008 |           -0.761 |              0.019 |            2.029 |             460 |            1114 |
| coinbase       |      0.9  |              3.25  |              0.103 |              0.095 |              2.116 |              0.009 |            0.863 |              0.021 |            2.248 |             340 |             979 |
| coinbase       |      1    |              2.174 |              0.806 |              0.342 |              1.857 |              0.008 |            0.726 |              0.021 |            2.072 |             258 |             853 |
| coinbase       |      1.1  |              1.16  |             -0.796 |             -0.283 |              1.239 |             -0     |           -0.023 |              0.01  |            0.9   |             186 |             714 |
| coinbase       |      1.25 |              2.407 |             -3.296 |             -0.823 |              1.842 |             -0.003 |           -0.17  |              0.01  |            0.767 |             142 |             549 |
| coinbase       |      1.5  |             -2.618 |             -2.024 |             -1.457 |              0.636 |             -0.02  |           -0.981 |              0.001 |            0.085 |              84 |             346 |
| mean cb+bs     |      0.8  |              1.029 |              1.226 |             -1.193 |              1.205 |             -0.008 |           -0.789 |              0.018 |            1.81  |             539 |            1160 |
| mean cb+bs     |      0.9  |              3.256 |              1.362 |             -0.121 |              1.685 |              0.008 |            0.769 |              0.022 |            2.299 |             417 |            1067 |
| mean cb+bs     |      1    |              3.598 |              1.216 |              0.761 |              1.955 |              0.016 |            1.502 |              0.023 |            2.413 |             320 |             948 |
| mean cb+bs     |      1.1  |              3.588 |              0.667 |              0.734 |              2.114 |              0.015 |            1.35  |              0.023 |            2.235 |             246 |             823 |
| mean cb+bs     |      1.25 |              3.265 |             -0.866 |             -0.408 |              0.823 |              0.003 |            0.228 |              0.006 |            0.474 |             186 |             666 |
| mean cb+bs     |      1.5  |              4.288 |             -0.162 |             -0.69  |              0.575 |              0.003 |            0.188 |              0.005 |            0.368 |             111 |             447 |

## E. Live lab spot vs the settlement (lab ticks)
usd_mid = (Coinbase mid + Kraken mid)/2 is what W03/W04 use live, without a basis correction.

| lab column   |   τ min |   raw n |   raw bias bp |   raw sd bp |   raw MAE bp |   raw p95 |e| bp |   basis-adj bias bp |   basis-adj sd bp |   basis-adj MAE bp |   basis-adj p95 |e| bp |
|:-------------|--------:|--------:|--------------:|------------:|-------------:|-----------------:|--------------------:|------------------:|-------------------:|-----------------------:|
| usd_mid      |     0.5 |      80 |          0.82 |        3    |         1.8  |             4.64 |                0.61 |              2.98 |               1.74 |                   4.51 |
| usd_mid      |     5   |      79 |          1.91 |       10.71 |         6.45 |            22.94 |                1.52 |             10.92 |               6.51 |                  23.71 |
| usd_mid      |    15   |      81 |          1.18 |       16.37 |        11.8  |            35    |                3    |             15.09 |              11.37 |                  29.75 |
| usd_mid      |    30   |      76 |         -0.81 |       34.31 |        19.55 |            52.94 |                0.84 |             32.85 |              18.85 |                  50.7  |
| cb_mid       |     0.5 |      80 |          1.32 |        3.06 |         2.03 |             5.44 |                0.63 |              3.16 |               1.82 |                   4.97 |
| cb_mid       |     5   |      79 |          2.49 |       10.95 |         6.86 |            22.62 |                1.32 |             11.2  |               6.78 |                  24.48 |
| cb_mid       |    15   |      81 |          1.69 |       16.5  |        11.89 |            35.25 |                2.49 |             15.36 |              11.6  |                  29.19 |
| cb_mid       |    30   |      76 |         -0.2  |       34.29 |        19.56 |            52.41 |                0.82 |             32.88 |              18.88 |                  51.04 |
| kr_mid       |     0.5 |      80 |          0.31 |        3.04 |         1.86 |             5.09 |                0.73 |              2.96 |               1.89 |                   4.85 |
| kr_mid       |     5   |      78 |          1.48 |       10.5  |         6.16 |            22.55 |                1.72 |             10.69 |               6.33 |                  23.28 |
| kr_mid       |    15   |      84 |          0.84 |       16.05 |        11.54 |            33.01 |                3.42 |             14.72 |              11.17 |                  29.36 |
| kr_mid       |    30   |      76 |         -1.41 |       34.35 |        19.62 |            53.47 |                0.89 |             32.85 |              18.87 |                  50.51 |
| bn_mid       |     0.5 |      80 |        -16.32 |        4    |        16.32 |            21.68 |               -0.5  |              4.58 |               3.61 |                   8.46 |
| bn_mid       |     5   |      79 |        -14.98 |       10.56 |        16.06 |            27.53 |               -0.03 |             10.92 |               6.81 |                  23.81 |
| bn_mid       |    15   |      85 |        -15.9  |       16.11 |        18.86 |            39.65 |                1.69 |             15.04 |              11.55 |                  29.9  |
| bn_mid       |    30   |      76 |        -17.75 |       34.16 |        25.83 |            57.26 |                0.58 |             33.17 |              19.73 |                  50.37 |
