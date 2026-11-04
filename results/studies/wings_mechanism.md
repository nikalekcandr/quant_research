# Why the hourly wings earn (04.11.2026) — result

**W03 and W04 earn for different reasons.**

| Test | W03 (10–20¢, 5–15 min, normal) | W04 (6–10¢, 15–30 min, t4) |
|---|---|---|
| Model edge on the same strike after 0 / 1 / 3 / 5 min | 3.5 → 2.0 → 0.65 → 0.2¢ | 1.7 → 1.4 → 0.9 → 0.5¢ |
| Same trade sold 0 / 1 / 2 / 3 / 5 min later | +3.2 → +2.1 → +1.3 → +0.7 → +0.05¢ | +1.55 → +1.4 → +1.4 → +1.5 → +1.3¢ |
| Excess fall of the wing's bid after our sale (vs unselected wings) | +0.6…+0.9¢ (t 2–3) | +0.2¢ (t 2) |
| Weekday / weekend | +4.6¢ (t 3.4) / −0.1¢ | +0.8¢ (t 1.1) / **+3.2¢ (t 3.8)** |
| Low / mid / high variance tercile | +3.4 / +5.4 / +0.8¢ | +1.1 / +0.3 / **+3.3¢ (t 3.9)** |

* **W03 catches quotes at the moment they are wrong** (timing / stale quotes): the edge is gone within 3–5 minutes,
  the price we sold into falls more than other wings afterwards. This is also why W03 is fragile to a ±2 bp change
  of the spot. Speed matters: every minute of delay costs ~1¢.
* **W04 collects a premium the market does not adjust**: it survives a 5-minute delay and sits on weekends and in
  high-variance hours. The raw market shows it without any model: every 6–10¢ wing with 15–30 min left, weekends,
  +2.2¢ per strike (t 4.2), weekdays −1.05¢ (t −2.6). Wing prices are the same on weekends (8.1¢) while they pay
  4.9% vs 8.1% of the time — BTC moves less on weekends and the ladder's makers do not reprice.
  Holds in Jul–Aug and Sep–Oct and in every band (3–6¢, 6–10¢, 10–20¢, 30–45 min), but **October ≈ 0** (+0.2…+0.5¢):
  possibly the market learning. Frozen as W08 (paper).
* No sign of the classic stale quote after spot moves: W03's P&L does not depend on the spot move of the last 1–5
  min in a consistent way; W04 is a bit better after moves away from the strike (+2.8…+3.4¢ vs ≈ 0 flat).

---

# Why the hourly wings earn: stale quotes or a risk premium? (studies/wings_mechanism.py)

Kalshi hourly ladder candles 2026-07-01 .. 2026-10-28; spot Binance × BRTI basis; frozen W03/W04 selection; t clustered by hour.

## 1. Markout: how much the cheap side's price falls after our sale
Excess = signal minus unselected cheap sides with the same price (¢) and minutes left. Positive excess = the quote we hit was stale and caught up afterwards.

| rule   |   minutes after | signal: bid fell by ¢   | baseline: bid fell by ¢   | excess (bid) ¢        | excess (mid) ¢        |
|:-------|----------------:|:------------------------|:--------------------------|:----------------------|:----------------------|
| W03    |               1 | +3.04 (t 12.5, n 523)   | +3.21 (t 92.5, n 14610)   | +0.05 (t 0.2, n 523)  | +0.05 (t 0.2, n 523)  |
| W03    |               2 | +5.32 (t 23.2, n 507)   | +5.02 (t 87.0, n 14113)   | +0.63 (t 3.1, n 507)  | +0.62 (t 3.0, n 507)  |
| W03    |               3 | +6.42 (t 19.4, n 503)   | +5.93 (t 66.5, n 13754)   | +0.70 (t 2.2, n 503)  | +0.70 (t 2.2, n 503)  |
| W03    |               5 | +6.84 (t 12.4, n 480)   | +5.63 (t 29.6, n 11448)   | +0.90 (t 1.7, n 480)  | +0.90 (t 1.7, n 480)  |
| W04    |               1 | +0.18 (t 2.2, n 1634)   | +0.21 (t 17.2, n 36649)   | +0.08 (t 1.0, n 1634) | +0.08 (t 0.9, n 1634) |
| W04    |               2 | +0.71 (t 7.4, n 1614)   | +0.61 (t 35.1, n 35933)   | +0.25 (t 2.6, n 1614) | +0.24 (t 2.5, n 1614) |
| W04    |               3 | +1.06 (t 9.9, n 1582)   | +1.02 (t 45.4, n 35293)   | +0.23 (t 2.1, n 1582) | +0.22 (t 2.1, n 1582) |
| W04    |               5 | +1.76 (t 14.9, n 1538)  | +1.77 (t 56.4, n 33936)   | +0.20 (t 1.7, n 1538) | +0.19 (t 1.7, n 1538) |

## 2. Entry delay: the same trade sold 1–5 minutes later at that minute's bid (¢ per trade)
Stale quotes ⇒ the edge vanishes within minutes; a premium ⇒ it stays.

| rule   | at signal             | +1 min                | +2 min                | +3 min                | +5 min                |
|:-------|:----------------------|:----------------------|:----------------------|:----------------------|:----------------------|
| W03    | +3.19 (t 2.6, n 647)  | +2.10 (t 1.7, n 523)  | +1.25 (t 1.1, n 507)  | +0.67 (t 0.6, n 503)  | +0.05 (t 0.1, n 480)  |
| W04    | +1.55 (t 2.7, n 1668) | +1.41 (t 2.5, n 1634) | +1.36 (t 2.5, n 1614) | +1.52 (t 2.9, n 1582) | +1.27 (t 2.5, n 1538) |

## 3. P&L by the spot move just before the signal (σ of the window; away = the wing got less likely)

| rule   |   window min | spot move before signal   |   n |   share % |   ¢/trade |     t |   mean edge at entry ¢ |
|:-------|-------------:|:--------------------------|----:|----------:|----------:|------:|-----------------------:|
| W03    |            1 | toward > 1σ               | 129 |     19.94 |      4    |  1.55 |                   3.81 |
| W03    |            1 | toward 0.3–1σ             |  85 |     13.14 |      6.53 |  2.34 |                   3.43 |
| W03    |            1 | flat                      | 123 |     19.01 |      0.72 |  0.23 |                   3.18 |
| W03    |            1 | away 0.3–1σ               | 134 |     20.71 |      4.41 |  1.75 |                   3.26 |
| W03    |            1 | away > 1σ                 | 176 |     27.2  |      1.77 |  0.73 |                   3.69 |
| W03    |            3 | toward > 1σ               |  91 |     14.06 |      4.55 |  1.5  |                   3.57 |
| W03    |            3 | toward 0.3–1σ             |  83 |     12.83 |     -3.23 | -0.78 |                   3.43 |
| W03    |            3 | flat                      | 120 |     18.55 |      2.33 |  0.8  |                   3.33 |
| W03    |            3 | away 0.3–1σ               | 147 |     22.72 |      5.44 |  2.41 |                   3.17 |
| W03    |            3 | away > 1σ                 | 206 |     31.84 |      4.07 |  1.97 |                   3.82 |
| W03    |            5 | toward > 1σ               | 105 |     16.23 |      7.02 |  2.83 |                   3.64 |
| W03    |            5 | toward 0.3–1σ             |  80 |     12.36 |      3.29 |  0.98 |                   3.56 |
| W03    |            5 | flat                      | 131 |     20.25 |      1.73 |  0.6  |                   3.25 |
| W03    |            5 | away 0.3–1σ               | 116 |     17.93 |      3.35 |  1.19 |                   2.84 |
| W03    |            5 | away > 1σ                 | 215 |     33.23 |      2.08 |  0.95 |                   3.91 |
| W04    |            1 | toward > 1σ               | 251 |     15.05 |      1.79 |  1.29 |                   1.8  |
| W04    |            1 | toward 0.3–1σ             | 249 |     14.93 |      1.99 |  1.47 |                   1.53 |
| W04    |            1 | flat                      | 419 |     25.12 |      1.52 |  1.34 |                   1.75 |
| W04    |            1 | away 0.3–1σ               | 361 |     21.64 |      0.95 |  0.74 |                   1.62 |
| W04    |            1 | away > 1σ                 | 388 |     23.26 |      1.7  |  1.45 |                   1.93 |
| W04    |            3 | toward > 1σ               | 268 |     16.07 |      0.27 |  0.18 |                   1.86 |
| W04    |            3 | toward 0.3–1σ             | 279 |     16.73 |      3.1  |  2.66 |                   1.59 |
| W04    |            3 | flat                      | 356 |     21.34 |     -0.06 | -0.04 |                   1.74 |
| W04    |            3 | away 0.3–1σ               | 374 |     22.42 |      2.78 |  2.6  |                   1.55 |
| W04    |            3 | away > 1σ                 | 391 |     23.44 |      1.61 |  1.39 |                   1.96 |
| W04    |            5 | toward > 1σ               | 261 |     15.65 |      0.23 |  0.14 |                   1.84 |
| W04    |            5 | toward 0.3–1σ             | 269 |     16.13 |      1.34 |  0.96 |                   1.51 |
| W04    |            5 | flat                      | 375 |     22.48 |      0.09 |  0.07 |                   1.65 |
| W04    |            5 | away 0.3–1σ               | 387 |     23.2  |      3.42 |  3.49 |                   1.77 |
| W04    |            5 | away > 1σ                 | 376 |     22.54 |      2.15 |  1.91 |                   1.9  |

## 4. Signals by regime

| sample   | split       | value               |    n |   ¢/trade |     t |   hit % |   mean bid ¢ |   paid % |
|:---------|:------------|:--------------------|-----:|----------:|------:|--------:|-------------:|---------:|
| W03      | vol tercile | low vol             |  216 |      3.4  |  1.66 |   89.81 |        14.79 |    10.19 |
| W03      | vol tercile | mid vol             |  215 |      5.42 |  2.87 |   91.63 |        15.04 |     8.37 |
| W03      | vol tercile | high vol            |  216 |      0.75 |  0.33 |   87.04 |        14.97 |    12.96 |
| W03      | weekend     | weekday             |  450 |      4.61 |  3.39 |   90.89 |        14.97 |     9.11 |
| W03      | weekend     | weekend             |  197 |     -0.07 | -0.03 |   86.29 |        14.85 |    13.71 |
| W03      | session     | Asia 00–08 UTC      |  220 |      4.21 |  2.18 |   90.91 |        14.51 |     9.09 |
| W03      | session     | Europe 08–13 / late |  219 |      1.83 |  0.84 |   88.13 |        14.94 |    11.87 |
| W03      | session     | US 13–21 UTC        |  208 |      3.53 |  1.64 |   89.42 |        15.38 |    10.58 |
| W03      | distance    | <1σ                 |   17 |      6.12 |  0.78 |   88.24 |        19.88 |    11.76 |
| W03      | distance    | 1–1.5σ              |  537 |      3.74 |  2.86 |   89.76 |        15.23 |    10.24 |
| W03      | distance    | 1.5–2σ              |   85 |     -0.42 | -0.12 |   88.24 |        12.38 |    11.76 |
| W03      | distance    | >2σ                 |    8 |     -1.75 | -0.15 |   87.5  |        11.75 |    12.5  |
| W04      | vol tercile | low vol             |  556 |      1.07 |  1.07 |   94.06 |         8.01 |     5.94 |
| W04      | vol tercile | mid vol             |  556 |      0.28 |  0.26 |   93.17 |         8.12 |     6.83 |
| W04      | vol tercile | high vol            |  556 |      3.29 |  3.9  |   95.86 |         8.43 |     4.14 |
| W04      | weekend     | weekday             | 1130 |      0.78 |  1.08 |   93.54 |         8.25 |     6.46 |
| W04      | weekend     | weekend             |  538 |      3.16 |  3.77 |   96.1  |         8.06 |     3.9  |
| W04      | session     | Asia 00–08 UTC      |  553 |      2.42 |  2.69 |   95.3  |         8.13 |     4.7  |
| W04      | session     | Europe 08–13 / late |  612 |      0.9  |  0.92 |   93.79 |         8.11 |     6.21 |
| W04      | session     | US 13–21 UTC        |  503 |      1.38 |  1.31 |   94.04 |         8.35 |     5.96 |
| W04      | distance    | 1–1.5σ              |  950 |      1.94 |  2.54 |   94.11 |         8.84 |     5.89 |
| W04      | distance    | 1.5–2σ              |  615 |      1.85 |  2.25 |   95.61 |         7.24 |     4.39 |
| W04      | distance    | >2σ                 |  103 |     -3.87 | -1.27 |   89.32 |         7.81 |    10.68 |

## 5. Unfiltered wings (every cheap side, first minute in the window, one per strike) by regime
The raw favourite–longshot premium the market pays a seller with no model.

| sample               | split       | value               |    n |   ¢/trade |     t |   hit % |   mean bid ¢ |   paid % |
|:---------------------|:------------|:--------------------|-----:|----------:|------:|--------:|-------------:|---------:|
| all 10–20¢, 5–15 min | vol tercile | low vol             | 2130 |      0.35 |  0.48 |   86.38 |        15.27 |    13.62 |
| all 10–20¢, 5–15 min | vol tercile | mid vol             | 2129 |     -0.42 | -0.59 |   85.81 |        15.04 |    14.19 |
| all 10–20¢, 5–15 min | vol tercile | high vol            | 2129 |      0.11 |  0.13 |   86.38 |        15    |    13.62 |
| all 10–20¢, 5–15 min | weekend     | weekday             | 5180 |     -0.32 | -0.63 |   85.89 |        15.07 |    14.11 |
| all 10–20¢, 5–15 min | weekend     | weekend             | 1208 |      1.44 |  1.59 |   87.5  |        15.25 |    12.5  |
| all 10–20¢, 5–15 min | session     | Asia 00–08 UTC      | 1880 |     -0.14 | -0.18 |   86.12 |        15.03 |    13.88 |
| all 10–20¢, 5–15 min | session     | Europe 08–13 / late | 1911 |      0.77 |  1.07 |   86.87 |        15.19 |    13.13 |
| all 10–20¢, 5–15 min | session     | US 13–21 UTC        | 2597 |     -0.43 | -0.56 |   85.75 |        15.1  |    14.25 |
| all 10–20¢, 5–15 min | distance    | <1σ                 | 4379 |     -0.22 | -0.42 |   85.29 |        15.84 |    14.71 |
| all 10–20¢, 5–15 min | distance    | 1–1.5σ              | 1726 |      0.85 |  1.01 |   88.35 |        13.63 |    11.65 |
| all 10–20¢, 5–15 min | distance    | 1.5–2σ              |  265 |     -1.52 | -0.62 |   86.79 |        12.72 |    13.21 |
| all 10–20¢, 5–15 min | distance    | >2σ                 |   18 |     -0.22 | -0.03 |   88.89 |        11.89 |    11.11 |
| all 6–10¢, 15–30 min | vol tercile | low vol             | 3216 |     -0.43 | -0.84 |   92.51 |         8.07 |     7.49 |
| all 6–10¢, 15–30 min | vol tercile | mid vol             | 3213 |     -0.62 | -1.1  |   92.31 |         8.07 |     7.69 |
| all 6–10¢, 15–30 min | vol tercile | high vol            | 3214 |     -0.29 | -0.44 |   92.66 |         8.05 |     7.34 |
| all 6–10¢, 15–30 min | weekend     | weekday             | 7837 |     -1.05 | -2.61 |   91.9  |         8.06 |     8.1  |
| all 6–10¢, 15–30 min | weekend     | weekend             | 1806 |      2.16 |  4.15 |   95.07 |         8.09 |     4.93 |
| all 6–10¢, 15–30 min | session     | Asia 00–08 UTC      | 2815 |     -0.67 | -1.11 |   92.29 |         8.04 |     7.71 |
| all 6–10¢, 15–30 min | session     | Europe 08–13 / late | 2943 |     -0.35 | -0.6  |   92.56 |         8.09 |     7.44 |
| all 6–10¢, 15–30 min | session     | US 13–21 UTC        | 3885 |     -0.36 | -0.62 |   92.59 |         8.05 |     7.41 |
| all 6–10¢, 15–30 min | distance    | <1σ                 | 1666 |     -0.44 | -0.6  |   91.66 |         8.9  |     8.34 |
| all 6–10¢, 15–30 min | distance    | 1–1.5σ              | 5538 |     -0.31 | -0.74 |   92.61 |         8.08 |     7.39 |
| all 6–10¢, 15–30 min | distance    | 1.5–2σ              | 1774 |     -0.16 | -0.21 |   93.35 |         7.49 |     6.65 |
| all 6–10¢, 15–30 min | distance    | >2σ                 |  665 |     -2.37 | -1.37 |   91.28 |         7.35 |     8.72 |

## 6. The model edge on the same strike 0–5 minutes later (it closes ⇒ the market caught up)

| rule   |   minutes after |   model edge bid − fair − fee ¢ |     t |    n |
|:-------|----------------:|--------------------------------:|------:|-----:|
| W03    |               0 |                            3.5  | 38.75 |  647 |
| W03    |               1 |                            1.98 | 15.02 |  522 |
| W03    |               3 |                            0.65 |  5.13 |  501 |
| W03    |               5 |                            0.22 |  1.58 |  471 |
| W04    |               0 |                            1.74 | 44.25 | 1668 |
| W04    |               1 |                            1.43 | 26.95 | 1633 |
| W04    |               3 |                            0.87 | 15.74 | 1581 |
| W04    |               5 |                            0.49 |  8.9  | 1537 |
