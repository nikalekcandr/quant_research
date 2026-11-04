# Kalshi wings on the Polymarket hourly ladder — result (04.11.2026)

* **History (2 849 hours, 57k strikes, Jul 1 – Nov 2, 1-minute prices ≈ book mid, half-spread 1¢ assumed):**
  W09 (= W03) by month +6.0 / +11.4 / +2.2 / −0.8¢ (Jul / Aug / Sep / Oct); W10 (= W04) +6.4 / −0.5 / +1.1 / −0.8¢.
  The edge was large in summer and is gone by October. Not a stale-price artefact: restricted to minutes where the
  price changed, the same pattern (W09 +4.8 / +10.7 / +1.4 / −1.4¢). History prices sit inside the live bid–ask in
  89% of wing minutes (median difference from the live mid 0).
* **Without a model** the Polymarket wings are fairly priced: 4.0¢ wings paid 4.8%, 8.3¢ → 7.1%, 12.6¢ → 12.3%,
  17.8¢ → 18.7% (−1.9…+0.6¢ per contract).
* **Live books 28.10–02.11:** median top-of-book size on a wing 10–18 shares (Kalshi 600–970), median spread 2¢.
  Capacity is 30–50× smaller than on Kalshi.
* Verdict: no edge now; W09/W10 stay on paper to see whether the summer effect comes back.

---

# Kalshi wing rules on the Polymarket hourly ladder (studies/pm_wings.py)

2849 hours, 56926 strikes, 2026-07-01 .. 2026-11-02; Binance close reproduces the outcome in 100.0% of rows.

## One trade per hour, held to settlement

| rule                                      |   half spread ¢ |   Jul–Aug n |   Jul–Aug ¢ |   Jul–Aug t |   Sep–Nov n |   Sep–Nov ¢ |   Sep–Nov t |   weekend ¢ |   weekend n |
|:------------------------------------------|----------------:|------------:|------------:|------------:|------------:|------------:|------------:|------------:|------------:|
| W09 = W03 (10–20¢, 5–15 min, normal, ≥1¢) |             0.5 |         338 |        8.12 |        8.55 |         179 |        2.2  |        0.92 |        1.05 |          81 |
| W10 = W04 (6–10¢, 15–30 min, t4, ≥0)      |             0.5 |         416 |        1.79 |        1.57 |         608 |        1.25 |        1.25 |        1.1  |         185 |
| no model, 10–20¢, 5–15 min                |             0.5 |         837 |        1.84 |        1.68 |         988 |       -0.8  |       -0.7  |        0.36 |         281 |
| no model, 6–10¢, 15–30 min                |             0.5 |         974 |       -0.53 |       -0.6  |        1186 |       -0.19 |       -0.25 |       -0.06 |         375 |
| W09 = W03 (10–20¢, 5–15 min, normal, ≥1¢) |             1   |         244 |        8.67 |        6.18 |         174 |        0.99 |        0.39 |       -1.02 |          60 |
| W10 = W04 (6–10¢, 15–30 min, t4, ≥0)      |             1   |         532 |        3.62 |        3.93 |         541 |        0.3  |        0.26 |        1.67 |         191 |
| no model, 10–20¢, 5–15 min                |             1   |         807 |        0.97 |        0.81 |         984 |       -0.96 |       -0.84 |       -0.41 |         275 |
| no model, 6–10¢, 15–30 min                |             1   |         990 |        1.37 |        1.74 |        1181 |       -0.71 |       -0.88 |        0.88 |         375 |
| W09 = W03 (10–20¢, 5–15 min, normal, ≥1¢) |             1.5 |         200 |        7.78 |        4.25 |         163 |        1.36 |        0.51 |       -0.43 |          53 |
| W10 = W04 (6–10¢, 15–30 min, t4, ≥0)      |             1.5 |         483 |        3.64 |        3.82 |         460 |        0.07 |        0.06 |        1.24 |         164 |
| no model, 10–20¢, 5–15 min                |             1.5 |         794 |        0.61 |        0.5  |         981 |       -1.53 |       -1.31 |       -1.48 |         275 |
| no model, 6–10¢, 15–30 min                |             1.5 |         973 |        0.35 |        0.41 |        1174 |       -0.7  |       -0.86 |        0.13 |         370 |

## Favourite–longshot on Polymarket (every cheap side, first minute in 5–30 min, h = 1¢)

| bk           |    n |   bid |   paid |   pnl |
|:-------------|-----:|------:|-------:|------:|
| (0.03, 0.06] | 2885 |  4.04 |   4.78 | -1.02 |
| (0.06, 0.1]  | 1758 |  8.28 |   7.11 |  0.64 |
| (0.1, 0.15]  | 1354 | 12.6  |  12.26 | -0.43 |
| (0.15, 0.2]  | 2010 | 17.8  |  18.71 | -1.93 |

## Staleness check: ¢ per trade by month (h = 1¢)

|                                     |   2026-07 |   2026-08 |   2026-09 |   2026-10 |   2026-11 |
|:------------------------------------|----------:|----------:|----------:|----------:|----------:|
| ('all minutes', 'W09 = W03')        |      5.99 |     11.35 |      2.22 |     -0.78 |     13.39 |
| ('all minutes', 'W10 = W04')        |      6.44 |     -0.49 |      1.13 |     -0.79 |      7.56 |
| ('fresh minutes only', 'W09 = W03') |      4.84 |     10.68 |      1.42 |     -1.38 |     13.39 |
| ('fresh minutes only', 'W10 = W04') |      2.81 |     -0.07 |      1.02 |     -0.93 |      7.56 |

trades:

|                                     |   2026-07 |   2026-08 |   2026-09 |   2026-10 |   2026-11 |
|:------------------------------------|----------:|----------:|----------:|----------:|----------:|
| ('all minutes', 'W09 = W03')        |       122 |       122 |        84 |        86 |         4 |
| ('all minutes', 'W10 = W04')        |       330 |       202 |       259 |       271 |        11 |
| ('fresh minutes only', 'W09 = W03') |        96 |       111 |        79 |        83 |         4 |
| ('fresh minutes only', 'W10 = W04') |       189 |       199 |       255 |       267 |        11 |
