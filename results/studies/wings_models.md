# Wings: model zoo, fragility, bootstrap (04.11.2026) — result

Code: `studies/wings_models.py` (tables in `wings_models.md`), `studies/wings_fragility.py` (`wings_fragility.md`).

1. **No model prices the cheap side better than the book.** 34 variance × shape models on 16 541 strike-hours: all
   worse on Brier (t 5–18). Best: short memory + fat tails (ewma30/t4, HAR/t4, HAR/empirical: +0.15 Brier×100 vs the
   book); worst: DVOL (+0.5…+1.1) and 24-hour means. Normal is always worse than t3/t4/empirical. Out of sample no
   model adds information to the book mid (Δ log loss > 0 for every model; in-sample β of the model 0.1–0.26).
2. **Trading with other models** (margin chosen on Jul–Aug, first minute with edge, no look-ahead): W03 window best
   test ewma60/t4 or t6 with a 2–3¢ margin, +2.9…+3.0¢ (t 2.2) — no better than the frozen W03 (+3.65¢). In the W04
   window almost every model is ≤ 0 on Jul–Aug and positive on Sep–Oct: the period matters more than the model.
3. **How the one strike per hour is chosen matters for W04:** dearest bid (frozen) +1.77¢ (t 2.5), random strike
   +1.11¢, largest edge +0.85¢, cheapest bid +0.38¢ on Sep–Oct. W03 is indifferent (+2.9…+3.65¢).
4. **Selling every signalling strike earns nothing** (W03 +0.4/+0.6¢, W04 +0.03/+0.08¢ per trade): the later signals of
   an hour come while the price moves toward the strikes. One bet per hour (the executor's slot rule) is essential.
   Counting how many strikes signal *in the decision minute* gives no usable filter (inconsistent across halves);
   the same count over the whole hour looks great but is look-ahead.
5. **Bootstrap (3-day blocks), frozen W03+W04, 1 contract per slot:** Jul–Aug $0.22/day, 95% CI [−0.03; 0.46],
   P(30-day loss) 11%; Sep–Oct $0.57/day [0.20; 0.93], 1.6%; Jul–Oct $0.39/day [0.18; 0.61], P(30-day loss) 4%.
   Leave-one-week-out: 1.83…2.31¢ per trade (all weeks 2.01¢); 72% of 18 weeks positive; worst week −$2.92.

---

# Model zoo for the wing fair value (studies/wings_models.py)

Cheap sides of the Kalshi hourly ladder Jul–Oct 2026; HAR fit before July: {'const': -0.232, 'h': 0.704, 'd': 0.21, 'w': 0.082}; empirical z from 25668 settlement × τ points Dec–Jun.

## A. Probability of the cheap side (16541 strikes × hours, 3–20¢, 5–30 min left)
Brier vs mid < 0 = better than the book. Encompassing: logit(won) on logit(mid) + logit(model), fit Jul–Aug (clustered by hour), Δ log loss on Sep–Oct vs mid-only (< 0 = the model adds information).

| forecast          |   Brier×100 |   log loss |   Brier vs mid ×100 |        t |   Sep–Oct log loss | encompassing β model (t)   |   Sep–Oct Δ log loss vs mid |
|:------------------|------------:|-----------:|--------------------:|---------:|-------------------:|:---------------------------|----------------------------:|
| book mid          |      8.8142 |     0.311  |              0      | nan      |             0.3064 | —                          |                      0      |
| ewma60/normal     |      9.2234 |     0.333  |              0.4091 |  10.8566 |             0.3325 | 0.11 (1.3)                 |                      0.001  |
| ewma60/t3         |      9.0138 |     0.3271 |              0.1996 |   5.8244 |             0.3254 | 0.10 (1.0)                 |                      0.0006 |
| ewma60/t4         |      9.0171 |     0.3251 |              0.2028 |   6.4109 |             0.3237 | 0.10 (1.0)                 |                      0.0007 |
| ewma60/t6         |      9.0741 |     0.3268 |              0.2599 |   7.9564 |             0.3256 | 0.11 (1.1)                 |                      0.0008 |
| ewma60/empirical  |      9.0125 |     0.3262 |              0.1983 |   5.9847 |             0.3247 | 0.10 (1.0)                 |                      0.0006 |
| ewma30/normal     |      9.1188 |     0.3273 |              0.3046 |   9.12   |             0.3261 | 0.18 (1.9)                 |                      0.0011 |
| ewma30/t4         |      8.9599 |     0.3214 |              0.1456 |   5.1054 |             0.3193 | 0.17 (1.5)                 |                      0.0007 |
| ewma30/empirical  |      8.9682 |     0.3229 |              0.154  |   5.0279 |             0.3207 | 0.15 (1.4)                 |                      0.0006 |
| ewma240/normal    |      9.4556 |     0.3474 |              0.6414 |  13.4635 |             0.3505 | 0.11 (1.7)                 |                      0.0017 |
| ewma240/t4        |      9.1738 |     0.336  |              0.3595 |   8.7566 |             0.3375 | 0.11 (1.6)                 |                      0.0014 |
| ewma240/empirical |      9.1487 |     0.3365 |              0.3344 |   8.0179 |             0.3379 | 0.11 (1.5)                 |                      0.0013 |
| day/normal        |      9.6117 |     0.3566 |              0.7975 |  14.6898 |             0.3614 | 0.11 (2.0)                 |                      0.002  |
| day/t4            |      9.2978 |     0.3439 |              0.4835 |  10.2979 |             0.3469 | 0.12 (1.9)                 |                      0.0017 |
| day/empirical     |      9.2638 |     0.3445 |              0.4495 |   9.5479 |             0.3473 | 0.11 (1.9)                 |                      0.0016 |
| blend/normal      |      9.3515 |     0.3364 |              0.5372 |  13.0527 |             0.3365 | 0.14 (1.6)                 |                      0.0013 |
| blend/t3          |      9.0435 |     0.3269 |              0.2292 |   6.2855 |             0.3258 | 0.10 (1.1)                 |                      0.0007 |
| blend/t4          |      9.0813 |     0.3264 |              0.267  |   7.6994 |             0.3254 | 0.12 (1.2)                 |                      0.0008 |
| blend/t6          |      9.165  |     0.329  |              0.3508 |   9.7319 |             0.3283 | 0.13 (1.4)                 |                      0.001  |
| blend/empirical   |      9.06   |     0.3267 |              0.2458 |   6.8605 |             0.3257 | 0.11 (1.1)                 |                      0.0007 |
| har/normal        |      9.1682 |     0.3299 |              0.3539 |   9.857  |             0.3305 | 0.26 (2.6)                 |                      0.0028 |
| har/t3            |      8.9803 |     0.3243 |              0.166  |   4.7846 |             0.3232 | 0.23 (2.1)                 |                      0.0016 |
| har/t4            |      8.9752 |     0.3223 |              0.1609 |   5.1278 |             0.3216 | 0.25 (2.3)                 |                      0.0019 |
| har/t6            |      9.0266 |     0.3239 |              0.2124 |   6.6903 |             0.3235 | 0.26 (2.4)                 |                      0.0022 |
| har/empirical     |      8.9742 |     0.3233 |              0.16   |   4.8054 |             0.3225 | 0.23 (2.2)                 |                      0.0017 |
| dvol/normal       |      9.9499 |     0.3637 |              1.1356 |  18.0227 |             0.3667 | 0.10 (1.8)                 |                      0.0015 |
| dvol/t3           |      9.3311 |     0.3439 |              0.5168 |  10.5968 |             0.3439 | 0.09 (1.6)                 |                      0.001  |
| dvol/t4           |      9.4889 |     0.347  |              0.6746 |  13.0199 |             0.3479 | 0.10 (1.7)                 |                      0.0012 |
| dvol/t6           |      9.6558 |     0.3523 |              0.8416 |  15.0927 |             0.354  | 0.10 (1.8)                 |                      0.0014 |
| dvol/empirical    |      9.4166 |     0.346  |              0.6023 |  11.8858 |             0.3464 | 0.09 (1.6)                 |                      0.0011 |

## B. Trading: one cheap side per hour, the first minute with edge ≥ margin (margin chosen on Jul–Aug)
Frozen W03 = ewma60/normal margin 1¢; W04 = ewma60/t4 margin 0.

| window     | model             |   margin chosen ¢ |   Jul–Aug n |   Jul–Aug ¢ |   Jul–Aug t |   Sep–Oct n |   Sep–Oct ¢ |   Sep–Oct t |
|:-----------|:------------------|------------------:|------------:|------------:|------------:|------------:|------------:|------------:|
| W03 window | ewma60/t4         |               3   |         326 |        1.8  |        0.99 |         601 |        2.9  |        2.25 |
| W03 window | ewma60/t6         |               2   |         283 |        3.2  |        1.74 |         530 |        3.02 |        2.23 |
| W03 window | har/normal        |               0   |         249 |        5.01 |        2.81 |         461 |        2.68 |        1.85 |
| W03 window | ewma30/normal     |               0   |         286 |        3.13 |        1.75 |         578 |        2.3  |        1.75 |
| W03 window | ewma30/t4         |               2   |         442 |        2.79 |        1.88 |         821 |        1.84 |        1.62 |
| W03 window | ewma60/normal     |               2   |         172 |        3.71 |        1.61 |         347 |        2.26 |        1.33 |
| W03 window | ewma30/empirical  |               3   |         491 |        1.84 |        1.25 |         869 |        1.45 |        1.29 |
| W03 window | har/empirical     |               2   |         525 |        1.56 |        1.09 |         879 |        1.32 |        1.19 |
| W03 window | har/t3            |               3   |         546 |        1.75 |        1.24 |         895 |        1.28 |        1.15 |
| W03 window | ewma240/normal    |               0   |         269 |        1.71 |        0.86 |         430 |        1.76 |        1.14 |
| W03 window | dvol/t3           |               2   |         410 |        2.46 |        1.55 |         549 |        1.45 |        1.03 |
| W03 window | blend/t6          |               3   |         160 |        4.15 |        1.73 |         287 |        1.95 |        1.01 |
| W03 window | ewma240/t4        |               0.5 |         455 |        2.83 |        1.91 |         706 |        1.16 |        0.93 |
| W03 window | dvol/empirical    |               0   |         439 |        2.45 |        1.62 |         587 |        1.23 |        0.91 |
| W03 window | ewma240/empirical |               2   |         452 |        2.87 |        1.93 |         705 |        1.1  |        0.88 |
| W03 window | ewma60/empirical  |               0   |         713 |        1.64 |        1.34 |        1088 |        0.76 |        0.75 |
| W03 window | har/t4            |               3   |         299 |        4.47 |        2.61 |         572 |        1.02 |        0.72 |
| W03 window | blend/t3          |               3   |         429 |        2.24 |        1.42 |         723 |        0.9  |        0.71 |
| W03 window | dvol/t4           |               0.5 |         350 |        1.39 |        0.79 |         450 |        0.86 |        0.55 |
| W03 window | ewma60/t3         |               1   |         736 |        1.62 |        1.34 |        1105 |        0.52 |        0.51 |
| W03 window | dvol/normal       |               0.5 |         230 |        2.29 |        1.09 |         271 |        0.97 |        0.48 |
| W03 window | blend/normal      |               0.5 |         174 |        3.73 |        1.65 |         311 |        0.87 |        0.46 |
| W03 window | blend/empirical   |               2   |         414 |        1.46 |        0.89 |         701 |        0.59 |        0.46 |
| W03 window | blend/t4          |               3   |         238 |        2.98 |        1.45 |         439 |        0.69 |        0.42 |
| W03 window | har/t6            |               2   |         255 |        6.07 |        3.57 |         483 |        0.57 |        0.37 |
| W03 window | day/normal        |               1   |         277 |        2.69 |        1.44 |         413 |        0.3  |        0.18 |
| W03 window | day/empirical     |               3   |         412 |        2    |        1.24 |         598 |        0.07 |        0.05 |
| W03 window | day/t4            |               2   |         387 |        1.65 |        0.99 |         565 |        0.04 |        0.03 |
| W03 window | dvol/t6           |               2   |         245 |        2.03 |        0.98 |         286 |       -0.84 |       -0.4  |
| W04 window | day/empirical     |               1   |         560 |       -2.05 |       -1.68 |         708 |        3.03 |        3.94 |
| W04 window | dvol/t3           |               0.5 |         471 |       -0.99 |       -0.78 |         560 |        3.3  |        3.86 |
| W04 window | blend/t3          |               1   |         584 |       -0.43 |       -0.39 |         807 |        2.81 |        3.76 |
| W04 window | ewma240/empirical |               0   |         675 |       -0.52 |       -0.51 |         874 |        2.55 |        3.57 |
| W04 window | dvol/empirical    |               0   |         465 |       -1.23 |       -0.96 |         542 |        3.06 |        3.46 |
| W04 window | blend/t4          |               1   |         358 |       -0.68 |       -0.47 |         533 |        3.09 |        3.44 |
| W04 window | blend/empirical   |               0.5 |         571 |       -0.76 |       -0.66 |         780 |        2.49 |        3.23 |
| W04 window | ewma240/t4        |               0   |         541 |       -1.17 |       -0.98 |         751 |        2.13 |        2.67 |
| W04 window | day/normal        |               3   |         274 |       -0.82 |       -0.5  |         335 |        2.92 |        2.59 |
| W04 window | ewma60/empirical  |               1   |         680 |        1.18 |        1.28 |         981 |        1.79 |        2.45 |
| W04 window | blend/t6          |               0.5 |         311 |       -0.57 |       -0.38 |         456 |        2.27 |        2.21 |
| W04 window | har/empirical     |               2   |         463 |       -0.05 |       -0.04 |         727 |        1.77 |        2.02 |
| W04 window | dvol/t4           |               0   |         389 |       -1.11 |       -0.79 |         451 |        2.1  |        1.99 |
| W04 window | har/t3            |               0.5 |         914 |       -0.31 |       -0.35 |        1155 |        1.31 |        1.88 |
| W04 window | ewma30/empirical  |               0.5 |         879 |        0.44 |        0.52 |        1158 |        1.21 |        1.74 |
| W04 window | ewma60/normal     |               0   |         408 |        0.48 |        0.39 |         681 |        1.46 |        1.7  |
| W04 window | ewma60/t3         |               2   |         592 |        1.38 |        1.38 |         900 |        1.37 |        1.7  |
| W04 window | ewma30/normal     |               3   |         170 |       -0.33 |       -0.16 |         297 |        2.06 |        1.56 |
| W04 window | ewma30/t4         |               0.5 |         684 |        0.71 |        0.75 |        1023 |        1.1  |        1.48 |
| W04 window | ewma60/t6         |               0   |         538 |        0.58 |        0.55 |         812 |        1.21 |        1.47 |
| W04 window | day/t4            |               3   |         319 |       -1.95 |       -1.19 |         407 |        1.64 |        1.38 |
| W04 window | blend/normal      |               0   |         276 |       -0.03 |       -0.02 |         401 |        1.42 |        1.25 |
| W04 window | ewma240/normal    |               2   |         259 |       -1.42 |       -0.82 |         346 |        1.42 |        1.13 |
| W04 window | ewma60/t4         |               0   |         688 |        1.62 |        1.86 |         980 |        0.85 |        1.1  |
| W04 window | har/t4            |               3   |         208 |       -0.52 |       -0.27 |         310 |        0.98 |        0.67 |
| W04 window | dvol/t6           |               2   |         236 |       -1.08 |       -0.6  |         267 |        1.02 |        0.66 |
| W04 window | dvol/normal       |               0.5 |         267 |        0.31 |        0.2  |         305 |        0.58 |        0.41 |
| W04 window | har/t6            |               3   |         155 |       -0.27 |       -0.13 |         225 |       -0.6  |       -0.32 |
| W04 window | har/normal        |               3   |         135 |        1.21 |        0.59 |         193 |       -0.85 |       -0.43 |

## C. Frozen W03+W04 book, Sep–Oct, 1 contract per slot: bootstrap

* 58 days, 1004 active hours, mean $0.568/day, worst day $-3.35, best $2.67, 29% losing days.
* Mean per day, iid days: 95% CI [0.222; 0.906], P(mean ≤ 0) = 0.07%.
* 3-day blocks: 95% CI [0.204; 0.929], P(mean ≤ 0) = 0.14%.
* 30 days: P(loss) = 1.3%, 5% quantile $4.69, median $17.24.
