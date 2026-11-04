## 1. How the one strike per hour is picked

| rule   | one strike per hour picked by               |   Jul–Aug ¢ |   Jul–Aug t |   Jul–Aug n |   Sep–Oct ¢ |   Sep–Oct t |   Sep–Oct n |
|:-------|:--------------------------------------------|------------:|------------:|------------:|------------:|------------:|------------:|
| W03    | frozen: first minute, dearest bid           |        2.29 |        1.06 |         220 |        3.65 |        2.51 |         427 |
| W03    | first minute, largest edge                  |        1.7  |        0.77 |         220 |        3.6  |        2.5  |         427 |
| W03    | first minute, cheapest bid                  |        2.04 |        0.95 |         220 |        2.94 |        2    |         427 |
| W03    | first minute, random strike                 |        2.19 |        1.02 |         220 |        3.62 |        2.52 |         427 |
| W03    | every signalling strike (first minute each) |        0.36 |        0.21 |         382 |        0.56 |        0.42 |         918 |
| W04    | frozen: first minute, dearest bid           |        1.23 |        1.37 |         688 |        1.77 |        2.45 |         980 |
| W04    | first minute, largest edge                  |        1.62 |        1.86 |         688 |        0.85 |        1.1  |         980 |
| W04    | first minute, cheapest bid                  |        1.46 |        1.67 |         688 |        0.38 |        0.49 |         980 |
| W04    | first minute, random strike                 |        1.26 |        1.42 |         688 |        1.11 |        1.48 |         980 |
| W04    | every signalling strike (first minute each) |        0.03 |        0.04 |        1417 |        0.08 |        0.12 |        3416 |

## 2. Day-block bootstrap of the frozen W03+W04 book (1 contract per slot)

| period   |   days |   $/day |   worst day $ |   losing days % | 95% CI $/day   |   P(mean ≤ 0) % |   P(30-day loss) % |   30-day 5% quantile $ |
|:---------|-------:|--------:|--------------:|----------------:|:---------------|----------------:|-------------------:|-----------------------:|
| Jul–Aug  |     62 |    0.22 |         -2.74 |           33.87 | [-0.03; 0.46]  |            4.56 |              11.24 |                  -2.5  |
| Sep–Oct  |     58 |    0.57 |         -3.35 |           29.31 | [0.20; 0.93]   |            0.14 |               1.6  |                   4.29 |
| Jul–Oct  |    120 |    0.39 |         -3.35 |           31.67 | [0.18; 0.61]   |            0.03 |               4.15 |                   0.65 |

## 3. Leave one week out (Jul–Oct, frozen book): ¢ per trade without that week

min 1.83¢, max 2.31¢ (all weeks 2.01¢); best week $7.55, worst $-2.92; weeks positive 72% of 18

## 4. The frozen trade of the hour by how many strikes signalled in that hour (first minute each)

| rule   | signalling strikes in the hour   | half    |   hours |   frozen trade ¢ |   frozen paid % |   all signalling strikes ¢ |
|:-------|:---------------------------------|:--------|--------:|-----------------:|----------------:|---------------------------:|
| W03    | 1                                | Jul–Aug |     154 |             3.8  |            9.74 |                       3.8  |
| W03    | 1                                | Sep–Oct |     241 |             4.53 |            9.13 |                       4.53 |
| W03    | 2                                | Jul–Aug |      33 |            -1.39 |           15.15 |                      -0.05 |
| W03    | 2                                | Sep–Oct |      95 |             3.74 |            9.47 |                      -0.16 |
| W03    | 3–4                              | Jul–Aug |      22 |             0.09 |           13.64 |                      -2.03 |
| W03    | 3–4                              | Sep–Oct |      45 |             0.4  |           13.33 |                       1.38 |
| W03    | 5+                               | Jul–Aug |      11 |            -3.45 |           18.18 |                      -3.29 |
| W03    | 5+                               | Sep–Oct |      46 |             2.04 |           13.04 |                      -2.28 |
| W04    | 1                                | Jul–Aug |     379 |             2.79 |            4.49 |                       2.79 |
| W04    | 1                                | Sep–Oct |     249 |             2.82 |            4.42 |                       2.82 |
| W04    | 2                                | Jul–Aug |     164 |             0.21 |            6.71 |                       0.92 |
| W04    | 2                                | Sep–Oct |     239 |             3.24 |            3.77 |                       1.93 |
| W04    | 3–4                              | Jul–Aug |     100 |             0.21 |            7    |                      -1.74 |
| W04    | 3–4                              | Sep–Oct |     274 |             2.19 |            4.74 |                       1.76 |
| W04    | 5+                               | Jul–Aug |      45 |            -5.89 |           13.33 |                      -1.95 |
| W04    | 5+                               | Sep–Oct |     218 |            -1.56 |            9.17 |                      -1.72 |

## 5. No look-ahead: strikes with a model edge at the decision minute itself

n_band = strikes in the rule's band with edge ≥ margin in that minute; n_wide = cheap sides 3–30¢ of the whole ladder with edge ≥ 0 in that minute (how widely the model disagrees with the book).

| rule   | count   | value   |   Jul–Aug hours |   Jul–Aug ¢ |   Jul–Aug t |   Jul–Aug paid % |   Sep–Oct hours |   Sep–Oct ¢ |   Sep–Oct t |   Sep–Oct paid % |
|:-------|:--------|:--------|----------------:|------------:|------------:|-----------------:|----------------:|------------:|------------:|-----------------:|
| W03    | n_band  | 1       |             204 |        2.76 |        1.27 |            10.78 |             372 |        3.86 |        2.52 |             9.68 |
| W03    | n_band  | 2       |              11 |      -13.09 |       -0.93 |            27.27 |              43 |       -1.91 |       -0.33 |            16.28 |
| W03    | n_band  | 3+      |               5 |       16.6  |       41.5  |             0    |              12 |       17.08 |       65.73 |             0    |
| W03    | n_wide  | 1       |             141 |        2.56 |        0.98 |            10.64 |             177 |        5.58 |        2.79 |             7.91 |
| W03    | n_wide  | 2       |              46 |       -0.67 |       -0.12 |            15.22 |             143 |        0.91 |        0.32 |            12.59 |
| W03    | n_wide  | 3–4     |              22 |        9.59 |        2.07 |             4.55 |              61 |        7.26 |        2.27 |             6.56 |
| W03    | n_wide  | 5+      |              11 |       -3.45 |       -0.29 |            18.18 |              46 |       -0.04 |       -0.01 |            15.22 |
| W04    | n_band  | 1       |             611 |        2.04 |        2.29 |             5.07 |             745 |        1.61 |        1.95 |             5.37 |
| W04    | n_band  | 2       |              64 |       -6.34 |       -1.45 |            14.06 |             188 |        3.41 |        2.33 |             4.26 |
| W04    | n_band  | 3+      |              13 |        0.85 |        0.11 |             7.69 |              47 |       -2.21 |       -0.49 |            10.64 |
| W04    | n_wide  | 1       |             405 |        2.85 |        2.78 |             4.44 |             310 |        0.85 |        0.61 |             6.45 |
| W04    | n_wide  | 2       |             191 |       -1.48 |       -0.74 |             8.38 |             291 |        2.76 |        2.34 |             4.12 |
| W04    | n_wide  | 3–4     |              53 |        3.23 |        1.23 |             3.77 |             223 |        1.69 |        1.12 |             5.38 |
| W04    | n_wide  | 5+      |              39 |       -5.03 |       -0.93 |            12.82 |             156 |        1.88 |        1    |             5.77 |
