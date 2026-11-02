Snapshots: {'binance': 184, 'bybit': 180, 'deribit': 234, 'okx': 236} from 28.10 23:40 to 02.11 21:00 UTC; 331,416 quotes after the lab's filters

Spot move between the snapshot and the lab's decision 3 minutes later (bp): | venue   |   count |   50% |   90% |   max |
|:--------|--------:|------:|------:|------:|
| binance |     178 |   0.1 |   8.8 |  39.6 |
| bybit   |     176 |   0.1 |   8.8 |  39.6 |
| deribit |     222 |  -0.1 |   8.9 |  39.6 |
| okx     |     222 |  -0.1 |   8.9 |  39.6 |

## O02 stages

| stage                                           | side           |   candidates |   signals |   snapshots with a signal |   sum $ (1 BTC each) |   median edge $ | by venue                    |
|:------------------------------------------------|:---------------|-------------:|----------:|--------------------------:|---------------------:|----------------:|:----------------------------|
| A. as the lab: snapshot vs perp 3 min later     | buy synthetic  |        30949 |        87 |                         5 |               3033.6 |            26.1 | {'deribit': 25, 'okx': 62}  |
| A. as the lab: snapshot vs perp 3 min later     | sell synthetic |        30949 |         0 |                         0 |                  0   |                 | {}                          |
| B. synchronous: perp at the snapshot            | buy synthetic  |        63558 |       219 |                        14 |               6947   |            26.4 | {'deribit': 60, 'okx': 159} |
| B. synchronous: perp at the snapshot            | sell synthetic |        63558 |        27 |                         2 |                895.6 |            30.5 | {'binance': 1, 'bybit': 26} |
| C. hedged with the same-expiry future (Deribit) | buy synthetic  |        19632 |         0 |                         0 |                  0   |                 | {}                          |
| C. hedged with the same-expiry future (Deribit) | sell synthetic |        19632 |         0 |                         0 |                  0   |                 | {}                          |

## O02 decomposition (median over strikes, bp of the forward)

| venue   |     n |   syn_vs_own_forward_bp |   syn_vs_own_forward_iqr |   own_forward_vs_perp_plus_carry_bp |
|:--------|------:|------------------------:|-------------------------:|------------------------------------:|
| binance |  8682 |                    3.28 |                     5.07 |                                0.97 |
| bybit   | 13528 |                   -0.27 |                     3.74 |                                2.54 |
| deribit | 19632 |                    0.65 |                     3.49 |                               -1.47 |
| okx     | 21716 |                   -0.16 |                     3.49 |                               -0.17 |

Share of the lab-stage edge variance common to all strikes of one expiry and snapshot: 76%

## O01 cross-venue

| filter                                     |   signals |   distinct options |   sum $ |   median $ | pairs                                                                      |
|:-------------------------------------------|----------:|-------------------:|--------:|-----------:|:---------------------------------------------------------------------------|
| all raw signals                            |        59 |                 26 |   608.1 |        0.6 | {('binance', 'bybit'): 5, ('bybit', 'binance'): 49, ('okx', 'deribit'): 5} |
| snapshots ≤ 30 s apart                     |        30 |                 11 |    64.8 |        1.9 | {('binance', 'bybit'): 1, ('bybit', 'binance'): 24, ('okx', 'deribit'): 5} |
| ≤ 30 s and still > 0 after the delta shift |        30 |                 11 |    64.8 |        1.9 | {('binance', 'bybit'): 1, ('bybit', 'binance'): 24, ('okx', 'deribit'): 5} |

### Longest-lived O01 opportunities

| key                                            | first                     | last                      |   snaps |   edge |   ask |   bid |   hours alive |
|:-----------------------------------------------|:--------------------------|:--------------------------|--------:|-------:|------:|------:|--------------:|
| bybit>binance|2026-11-06 08:00:00+00:00|92000C | 2026-11-01 09:20:02+00:00 | 2026-11-01 13:20:02+00:00 |      14 |    1.9 |  10   |  15   |           4   |
| bybit>binance|2026-11-01 08:00:00+00:00|79000C | 2026-10-31 19:30:03+00:00 | 2026-10-31 20:40:02+00:00 |       4 |    1.9 |  10   |  15   |           1.2 |
| bybit>binance|2026-11-01 08:00:00+00:00|78500C | 2026-10-31 23:10:03+00:00 | 2026-10-31 23:20:03+00:00 |       2 |    3.4 |  12.5 |  20   |           0.2 |
| bybit>binance|2026-11-06 08:00:00+00:00|90000C | 2026-11-01 10:20:33+00:00 | 2026-11-01 11:30:03+00:00 |       2 |    0.6 |  15   |  20   |           1.2 |
| bybit>binance|2026-11-02 08:00:00+00:00|75000P | 2026-11-01 16:20:37+00:00 | 2026-11-01 16:40:03+00:00 |       2 |    1.9 |  10   |  15   |           0.3 |
| binance>bybit|2026-11-01 08:00:00+00:00|86000C | 2026-10-29 16:20:03+00:00 | 2026-10-29 16:20:03+00:00 |       1 |    0.6 |  15   |  20   |           0   |
| okx>deribit|2026-10-30 08:00:00+00:00|77000P   | 2026-10-30 05:00:02+00:00 | 2026-10-30 05:00:02+00:00 |       1 |    2   |  46.9 |  62.5 |           0   |
| okx>deribit|2026-10-31 08:00:00+00:00|80000C   | 2026-10-30 08:40:02+00:00 | 2026-10-30 08:40:02+00:00 |       1 |    1   |  23.4 |  31.2 |           0   |
| okx>deribit|2026-11-02 08:00:00+00:00|79000C   | 2026-11-01 09:30:02+00:00 | 2026-11-01 09:30:02+00:00 |       1 |    4.9 |  62.3 |  85.7 |           0   |
| okx>deribit|2026-11-02 08:00:00+00:00|79500C   | 2026-11-01 09:30:02+00:00 | 2026-11-01 09:30:02+00:00 |       1 |    7.8 |  23.4 |  38.9 |           0   |
| okx>deribit|2026-11-02 08:00:00+00:00|80000C   | 2026-11-01 09:20:02+00:00 | 2026-11-01 09:20:02+00:00 |       1 |    2.9 |  15.6 |  23.4 |           0   |