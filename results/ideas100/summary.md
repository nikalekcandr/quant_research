# 100 ideas — results summary

Panel kx1h: 32763 occasions, 2023-02-01 23:05:00+00:00 .. 2026-10-29 01:05:00+00:00.
Baseline = idea 01 (HAR). DM < 0 means the idea beats HAR on QLIKE.
Holm-adjusted p across all tested ideas. See results.csv for full numbers.

## Volatility ideas ranked by QLIKE (lower better)

| id | idea | n | QLIKE | CRPS | DM vs HAR | p (Holm) |
|---|---|---|---|---|---|---|
| 69 | HAR + ladder vol (mkt_s) | 5611 | 0.2420 | 0.3363 | -6.83 | 8.88e-10 |
| 68 | HAR + ladder + fit RMSE | 5611 | 0.2440 | 0.3389 | -6.22 | 4.84e-08 |
| 67 | HAR + ladder + skew (nu) | 5421 | 0.2516 | 0.3395 | -5.93 | 2.72e-07 |
| 77 | GBM all features | 2733 | 0.2557 | 0.3575 | -3.71 | 0.016 |
| 64 | HAR-X + DVOL + ladder (harx_all) | 6051 | 0.2640 | 0.3464 | -4.26 | 0.00166 |
| 63 | HAR-X + ladder MFIV | 6051 | 0.2645 | 0.3459 | -4.38 | 0.000956 |
| 81 | Combo HAR + DVOL-MZ + ladder-MZ | 6051 | 0.2685 | 0.3507 | -5.02 | 4.5e-05 |
| 62 | HAR + ladder MFIV | 6051 | 0.2687 | 0.3498 | -5.37 | 6.78e-06 |
| 76 | GBM HAR-X+DVOL+ladder | 2733 | 0.2725 | 0.3698 | -2.55 | 0.643 |
| 90 | HAR + ladder drift | 5611 | 0.2735 | 0.3581 | 0.73 | 1 |
| 79 | GBM HAR + short lags (5m/15m) | 12338 | 0.2856 | 0.3639 | -1.57 | 1 |
| 74 | GBM HAR-X | 12338 | 0.2865 | 0.3597 | -1.34 | 1 |
| 75 | RandomForest HAR-X | 12338 | 0.2898 | 0.3628 | -0.38 | 1 |
| 73 | ElasticNet HAR-X | 12338 | 0.2909 | 0.3633 | -0.09 | 1 |
| 72 | Lasso HAR-X | 12338 | 0.2912 | 0.3639 | 0.03 | 1 |
| 78 | ExtraTrees HAR-X | 12338 | 0.2952 | 0.3669 | 1.13 | 1 |
| 61 | Ladder MFIV alone (raw) | 6555 | 0.3220 | 0.4094 | 1.99 | 1 |
| 43 | HAR + Fourier K=6 (hour-of-week) | 31921 | 0.3312 | 0.3668 | -2.91 | 0.234 |
| 50 | HAR + macro x hour | 31921 | 0.3328 | 0.3714 | -4.03 | 0.00432 |
| 83 | HAR-X + short lags 5m/15m | 31921 | 0.3329 | 0.3694 | -3.07 | 0.151 |
| 13 | HAR + macro dummy | 31921 | 0.3332 | 0.3728 | -4.52 | 0.000516 |
| 5 | HAR-X (semivar+jump+HARQ+dummies) | 31921 | 0.3335 | 0.3702 | -2.93 | 0.225 |
| 71 | Ridge HAR-X (alpha=1) | 31921 | 0.3335 | 0.3703 | -2.99 | 0.197 |
| 15 | HAR + dummies + semivar | 31921 | 0.3337 | 0.3724 | -3.38 | 0.053 |
| 59 | HAR + DVOL + Fourier | 31921 | 0.3347 | 0.3680 | -2.17 | 1 |
| 53 | HAR + DVOL change 24h | 31897 | 0.3352 | 0.3738 | -4.21 | 0.00203 |
| 42 | HAR + Fourier K=3 (hour) | 31921 | 0.3352 | 0.3689 | -2.63 | 0.527 |
| 41 | HAR + hour dummies (24) | 31921 | 0.3354 | 0.3646 | -1.06 | 1 |
| 49 | HAR + weekend x hour (Fourier) | 31921 | 0.3366 | 0.3684 | -1.29 | 1 |
| 57 | HAR + DVOL z-score | 31202 | 0.3368 | 0.3738 | -2.40 | 0.959 |
| 48 | HAR + day-of-week dummies | 31921 | 0.3373 | 0.3743 | -1.72 | 1 |
| 3 | HAR + jump share | 31921 | 0.3383 | 0.3740 | -1.78 | 1 |
| 38 | HAR + Amihud illiquidity | 31920 | 0.3386 | 0.3750 | -2.67 | 0.488 |
| 11 | HAR + RV x jump interaction | 31921 | 0.3388 | 0.3741 | -1.22 | 1 |
| 82 | HAR + short lags 5m/15m | 31921 | 0.3388 | 0.3738 | -0.97 | 1 |
| 52 | HAR + DVOL change 1h | 31920 | 0.3390 | 0.3751 | -2.13 | 1 |
| 88 | HAR + |funding| | 31921 | 0.3391 | 0.3755 | -0.98 | 1 |
| 56 | HAR + DVOL + semivar | 31921 | 0.3393 | 0.3736 | -0.42 | 1 |
| 4 | HAR + HARQ | 31921 | 0.3395 | 0.3726 | -0.40 | 1 |
| 89 | HAR + VRP proxy (DVOL-RV) | 31921 | 0.3395 | 0.3742 | -0.29 | 1 |
| 51 | HAR + DVOL level | 31921 | 0.3395 | 0.3742 | -0.29 | 1 |
| 9 | HAR + naive | 31921 | 0.3395 | 0.3749 | -1.11 | 1 |
| 2 | HAR + semivariances | 31921 | 0.3397 | 0.3745 | -0.32 | 1 |
| 87 | HAR + funding level | 31921 | 0.3397 | 0.3756 | -0.33 | 1 |
| 35 | HAR + volume imbalance proxy | 31920 | 0.3397 | 0.3749 | -0.23 | 1 |
| 30 | HAR + leverage (r x RV) | 31897 | 0.3398 | 0.3754 | -0.17 | 1 |
| 44 | HAR + US session dummy | 31921 | 0.3398 | 0.3738 | -0.20 | 1 |
| 47 | HAR + hour-0 dummy (:00 spike) | 31921 | 0.3399 | 0.3741 | -0.12 | 1 |
| 54 | HAR + DVOL x RV | 31921 | 0.3399 | 0.3741 | -0.01 | 1 |
| 1 | HAR baseline | 31921 | 0.3399 | 0.3752 | nan | nan |
| 22 | HAR + jump share 1h | 31920 | 0.3402 | 0.3741 | 0.42 | 1 |
| 27 | HAR + HARQ level (not scaled) | 31897 | 0.3403 | 0.3724 | 0.21 | 1 |
| 58 | HAR + DVOL + volume | 31897 | 0.3404 | 0.3745 | 0.23 | 1 |
| 45 | HAR + session dummies (Asia/EU/US) | 31921 | 0.3405 | 0.3719 | 0.35 | 1 |
| 25 | HAR + RV5 | 31897 | 0.3407 | 0.3755 | 1.00 | 1 |
| 26 | HAR + quarticity level | 31897 | 0.3409 | 0.3746 | 1.24 | 1 |
| 10 | HAR + squared term | 31921 | 0.3410 | 0.3754 | 2.23 | 1 |
| 39 | HAR + volume + funding | 31897 | 0.3410 | 0.3761 | 0.89 | 1 |
| 46 | HAR + funding-hour dummy | 31921 | 0.3410 | 0.3746 | 3.41 | 0.0476 |
| 31 | HAR + log volume 24h | 31897 | 0.3412 | 0.3756 | 1.40 | 1 |
| 36 | HAR + volume volatility | 31897 | 0.3412 | 0.3750 | 2.65 | 0.511 |
| 86 | HAR + RV x vol-of-vol | 31921 | 0.3412 | 0.3757 | 2.43 | 0.893 |
| 34 | HAR + volume x RV | 31897 | 0.3413 | 0.3755 | 1.62 | 1 |
| 18 | HAR + BV + RV | 31753 | 0.3413 | 0.3760 | 2.94 | 0.222 |
| 14 | HAR + weekend dummy | 31921 | 0.3413 | 0.3755 | 1.28 | 1 |
| 85 | HAR + vol-of-vol | 31921 | 0.3413 | 0.3756 | 2.30 | 1 |
| 60 | HAR + DVOL + jump + volume | 31897 | 0.3413 | 0.3746 | 0.81 | 1 |
| 12 | HAR + RV x weekend interaction | 31921 | 0.3413 | 0.3754 | 1.38 | 1 |
| 33 | HAR + volume change | 31897 | 0.3413 | 0.3745 | 0.97 | 1 |
| 28 | HAR + BV x jump interaction | 31897 | 0.3415 | 0.3755 | 1.51 | 1 |
| 32 | HAR + log volume 1h | 31920 | 0.3417 | 0.3744 | 1.44 | 1 |
| 19 | HAR + MedRV + RV | 31753 | 0.3417 | 0.3757 | 3.80 | 0.0111 |
| 29 | HAR + semivariance asymmetry | 31897 | 0.3417 | 0.3748 | 1.49 | 1 |
| 21 | HAR + jump share 24h (BV-based) | 31897 | 0.3419 | 0.3754 | 2.96 | 0.213 |
| 40 | HAR + volume + jump | 31897 | 0.3422 | 0.3758 | 2.39 | 0.959 |
| 80 | HAR log + level (both) | 31921 | 0.3477 | 0.3758 | 1.57 | 1 |
| 37 | HAR + volume HAR (d/w) | 31753 | 0.3485 | 0.3777 | 2.64 | 0.511 |
| 6 | HAR daily only | 31921 | 0.3499 | 0.3927 | 3.33 | 0.0623 |
| 23 | HAR-truncated RV (winsorised 50bp) | 31897 | 0.3574 | 0.3843 | 7.86 | 3.81e-13 |
| 66 | Ladder Student-t var alone (raw) | 5925 | 0.3597 | 0.4384 | 5.99 | 2e-07 |
| 8 | Naive log RV only | 31921 | 0.3607 | 0.4008 | 5.07 | 3.31e-05 |
| 16 | HAR-BV (bipower instead of RV) | 31753 | 0.3655 | 0.3842 | 5.68 | 1.14e-06 |
| 24 | HAR-RV5 (5-min returns) | 31897 | 0.3660 | 0.3911 | 5.95 | 2.4e-07 |
| 17 | HAR-MedRV | 31753 | 0.3815 | 0.3842 | 2.31 | 1 |
| 20 | HAR-BV naive | 31920 | 0.4014 | 0.4108 | 4.34 | 0.00115 |
| 84 | HAR + EWMA RV | 31921 | 0.4279 | 0.4425 | 13.19 | 1.22e-37 |
| 7 | HAR weekly only | 31921 | 0.4756 | 0.4714 | 14.02 | 1.46e-42 |
| 65 | Ladder lognormal var alone (raw) | 6115 | 0.5264 | 0.3341 | 0.85 | 1 |
| 70 | Ladder + DVOL average (raw combo) | 32425 | 1.0815 | 0.7502 | 39.78 | 0 |
| 55 | DVOL alone (raw) | 32425 | 1.1582 | 0.7893 | 44.23 | 0 |

## Directional ideas

| id | idea | n | OOS R2 | corr | Sharpe net | t (NW) | p |
|---|---|---|---|---|---|---|---|
| 91 | Return: overnight/US session drift | 12338 | -0.0001 | 0.009 | 0.19 | 0.22 | 0.827 |
| 92 | Return: funding -> next return | 12338 | -0.0002 | 0.004 | 0.19 | 0.22 | 0.827 |
| 93 | Return: DVOL change -> next return | 12338 | 0.0003 | 0.022 | -4.85 | -5.97 | 2.43e-09 |
| 94 | Return: volume -> next return | 12338 | -0.0002 | -0.004 | -1.19 | -1.39 | 0.163 |
| 95 | Return: ladder drift -> next return | 2557 | -0.0046 | -0.002 | -5.54 | -2.95 | 0.00316 |
| 96 | Return: VRP -> next return | 12338 | -0.0003 | 0.003 | -1.96 | -2.29 | 0.0221 |
| 97 | Return: 24h momentum -> next return | 12338 | -0.0003 | -0.007 | -0.24 | -0.29 | 0.775 |
| 98 | Return: 1h reversal -> next return | 12338 | -0.0012 | -0.007 | -7.34 | -9.46 | 3.71e-21 |
| 99 | Return: funding x volume squeeze | 12338 | -0.0002 | 0.005 | 0.19 | 0.22 | 0.827 |
| 100 | Abs return: HAR predicts |r| | 31923 | nan | nan | nan | nan | 1.88e-72 |

## Skipped / errors
