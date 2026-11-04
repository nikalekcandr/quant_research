# Sub-minute lead–lag: spot vs 15-minute binaries (studies/fast_leadlag.py)

30884 snapshots, 567 markets, 2026-10-29 07:45 .. 2026-11-02 22:04; median gap between snapshots 15 s.

β gap > 0: the mid later moves toward the value implied by the current spot (the book lags). Taker rows: markout = mid k snapshots later − entry price − fee, in ¢ (the 'β gap' column).

| market        |   horizon s | spec                        |     n |   β gap |       t |   β past move |      R² |
|:--------------|------------:|:----------------------------|------:|--------:|--------:|--------------:|--------:|
| kx_15m        |          15 | gap                         | 12159 |  -0     |  -0.014 |       nan     |   0     |
| kx_15m        |          15 | gap + past move             | 12159 |   0.007 |   0.553 |         0.018 |   0     |
| kx_15m        |          30 | gap                         | 11928 |  -0.006 |  -0.376 |       nan     |   0     |
| kx_15m        |          30 | gap + past move             | 11928 |   0.018 |   0.796 |         0.059 |   0.001 |
| kx_15m        |          60 | gap                         | 11472 |  -0.009 |  -0.352 |       nan     |   0     |
| kx_15m        |          60 | gap + past move             | 11472 |   0.014 |   0.359 |         0.058 |   0.001 |
| kx_15m        |          15 | taker, edge ≥ 0¢: markout ¢ |  9119 |  -2.059 | -31.399 |       nan     | nan     |
| kx_15m        |          30 | taker, edge ≥ 0¢: markout ¢ |  8983 |  -2.088 | -21.903 |       nan     | nan     |
| kx_15m        |          60 | taker, edge ≥ 0¢: markout ¢ |  8702 |  -2.076 | -11.72  |       nan     | nan     |
| kx_15m        |          15 | taker, edge ≥ 2¢: markout ¢ |  6638 |  -2.094 | -27.636 |       nan     | nan     |
| kx_15m        |          30 | taker, edge ≥ 2¢: markout ¢ |  6533 |  -2.156 | -18.476 |       nan     | nan     |
| kx_15m        |          60 | taker, edge ≥ 2¢: markout ¢ |  6309 |  -2.167 |  -9.794 |       nan     | nan     |
| kx_15m        |          15 | taker, edge ≥ 4¢: markout ¢ |  4837 |  -2.093 | -23.131 |       nan     | nan     |
| kx_15m        |          30 | taker, edge ≥ 4¢: markout ¢ |  4746 |  -2.152 | -14.82  |       nan     | nan     |
| kx_15m        |          60 | taker, edge ≥ 4¢: markout ¢ |  4556 |  -2.213 |  -8.139 |       nan     | nan     |
| pm_updown_15m |          15 | gap                         | 16065 |   0.001 |   0.292 |       nan     |   0     |
| pm_updown_15m |          15 | gap + past move             | 16065 |  -0.001 |  -0.293 |        -0.014 |   0     |
| pm_updown_15m |          30 | gap                         | 15812 |  -0.004 |  -0.494 |       nan     |   0     |
| pm_updown_15m |          30 | gap + past move             | 15812 |  -0.003 |  -0.313 |         0.006 |   0     |
| pm_updown_15m |          60 | gap                         | 15254 |  -0.01  |  -0.701 |       nan     |   0     |
| pm_updown_15m |          60 | gap + past move             | 15254 |  -0.002 |  -0.152 |         0.04  |   0     |
| pm_updown_15m |          15 | taker, edge ≥ 0¢: markout ¢ | 13748 |  -1.655 | -30.662 |       nan     | nan     |
| pm_updown_15m |          30 | taker, edge ≥ 0¢: markout ¢ | 13524 |  -1.681 | -20.05  |       nan     | nan     |
| pm_updown_15m |          60 | taker, edge ≥ 0¢: markout ¢ | 13062 |  -1.761 | -11.698 |       nan     | nan     |
| pm_updown_15m |          15 | taker, edge ≥ 2¢: markout ¢ | 10262 |  -1.696 | -27.013 |       nan     | nan     |
| pm_updown_15m |          30 | taker, edge ≥ 2¢: markout ¢ | 10102 |  -1.732 | -17.386 |       nan     | nan     |
| pm_updown_15m |          60 | taker, edge ≥ 2¢: markout ¢ |  9768 |  -1.825 | -10.025 |       nan     | nan     |
| pm_updown_15m |          15 | taker, edge ≥ 4¢: markout ¢ |  7655 |  -1.695 | -22.476 |       nan     | nan     |
| pm_updown_15m |          30 | taker, edge ≥ 4¢: markout ¢ |  7515 |  -1.715 | -13.867 |       nan     | nan     |
| pm_updown_15m |          60 | taker, edge ≥ 4¢: markout ¢ |  7226 |  -1.806 |  -7.973 |       nan     | nan     |
| pm_updown_5m  |          15 | gap                         |  1346 |   0.039 |   1.696 |       nan     |   0.003 |
| pm_updown_5m  |          15 | gap + past move             |  1346 |   0.038 |   1.245 |        -0.004 |   0.003 |
| pm_updown_5m  |          30 | gap                         |  1240 |   0.051 |   1.413 |       nan     |   0.003 |
| pm_updown_5m  |          30 | gap + past move             |  1240 |   0.068 |   1.341 |         0.04  |   0.003 |
| pm_updown_5m  |          60 | gap                         |  1023 |   0.109 |   1.639 |       nan     |   0.005 |
| pm_updown_5m  |          60 | gap + past move             |  1023 |   0.163 |   1.61  |         0.129 |   0.008 |
| pm_updown_5m  |          15 | taker, edge ≥ 0¢: markout ¢ |  1366 |  -1.524 |  -5.308 |       nan     | nan     |
| pm_updown_5m  |          30 | taker, edge ≥ 0¢: markout ¢ |  1262 |  -1.183 |  -2.217 |       nan     | nan     |
| pm_updown_5m  |          60 | taker, edge ≥ 0¢: markout ¢ |  1049 |  -0.526 |  -0.512 |       nan     | nan     |
| pm_updown_5m  |          15 | taker, edge ≥ 2¢: markout ¢ |  1236 |  -1.378 |  -4.319 |       nan     | nan     |
| pm_updown_5m  |          30 | taker, edge ≥ 2¢: markout ¢ |  1145 |  -1.056 |  -1.841 |       nan     | nan     |
| pm_updown_5m  |          60 | taker, edge ≥ 2¢: markout ¢ |   950 |  -0.494 |  -0.454 |       nan     | nan     |
| pm_updown_5m  |          15 | taker, edge ≥ 4¢: markout ¢ |  1081 |  -1.179 |  -3.437 |       nan     | nan     |
| pm_updown_5m  |          30 | taker, edge ≥ 4¢: markout ¢ |  1004 |  -0.73  |  -1.173 |       nan     | nan     |
| pm_updown_5m  |          60 | taker, edge ≥ 4¢: markout ¢ |   825 |  -0.124 |  -0.109 |       nan     | nan     |
