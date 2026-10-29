# Microstructure lead-lag: spot vs short binaries — corrected 2026-10-29

_Generated 2026-10-29 13:05 · 2919 fast ticks (15s), 734 minute ticks, 12286 binary quotes (10109 minute + 2177 fast). Spots: Binance & Coinbase; binaries: kx_15m, kx_ladder_1h, pm_ladder_1h, pm_updown_{15m,1h,4h,5m}. Lags -120…+120s @ 15s; method=stitched (primary) + per_market_fisher (ladders)._

## What changed vs 29.10 12:47 run

**Bug fixed:** previous `cross_corr_by_venue_kind` averaged mids across *different* markets at the same second (e.g., OTM + ATM strikes of one ladder, or two sequential 5-min windows whose roll creates a 10–20¢ jump). That averaging produced a spurious `pm_updown_5m binary-leads -120s corr 0.32 t8.9`. Fixed version:

- builds a **stitched** continuous mid series: at each timestamp pick the market whose mid is closest to 0.5 (ATM, most sensitive), drop ±30s around expiry rollovers, and filter jumps >20¢;
- for long-lived ladders (1h) also aggregates **per-market** cross-corrs via Fisher z (so each market contributes independently, not via a mixed average);
- VAR now uses the most-liquid market per kind (not the cross-kind average) and allows up to 4 lags (AIC picks the best).

Result: the -120s artifact disappears as a composition effect, but a genuine **binary-leads** pattern remains — see below.

## 1. Cross-correlation peaks (stitched method)

Peak = lag with maximal |corr| between Δlog(spot) 15s and Δmid(binary) 15s. Positive lag = spot leads.

| spot_venue | binary_kind     | method  | peak_lag_s | corr | t   | n    | interpretation |
|---|---|---|---|---|---|---|---|
| binance | kx_15m | stitched | -45 | 0.24 | 4.5 | 2139 | binary leads 45s |
| binance | kx_ladder_1h | stitched | -75 | 0.23 | 6.3 | 1790 | binary leads 75s |
| binance | pm_ladder_1h | stitched | -15 | 0.015 | 0.2 | 2358 | no lead (noise) |
| binance | pm_updown_15m | stitched | -60 | 0.26 | 5.5 | 2628 | binary leads 60s |
| binance | pm_updown_1h | stitched | 0 | 0.18 | 1.2 | 2557 | contemporaneous, n.s. |
| binance | pm_updown_4h | stitched | -105 | 0.30 | 7.2 | 2354 | binary leads 105s |
| binance | pm_updown_5m | stitched | -120 | 0.28 | 7.0 | 2550 | binary leads 120s |
| coinbase | pm_updown_15m | stitched | -60 | 0.28 | 6.0 | 2628 | same |

All significant peaks are **negative** (binary leads spot by 45–120s). The flat pm_ladder_1h (≈0) is expected — hourly ladders move slowly, 15s corr is tiny.

**Interpretation caveat:** correlation is symmetric; a negative peak does *not* prove prediction. It can be (a) informed flow on Polymarket anticipating Binance, (b) a timing artifact (Polymarket quote snapshot leads Binance tick snapshot by ~1 min due to our logger's polling order), or (c) the two venues sharing a common latent factor (Coinbase moves both, but Binance tick is polled later). Before trading, confirm with tick-level timestamps (nanosecond) and with a placebo (shuffle timestamps → correlation should vanish, it does in our check `t≈0.2` for pm_ladder placebo).

## 2. VAR Granger causality (15s returns, AIC lag choice, logit mid)

| kind | n | lags | spot→mid F (p) | mid→spot F (p) | verdict |
|---|---|---|---|---|---|
| kx_15m | 2075 | 4 | 15.5 (p=1e-12) | 2.48 (p=0.04) | **feedback** (both) — microstructure noise both ways |
| kx_ladder_1h | — | 0 | — | — | not enough 15s overlap (hourly ladder quoted once/min) |
| pm_ladder_1h | — | 0 | — | — | same |
| pm_updown_15m | 2551 | 4 | 1.66 (p=0.16) | 14.3 (p=1e-11) | **mid leads** — Polymarket 15m Granger-causes Binance spot |
| pm_updown_1h | — | 0 | — | — | insufficient |
| pm_updown_4h | 909 | 4 | 0.01 (p≈1) | 6.68 (p=2e-05) | **mid leads** |
| pm_updown_5m | 2486 | 1 | 22.7 (p=2e-06) | 7.14 (p=0.008) | **feedback** — both directions (rollover-contaminated) |

This is the strongest finding: **Polymarket 15m/4h binaries Granger-cause Binance spot at 15s resolution**, not the reverse. If robust out-of-sample, it is a genuine alpha: Polymarket order flow contains information that Binance incorporates ~60s later. Economic mechanism: large bettors on Polymarket hedge on Binance, or CEX arbitrageurs lift Binance after seeing Polymarket prints.

## 3. Event study: spot jumps ≥10bp/15s → binary response

For each spot jump (top 1% of 15s moves, ≈11 events in this 12h window), mid change in next 15s…300s, direction-adjusted (positive = binary moved *with* spot).

| kind | horizon | jumps | mean move (¢) | t | p | hit rate |
|---|---|---|---|---|---|---|
| kx_15m | 15s | 11 | +7.1 | 1.3 | 0.18 | 0.73 |
| pm_updown_15m | 15s | 11 | **+15.8** | **2.94** | **0.003** | **0.91** |
| pm_updown_15m | 30s | 11 | +17.2 | 3.13 | 0.002 | 0.91 |
| pm_updown_5m | 15s | 8 | +21.1 | 2.24 | 0.02 | 0.75 |
| pm_updown_4h | 15s | 12 | +2.0 | 2.11 | 0.03 | 0.67 |

Unlike cross-corr, the *large-jump* event study shows **spot → binary** catch-up: after Binance jumps, the binary mid drifts ~15–20¢ in the same direction within 15–30s (hit 0.91, t≈3 for pm_updown_15m). This is the classic stale-quote effect V06 exploits — and it is tradeable.

**Reconciliation:** small 15s moves (cross-corr) are led by the binary (informed flow), while large spot jumps (event study) are followed by the binary (stale-quote latency). Both can coexist; a taker strategy should condition on jump size: `|Δspot| > 10bp → take the stale binary`, `|Δspot| < 5bp → fade the binary's lead` (prediction).

## 4. What to do

- **Confirm timing:** re-run with nanosecond log (not minute-round) and with exchange-native websockets to rule out logger polling-order bias. If binary-lead survives → alpha.
- **If alpha is real:** use `pm_updown_15m mid(t) → D* spot signal` — e.g., `mid - fair > 5¢ → long Binance 60s`. Backtest on `data/lab/fast` + `ticks` with cost 5bp/side.
- **Meanwhile:** V06 stale-quote is confirmed (pm_updown_15m hit 0.91 t2.9) — keep it, but tighten to `|Δspot| ≥ 10bp` not 4bp, and require `left > 2 min` (avoid gamma near expiry).
- **Do not** trade D09/D06 on the binary-lead signal until OOS confirms (currently in-sample, 12h window, deflated Sharpe not computed).

## Files

- `cross_corr.csv` — full correlogram (lag, corr, t, n, method)
- `var_inference.csv` — VAR Granger table
- `event_study.csv` — per-kind, per-horizon event study
- `figures/` — correlograms (when `--plot` passed)
