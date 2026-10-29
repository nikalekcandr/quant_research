# Microstructure lead-lag: spot vs short binaries

_Generated 2026-10-29 12:47 in 0.0 min._
Data: 2919 fast ticks, 734 minute ticks, 12286 binary quotes.
Spot venues: binance, coinbase; binary kinds: kx_15m, kx_ladder_1h, pm_ladder_1h, pm_updown_15m, pm_updown_1h, pm_updown_4h, pm_updown_5m.
Lags tested: -120s … 120s at 15s resolution; event-study threshold 10.0bp/15s.

## How to read

- **Cross-correlation peak at +15s** → spot at t predicts binary at t+15s (spot leads by 15s).
- **Granger p < 0.05 (spot→mid) and p ≥ 0.05 (mid→spot)** → one-way causality: spot leads.
- **Event study hit rate > 0.5 + t > 2** → binary reliably follows spot jumps — tradeable.
- **Half-life**: seconds until binary covers 50% of the spot-implied move.

## Cross-correlation peaks

## Event study: binary response to spot jumps

## What to do with this

- If spot leads by 10–30s with corr > 0.15 and event-study t > 3 → build a taker strategy (D09) or a maker that fades the stale quote (V06).
- If the VAR shows feedback (both directions) → spread is too tight, adverse selection dominates — maker needs wider quotes.
- If no lead-lag is found → close D06/D07/D09 and reallocate to funding carry (F01) or vol zoo.
