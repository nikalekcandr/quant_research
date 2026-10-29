# Microstructure tests (29.10.2026, 17:00 MSK)

## 1. Kyle–Obizhaeva / trading activity → volatility (`studies/invariance_vol.py`, Binance 1m 2023–2026)
Out of sample 2024-01 → 2026-10, expanding window refit monthly, QLIKE, DM with Newey–West vs seasonal HAR.

| Horizon | HAR | + trades N | + $ volume | + Kyle λ | + abs(flow imbalance) | + all | invariance (var/trade × trades) |
|---|---|---|---|---|---|---|---|
| 15 min | 0.459 | 0.435 (t 3.2) | 0.438 (t 3.5) | 0.452 (t 4.0) | 0.450 (t 3.9) | **0.430 (t 3.7)** | 0.449 (t 2.0) |
| 1 hour | 0.330 | 0.331 (t −2.4) | 0.329 (t 3.5) | 0.329 (t 3.6) | 0.330 (t 2.9) | 0.331 | 0.334 (t −4.1) |

Activity improves the 15-minute variance forecast by ~6% QLIKE, every year (2024, 2025, 2026); at 1 hour the gain
is ~0.5%. Use for short binaries' fair value and in the course project as a microstructure extension.

## 2. Signed taker flow → future returns (Binance, 2 M minutes)
Impact 3.1 bp per sd of flow in the same minute (t 152). Next minute: 0.003 bp (t 0.2); 5 min: −0.09 bp
(t −3.6); after the 0.1% largest sell flows +3.5 bp in 5 min. Real but far below a 10 bp round trip: no trade.

## 3. Who earns on Kalshi 15-minute markets (`studies/kx15_makers.py`, 105 markets, 2.25 M prints, Oct 2026)
Favourite–longshot bias in executed trades: takers who buy the cheap side lose −1.9¢ (≤5¢, t −19.7), −6.0¢
(5–10¢, t −6.5), −9.8¢ (10–20¢, t −4.3) per contract; takers who buy the favourite at 90–100¢ win +1.9…+5.3¢
(t 3.9…11.8). The resting side of lottery-ticket buys earns +1.9¢ (t 19.5) after the maker fee.
Only October 2026 so far; the 220-day sample is downloading (`jobs/hist_kx15_trades_sample_now.py`).
