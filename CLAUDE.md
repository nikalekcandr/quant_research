# AGENTS.md — guide for AI agents and new contributors

Read this before changing anything. It is the shortest path to working on this project without repeating
mistakes that already cost us days. (Same file as `CLAUDE.md`.)

## What the project is

Course project, NES "Introduction to Forecasting" (due 1 Nov 2026): **forecast Bitcoin volatility from retail
prediction-market strike ladders (Polymarket, Kalshi)**, compare against time-series models and Deribit options,
and test whether the gaps are tradeable after spreads and fees. On top of that sits a 24/7 paper-trading
"alpha lab" (`lab/`) that tests ~50 frozen hypotheses on live quotes.

Language: code and comments in English, most docs (`README.md`, `RUN_ME.md`, reports) in Russian.

## Layout

| Path | What |
|---|---|
| `config.py` | every date, window, threshold, path |
| `src/download/` | Binance, Polymarket, Kalshi, Deribit, funding, other venues — verified endpoints documented in each file's docstring |
| `src/measure/` | realised variance, ladder → distribution fit, Deribit ATM IV |
| `src/build.py, models.py, evaluate.py, extensions.py, alpha.py` | panels, 13+ models with rolling fits, scoring (QLIKE, CRPS, Brier, DM, MCS), alpha search |
| `run_all.py`, `backfill.py`, `daily_update.py` | full pipeline, history backfill, nightly update |
| `live.py`, `live_once.py`, `src/live_forecast.py`, `forecasts_live/` | live out-of-sample forecasts, SHA-256 chained — **never edit files in `forecasts_live/`** |
| `research.py` | research layer R1–R21 (regimes, classic inefficiencies, vol model zoo) → `reports/research.md` |
| `lab/` + `lab_daemon.py` | the alpha lab: feeds (7 venues), short binaries (Polymarket 5m/15m/1h, Kalshi 15m and hourly ladder), hypothesis registry `lab/signals.py`, paper book + hash-chained ledger `lab/core.py`, vol zoo, ML, report `reports/lab.html` |
| `jobs/` | one-off background scripts, started by the lab daemon (`lab/jobs.py`); `jobs/done/*.json` = finished |
| `studies/` | clean, self-contained studies (e.g. `kx15_study.py`: Kalshi 15-minute markets on book candles) |
| `data/raw/` | API responses as downloaded; `data/interim/` cleaned tables; `data/live/` 10-minute logger snapshots; `data/lab/` lab logs (ticks, market quotes, ledger) |
| `results/`, `reports/` | tables, figures, markdown/HTML reports |
| `tests/` | parser tests on real responses, pipeline tests |

## Hard rules

1. **No look-ahead.** A forecast at time t may use only data with timestamps < t. Variance forecasts, basis
   estimates, seasonal profiles: all rolling and lagged.
2. **Kalshi quotes: use the order book, never the `/markets` listing.** The listing (`yes_bid`/`yes_ask` in
   `GET /markets`) lags the live book by 30–60 s (measured 29.10.2026: 6.1¢ mean gap on the 15-minute market).
   Every "edge" we found against it was an artifact. Use `GET /markets/{ticker}/orderbook` live and
   `/markets/candlesticks` (or `/historical/markets/{ticker}/candlesticks`) for history: those carry book bid/ask.
3. **Kalshi reports an empty book side as 0.** Treat ≤ 0 as missing, not as a price of zero (cost us 720 fake
   arbitrage trades).
4. **Settlement sources differ.** Kalshi BTC: CF Benchmarks BRTI, 60-second average before the close (USD).
   Polymarket daily/hourly ladders: Binance BTC/USDT 1-minute candle close. Polymarket 5m/15m Up/Down:
   Chainlink BTC/USD. Use the right spot and correct the USDT/USD basis.
5. **Fees are part of every P&L.** Kalshi taker `ceil(0.07·P·(1−P))` to the cent, maker `ceil(0.0175·P·(1−P))`;
   Polymarket crypto markets `0.07·P·(1−P)`; Binance perp 5 bp/side; options 0.03% of spot capped at 12.5% of price.
6. **Lab hypotheses are frozen.** Never change the rule of an existing ID in `lab/signals.py`; add a new ID with
   a `frozen=` date. The ledger (`data/lab/ledger.csv`) is append-only and hash-chained (`verify_ledger`).
   Trades produced by buggy code are excluded in `lab/report.py` (`VOID`, `KALSHI_LISTING_VERSIONS`), never deleted.
7. **Never compare quotes taken at different moments.** Three fake "alphas" so far had one cause: the Kalshi
   listing vs the live book, a 10-minute option snapshot vs the live perpetual (O02: $6,139 from one $200 move,
   29.10 13:33 UTC), stale Kalshi sides. Check the age of every quote; if the underlying moved more than the edge
   since the oldest one, skip. Many trades from one snapshot are one observation, not many.
8. **One month is not a sample.** The maker-on-longshots edge (+8.7¢, t 3.4 on October) was −2¢ on Mar–Oct.
   Validate every finding on the longest history available before freezing a rule.
9. **Statistics.** Cluster t-stats by market / entry hour (trades on one market are not independent), HAC for
   time series, Holm / BH for many hypotheses, deflated Sharpe for strategy search, train/test split in time.

## Where things stand (29.10.2026 18:30)

* **Volatility forecasting (the course question):** out of sample on the Kalshi hourly window the ladder centre
  CRPS 0.336 beats Deribit 0.699 (p < 0.001) and ties our HAR-X + ladder. Gradient boosting trained on three
  years of minutes beats HAR by 23–27% QLIKE intraday (2024–2026, every year) and, averaged with the ladder, is the
  best variance forecast over Jan–Oct 2026 (CRPS −2.4% vs the ladder, t −5.3); not significant on the last two
  weeks. `results/studies/vol_models_summary.md`.
* **Probabilities of strikes:** the market price beats every model on Brier (Kalshi hourly ladder and 15-minute
  market, 20 317 markets): better variance does not beat the book.
* **Alpha:** nothing survives. Taker vs model, wings, lead–lag, Kyle-λ flow, 15m vs hourly ladder, Kalshi vs
  Polymarket, maker on longshots (October only), copy/fade the tape — all ≈ 0 or negative after fees on the full
  samples. Several fake edges were timing artifacts (rule 7). `STRATEGIES.md` lists active and retired rules.
* Still live: W02, A12/A12F (cross-venue on simultaneous books), V07 (60-s settlement average), O-family with the
  snapshot-age guard, D/F families.

## Good next tasks

* Run `studies/kx15_study.py` on the full Kalshi 15-minute history; the same design for Polymarket 5m/15m.
* Lead–lag between Binance/Coinbase and short binaries at sub-minute resolution (data in `data/lab/fast`, `flow`).
* Maker strategies: simulated fills must be strict (only prints through our price).
* Clean up README into one document; the live sections at the top of `RUN_ME.md` are the current ones.

## Running

Windows, Python 3.11+ (developed on 3.13): `pip install -r requirements.txt`, then see `RUN_ME.md`.
`python tests/test_parsers.py` checks the parsers. Most scripts take `--help`.
