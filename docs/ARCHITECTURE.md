# Архитектура btcvol — карта проекта

> **Статус данных:** снепшот 29.10.2026 · 546k строк лаб-леджера · 60+ гипотез в live · 13 моделей волатильности
> **Заморозка спецификации:** 15.10.2026 — код моделей (`src/models.py`, `build.py`, `evaluate.py` и др.) байт-идентичен замороженной версии (проверяет `oos_report.py`).

## 1. Что прогнозируем и зачем

**Сырьё:** лесенки бинарных контрактов Polymarket (`Bitcoin above K on date`) и Kalshi (`KXBTCD` часовые) — это полоски цифровых опционов. По их ценам восстанавливается распределение цены BTC к моменту `T`, а ширина распределения — это прогноз рынка о волатильности на окно `[t, T)`.

**Цель:** спрогнозировать ту же реализованную дисперсию (сумма квадратов 1-минутных лог-доходностей Binance BTC/USDT) лучше, чем:
- наивный бенчмарк (дисперсия предыдущего окна той же длины);
- HAR-RV (Corsi 2009) и его расширения;
- Deribit (DVOL / ATM implied vol из сделок);
- сама розничная лесенка (центр + хвосты).

**Горизонты (config.HORIZONS):**

| id | площадка | окно | снепшот |
|---|---|---|---|
| `kx1h` | Kalshi KXBTCD | 55 минут | `HH:05` каждого часа |
| `pm24h` | Polymarket daily | 24 часа | `HH:00` за сутки до 12:00 ET |
| `pm6d` | Polymarket daily | 6 дней | `HH:00` за 6 суток до 12:00 ET |

Дополнительно: `pm1h` — та же daily-лесенка за 55 минут до расчёта (пересекается с Kalshi часовой, один и тот же исход двумя ценами).

---

## 2. Поток данных

```
Binance 1m (2023→)  ─┐
Deribit DVOL+bchain ─┤
Polymarket ladders  ──┼─► data/raw/  (дневные .csv.gz, докачка с места остановки)
Kalshi ladders     ──┤         │
Deribit trades    ──┘         ▼
                    data/interim/  minutes.parquet, pm_ladder.parquet, deribit_iv.parquet
                                  panel_{kx1h,pm24h,pm6d}.parquet — одна строка = один прогноз-случай
                                         признаки (HAR, semivariance, jumps, календарь, опционы, лесенка) + цель y=log RV
                                  │
                                  ▼
                    models.py ──► results/forecasts_{h}.parquet (rolling, refit каждые 24/1/1 случаев, окно 90/365 дней)
                                  results/scores.parquet (CRPS, log-score, QLIKE, покрытие, PIT)
                                  │
                                  ├─► evaluate.py / extensions.py / alpha.py ──► results/tables/*.csv
                                  ├─► figures.py ──► results/figures/*.png
                                  └─► report.py ──► reports/draft_report.md

live.py (каждые 10 минут) ──► data/live/{pm_books,kalshi,deribit_chain,binance}/*.csv.gz
                           ──► src/live_forecast.py ──► forecasts_live/{kx1h,pm24h,pm6d}/*.json (git + hash-chain)
lab_daemon.py (каждую минуту) ──► data/lab/{ticks,fast,flow,markets,ledger.csv,positions.json}
                              lab/signals.py (50+ гипотез) ──► lab/report.py ──► reports/lab.html
```

**Ключевые инварианты (AGENTS.md):**

1. **No look-ahead:** любой прогноз в `t` использует только данные с `e_j ≤ t`.
2. **Kalshi котировки — только стакан** (`/markets/{ticker}/orderbook`, `/markets/candlesticks` с bid/ask), не листинг `/markets` (лагает 30-60с, 6.1¢ средняя ошибка на 15-мин рынке).
3. **Стакан ≠ 0:** пустая сторона Kalshi = 0 → трактуем как missing.
4. **Разные базы расчёта:** Kalshi = среднее BRTI за 60с (USD), Polymarket daily = close свечи Binance BTC/USDT, Polymarket 5m/15m = Chainlink BTC/USD. Корректируем базис USDT/USD (медиана BRTI/Binance).
5. **Комиссии в каждом P&L:** Kalshi `ceil(0.07·P·(1-P))`, Polymarket `0.07·P·(1-P)` (сейчас 0 на крипто), perp 5bp, опционы 0.03% от спота cap 12.5%.
6. **Гипотезы заморожены:** правило существующего ID в `lab/signals.py` не меняем; новый ID с `frozen=дата`. Леджер append-only, hash-chained.

---

## 3. Модели (src/models.py — 13+ штук)

Нумерация по плану проекта:

| # | id | что внутри |
|---|---|---|
| 5 | `naive` | лог RV предыдущего окна той же длины (для `kx1h` — с сезонной поправкой `l_F`) |
| 6 | `garch` | GARCH(1,1)-t на 24-часовых доходностях (только `pm24h`/`pm6d`) |
| 7 | `har` | HAR-RV: daily + weekly + monthly (+ сезонный фактор `l_F` для `kx1h`) |
| 8 | `harx` | HAR + semivariances, jump share, HARQ, дамми макро-релизов и выходных |
| 9 | `harx_d` | HAR-X + Deribit: DVOL и ATM implied variance |
| 7m | `har_m` | HAR + ladder-implied variance (BJN, model-free) |
| 10 | `harx_m` | HAR-X + prediction-market implied variance |
| 11 | `harx_all` | HAR-X + Deribit + prediction market (encompassing test) |
| 12 | `iv_raw` | Deribit implied variance как есть |
| 13 | `iv_mz` | то же, rolling Mincer-Zarnowitz перекалибровка |
| 14 | `mkt_raw/ln/t` | ladder BJN / lognormal / Student-t — как есть |
| 15 | `mkt_mz` | ladder, rolling MZ |
| 16 | `combo` | равновзвешенная смесь `harx`+`iv_mz`+`mkt_mz` |
| 17 | `gbm_all` | gradient boosting на признаках `harx_all` (нелинейности, пропуски нативны) |

Плотность: `N(m, s²)` (или смесь 3 гауссиан для `combo`). `m` — OLS/GBM прогноз `y=log RV`, `s` — RMSE прошлых ошибок на том же окне.

---

## 4. Оценка (src/evaluate.py, extensions.py, alpha.py, research.py)

| Модуль | метрики | статистика |
|---|---|---|
| `evaluate.py` | QLIKE, CRPS, log-score, покрытие 50/90%, PIT | Diebold-Mariano (HAC, лаг = перекрытие окон), MCS (Hansen), fluctuation test |
| `extensions.py` | оптимальный пул (лог-скор), калибровка крыльев, календарь | White HAC, Hansen SPA |
| `alpha.py` | P&L на страйк-лесенках (пороги 2/5/10¢), Brier | HAC по снепшотам, SPA, hold-out пополам |
| `research.py` | R1–R21: сезонность, overnight, momentum, macro, VRP, хвосты и т.д. | Newey-West, Holm/BH, deflated Sharpe |

**Где смотреть цифры (29.10.2026):**

- `reports/oos_report.md` — CRPS/Brier/DM по 3 сэмплам (history / new / live), спреды логгера, code check.
- `reports/research.md` — вердикты R1–R21 + 2 графика на гипотезу.
- `reports/alpha_live.md`, `alpha_extra.md` — торговля на реальных стаканах.
- `reports/lab.html` — live-борд 50 гипотез (Holm + DSR), Brier model vs market, vol zoo, ML IC.

---

## 5. Alpha Lab (lab/)

```
lab/feeds.py      тикеры 7 площадок (Binance spot+perp, Bybit, OKX, Coinbase, Kraken, Deribit) + funding/OI/Liquidations
lab/markets.py    снимки коротких бинаров (Polymarket 5m/15m/1h/4h, Kalshi 15m, hourly ladder) + резолв исходов
lab/history.py    Minutes: цены, сезонность (hour-of-week), forecast_var()
lab/signals.py    REGISTRY: D(20) + P(7) + A(4) + V(6) + W(2) + M(2) + O(4) + F(1) гипотез, frozen 2026-10-29
lab/core.py       Book (позиции + hash-chained ledger), fees, heartbeat
lab/engine.py     Lab.minute() каждую минуту + fast_loop (15с) + hourly (volzoo, ML retrain, report)
lab/ml.py         GBM + Ridge на 20 price/volume/calendar признаках, walk-forward monthly
lab/volzoo.py     зоопарк волатильности (HAR, GARCH, EGARCH, GJR, EWMA, GBM, DVOL…) — QLIKE на 60-мин окнах
lab/report.py     leaderboard (Holm, DSR), calibration (Brier), health
lab/analysis.py   markouts, buckets, capacity, lead-lag
lab/flow.py       taker flow, liquidation flow (Binance)
lab/maker.py      симуляция мейкера (строгий fill: только print сквозь цену)
data/lab/ticks/   минутные тики (bid/ask/mid, funding, DVOL, premium index, OI)
data/lab/fast/    15-секундные mids Binance/Coinbase/Bybit
data/lab/flow/    минутный flow (buy/sell USD, liq long/short)
data/lab/markets/ котировки коротких бинаров + fair values всех моделей
data/lab/ledger.csv  OPEN/CLOSE/INSTANT, chain hash, code_version
```

Семейства гипотез — см. шапку `lab/signals.py` и `ROADMAP.md`.

---

## 6. Live-прогнозы (src/live_forecast.py)

- Делаются в `12:00 ET` (pm24h/pm6d) и `HH:05` (kx1h), пишутся в `forecasts_live/{h}/YYYYMMDDTHHMM.json` и коммитятся в git **до** исхода.
- `data/lab/ledger.csv` для live-трейдов тоже hash-chained; `lab.report.VOID` помечает сделки бажных версий (не удаляем, исключаем в отчёте).
- `daily_update.py` (tasks `btcvol-daily` 05:10) перекачивает новые сутки, пересобирает `results/`, зовёт `oos_report.py` + `alpha_live.py` + `research.py`.
- `status.py` — здоровье логгера (пишет ли, нет ли `late`).

---

## 7. Куда идти дальше

См. `ROADMAP.md` (приоритизированный бэклог) и `docs/IDEAS.md` (20 новых торговых идей).
Коротко: lead-lag на секунды, мейкер со строгими филами, фанд-арбитраж, кросс-venue опционы, regime-aware пулы, портфель/Kelly.

---

## 8. Быстрый старт

```powershell
pip install -r requirements.txt
python tests/test_parsers.py          # 7× ok
python run_all.py --synthetic         # проверка конвейера на синтетике (5 мин)
python run_all.py                     # вся история: 1.5–2.5 ч (Kalshi — отдельно долго)
python research.py --quick            # research layer, 3 мин
python oos_report.py                  # сводка history/new/live
python -m studies.kx15_study data/raw  # Kalshi 15-min book-candle study
python -m studies.microstructure_leadlag  # новый: lead-lag за секунды (см. ROADMAP)
```

Windows tasks (логгер + прогнозы + быстрый логгер в последний час перед полуднем ET):

```powershell
powershell -ExecutionPolicy Bypass -File install_tasks.ps1
python status.py   # проверка
```

---

## 9. Конвенции

- Код и комментарии — английский; доки/отчёты — русский.
- Даты/окна/пороги — только `config.py`.
- Никаких блокнотов в `main` без ревью; крупные датасеты вне git (см. `.gitignore`).
- Каждый новый результат — с HAC/кластерами, поправкой на множественность и net-of-fees.
