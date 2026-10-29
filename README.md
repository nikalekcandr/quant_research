# Волатильность биткоина по лесенкам Polymarket и Kalshi

> **Курсовой проект РЭШ «Introduction to Forecasting» · дедлайн 01.11.2026 · снепшот данных 29.10.2026 12:00 UTC**
> **Архитектура и дорожная карта:** `docs/ARCHITECTURE.md` · `ROADMAP.md` · `docs/IDEAS.md`

На Polymarket и Kalshi торгуются лесенки бинарных контрактов «BTC выше K к моменту T». Лесенка — это цифровой опцион-стрип: по её ценам восстанавливается распределение цены к моменту T, а ширина распределения — прогноз рынка о волатильности на окно до T. Мы прогнозируем ту же волатильность своими моделями и проверяем: кто точнее — time-series модель, опционы Deribit или розничная лесенка, и можно ли на разнице заработать после спредов и комиссий.

---

## 30-секундный старт

```powershell
# 1. окружение
python -m venv .venv; .venv\Scripts\activate; pip install -r requirements.txt
python tests/test_parsers.py          # 7× ok — парсеры на реальных ответах API

# 2. проверка конвейера на синтетике (5 мин, без скачивания)
python run_all.py --synthetic

# 3. реальные данные (1.5–2.5 ч; докачка с места остановки)
python run_all.py                     # Binance 5м + Polymarket 30м + Deribit 10м + Kalshi 1–2ч

# 4. результаты
ls results/tables/*.csv results/figures/*.png reports/draft_report.md
python oos_report.py                  # сводка history / new / live
python research.py --quick            # R1–R21, 3 мин → reports/research.md
python -m studies.microstructure_leadlag --plot  # новый: lead-lag на секундах
```

**Live-режим** (логгер стаканов + прогнозы + alpha lab) — см. `RUN_ME.md` §§ 2–4.

---

## Что уже проверено (на 29.10.2026)

### Волатильность

| Горизонт | Лучший прогноз | CRPS (new, 13–28.10, n=344×1ч) | vs рынок |
|---|---|---|---|
| **55 мин (Kalshi)** | `harx_all` (HAR-X + Deribit + лесенка) | **0.333** | ничья с центром лесенки 0.336 (p=0.76), бьёт Deribit 0.699 (p<0.001) |
| **24 ч (Polymarket)** | `+ Options` 0.191 / `+ Ladder` 0.195 | — | ничья с рынком 0.201 (p=0.58) на n=11 (мало) |
| **6 д (Polymarket)** | `Equal-weight blend` 0.151 / `+ Ladder` 0.153 | — | ничья, Deribit 0.123 лучше на n=10 (шум) |

**Вывод:** на коротком горизонте лесенка — сильный сигнал (добавляет −0.024 CRPS к HAR-X, p<0.001); на сутках и неделях рынок уже эффективен — наши модели догоняют, но не бьют. Live-борд подтверждает: `harx_all` на 6 новых лесенках 13–28.10 держит паритет.

Полные таблицы + DM-тесты — `reports/oos_report.md` (3 сэмпла: history / new / live) и `reports/draft_report.md`.

### Research R1–R21 (Binance 1m 2023–2026)

| Статус | Гипотезы |
|---|---|
| **Подтверждены (Holm p<0.05)** | R1 сезонность волатильности (corr год-к-году 0.96), R10 хвосты лесенки переоценены (2.6pp, t=5.3), R13 всплеск в :00 минуту (1.28×, p<1e-8), R21 HAR бьёт GARCH(1,1) (QLIKE +0.10, p<1e-17) |
| **Likely (FDR 10%)** | R2 overnight drift 21bp/д (p=0.031), R8a variance risk premium (IV²>RV, 688→ p=0.010), R12 ΔDVOL ведёт доходность (t=2.75), R19 regime-HAR (но отрицательный) |
| **Не подтверждены** | R3–R7, R9, R11, R14–R18, R20 — все p>0.1 после Holm; ничего tradeable после 5bp комиссий, кроме funding carry (Sharpe 6.0, см. ниже) |

→ `reports/research.md` (8 графиков) и `results/research/*.csv`.

### Alpha Lab — live бумажные торги (50 гипотез, frozen 29.10)

- **Аннулировано (ошибки сбора, не результаты):** все сделки по Kalshi до 29.10 14:35 МСК — котировки брались из
  листинга `/markets`, который отстаёт от стакана на 30–60 с (в среднем 6¢); A11 «арбитраж» — Kalshi отдаёт пустую
  сторону стакана как 0. Отсюда «P06 +6–12¢», «A11/A12 700+ случаев, hit 1.00». С 14:35 Kalshi торгуется только по
  стакану, см. `lab/report.py` (`VOID`, `void_mask`) и `AGENTS.md`.
- **P02 Polymarket 15m:** +6.0¢ на контракт на малой выборке, p > 0.2, Holm не пройден.
- **D-группа (направленные):** D06/D07 значимо убыточны (t −8.6 / −3.1).
- **W01/W02 (продажа крыльев 2–6¢)** запущены 29.10 14:35, на истории ≈ 0 или минус после спреда и комиссии.
- **V (15-сек fair):** все 5 моделей около 0; V06 stale-quote — 1 сделка.
- **Vol zoo:** лучшее — `gbm_vol` QLIKE 0.22 vs `har_hourly` 0.31.

→ `reports/lab.html` (auto-refresh 300с) и `reports/lab_analysis.md` (markouts, buckets, capacity).

### Торговля на истории (alpha.py — лесенки как бинары)

- SPA-тест: ни один вариант (модель×порог 2/5/10¢) не бьёт buy-and-hold после поправки на множественность.
- Hold-out пополам: выбранный на первой половине вариант уходит в 0 на второй.
- Brier: модели не лучше рынка (DM p>0.1), кроме `recal` (перекалибровка longshot bias).

---

## Структура

```
config.py               все даты/окна/пороги/пути
run_all.py              весь проект одной командой
live.py / live_once.py  логгер стаканов + live-прогнозы (каждые 10 мин)
daily_update.py         ночная докачка + пересборка + oos_report + alpha_live
backfill.py             докачка истории (Kalshi январь–июнь — 4ч)
discover_venues.py      разовая проверка всех площадок

src/download/           binance, polymarket, kalshi, deribit, funding, venues
src/measure/            realized.py (RV, semivar, jumps, сезонность), ladder.py, deribit_iv.py
src/build.py            панели: строка = момент прогноза, признаки + цель
src/models.py           13+ моделей, rolling fits
src/evaluate.py         QLIKE/CRPS/logscore/DM/MCS/Brier/торговля
src/extensions.py       MCS, оптимальный пул, флуктуации, крылья
src/alpha.py            гибриды, перекалибровка, Kalshi vs Polymarket, SPA, hold-out
src/portfolio.py        ★ новое: Kelly, vol-targeting, drawdown brake, корреляционный cap
src/risk.py             ★ новое: VaR/ES, drawdown, PIT/Christoffersen, QLIKE-decomp
src/pipeline.py         порядок шагов run_all
src/figures.py          7 графиков отчёта
src/report.py           черновик отчёта с цифрами
src/live_forecast.py    live-прогнозы + git + hash-chain
src/synthetic.py        фальшивые данные для проверки кода

research.py             R1–R21 → reports/research.md
oos_report.py           history / new / live → reports/oos_report.md
alpha_live.py           реальные стаканы → reports/alpha_live.md + alpha_extra.md
lab/                    alpha lab: 50 гипотез, paper book, hash-ledger
  signals.py            REGISTRY (D/P/A/V/W/M/O/F, frozen 2026-10-29)
  engine.py             минутный + 15-секундный лупы, volzoo, ML, settlement
  feeds.py / markets.py / flow.py / maker.py / ml.py / volzoo.py
  report.py / analysis.py
studies/
  kx15_study.py         Kalshi 15-min на book candles (чистый тест)
  microstructure_leadlag.py ★ новое: lead-lag спот ↔ короткие бинары (15с)

docs/
  ARCHITECTURE.md       карта проекта (поток данных → модели → lab → live)
  IDEAS.md              20 новых торговых идей с дизайном эксперимента
ROADMAP.md              приоритизированный бэклог (P0/P1/P2/Infra)
RUN_ME.md               пошаговый запуск на Windows (логгер, tasks, бэкапы)
AGENTS.md / CLAUDE.md   правила для контрибьюторов (no look-ahead, Kalshi только стакан, …)

data/raw/               ответы API как есть
data/interim/           чистые таблицы и панели (minutes, pm_ladder, panel_*, strikes_*)
data/live/              снимки логгера (pm_books, kalshi, deribit_chain, binance, …)
data/lab/               ticks/fast/flow/markets/ledger.csv/positions.json
results/                forecasts_*, scores, trades, figures, research/, lab/
reports/                draft_report, research, oos_report, alpha_*, lab.html
forecasts_live/         живые прогнозы с hash-chain (не редактировать!)
```

---

## Данные: откуда и подводные камни

| Источник | Что | Подводные камни (проверено) |
|---|---|---|
| **Binance** `data.binance.vision` | 1m свечи BTC/USDT с 2023 | С 2025 время в микросекундах; текущий месяц — дневные файлы + REST |
| **Polymarket** `gamma-api` + `clob` | Daily лесенки «BTC above ___ on date» с 01.09.2025, 8–11 страйков × $2000 | Расчёт по close свечи Binance 12:00 ET; лесенка открывается за 7д; slug `…-october-13-2026` vs `…-march-2`; шаг 0.1¢, спреды 0.1–1¢; в истории нет bid/ask — только логгер |
| **Kalshi** `api.elections.kalshi.com` | KXBTCD часовые, ~190 страйков × $100, минутки с bid/ask | Расчёт по среднему BRTI за 60с (не Binance); поправка basis = медиана BRTI/Binance; рынок 1ч; листинг лагает 30–60с — используем только `orderbook`/`candlesticks` |
| **Deribit** | DVOL + сделки опционов → ATM IV на любое окно | Сделки только в окне 60м перед снепшотом; премии сейчас в BTC, не в USD |
| **Binance/Bybit/OKX Options** | цепочки для B3/D1/O-группы | Разные индексы расчёта (USD vs USDT) — не арбитраж, а базис |
| **Limitless** | зеркало Polymarket Up-or-Down | `externalSlug` = Polymarket slug; A3 нашёл 29 crossing quotes |

Сырые ответы — в `data/raw/`; тесты парсеров — `tests/test_parsers.py` на реальных ответах.

---

## Как запустить

### Разовая установка (5 мин)

```powershell
cd C:\Users\<ты>\btcvol
python -m venv .venv; .venv\Scripts\activate
pip install -r requirements.txt
python tests/test_parsers.py         # 7× ok
```

### Логгер — запустить сразу

```powershell
python live.py --no-forecast         # каждые 10 мин в data/live/; не закрывать окно, отключить сон
```

### Скачать историю и прогнать всё (второй терминал)

```powershell
python run_all.py                    # 1.5–2.5 ч; --no-kalshi чтобы отложить Kalshi (45–90 мин)
# результат: results/tables/*.csv, results/figures/*.png, reports/draft_report.md
python run_all.py --skip-download    # после правки кода — без перекачки
```

### Live-прогнозы (после прогона)

```powershell
# Ctrl+C в первом окне, затем:
python live.py                       # прогнозы в 12:00 ET (24ч/6д) и HH:05 (55м) → forecasts_live/ + git
```

### Alpha Lab (24/7 бумажные торги)

```powershell
python lab_daemon.py                 # минутный + 15с лупы, 50 гипотез, ledger + hash
# отчёт: reports/lab.html (auto-refresh), reports/lab.md
```

### Ежедневная проверка

```powershell
python status.py                     # пишет ли логгер, нет ли late, ledger head
python oos_report.py                 # history / new / live
python research.py --quick           # 3 мин
python alpha_live.py                 # реальные стаканы → reports/alpha_live.md
```

**Windows Tasks (рекомендуется):** `install_tasks.ps1` создаёт 3 задачи (`btcvol-live` каждые 5м, `btcvol-daily` 05:10, `btcvol-fast` 17:55/18:55). Требует «Разрешить таймеры пробуждения». Удаление — `uninstall_tasks.ps1`.

Подробно — `RUN_ME.md`.

---

## Методология (коротко)

- **HAR(-X)** с сезонностью (Andersen & Bollerslev), semivariance (Patton & Sheppard), jumps, HARQ, календарём.
- **Ladder → variance:** BJN model-free variance + lognormal/Student-t fits; центр vs рынок (encompassing).
- **Deribit IV:** ATM vol из сделок в окне 60м → интерполяция на горизонт.
- **Оценка:** QLIKE (Patton 2011), CRPS, log-score, PIT, coverage; DM (Newey-West, лаг = перекрытие), MCS (Hansen), SPA (Hansen), fluctuation test (Giacomini-Rossi); Brier + торговля на страйках (пороги 2/5/10¢, half-spread 0.5/1/2¢ для Polymarket); deflated Sharpe (Bailey & López de Prado).
- **Research R1–R21:** NW t, Holm/BH, DSR, regime breakdown (Markov high/low, vol tercile, trend, era, year).
- **Lab:** каждая гипотеза — `f(ctx) → list[Order]`, frozen дата, real bid/ask + fees, held to settlement, t по кластерам (час/рынок), Holm + DSR.

---

## Результаты — что запомнить для отчёта

1. **Сезонность волатильности — железобетонная** (R1, p<1e-300, corr 0.96), но уже в HAR.
2. **Лесенка знает vol лучше Deribit на часе** (CRPS 0.333 vs 0.357, p<0.001; vs Deribit 0.699), на сутках/неделе — паритет с рынком (рынок эффективен).
3. **Хвосты лесенки переоценены** (R10, 2.6pp, t=5.3), но продать их после спреда+комиссий нельзя (W01 0 сделок, история −2¢).
4. **HAR бьёт GARCH** (R21, QLIKE +0.10, p<1e-17) — realised measures рулят.
5. **Funding carry — единственная стратегия с DSR=1.0** (Sharpe net 6.0, 7% годовых за счёт фандинга) — собирать фандинг шортом перпа + лонгом спота, а не fade funding.
6. **Всё остальное (momentum, FOMC, CME gaps, round numbers, ML) — не проходит Holm** после комиссий.

---

## Куда дальше

См. **`ROADMAP.md`** — 20 пунктов в 4 приоритетах:

- **P0 (2 недели):** lead-lag на секундах, строгий мейкер, фандинг-кэрри как портфель, `src/portfolio.py` + `src/risk.py` уже в репо.
- **P1 (месяц):** regime-aware пулы, хвосты/крылья с модельным фильтром, опционная term structure, кросс-веню pm1h.
- **P2 (зима):** volume-vol, jump-robust HAR, ETF flows, on-chain, multi-asset, RL execution, conformal.
- **Infra:** алерты, CI, DVC, Grafana.

Новые идеи — `docs/IDEAS.md` (20 штук с дизайном эксперимента). Новый анализ — `studies/microstructure_leadlag.py` (spot 15с × бинары 5m/15m, кросс-корреляция + VAR + event study).

---

## Правила для контрибьюторов

См. `AGENTS.md` (кратко):

1. **No look-ahead:** прогноз в `t` — только `e_j ≤ t`.
2. **Kalshi — только стакан** (`orderbook`/`candlesticks`), не `markets`.
3. **Пустой стакан = missing**, не 0.
4. **Правильная база расчёта** (BRTI vs Binance vs Chainlink, USDT/USD basis).
5. **Комиссии в каждом P&L.**
6. **Гипотезы frozen** — правило ID не меняем, новый ID с новой датой.
7. **Статистика:** HAC/кластеры, Holm/BH, deflated Sharpe, train/test пополам во времени.

Код/комментарии — английский, доки — русский. Даты/окна/пороги — только `config.py`.

---

## Лицензия и контакты

MIT · Данные — публичные API площадок · Вопросы — issue в GitHub или `lab_daemon.py` heartbeat.
