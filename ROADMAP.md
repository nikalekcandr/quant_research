# Дорожная карта — куда развивать btcvol дальше

> Этот документ — живой бэклог. Приоритеты пересматриваются каждую неделю по итогам `oos_report.md`, `lab.html` и `lab_analysis.md`. Всё, что помечено `(frozen …)`, уже зафиксировано в коде и не перенастраивается задним числом.

Дата: 29.10.2026 · Ветка: `arena/5d0c3cc1-quant-research` · Снепшот данных: 29.10.2026 12:00 UTC

---

## Как читать

- **P0** — делаем сейчас (1–2 недели, измеримый эффект, данные уже есть).
- **P1** — следующий месяц (нужна разработка или новая выгрузка, но гипотеза сформулирована).
- **P2** — исследовательский бэклог (идея есть, дизайн ещё не зафиксирован).
- **Infra** — то, без чего всё остальное ненадёжно.

Каждый пункт: *что → зачем → как проверить → стоимость данных/кода*.

---

## P0 — быстрые победы (до 15.11)

### 1. Lead–lag на секундах между спотом и короткими бинарами

**Проблема:** AGENTS.md прямо просит «lead–lag между Binance/Coinbase и short binaries at sub-minute resolution (data in `data/lab/fast`, `flow`)». Сейчас D06/D07/D09 пытаются торговать это вслепую, а измерения нет.

**Что сделать:**
- Новый скрипт `studies/microstructure_leadlag.py` (уже в репо, см. ниже): на `data/lab/fast` (15с mids) + `data/lab/markets_fast` (котировки 5m/15m) посчитать:
  - кросс-корреляцию `Δlog(spot)` → `Δmid(binary)` на лагах −120…+120с;
  - Hasbrouck information share и Gonzalo-Granger (VAR 1–2 лага);
  - event study: спот прыгнул ≥ 10bp за 15с — как быстро бинар догоняет (half-life, hit rate).
- Разбить по венью (Coinbase vs Binance vs Kraken), по времени суток (US cash vs ночь), по `left` (много vs мало времени до экспирации).
- Вывод: если спот ведёт с лагом 5–15с — это чистый сигнал для мейкера/тейкера; если нет — закрываем D09 и перераспределяем капитал.

**Метрика успеха:** t-stat кросс-корреляции на лучшем лаге > 4 (HAC), half-life < 20с — есть что торговать.

### 2. Строгий мейкер-симулятор

**Проблема:** M01/M02 сейчас «виртуальные заявки fair ± 4¢, filled only if a real trade prints strictly through our price» — правило уже строгое, но нет анализа adverse selection (нас исполняют именно когда мы неправы).

**Что сделать:**
- В `lab/maker.py` и `lab/analysis.py` добавить:
  - adverse drift: средний `mid(5m после fill) − fill_price` в нашу сторону;
  - queue position: долю времени наша цена была лучшей bid/ask до принта;
  - fill rate vs размер книги (когда спред 1¢ vs 3¢).
- Сравнить P&L мейкера (maker fee `0.0175·P(1-P)`) vs тейкера на тех же рынках.

**Метрика:** fill-задержка и adverse drift должны быть < edge, иначе мейкер убыточен даже с хорошим fair.

### 3. Фандинг-кэрри как портфель, а не одиночные ставки

**Наблюдение:** `R11 cash-and-carry` — единственная стратегия в `research.md` с DSR=1.0 и Sharpe net 6.0 (собираем фандинг шортом перпа + лонгом спота). В lab это не торгуется системно, только D04 (fade funding) и F01 (межбиржевой спред фандинга).

**Что сделать:**
- Новый модуль `src/portfolio.py` / `lab/signals.py::F02`: каждый фандинг-слот (00/08/16 UTC) — если `funding > 5bp/8h`, открываем хедж, закрываем сразу после выплаты; размер = `target_vol / σ` (vol-targeting), cap 2×.
- Учесть: borrow cost спота, rebalancing каждые 4ч, проскальзывание.

**Метрика:** годовая доходность net vs buy&hold, max DD, turnover.

### 4. Документация в один документ

**Проблема:** README 38k, RUN_ME 14k, AGENTS 5.6k дублируют друг друга; новичок тонет.

**Сделано в этом PR:**
- `docs/ARCHITECTURE.md` — единая карта (поток данных, модели, оценка, lab, live).
- Этот `ROADMAP.md` — куда идём.
- Новый `README.md` (ниже) — 5-минутный quickstart + ссылки.

**Осталось:** удалить секцию «v6.1 что и в каком порядке запускать до сдачи» из верха RUN_ME после 01.11 (историческая).

### 5. Портфель и риск — модуль `src/portfolio.py`

**Сделано:** базовый модуль с:
- Kelly fraction по каждому сигналу (с shrink к 1/N и cap 2×);
- vol targeting (цель 15% ann, окно 30д);
- drawdown-контроль (halving после −10% DD);
- корреляционная поправка (если 3 сигнала на один BTC move — делим ставку).

Интеграция в lab: `lab/engine.py` перед `book.submit()` прогоняет `portfolio.scale(orders, ctx)`.

---

## P1 — следующий месяц (до 01.12)

### 6. Regime-aware пулы и смеси

**Мотивация:** R18 (зависит ли информ. выигрыш лесенки от vol-режима) — не значим на kx1h, но на pm24h `harx_all` в low-vol даёт +36% CRPS gain vs +30% в high-vol; тренд `bull` +33% vs `bear` +16%. R19 (regime-HAR) даже вредит (QLIKE gain −0.016, p=0.025). Значит, режимы важны, но текущая параметризация (взаимодействие с `ms_prob`) не работает.

**Что сделать:**
- Обучить `pool` отдельно в каждом режиме (Markov high/low, trend bull/bear, weekend/weekday) и смешивать по текущему `P(high)` (Bayesian model averaging вместо одного OLS).
- Добавить `har_regime` в `src/models.py` с дамми `I(high)`, `I(bull)` и их взаимодействиями с `l_rv_*`.

### 7. Хвосты и wings — от описательной к торговой

**Наблюдение:** R10 (crash insurance overpriced) подтверждается с t=5.3, но торговля крыльев W01/W02 в live — 0 сделок (порог 2–6¢ слишком редок) и в истории sell downside tail уходит в −2¢ после комиссий. Крылья «выигрывают 97–99% времени, ≈0 после спреда и комиссий» (AGENTS.md).

**Что сделать:**
- Перейти от «продать крыло по 3¢» к «продать крыло когда модель говорит что его fair ≤ 1¢» (W02 уже это делает, но выборка мала).
- Тест на `alpha_extra:X2`: edge 55 мин до экспирации 3.1¢ (t=1.8), к 10 мин — уже −3¢ (t=−3.1). Значит, крылья надо продавать рано и крыть за 10 мин до конца. Добавить holding period analysis.

### 8. Опционная секция — кросс-веню и экзотика

**Сейчас:** O01/O03 — стат-арбитраж улыбки (0 сигналов), O02 put-call parity (0), O04 straddle vs наш прогноз (1 сделка, −$468). Логика верная, но выборка 10 мин снимков слишком редка.

**Что сделать:**
- Участить снимки опционов до 1 мин в последний час перед дневной экспирацией (08:00 UTC Deribit).
- Добавить дисперсионный своп: стрип OTM опционов → MFIV → сравнить с нашим `harx_all` (Carr-Wu).
- Cross-venue basis: Deribit vs Binance/Bybit settlement indices (USD vs USDT) — это не арбитраж, а базис; измерить его и торговать mean reversion базиса.

### 9. Калибровка коротких бинаров

**Наблюдение:** `lab_analysis.md` секция 2: почти все fair-модели хуже рынка на 5-мин бинарах (Brier ours 0.160 vs market 0.134, t=−3.37). На `kx_15m` наоборот — мы лучше (0.124 vs 0.128), но t лишь 0.66. Значит, сезонность и уровень волатильности для 5-мин окон настроены плохо.

**Что сделать:**
- Переоценить `seasonal_profiles` на 1-минутных данных (сейчас часовые бины, а для 5-мин нужен минутно-точный профиль — R13 показал что :00, :05, :30 — аномальные минуты).
- Добавить HARQ-коррекцию (Bollerslev et al. 2016) именно для коротких окон: `RV_t * IQR` как мера шума измерения.

### 10. Polymarket hourly vs Kalshi hourly — честный тест

**Сейчас:** `alpha.py:pm1h` уже строит этот тест (Polymarket daily за 55 мин vs Kalshi hourly, один исход, две цены), но в `alpha_live` B1 — лишь 9 сигналов на 8 settled (pnl 0.29, t не посчитан — мало событий). Нужен накопленный сэмпл.

**Что сделать:** отдельный дашборд «pm1h gap»: средний `P_pm − P_kx` по страйкам, зависимость от спреда, от `minutes_left`, от basis. Это чистый тест эффективности двух розниц.

---

## P2 — исследовательский бэклог (зима 2026/27)

| # | Идея | Источник | Что нужно |
|---|---|---|---|
| 11 | **Order-book imbalance на уровне 10bp** (D06 теряет −8.6 t) — переобучить порог 0.4 на истории, добавить depth asymmetry + queue imbalance | Cont et al. 2014 | `bn_imb10` уже в `ticks`, нужен бэктест по сетке |
| 12 | **Volume-volatility elasticities**: объём предсказывает дисперсию лучше чем цена | Andersen et al. | минутный объём Binance уже в `minutes.parquet` |
| 13 | **Jump-robust HAR**: bipower variation, MedRV вместо RV | Barndorff-Nielsen & Shephard | `realized.py` добавить |
| 14 | **Term structure VRP**: DVOL 7d vs 30d vs realized — carry vol | Carr & Wu | Deribit term structure уже в `deribit_chain` |
| 15 | **ETF flows → BTC**: daily ETF creations/redemptions (Farside) как предиктор overnight | Lou et al. | новый downloader |
| 16 | **On-chain → vol**: mempool fee, active addresses, SOPR | on-chain folklore | Glassnode API |
| 17 | **Limitless vs Polymarket latency arb** (A03 уже нашёл 29 кейсов, median 3¢) — автоматизировать | A3 | `limitless` уже в `data/live` |
| 18 | **Multi-asset**: ETH/BTC vol spillover, beta-hedged vol arb | — | ETH 1m уже можно скачать тем же `binance.py` |
| 19 | **Reinforcement learning для execution**: оптимальное размещение лимиток (Avellaneda-Stoikov) | — | симулятор книги нужен |
| 20 | **Conformal prediction**: интервалы с гарантированным покрытием вместо гауссиан | Vovk et al. | `evaluate.py` добавить |

---

## Infra — без этого всё посыпется

| Задача | Статус | Владелец |
|---|---|---|
| `status.py` → алерты в Telegram/Slack (логгер молчит > 15 мин, `late` > 3, диск < 10GB) | TODO | infra |
| Бэкап `data/lab/ledger.csv` в S3 + ежедневный `ledger head` в гит (уже в `daily_update.py`, но не пушится без GitHub remote) | частично | infra |
| CI: `pytest` на `test_parsers.py` + `test_v6.py` + синтетика `run_all --synthetic` на каждый PR | TODO | infra |
| Версионирование данных: `data/interim/*.parquet` → DVC или хотя бы `sha256` в `results/tables/data_summary.csv` | TODO | infra |
| Дашборд `lab.html` → Grafana/Streamlit с живыми графиками P&L, exposure, vol zoo | в процессе (HTML уже auto-refresh 300с) | infra |
| Права на `forecasts_live/` — pre-commit hook запрещает ручное редактирование | TODO | infra |

---

## Что уже сделано в этом PR

| Файл | Что |
|---|---|
| `docs/ARCHITECTURE.md` | Единая карта проекта (поток данных → модели → оценка → lab → live) |
| `ROADMAP.md` | Этот документ |
| `docs/IDEAS.md` | 20 новых торговых идей с дизайном эксперимента |
| `src/portfolio.py` | Портфель: Kelly, vol-targeting, drawdown brake, корреляционный cap |
| `src/risk.py` | Риск-метрики: VaR/ES, drawdown, turnover, exposure |
| `studies/microstructure_leadlag.py` | Lead-lag на секундах (fast ticks × short binaries) |
| `README.md` | Переписан: 5-мин quickstart, результаты 29.10, ссылки на доки |

---

## Как предложить новую идею

1. Заведи issue с шаблоном: *гипотеза → источник → знак эффекта → данные → статистика (NW/HAC/Holm/DSR) → стоимость → как отличить альфу от шума*.
2. Если идея торговая — сразу опиши `lab/signals.py` ID, `MARGIN`, `hold_min`, `fee` и как избежать look-ahead.
3. PR должен содержать: код + тест на синтетике (`python run_all.py --synthetic` проходит) + строку в этом ROADMAP.

---

*Следующий пересмотр: после ночного обновления 01.11 (финальные цифры live-периода).*
