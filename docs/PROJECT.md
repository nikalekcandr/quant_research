# Полная документация проекта `quant_research`

> Прогнозирование реализованной волатильности BTC на данных розничных рынков предсказаний (Polymarket, Kalshi), опционов (Deribit) и минутных свечей (Binance), а также круглосуточная лаборатория бумажной торговли («Alpha Lab»).

---

## 1. Назначение и главные выводы

Проект отвечает на три вопроса:
1. **Несут ли розничные лесенки страйков (Kalshi, Polymarket) информацию о будущей волатильности BTC?**
   Да: на часовом горизонте Kalshi (`kx1h`, окно 55 минут) CRPS центра распределения равен **0.336** против **0.699** у опционов Deribit (p < 0.001, OOS 13–28.10.2026, 344 ч), потому что рынок предсказаний учитывает внутридневную сезонность (открытие США vs азиатская ночь), а дневная IV Deribit — плоская. Комбинация `HAR-X + Deribit + лесенка` (0.333) сравнивается с рынком.
2. **Могут ли модели превзойти рынок в вероятностях страйков и точечной волатильности?**
   На точечной волатильности градиентный бустинг на 3 годах минутных данных бьёт HAR на **23–27% по QLIKE** внутри дня (лучший результат — янв–окт 2026 с признаками лесенки, CRPS 0.3255). Но по **Brier score на бинарных страйках** рынок бьёт все модели дисперсии — калибровка самого рынка по страйкам точнее.
3. **Есть ли торговая альфа после комиссий и поправки на множественное тестирование?**
   Внутридневной альфы после спредов/комиссий и поправки Холма не найдено (5 первичных «альф» оказались артефактами данных). Активный фокус с 03.11.2026 — только **продажа дешёвых крыльев** (`W03–W06` в `lab/signals.py`, `docs/WINGS.md`).

---

## 2. Структура репозитория

```
config.py                 Все даты, окна, пороги, пути к данным и результатам
run.py                    Основной пайплайн: download -> measure -> build -> models -> evaluate
research.py               Блок R1–R21: расширенные тесты, микроструктура, макро, отложенная выборка
requirements.txt          Python-зависимости

src/
  download/               Загрузка данных:
    binance.py            Минутные свечи BTCUSDT (2023–2026)
    polymarket.py         Лесенки и котировки Polymarket (24h, 6d)
    kalshi.py             Серии KXBTC (1h) и KXBTC15M (15m)
    deribit.py            Опционы Deribit и индекс DVOL
    funding.py            Ставки фандинга (Binance/Bybit/OKX)
  measure/                Измерение волатильности и распределений:
    realized.py           RV, BV, MedRV, RS+/RS-, RQ, Jump share, маскировка аутэйджей
    ladder.py             Восстановление распределения из лесенки страйков (MFIV, lognormal, Student-t)
    deribit_iv.py         Интерполяция IV Deribit на горизонт
  build.py                Построение панелей (panel_kx1h, panel_pm24h, panel_pm6d) без look-ahead
  models.py               Спецификации моделей (HAR, HAR-X, IV, Market, Combo, GBM), скользящая оценка
  evaluate.py             QLIKE, MSE_log, CRPS, Brier, PIT, Mincer-Zarnowitz, DM (HAC+HLN), MCS
  extensions.py           MCS, оптимальные смеси, SPA-тест, отложенная выборка
  alpha.py                Торговля лесенками, симуляция исполнения с учётом спреда/комиссий

lab/                      Круглосуточная лаборатория бумажной торговли («Alpha Lab»):
  engine.py               Цикл сбора тиков, расчёта сигналов и записи бумажных сделок
  signals.py              51 гипотеза (семейства A, D, F, O, P/V, W, M, T); FOCUS = W03–W06
  markets.py              Коннекторы к рыночным данным (Kalshi, Polymarket, Binance, Deribit)
  maker.py                Симуляция мейкерских ордеров
  tape.py                 Лента сделок и агрегация
  volzoo.py               Зоопарк волатильностных моделей в реальном времени
  report.py               Генерация отчётов лаборатории
  jobs.py                 Планировщик задач

trader/                   Исполнитель для Kalshi (shadow / demo / live), см. docs/TRADER.md
studies/                  Самостоятельные исследования:
  test_100_ideas.py       Тест 100 идей/моделей на kx1h (см. docs/IDEAS_100.md)
  gen_ideas100_doc.py     Генератор docs/IDEAS_100.md из results/ideas100/results.csv
  gbm_kx1h.py             Бустинг на цели kx1h
  short_lags.py           Короткие лаги (15m–4h)
  kx15_study.py           15-минутная серия Kalshi
  tape_strategies.py      Стратегии по ленте сделок

tests/                    Юнит-тесты (pytest)
docs/                     Документация (ARCHITECTURE, PROJECT, IDEAS, IDEAS_100, LITERATURE, WINGS, EXECUTION, TRADER, SERVER)
data/                     raw/, interim/, processed/, lab/, live/ (крупные файлы вне Git)
results/                  Таблицы, отчёты, прогнозы (forecasts_kx1h.parquet, ideas100/, studies/)
```

---

## 3. Данные

| Источник | Файл / путь | Покрытие | Примечание |
|---|---|---|---|
| Binance 1m | `data/interim/minutes.parquet` | 2023-01-01 … 2026-10-29 (2,011,811 строк) | Аутэйджи биржи (forward-filled close, `filled=True`, `volume=NaN`) маскируются при расчёте RV/BV/MedRV |
| Kalshi 1h | `data/processed/panel_kx1h.parquet` | 2023-02-01 … 2026-10-29 (32,763 строки) | Окно 55 минут; лесенка страйков `strikes_kx1h.parquet` (44,177 строк) с 2025-12-01 |
| Polymarket | `panel_pm24h.parquet`, `panel_pm6d.parquet` | — | Горизонты 24 часа и 6 дней |
| Deribit | `deribit_iv.parquet`, `data/raw/deribit/dvol.csv.gz` | `deribit_iv`: с 2025-06-01 (515 строк); DVOL: часовой OHLC | Дневная IV и индекс DVOL |
| Фандинг | `data/raw/funding/btc_funding.csv.gz` | 8-часовые ставки | `ts`, `rate`, `venue` |
| Lab / Live | `data/lab/`, `data/live/` | 24/7 сбор | Тики и снапшоты стаканов |

---

## 4. Модели и протокол оценки

### 4.1 Признаки и модели (`src/models.py`)
- **HAR:** `l_rv_d` (24ч), `l_rv_w` (7д), `l_rv_m` (30д) + `l_F` (интенсивность/уровень для `kx1h`).
- **EXTRA:** `l_rs_pos`, `l_rs_neg` (полудисперсии Patton & Sheppard 2015), `jump_share`, `harq` (`l_rv_d * sqrt(rq_d)`, Bollerslev et al. 2016), `macro` (FOMC/CPI/NFP), `weekend`.
- **Модельный ряд:** `naive`, `garch` (только `pm`), `har`, `harx`, `harx_d`, `har_m`, `harx_m`, `harx_all`, `iv_raw`, `iv_mz`, `mkt_raw`, `mkt_ln`, `mkt_t`, `mkt_mz`, `combo`, `gbm_all`.
- **Защита от утечки будущего (no look-ahead):** признаки в момент `t` используют только историю до `t`; скользящая OLS/Ridge (окно 2,160 часов для `kx1h`, переобучение раз в неделю) обучается строго на наблюдениях с `e_j <= t`.

### 4.2 Метрики и тесты (`src/evaluate.py`)
- **Точечные:** QLIKE (Patton 2011 — устойчива к шуму прокси дисперсии), MSE на `log RV`.
- **Вероятностные:** гауссова CRPS, log-score, PIT/coverage, Brier score на бинарных страйках.
- **Статистические тесты:** Diebold–Mariano с HAC Newey–West и поправкой Harvey–Leybourne–Newbold (HLN), Model Confidence Set (Hansen et al. 2011), тест Минцера–Зарновица (калибровка), поправка Холма на множественное тестирование.

---

## 5. Лаборатория альфы (`lab/`) и исполнитель (`trader/`)

- **51 гипотеза** в 8 семействах: `A` (безарбитражность), `D` (направленные), `F` (фандинг), `O` (опционы), `P/V` (справедливая стоимость), `W` (дешёвые крылья), `M` (мейкерство), `T` (лента).
- С 03.11.2026 активный фокус сужен до **`W03–W06`** (продажа переоценённых дальних крыльев на Kalshi).
- Режимы исполнителя `trader/`: `shadow` (логирование без ордеров), `demo` (тестовая среда Kalshi), `live` (реальный счёт с лимитами риска).

---

## 6. Запуск и воспроизведение

```bash
# Установка зависимостей
pip install -r requirements.txt

# Полный пайплайн проекта
python run.py

# Исследование 100 идей и генерация отчёта
python studies/test_100_ideas.py
python studies/gen_ideas100_doc.py

# Юнит-тесты
pytest tests/
```
