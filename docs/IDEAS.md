# 20 новых торговых идей — дизайн эксперимента

> Каждая идея — в формате *гипотеза → источник → знак → данные → статистика → как не обмануться*.
> Все идеи используют только данные, которые уже качаются (или легко докачиваются тем же `src/download/`).
> Порог значимости: Holm p < 0.05 или DSR > 0.95, HAC/кластеры, net-of-fees.

---

## Направление A — микроструктура и исполнение

### 1. Queue imbalance + OFI на Binance spot

**Гипотеза:** `OFI = Δ(bid_depth) − Δ(ask_depth)` за 1с предсказывает следующую 1-минутную доходность.
**Источник:** Cont, Kukanov & Stoikov (2014).
**Данные:** `data/lab/ticks` уже содержит `bn_imb10` (глубина в 10bp), но не содержит OFI. Добавить в `feeds.py`: подписка на `depth@100ms` Binance, считать OFI каждую секунду.
**Дизайн:** regression `r_{t+1m} = α + β·OFI_t + ε`, NW t-stat, IC. Стратегия: `|OFI| > 2σ → long/short 1m`, hold 60с, fee 5bp.
**Ловушка:** OFI скоррелирован с прошлым return (feedback loop) → контролировать `r_t` в регрессии.

### 2. Stale-quote sniping с учётом размера

**Гипотеза:** V06 (stale-quote при `|Δspot|≥4bp` и `Δmid≤1¢`) прав, но `edge` должен масштабироваться на `|Δspot| / spread`.
**Источник:** Budish, Cramton & Shim (2015) latency arbitrage.
**Данные:** `fast` (15с) + `markets_fast` уже есть.
**Дизайн:** вместо фиксированного `MARGIN 3¢` — `edge = |Δspot| / σ√τ − |Δmid|`. Торговать только когда `edge > 2×spread`.
**Проверка:** markout анализ (уже в `lab_analysis.md`): если `markout 15с > 0` и растёт — котировка была stale.

### 3. Maker P&L decomposition

**Гипотеза:** мейкер убыточен из-за adverse selection, а не из-за плохого fair.
**Данные:** `maker.py` симуляция уже строгая (fill только если print сквозь цену).
**Дизайн:** разложить P&L = `edge_at_entry − adverse_drift(5m) − fees`. Если `adverse_drift ≈ edge` — нас выбирают когда мы неправы. Решение: квотировать только когда `spread > 2σ` и `time_left > 10m`.

### 4. Funding + OI squeeze

**Гипотеза:** экстремальный funding (> 15bp/8h) + рост OI (> 2% за час) → squeeze, за которым следует reversal в следующий фандинг-слот.
**Источник:** He et al. (2022) + anecdotal.
**Данные:** `bf_funding`, `bf_next_funding`, OI уже в `ticks`.
**Дизайн:** event study: `funding > 90th percentile & ΔOI > 1% → r_{next 8h}`. NW t, разбивка по regime (high/low vol).

---

## Направление B — кросс-веню

### 5. Basis USDT/USD mean reversion

**Гипотеза:** `basis = log(BRTI / Binance) − log(USDT/USD)` mean-reverts за минуты; когда `|basis| > 2σ` — торговать basis (long BTC на дешёвой площадке, short на дорогой).
**Источник:** Lyons (2001) FX microstructure; basis уже в `alpha.py` как `basis = BRTI/Binance`.
**Данные:** `ticks` содержит `usd_mid`, `bn_mid`, `usdt_usd`.
**Дизайн:** cointegration test (Engle-Granger), half-life, trade: `z(basis) > 2 → short basis 5m`, fee 2×5bp.

### 6. Deribit vs Binance/Bybit basis trade (perpetual)

**Гипотеза:** Deribit perp (USD index) vs Binance perp (USDT index) — тот же BTC, разный индекс; спред mean-reverts после стресс-событий.
**Данные:** `db_mid`, `bf_mid`, `by_mid`, `ok_mid` в `ticks`.
**Дизайн:** как #5, но на перпах; добавить funding differential в fair.

### 7. Polymarket vs Kalshi hourly — term structure

**Гипотеза:** `gap = P_pm − P_kx` на один и тот же час (pm1h) зависит от `time_left`: близко к экспирации Polymarket sticky, Kalshi — нет.
**Данные:** `alpha.py:pm1h` уже строит этот сэмпл; `alpha_live:B1` — live версия.
**Дизайн:** regression `gap = α + β₁·time_left + β₂·spread + β₃·volatility + ε`, HAC. Торговать `|gap| > 3¢ + fees`.

### 8. OP-expiry pinning

**Гипотеза:** BTC стремится к max pain (страйк с макс открытым интересом) в день экспирации Deribit (пятница 08:00 UTC).
**Источник:** Ni, Pearson & Poteshman (2005).
**Данные:** `deribit_chain` уже содержит весь OI по страйкам.
**Дизайн:** `distance_to_maxpain = |S − K_maxOI| / S`, `r_{to expiry} = α + β·distance + ε`. Торговать только в пятницу 00–08 UTC.

---

## Направление C — волатильность

### 9. Intraday HARQ

**Гипотеза:** шум измерения RV (realized quarticity) коррелирует с ошибкой прогноза; HARQ (`HAR + RQ^{1/2}`) лучше HAR на коротких окнах.
**Источник:** Bollerslev, Patton & Quaedvlieg (2016).
**Данные:** `minutes.parquet` → `RQ = (n/3)·Σ r_i^4`.
**Дизайн:** добавить `harq = sqrt(RQ)` в `src/models.py:EXTRA`, сравнить QLIKE OOS.

### 10. Jump-robust HAR

**Гипотеза:** jumps (BNS test) — шум, а не сигнал; MedRV лучше RV для прогноза.
**Источник:** Andersen, Bollerslev & Diebold (2007).
**Данные:** те же минуты.
**Дизайн:** `MedRV = median(|r_i|,|r_{i+1}|,|r_{i+2}|)` вместо `RV`; сравнить с обычным HAR.

### 11. VRP term structure carry

**Гипотеза:** `VRP_7d − VRP_30d > 0` → term structure крутая → продавать ближний vol, покупать дальний.
**Источник:** Carr & Wu (2009).
**Данные:** DVOL уже есть (1d fix), но Deribit chain даёт всю term structure IV.
**Дизайн:** `carry = IV_7d² − IV_30d² − (RV_7d − RV_30d)`; regression `r_straddle_7d − r_straddle_30d` на `carry`.

### 12. Volatility-of-volatility timing

**Гипотеза:** когда vol-of-vol высока (дисперсия log RV за 30д), следующий RV более непредсказуем → уменьшать размер.
**Источник:** Corsi et al. (2008) HAR with vol-of-vol.
**Дизайн:** `vol_of_vol = std(log RV, 30d)`, `weight = 1 / vol_of_vol`, сравнить Sharpe vol-managed с и без.

---

## Направление D — режимы и календарь

### 13. Weekend vs weekday vol seasonality — торговля

**Гипотеза:** weekend RV = 0.4× weekday (research R5), но опционы/лесенки этого не учитывают полностью → продавать weekend vol в пятницу.
**Данные:** R1 seasonality уже измерена (корр 0.96 год-к-году).
**Дизайн:** пятничный close: `fair_weekend = level × seasonal_factor(weekend)`, сравнить с implied; если `implied > fair + 2σ` → short weekend straddle.

### 14. FOMC/CPI volatility crush

**Гипотеза:** IV завышена перед macro (R6: vol 5–7× выше после релиза), а после — crush; продавать straddle за 1ч до релиза, крыть через 1ч после.
**Источник:** Savor & Wilson (2013) + R6 таблица (CPI 7.7×, FOMC 5.5×).
**Данные:** `us_macro_calendar.csv` + `deribit_chain` snapshots.
**Дизайн:** event study: `P&L = premium_sold − |S_T − K|`, net of fees, HAC.

### 15. ETF flow → overnight drift

**Гипотеза:** daily ETF net inflow (Farside) предсказывает следующий overnight return (16:00→09:30 ET).
**Источник:** Lou et al. (2019) overnight drift уже в R2 (21bp/day, p=0.031).
**Данные:** новый downloader `src/download/etf_flows.py` (Farside API, daily).
**Дизайн:** `r_overnight = α + β·flow_{t-1} + ε`, NW t, `flow` z-scored.

---

## Направление E — кросс-актив и on-chain

### 16. ETH/BTC vol spillover

**Гипотеза:** скачок vol ETH предсказывает скачок vol BTC через час (crypto vol factor).
**Источник:** Bouri et al. (2017).
**Данные:** ETH 1m уже можно скачать тем же `binance.py` (добавить `ETHUSDT`).
**Дизайн:** VAR(1) `log RV_BTC, log RV_ETH`, Granger test, OOS QLIKE gain от добавления `l_rv_eth_*` в HAR.

### 17. Mempool fee → volatility

**Гипотеза:** всплеск Bitcoin mempool fees (перегрузка сети) предшествует всплеску vol (стресс-событие).
**Данные:** mempool.space API (бесплатно, 1 запрос/мин) → новый `src/download/mempool.py`.
**Дизайн:** `log RV_{t+1} = α + β·log(fee_t) + HAR_controls + ε`.

### 18. Stablecoin premium → funding

**Гипотеза:** `USDT/USD` premium (уже в `ticks` как `usdt_usd`) предсказывает funding rate (спрос на леверидж).
**Данные:** `usdt_usd` уже в `ticks`.
**Дизайн:** `funding_{t+1} = α + β·premium_t + ε`, 8-часовой горизонт.

---

## Направление F — исполнение и портфель

### 19. Optimal execution — Avellaneda-Stoikov

**Гипотеза:** оптимальная лимитка `reservation_price = mid − γ·σ²·(T−t)·inventory` + `spread/2 + γ·σ²·T` улучшает P&L мейкера vs наивной `fair ± 4¢`.
**Источник:** Avellaneda & Stoikov (2008).
**Данные:** `ticks` + `flow` для калибровки `γ` (risk aversion) и `κ` (arrival rate).
**Дизайн:** бэктест: симулировать оба размещения на исторических книгах, сравнить fill rate и adverse selection.

### 20. Kelly + drawdown brake — портфель

**Гипотеза:** текущий размер 1 контракт на сигнал — неоптимален; Kelly с fractional (0.5×) и vol targeting + DD brake улучшает Sharpe и снижает max DD.
**Источник:** Kelly (1956), Moreira & Muir (2017).
**Данные:** `ledger.csv` — история P&L всех гипотез.
**Дизайн:** `src/portfolio.py` уже реализует: `f* = μ/σ²` по каждому сигналу (EWMA 30д), cap 2×, `dd_brake = 1 if DD < 10% else 0.5`, `corr_cap = 1/√n` для n сигналов на один underlying. Бэктест на истории ledger с HAC.

---

## Как добавить новую идею

1. Скопируй шаблон выше.
2. Зафиксируй знак эффекта *до* просмотра данных (AGENTS.md правило 1).
3. Опиши точный `lab/signals.py` ID, `MARGIN`, `hold_min`, `fee` и как избежать look-ahead.
4. PR: код + тест на синтетике + строка в ROADMAP.

*Следующий пересмотр — после 01.11 live-финала.*
