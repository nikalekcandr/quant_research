# Перечень квант-литературы для дальнейшего развития проекта

> **Новое (26.11).** Появился структурированный корпус: `data/lit/bibliography.csv` (201 работа, 13 доменов,
> статус «использовано / кандидат / нужны данные») и сгенерированная из него
> **[`docs/MEGA_ENCYCLOPEDIA.md`](MEGA_ENCYCLOPEDIA.md)**. Этот файл остаётся коротким списком «что читать
> по приоритету»; всё, что нужно для отбора гипотез и для расширения до тысяч записей импортом
> (`python -m src.lit.import_openalex`), — там.

> Ниже собрана литература, **ещё не использованная** в проекте (в список намеренно не включены уже применённые работы: Corsi 2009 HAR, HARQ, Barndorff-Nielsen & Shephard BV/MedRV, Patton & Sheppard 2015, Carr & Wu 2009, Britten-Jones & Neuberger 2000, Carr & Madan 2005, Avellaneda & Stoikov 2008, Kyle 1985, Hasbrouck 1995, Cont et al. 2014, Gu-Kelly-Xiu 2020, Hansen MCS, DM/HLN, Grimit CRPS, R1–R21 и др.).
> Для каждой работы указано, **как именно её можно применить** в `quant_research`.

---

## 1. Моделирование и прогнозирование волатильности

1. **Patton, A. J. (2011).** *Volatility forecast comparison using imperfect volatility proxies.* Journal of Econometrics, 160(1), 246–256. — Обоснование выбора QLIKE и семейства робастных функций потерь Bregman при оценке на шумной RV.
2. **Gatheral, J., Jaisson, T., & Rosenbaum, M. (2018).** *Volatility is rough.* Quantitative Finance, 18(6), 933–949. — Модель RFSV / rough Bergomi (Hurst $H \approx 0.1$): замена лог-линейных лагов HAR на дробное интегрирование $(t-s)^{H-1/2}$ для часовой и 15-минутной волатильности BTC.
3. **Bennedsen, M., Lunde, A., & Pakkanen, M. S. (2022).** *Decoupling the short- and long-term behavior of stochastic volatility.* Journal of Financial Econometrics, 20(5), 961–1006. — Разделение шероховатости ($H$) и долгой памяти: двухфакторная модель для внутридневного BTC.
4. **Clement, E., & Gloter, A. (2011)** / **Jacod, J., Li, Y., Mykland, P. A., Podolskij, M., & Vetter, M. (2009).** *Microstructure noise in the continuous case: the pre-averaging approach.* Stochastic Processes and their Applications, 119(7), 2249–2276. — Pre-averaged RV для 1-минутных и тиковых данных Binance вместо обычного RV.
5. **Zhang, L., Mykland, P. A., & Aït-Sahalia, Y. (2005).** *A tale of two time scales: determining integrated volatility with noisy high-frequency data.* JASA, 100(472), 1394–1411. — Two-Scale Realized Volatility (TSRV) для борьбы с микроструктурным шумом на 15m/1h.
6. **Barndorff-Nielsen, O. E., Hansen, P. R., Lunde, A., & Shephard, N. (2008).** *Designing realized kernels to measure the ex post variation of equity prices in the presence of noise.* Econometrica, 76(6), 1481–1536. — Realized Kernel (Parzen/Tukey-Hanning) как альтернатива RV на часовом горизонте.
7. **Hansen, P. R., Huang, Z., & Shek, H. H. (2012).** *Realized GARCH: a joint model of returns and realized measures of volatility.* Journal of Applied Econometrics, 27(6), 877–906. — Совместное моделирование доходностей и RV с уравнением измерения; прямой апгрейд `garch` на `pm24h`/`kx1h`.
8. **engle, R. F., & Gallo, G. M. (2006).** *A multiple indicators model for volatility using intra-daily data.* Journal of Econometrics, 131(1-2), 3–27. — MEM (Multiplicative Error Model) для положительных величин (RV, объём, спред) без логарифмирования.
9. **Cipollini, F., Engle, R. F., & Gallo, G. M. (2013).** *Semiparametric vector MEM.* Journal of Applied Econometrics, 28(7), 1067–1086. — Векторная MEM для совместной динамики RV, объёма и бид-аск спреда.
10. **Shephard, N., & Sheppard, K. (2010).** *Realising the future: forecasting with high-frequency-based volatility (HEAVY) models.* Journal of Applied Econometrics, 25(2), 197–231. — HEAVY-модель: быстрая адаптация условной дисперсии к недавним минутным скачкам.
11. **Corsi, F., & Renò, R. (2012).** *Discrete-time volatility forecasting with persistent leverage effect and the link with continuous-time volatility.* Journal of Financial Econometrics, 10(1), 67–98. — LHAR-CJ: разделение непрерывной части, скачков и эффекта рычага на разных горизонтах.
12. **Audrino, F., & Knaus, S. D. (2016).** *Lassoing the HAR model: A model selection perspective on realized volatility dynamics.* Econometric Reviews, 35(8-10), 1485–1521. — Разреженный отбор лагов RV (1h..30d) через L1-регуляризацию вместо фиксированных 3 лагов.
13. **Hillebrand, E., & Medeiros, M. C. (2010).** *The benefits of bagging for forecast models of realized volatility.* Econometric Reviews, 29(5-6), 571–593. — Бэггинг и случайные леса поверх HAR-признаков для снижения дисперсии прогноза.
14. **Bucci, A. (2020).** *Realized volatility forecasting with neural networks.* Journal of Financial Econometrics, 18(3), 502–531. — Сравнение LSTM/GRU и HAR на лог-RV (актуально для 3-летней минутной выборки BTC).
15. **Christensen, K., Siggaard, M., & Veliyev, B. (2023).** *A machine learning approach to volatility forecasting.* Journal of Financial Econometrics, 21(5), 1680–1727. — Систематический бенчмарк ML-моделей (Ridge, ElasticNet, RF, GBDT, NN) против HAR с учётом интерпретируемости.

---

## 2. Опционы, безмодельная IV и рынки предсказаний

16. **Bakshi, G., Kapadia, N., & Madan, D. (2003).** *Stock return characteristics, skew laws, and the differential pricing of individual equity options.* Review of Financial Studies, 16(1), 101–143. — Безмодельные моменты (MFIV, MFSKEW, MFKURT) из лесенки страйков Kalshi/Polymarket как признаки для HAR-X.
17. **Breeden, D. T., & Litzenberger, R. H. (1978).** *Prices of state-contingent claims implicit in option prices.* Journal of Business, 51(4), 621–651. — Непараметрическая риск-нейтральная плотность (RND) как вторая производная цены колла/бинарного контракта по страйку.
18. **Ross, S. A. (2015).** *The recovery theorem.* Journal of Finance, 70(2), 615–648. — Восстановление физической меры $P$ из риск-нейтральной $Q$ по матрице переходов лесенки.
19. **Martin, I. (2017).** *What is the expected return on the market?* Quarterly Journal of Economics, 132(1), 367–433. — Нижняя граница ожидаемой доходности и SVIX из опционной лесенки.
20. **Wolfers, J., & Zitzewitz, E. (2004).** *Prediction markets.* Journal of Economic Perspectives, 18(2), 107–126. — Агрегация информации и систематические искажения (favourite–longshot bias) на рынках предсказаний — прямое объяснение перекоса крыльев `W03–W06`.
21. **Manski, C. F. (2006).** *Interpreting the predictions of prediction markets.* Economics Letters, 91(3), 425–429. — Почему цена бинарного контракта может отличаться от средней веры участников при ограниченном бюджете/неприятии риска.
22. **Gjerstad, S. (2005).** *Risk aversion, beliefs, and prediction market equilibrium.* Economic Theory, 26(2), 349–370. — Калибровка бинарных цен с учётом CRRA-неприятия риска (коррекция Brier score лесенки).
23. **Berg, J. E., Nelson, F. D., & Rietz, T. A. (2008).** *Prediction market accuracy in the long run.* International Journal of Forecasting, 24(2), 285–300. — Долгосрочная точность и калибровка лесенок на разных горизонтах.
24. **Gneiting, T., & Raftery, A. E. (2007).** *Strictly proper scoring rules, prediction, and estimation.* JASA, 102(477), 359–378. — Теория строго собственных правил оценки (CRPS, log-score, Brier, Dawid-Sebastiani) для вероятностных прогнозов страйков.
25. **Gneiting, T., Ranjan, R., & Weinberg, J. (2013).** *Combining predictive distributions.* Operations Research, 61(4), 939–953. — Линейный и логарифмический пулинг плотностей (`HAR-X` + `Deribit` + `Kalshi` на уровне распределений, а не только точечных прогнозов).

---

## 3. Криптовалютная микроструктура, фандинг и бессрочные фьючерсы

26. **Ackerer, D., Hugonnier, J., & Jermann, U. (2024).** *Perpetual futures pricing.* Mathematical Finance (preprint / SSRN). — Теория ценообразования бессрочных фьючерсов и связи ставки фандинга со спот-премией и волатильностью.
27. **He, S., Manela, A., Ross, O., & von Wachter, V. (2022)** — *уже использована*; дополнение: **Shiller, R. J. (1993)** / **Ang, A., & Chen, J. (2002)** и **Makarov, I., & Schoar, A. (2020).** *Trading and arbitrage in cryptocurrency markets.* Journal of Financial Economics, 135(2), 293–319. — Межбиржевые спреды, сегментация ликвидности и арбитражные границы на крипторынке.
28. **Easley, D., López de Prado, M. M., & O'Hara, M. (2012).** *Flow toxicity and liquidity in a high-frequency world.* Review of Financial Studies, 25(5), 1457–1493. — VPIN (Volume-Synchronized Probability of Informed Trading) на минутных/тиковых данных Binance как предвестник скачков волатильности.
29. **Cartea, Á., Jaimungal, S., & Penalva, J. (2015).** *Algorithmic and High-Frequency Trading.* Cambridge University Press. — Оптимальное исполнение и маркет-мейкинг с учётом неблагоприятного отбора (для `lab/maker.py` и `trader/`).
30. **Guéant, O. (2016).** *The Financial Mathematics of Market Liquidity: From Optimal Execution to Market Making.* Chapman & Hall/CRC. — Управление инвентарём и квотирование на бинарных/ограниченных рынках.
31. **Tóth, B., Lemperiere, Y., Deremble, C., De Lataillade, J., Kockelkoren, J., & Bouchaud, J.-P. (2011).** *Anomalous price impact and the critical nature of liquidity in financial markets.* Physical Review X, 1(2), 021006. — Закон квадратного корня для рыночного воздействия при оценке проскальзывания.

---

## 4. Скачки, режимы, хвостовые риски и статистические тесты

32. **Lee, S. S., & Mykland, P. A. (2008).** *Jumps in financial markets: A new nonparametric test and jump dynamics.* Review of Financial Studies, 21(6), 2535–2563. — Тест скачков на каждой минуте с учётом внутридневной периодичности (особенно важно для BTC).
33. **Aït-Sahalia, Y., & Jacod, J. (2009).** *Testing for jumps in a discretely observed process.* Annals of Statistics, 37(1), 184–222. — Тест на наличие скачков и разделение бесконечной/конечной активности.
34. **Hamilton, J. D. (1989).** *A new approach to the economic analysis of nonstationary time series and the business cycle.* Econometrica, 57(2), 357–384. — Марковские переключения режимов (высокая/низкая волатильность) для фильтрации крыльев `W03–W06`.
35. **White, H. (2000).** *A reality check for data snooping.* Econometrica, 68(5), 1097–1126. — Reality Check (дополнение к SPA Hansen 2005) при массовом переборе стратегий и 100+ моделей.
36. **Romano, J. P., & Wolf, M. (2005).** *Stepwise multiple testing as formalized data snooping.* Econometrica, 73(4), 1237–1247. — Пошаговая процедура Romano–Wolf (мощнее поправки Холма при коррелированных тестах).
37. **Harvey, C. R., Liu, Y., & Zhu, H. (2016).** *… and the cross-section of expected returns.* Review of Financial Studies, 29(1), 5–68. — Порог $|t| > 3.0$ и оценка доли ложных открытий (FDR/BHY) при поиске альфы.
38. **Bailey, D. H., Borwein, J. M., López de Prado, M., & Zhu, Q. J. (2014).** *Pseudo-mathematics and financial charlatanism: The effects of backtest overfitting on out-of-sample performance.* Notices of the AMS, 61(5), 458–471. — Probability of Backtest Overfitting (PBO) и Deflated Sharpe Ratio (DSR) для аудита 51 гипотезы в `lab/`.
39. **López de Prado, M. (2018).** *Advances in Financial Machine Learning.* Wiley. — Purged K-Fold CV и embargoing (использовано в `test_100_ideas.py`, можно перенести в `src/models.py` для GBM).
40. **McNeil, A. J., Frey, R., & Embrechts, P. (2015).** *Quantitative Risk Management: Concepts, Techniques and Tools* (2nd ed.). Princeton University Press. — Теория экстремальных значений (EVT / Peaks-Over-Threshold) для оценки риска хвостовых убытков при продаже крыльев `W03–W06`.

---

## Приоритетный план внедрения (топ-5)

1. **Baker–Kapadia–Madan (2003) MFSKEW / MFKURT** из лесенки Kalshi → новые признаки для `HAR-X` и бустинга.
2. **Gatheral et al. (2018) Rough Volatility** → дробные лаги на минутных данных BTC.
3. **Lee & Mykland (2008)** с десезонализацией → чистая идентификация скачков на 1-минутных данных вместо BV/MedRV.
4. **Bailey et al. (2014) DSR / PBO** → аудит всех 51 гипотезы `lab/signals.py`.
5. **EVT / Peaks-Over-Threshold (McNeil et al. 2015)** → контроль хвостового риска для стратегии продажи крыльев `W03–W06`.
