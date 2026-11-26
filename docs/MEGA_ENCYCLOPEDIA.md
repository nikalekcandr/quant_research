# Мега-энциклопедия литературы проекта

_Сгенерировано 2026-11-26 23:10 UTC из `data/lit/bibliography.csv` (`src/lit/build_encyclopedia.py`). Руками этот файл не правится: правится CSV._

## Что это и чего здесь нет

В хранилище **201** строк в 13 доменах. Из них 0 с DOI, 198 с высокой уверенностью в выходных данных. Это **курируемый корпус**, а не выгрузка базы: каждая строка прочитана и связана с конкретным местом проекта (колонка `why`).

Здесь **намеренно нет** ссылок, которых я не могу подтвердить: в песочнице нет доступа к arXiv и издателям, поэтому DOI и страницы не придумываются. Строки без DOI помечены `confidence=medium` или `unverified` — их надо открыть и сверить. Массовое расширение до тысяч работ делается импортом настоящих метаданных (`src/lit/import_openalex.py`), а не генерацией текста.

## Как читать

`status` — это стадия работы с источником, а не качество статьи:

| status | значение |
|---|---|
| `anchor_used` | использовано в коде проекта |
| `anchor_tested` | проверено и описано в `results/studies` |
| `needs_data` | нужны данные, которых пока нет |
| `candidate` | кандидат: не пробовали |
| `imported` | импортировано автоматически, не читано |
| `rejected` | проверено и отклонено |

## Счёт по доменам

| домен | что | зачем проекту | работ | уже в коде |
|---|---|---|---|---|
| vol-forecasting | Прогноз волатильности | дисперсия до расчёта — единственный вход модельной вероятности | 25 | 10 |
| hf-econometrics | Высокочастотная эконометрика | что вообще можно оценить по минутным данным | 20 | 7 |
| jumps-tails | Скачки и хвосты | крыло продаёт хвост, значит хвост надо моделировать отдельно | 11 | 3 |
| prediction-markets | Рынки предсказаний | почему бинарная цена систематически смещена | 11 | 5 |
| probability-scoring | Оценка вероятностных прогнозов | Brier, лог-лосс, калибровка, пулинг | 12 | 5 |
| backtest-overfitting | Переподгонка и множественные тесты | сколько испытаний выдержала гипотеза | 14 | 10 |
| ml-methods | Методы машинного обучения | что реально работает на таблицах и сколько данных нужно | 34 | 9 |
| asset-pricing | Эмпирические факторы | откуда берутся премии и как их не спутать с шумом | 18 | 3 |
| options-iv | Опционы и IV | улыбка, RND, безмодельные моменты | 16 | 6 |
| crypto | Крипторынок | перпы, фандинг, базис, сегментация площадок | 12 | 0 |
| portfolio-risk | Портфель и риск | размер позиции, Келли, просадка | 7 | 3 |
| econometrics-tools | Эконометрический инструментарий | тесты, бутстрэп, HAC | 15 | 8 |
| infra-repro | Воспроизводимость | чтобы через полгода всё считалось тем же | 6 | 1 |

По статусам: `candidate` 130, `anchor_used` 70, `needs_data` 1.

## Оглавление

1. [Прогноз волатильности](#1-vol-forecasting) — 25 работ
2. [Высокочастотная эконометрика](#2-hf-econometrics) — 20 работ
3. [Скачки и хвосты](#3-jumps-tails) — 11 работ
4. [Рынки предсказаний](#4-prediction-markets) — 11 работ
5. [Оценка вероятностных прогнозов](#5-probability-scoring) — 12 работ
6. [Переподгонка и множественные тесты](#6-backtest-overfitting) — 14 работ
7. [Методы машинного обучения](#7-ml-methods) — 34 работ
8. [Эмпирические факторы](#8-asset-pricing) — 18 работ
9. [Опционы и IV](#9-options-iv) — 16 работ
10. [Крипторынок](#10-crypto) — 12 работ
11. [Портфель и риск](#11-portfolio-risk) — 7 работ
12. [Эконометрический инструментарий](#12-econometrics-tools) — 15 работ
13. [Воспроизводимость](#13-infra-repro) — 6 работ

---

## 1. Прогноз волатильности

_vol-forecasting: дисперсия до расчёта — единственный вход модельной вероятности._

| id | год | авторы | работа | где | зачем проекту | status | уверенность |
|---|---|---|---|---|---|---|---|
| L177 | 2023 | Christensen, K., Siggaard, M., Veliyev, B. | A machine learning approach to volatility forecasting | J. Financial Econometrics | Систематический ML-бенчмарк против HAR | candidate | high |
| L178 | 2022 | Bennedsen, M., Lunde, A., Pakkanen, M. S. | Decoupling the short- and long-term behavior of stochastic volatility | J. Financial Econometrics | Двухфакторная шероховатость+память | candidate | high |
| L179 | 2020 | Bucci, A. | Realized volatility forecasting with neural networks | J. Financial Econometrics | LSTM/GRU против HAR на лог-RV | candidate | high |
| L180 | 2019 | Livieri, G., Masi, S., Rabbachin, S., Toscano, A. | Rough volatility: a reality check | Quantitative Finance | Скепсис к rough vol; проверка H на реальных данных | candidate | high |
| L181 | 2018 | Gatheral, J., Jaisson, T., Rosenbaum, M. | Volatility is rough | Quantitative Finance | Rough vol, H~0.1; дробные лаги вместо HAR | candidate | high |
| L182 | 2016 | Audrino, F., Knaus, S. D. | Lassoing the HAR model: a model selection perspective on realized volatility dynamics | Econometric Reviews | Разреженный отбор лагов | candidate | high |
| L183 | 2015 | Patton, A. J., Sheppard, K. | Good volatility, bad volatility: signed jumps and the persistence of volatility | Review of Economics and Statistics | Разделение знака скачков; HAR-SJ | anchor_used | high |
| L184 | 2014 | Taylor, S. J. | Modelling financial time series | Wiley (book) | Классика ARCH/стохастической волатильности | anchor_used | high |
| L185 | 2013 | Corsi, F., Renò, R. | HAR volatility modelling with heterogeneous leverage and jumps | Advances in Econometrics | Расширение LHAR | candidate | high |
| L186 | 2012 | Hansen, P. R., Huang, Z., Shek, H. H. | Realized GARCH: a joint model of returns and realized measures of volatility | J. Applied Econometrics | Уравнение измерения RV; апгрейд GARCH | needs_data | high |
| L187 | 2012 | Corsi, F., Reno, R. | Discrete-time volatility forecasting with persistent leverage effect and the link with continuous-time volatility | J. Financial Econometrics | LHAR-CJ: рычаг и скачки по горизонтам | candidate | high |
| L188 | 2012 | Bauwens, L., Hautsch, N. | Modelling financial high frequency data using point processes | Handbook of Financial Time Series | ACD/точечные процессы для приходов сделок | candidate | high |
| L189 | 2011 | Patton, A. J. | Volatility forecast comparison using imperfect volatility proxies | J. Econometrics | QLIKE и робастные функции потерь | anchor_used | high |
| L190 | 2011 | Bollerslev, T., Patton, A. J., Quaedvlieg, R. | Exploiting the errors: a simple approach for improved volatility forecasting | J. Econometrics | HARQ: поправка на ошибку прошлого прогноза | anchor_used | high |
| L191 | 2010 | Shephard, N., Sheppard, K. | Realising the future: forecasting with high-frequency-based volatility (HEAVY) models | J. Applied Econometrics | HEAVY: быстрая адаптация к минутным скачкам | candidate | high |
| L192 | 2010 | Hillebrand, E., Medeiros, M. C. | The benefits of bagging for forecast models of realized volatility | Econometric Reviews | Бэггинг поверх HAR | candidate | high |
| L193 | 2009 | Corsi, F. | A simple approximate long-memory model of realized volatility | J. Financial Econometrics | HAR: базовая модель дисперсии, уже в проекте | anchor_used | high |
| L194 | 2009 | Jacod, J., Li, Y., Mykland, P. A., Podolskij, M., Vetter, M. | Microstructure noise in the continuous case: the pre-averaging approach | Stochastic Processes and their Applications | Pre-averaged RV для минутных данных | candidate | high |
| L195 | 2008 | Andersen, T. G., Bollerslev, T., Diebold, F. X., Labys, P. | Modeling and forecasting realized volatility | Econometrica | RV как объект прогноза; лог-нормальность | anchor_used | high |
| L196 | 2008 | Barndorff-Nielsen, O. E., Hansen, P. R., Lunde, A., Shephard, N. | Designing realized kernels to measure the ex post variation of equity prices in the presence of noise | Econometrica | Realized kernel против шума | candidate | high |
| L197 | 2006 | Engle, R. F., Gallo, G. M. | A multiple indicators model for volatility using intra-daily data | J. Econometrics | MEM для положительных величин | candidate | high |
| L198 | 2003 | Barndorff-Nielsen, O. E., Shephard, N. | Realized power variation and stochastic volatility models | Bernoulli | BV/MedRV, устойчивость к скачкам | anchor_used | high |
| L199 | 2001 | Andersen, T. G., Bollerslev, T. | Intraday periodicity and volatility persistence in financial markets | J. Empirical Finance | Внутридневной профиль дисперсии — наш profile() | anchor_used | high |
| L200 | 1986 | Engle, R. F. | Autoregressive conditional heteroscedasticity with estimates of the variance of United Kingdom inflation | Econometrica | ARCH; исток всей ветви | anchor_used | high |
| L201 | 1986 | Bollerslev, T. | Generalized autoregressive conditional heteroskedasticity | J. Econometrics | GARCH | anchor_used | high |

## 2. Высокочастотная эконометрика

_hf-econometrics: что вообще можно оценить по минутным данным._

| id | год | авторы | работа | где | зачем проекту | status | уверенность |
|---|---|---|---|---|---|---|---|
| L060 | 2019 | Eisler, Z., Bouchaud, J.-P., Kockelkoren, J. | The impact of order book events: market orders, limit orders and cancellations | Quantitative Finance | Разложение импакта по типам событий | candidate | high |
| L061 | 2018 | Cont, R., Cucuringu, M., Zhang, C. | Cross-impact of order flow imbalance in equity markets | Quantitative Finance | Кросс-импакт между соседними страйками | candidate | high |
| L062 | 2016 | Gueant, O. | The Financial Mathematics of Market Liquidity | CRC Press | Инвентарь и квотирование: продолжение Avellaneda-Stoikov | candidate | high |
| L063 | 2015 | Bouchaud, J.-P., Bonart, J., Donier, J., Gould, M. | Trades, Quotes and Prices | Cambridge University Press | Полный учебник микроструктуры | candidate | high |
| L064 | 2014 | Cont, R., Kukanov, A., Stoikov, S. | The price impact of order book events | J. Financial Econometrics | Импакт OFI; поток стакана | anchor_used | high |
| L065 | 2013 | Cartea, A., Jaimungal, S., Penalva, J. | Algorithmic and High-Frequency Trading | Cambridge University Press | Исполнение и мейкинг с adverse selection | anchor_used | high |
| L066 | 2012 | Easley, D., Lopez de Prado, M. M., O'Hara, M. | Flow toxicity and liquidity in a high-frequency world | Review of Financial Studies | VPIN как предиктор скачков | candidate | high |
| L067 | 2012 | Ait-Sahalia, Y., Jacod, J. | Analyzing the impact of noise on integrated volatility | J. Financial Econometrics | Шум и его влияние на оценки | candidate | high |
| L068 | 2011 | Toth, B., Lemperiere, Y., Deremble, C., de Lataillade, J., Kockelkoren, J., Bouchaud, J.-P. | Anomalous price impact and the critical nature of liquidity in financial markets | Physical Review X | Квадратный корень импакта; ёмкость | candidate | high |
| L069 | 2008 | Avellaneda, M., Stoikov, S. | High-frequency trading in a limit order book | Quantitative Finance | Квотирование вокруг резервации | anchor_used | high |
| L070 | 2007 | Bouchaud, J.-P., Farmer, J. D., Lillo, F. | How markets slowly digest changes in supply and demand | Handbook of Financial Markets | Кроссовер импакта; метаордера | candidate | high |
| L071 | 2005 | Zhang, L., Mykland, P. A., Ait-Sahalia, Y. | A tale of two time scales: determining integrated volatility with noisy high-frequency data | JASA | TSRV: две шкалы против шума | candidate | high |
| L072 | 2005 | Ait-Sahalia, Y., Mykland, P. A., Zhang, L. | How often to sample a continuous-time process in the presence of market microstructure noise | Review of Financial Studies | Оптимальная частота сэмплирования | anchor_used | high |
| L073 | 2002 | Amihud, Y. | Illiquidity and stock returns: cross-section and time-series effects | J. Financial Markets | Amihud ILLIQ: /r//объём | candidate | high |
| L074 | 2000 | Hasbrouck, J. | Trade and quote relationships: an econometric analysis | J. Finance | VAR/информационная доля сделок | anchor_used | high |
| L075 | 2000 | Almgren, R., Chriss, N. | Optimal execution of portfolio transactions | J. Risk | Оптимальное исполнение; проскальзывание | candidate | high |
| L076 | 1997 | Easley, D., Kiefer, N. M., O'Hara, M., Paperman, J. B. | The information content of the trading process | J. Empirical Finance | PIN: вероятность информированной торговли | candidate | high |
| L077 | 1995 | Kyle, A. S. | Continuous auctions and insider trading | Econometrica | Лямбда Кайла: импакт на единицу потока | anchor_used | high |
| L078 | 1991 | Roll, R. | A simple implicit measure of the effective bid-ask spread in an efficient market | J. Finance | Спред из автоковариации доходностей | candidate | high |
| L079 | 1985 | Glosten, L. R., Milgrom, P. R. | Bid, ask and transaction prices in a specialist market with heterogeneously informed traders | J. Financial Economics | Adverse selection как источник спреда | anchor_used | high |

## 3. Скачки и хвосты

_jumps-tails: крыло продаёт хвост, значит хвост надо моделировать отдельно._

| id | год | авторы | работа | где | зачем проекту | status | уверенность |
|---|---|---|---|---|---|---|---|
| L086 | 2016 | Bollerslev, T., Todorov, V. | Estimation of jump tails | Econometrica | Оценка параметров хвоста по опционам и доходностям | candidate | high |
| L087 | 2015 | McNeil, A. J., Frey, R., Embrechts, P. | Quantitative Risk Management (2nd ed.) | Princeton University Press | EVT/POT для хвоста крыла | candidate | high |
| L088 | 2012 | Kelly, B., Pastor, L. | Tail risk and asset prices | Review of Financial Studies | Хвостовой риск как фактор | candidate | high |
| L089 | 2011 | Creal, D., Koopman, S. J., Lucas, A. | A dynamic multivariate heavy-tailed model for time-varying volatilities and correlations | J. Business & Economic Statistics | GAS с t-хвостами | candidate | high |
| L090 | 2011 | Andersen, T. G., Bollerslev, T., Diebold, F. X. | Roughing it up: including jump components in the measurement, modeling and forecasting of return volatility | Review of Economics and Statistics | HAR-CJ | anchor_used | high |
| L091 | 2009 | Ait-Sahalia, Y., Jacod, J. | Testing for jumps in a discretely observed process | Annals of Statistics | Разделение скачков и диффузии | candidate | high |
| L092 | 2008 | Lee, S. S., Mykland, P. A. | Jumps in financial markets: a new nonparametric test and jump dynamics | Review of Financial Studies | Тест скачков поминутно с десезонализацией | candidate | high |
| L093 | 2007 | Barndorff-Nielsen, O. E., Shephard, N. | Impact of jumps on returns and realised variances | Econometrica | Bipower variation: разделение непрерывного и скачков | anchor_used | high |
| L094 | 2004 | Carr, P., Wu, L. | Time-changed Levy processes and option pricing | J. Financial Economics | Скачки в риск-нейтральной мере | anchor_used | high |
| L095 | 1989 | Hamilton, J. D. | A new approach to the economic analysis of nonstationary time series and the business cycle | Econometrica | Марковские режимы | candidate | high |
| L096 | 1975 | Pickands, J. | Statistical inference using extreme order statistics | Annals of Statistics | Оценка хвостового индекса | candidate | high |

## 4. Рынки предсказаний

_prediction-markets: почему бинарная цена систематически смещена._

| id | год | авторы | работа | где | зачем проекту | status | уверенность |
|---|---|---|---|---|---|---|---|
| L154 | 2020 | He, S., Manela, A., Ross, O., von Wachter, V. | Fundamentals of perpetual futures | Working paper | Фандинг и базис как цена срочности | anchor_used | high |
| L155 | 2017 | Rothschild, D. | Forecasting elections: comparing prediction markets, polls, and their biases | Public Opinion Quarterly | Сравнение с опросами; смещения | candidate | high |
| L156 | 2014 | Thaler, R. H., Ziemba, W. T. | Anomalies: parimutuel betting markets: racetracks and lotteries | J. Economic Perspectives | Longshot bias в ставках — прямой аналог | anchor_used | high |
| L157 | 2013 | Arrow, K. J. | The role of securities in the optimal allocation of risk-bearing | Review of Economic Studies | Стрелки-Дебрё: бинарный контракт = элементарная ценная бумага | anchor_used | high |
| L158 | 2012 | Page, L., Clemen, R. T. | Do prediction markets produce well-calibrated probability forecasts? | Economic Journal | Калибровка на длинной истории | candidate | high |
| L159 | 2008 | Berg, J. E., Nelson, F. D., Rietz, T. A. | Prediction market accuracy in the long run | Int. J. Forecasting | Долгосрочная калибровка | candidate | high |
| L160 | 2008 | Snowberg, E., Wolfers, J. | Examining preferences over ambiguity and risk with prediction markets | NBER working paper | Отношение к риску из цен контрактов | candidate | high |
| L161 | 2008 | Hanson, R. | Logarithmic market scoring rules for modular combinatorial information aggregation | J. Prediction Markets | LMSR: механика цены и связь с лог-скором | candidate | high |
| L162 | 2006 | Manski, C. F. | Interpreting the predictions of prediction markets | Economics Letters | Цена != средняя вера при ограниченном бюджете | anchor_used | high |
| L163 | 2005 | Gjerstad, S. | Risk aversion, beliefs, and prediction market equilibrium | Economic Theory | Равновесие с CRRA: систематический сдвиг цены | candidate | high |
| L164 | 2004 | Wolfers, J., Zitzewitz, E. | Prediction markets | J. Economic Perspectives | Обзор; favourite-longshot bias — почему крылья дороги | anchor_used | high |

## 5. Оценка вероятностных прогнозов

_probability-scoring: Brier, лог-лосс, калибровка, пулинг._

| id | год | авторы | работа | где | зачем проекту | status | уверенность |
|---|---|---|---|---|---|---|---|
| L165 | 2019 | Lakshminarayanan, B., Pritzel, A., Blundell, C. | Simple and scalable predictive uncertainty estimation using deep ensembles | NeurIPS | Дип-ансамбли как практическая неопределённость | candidate | high |
| L166 | 2017 | Kuleshov, V., Fenner, N., Ermon, S. | Accurate uncertainties for deep learning using calibrated regression | ICML | Калибровка регрессионных интервалов | candidate | high |
| L167 | 2016 | Vovk, V., Gammerman, A., Shafer, G. | Algorithmic Learning in a Random World | Springer | Конформные интервалы предсказаний | candidate | high |
| L168 | 2015 | Guo, C., Pleiss, G., Sun, Y., Weinberger, K. Q. | On calibration of modern neural networks | ICML | Современные сети недокалиброваны; temperature scaling | candidate | high |
| L169 | 2013 | Gneiting, T., Ranjan, R., Weinberg, J. | Combining predictive distributions | Operations Research | Пулинг плотностей вместо точечных прогнозов | candidate | high |
| L170 | 2007 | Gneiting, T., Raftery, A. E. | Strictly proper scoring rules, prediction, and estimation | JASA | Теория собственных правил; CRPS | anchor_used | high |
| L171 | 2007 | Gneiting, T., Balabdaoui, F., Raftery, A. E. | Probabilistic forecasts, calibration and sharpness | J.R. Stat. Soc. B | Калибровка и резкость как два качества | anchor_used | high |
| L172 | 2002 | Zadrozny, B., Elkan, C. | Transforming classifier scores into accurate multiclass probability estimates | KDD | Изотоническая регрессия | candidate | high |
| L173 | 1999 | Platt, J. C. | Probabilistic outputs for support vector machines | Advances in Large Margin Classifiers | Platt scaling | candidate | high |
| L174 | 1973 | Murphy, A. H. | A new vector partition of the probability score | J. Applied Meteorology | Разложение Brier на надёжность/разрешение | anchor_used | high |
| L175 | 1973 | Dawid, A. P. | The well-calibrated Bayesian | JASA | Понятие калибровки | anchor_used | high |
| L176 | 1950 | Brier, G. W. | Verification of forecasts expressed in terms of probability | Monthly Weather Review | Brier score | anchor_used | high |

## 6. Переподгонка и множественные тесты

_backtest-overfitting: сколько испытаний выдержала гипотеза._

| id | год | авторы | работа | где | зачем проекту | status | уверенность |
|---|---|---|---|---|---|---|---|
| L019 | 2020 | Grinold, R. C., Kahn, R. N. | Advances in Active Portfolio Management | McGraw-Hill | Информационный коэффициент и breadth | anchor_used | high |
| L020 | 2019 | Harvey, C. R., Liu, Y. | False (and missed) discoveries in financial economics | J. Finance | Ошибки второго рода в финансах | candidate | high |
| L021 | 2019 | Fabozzi, F. J., Lopez de Prado, M. | Being honest in backtest reporting: a template for disclosing multiple tests | J. Portfolio Management | Как отчитываться о числе испытаний | candidate | high |
| L022 | 2018 | Lopez de Prado, M. | Advances in Financial Machine Learning | Wiley | Purged K-fold, embargo, метабэггинг | anchor_used | high |
| L023 | 2017 | Arnott, R., Harvey, C. R., Markowitz, H. | A backtesting protocol in the era of machine learning | J. Financial Data Science | Протокол бэктеста | candidate | high |
| L024 | 2016 | Harvey, C. R., Liu, Y., Zhu, H. | ... and the cross-section of expected returns | Review of Financial Studies | Порог /t/>3 и FDR при поиске альфы | anchor_used | high |
| L025 | 2014 | Bailey, D. H., Borwein, J. M., Lopez de Prado, M., Zhu, Q. J. | Pseudo-mathematics and financial charlatanism | Notices of the AMS | PBO и DSR | anchor_used | high |
| L026 | 2014 | Bailey, D. H., Lopez de Prado, M. | The deflated Sharpe ratio: correcting for selection bias, backtest overfitting and non-normality | J. Portfolio Management | Дефлированный Шарп, формула | anchor_used | high |
| L027 | 2011 | Bajgrowicz, P., Scaillet, O. | Technical trading revisited: false discoveries, persistence tests, and transaction costs | J. Financial Economics | FDR на технических правилах | candidate | high |
| L028 | 2005 | Hansen, P. R. | A test for superior predictive ability | J. Business & Economic Statistics | SPA-тест | anchor_used | high |
| L029 | 2005 | Romano, J. P., Wolf, M. | Stepwise multiple testing as formalized data snooping | Econometrica | Stepdown: мощнее Холма при корреляции | anchor_used | high |
| L030 | 2000 | White, H. | A reality check for data snooping | Econometrica | Reality Check для массового перебора | anchor_used | high |
| L031 | 1995 | Benjamini, Y., Hochberg, Y. | Controlling the false discovery rate | J.R. Stat. Soc. B | BHY-процедура | anchor_used | high |
| L032 | 1961 | Bonferroni, C. | Teoria statistica delle classi e calcolo delle probabilita | Pubblicazioni Istituto Superiore di Scienze Economiche | Поправка Бонферрони | anchor_used | high |

## 7. Методы машинного обучения

_ml-methods: что реально работает на таблицах и сколько данных нужно._

| id | год | авторы | работа | где | зачем проекту | status | уверенность |
|---|---|---|---|---|---|---|---|
| L097 | 2022 | Grinsztajn, L., Oyallon, E., Varoquaux, G. | Why do tree-based models still outperform deep learning on typical tabular data? | NeurIPS Datasets and Benchmarks | Табличные данные: деревья сильнее DL | candidate | high |
| L098 | 2022 | Shwartz-Ziv, R., Armon, A. | Tabular data: deep learning is not all you need | Information Fusion | То же, независимая проверка | candidate | high |
| L099 | 2022 | Hoffmann, J. et al. | Training compute-optimal large language models | NeurIPS | Chinchilla: 20 токенов на параметр | candidate | high |
| L100 | 2021 | Gu, S., Kelly, B., Xiu, D. | Autoencoder asset pricing models | J. Econometrics | Латентные факторы через автоэнкодеры | candidate | high |
| L101 | 2021 | Gorishniy, Y., Rubachev, I., Khrulkov, V., Babenko, A. | Revisiting deep learning models for tabular data | NeurIPS | FT-Transformer и честные бейзлайны | candidate | high |
| L102 | 2021 | Dosovitskiy, A. et al. | An image is worth 16x16 words: transformers for image recognition at scale | ICLR | ViT: масштаб важнее индуктивного смещения | candidate | high |
| L103 | 2021 | Bouthillier, X. et al. | Accounting for variance in machine learning benchmarks | MLSys | Дисперсия результатов: один сид — не результат | candidate | high |
| L104 | 2020 | Gu, S., Kelly, B., Xiu, D. | Empirical asset pricing via machine learning | Review of Financial Studies | Trees/NN против OLS на премиях | anchor_used | high |
| L105 | 2020 | Feng, G., Giglio, S., Xiu, D. | Taming the factor zoo: a test of new factors | J. Finance | Оценка вклада фактора при многих тестах | candidate | high |
| L106 | 2020 | Kaplan, J. et al. | Scaling laws for neural language models | arXiv preprint | Законы масштабирования: сколько данных нужно модели | candidate | high |
| L107 | 2019 | Kelly, B. T., Pruitt, S., Su, Y. | Characteristics are covariances: a unified model of risk and return | J. Financial Economics | IPCA | candidate | high |
| L108 | 2019 | Izmailov, P., Podoprikhin, D., Garipov, T., Vetrov, D., Wilson, A. G. | Averaging weights leads to wider optima and better generalization | UAI | SWA: усреднение весов | candidate | high |
| L109 | 2019 | Zhang, H., Cisse, M., Dauphin, Y. N., Lopez-Paz, D. | mixup: beyond empirical risk minimization | ICLR | Аугментация через смешивание | candidate | high |
| L110 | 2019 | Sutton, R. S. | The bitter lesson | Essay | Масштаб и общие методы бьют ручные эвристики | candidate | high |
| L111 | 2018 | Prokhorenkova, L., Gusev, G., Vorobev, A., Dorogush, A. V., Gulin, A. | CatBoost: unbiased boosting with categorical features | NeurIPS | Ordered boosting против утечки | candidate | high |
| L112 | 2018 | Vanschoren, J. | Meta-learning: a survey | arXiv preprint | Метаобучение и выбор модели | candidate | high |
| L113 | 2017 | Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q., Liu, T.-Y. | LightGBM: a highly efficient gradient boosting decision tree | NeurIPS | Гистограммный бустинг | anchor_used | high |
| L114 | 2017 | Huang, G., Li, Y., Pleiss, G., Liu, Z., Hopcroft, J. E., Weinberger, K. Q. | Snapshot ensembles: train 1, get M for free | ICLR | Ансамбль из одной тренировки | candidate | high |
| L115 | 2017 | Szegedy, C., Ioffe, S., Vanhoucke, V., Alemi, A. | Inception-v4, Inception-ResNet and the impact of residual connections on learning | AAAI | Остаточные связи | candidate | high |
| L116 | 2017 | Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, L., Polosukhin, I. | Attention is all you need | NeurIPS | Трансформер | candidate | high |
| L117 | 2017 | Gal, Y., Ghahramani, Z. | Dropout as a Bayesian approximation: representing model uncertainty in deep learning | ICML | MC-dropout как неопределённость | candidate | high |
| L118 | 2017 | Chollet, F. | Deep Learning with Python | Manning | Практика DL | candidate | medium |
| L119 | 2016 | Chen, T., Guestrin, C. | XGBoost: a scalable tree boosting system | KDD | Бустинг: стандарт для таблиц | anchor_used | high |
| L120 | 2016 | Blundell, C., Cornebise, J., Kavukcuoglu, K., Wierstra, D. | Weight uncertainty in neural networks | ICML | Bayes by Backprop | candidate | high |
| L121 | 2015 | Ioffe, S., Szegedy, C. | Batch normalization | ICML | Нормализация слоёв | anchor_used | high |
| L122 | 2015 | He, K., Zhang, X., Ren, S., Sun, J. | Deep residual learning for image recognition | CVPR | ResNet | candidate | high |
| L123 | 2015 | Snoek, J., Larochelle, H., Adams, R. P. | Practical Bayesian optimization of machine learning algorithms | NeurIPS | Байесовская оптимизация гиперпараметров | candidate | high |
| L124 | 2014 | Kingma, D. P., Ba, J. | Adam: a method for stochastic optimization | ICLR | Оптимизатор | anchor_used | high |
| L125 | 2014 | Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I., Salakhutdinov, R. | Dropout: a simple way to prevent neural networks from overfitting | JMLR | Регуляризация | anchor_used | high |
| L126 | 2013 | Bergstra, J., Bengio, Y. | Random search for hyper-parameter optimization | JMLR | Random search вместо сетки | candidate | high |
| L127 | 2012 | Hinton, G., Srivastava, N., Swersky, K. | Neural networks for machine learning (lectures) | Coursera | Основы | candidate | medium |
| L128 | 2011 | Wolpert, D. H. | The no free lunch theorems | J. Machine Learning Research | Нет универсально лучшего алгоритма | anchor_used | high |
| L129 | 2007 | Hastie, T., Tibshirani, R., Friedman, J. | The Elements of Statistical Learning | Springer | Регуляризация, смещение-разброс | anchor_used | high |
| L130 | 2001 | Breiman, L. | Random forests | Machine Learning | Бэггинг деревьев | anchor_used | high |

## 8. Эмпирические факторы

_asset-pricing: откуда берутся премии и как их не спутать с шумом._

| id | год | авторы | работа | где | зачем проекту | status | уверенность |
|---|---|---|---|---|---|---|---|
| L001 | 2020 | Bianchi, D., Buchner, M., Tamoni, A. | Bond risk premia with machine learning | Review of Financial Studies | ML на премиях | candidate | high |
| L002 | 2015 | Fama, E. F., French, K. R. | A five-factor asset pricing model | J. Financial Economics | Пятифакторная | candidate | high |
| L003 | 2015 | Barroso, P., Santa-Clara, P. | Momentum has its moments | J. Financial Economics | Волатильность-таргетинг моментума | candidate | high |
| L004 | 2014 | Garleanu, N., Pedersen, L. H. | Dynamic trading with predictable returns and transaction costs | J. Finance | Динамическая торговля с издержками | candidate | high |
| L005 | 2014 | Harvey, C. R., Liu, Y., Zhu, H. | Comment on "The cross-section of expected returns" | Working paper | Множественные тесты в факторах | candidate | high |
| L006 | 2014 | Kelly, G. W., Pruitt, S. | Market expectations in the cross-section of present values | J. Finance | Ожидания рынка как предиктор | candidate | high |
| L007 | 2013 | Koijen, R. S. J., Moskowitz, T. J., Pedersen, L. H., Vrugt, E. B. | Carry | J. Financial Economics | Кэрри как универсальный фактор | anchor_used | high |
| L008 | 2013 | Daniel, K., Moskowitz, T. J. | Momentum crashes | J. Financial Economics | Обвалы моментума в разворотах | candidate | high |
| L009 | 2013 | Frazzini, A., Pedersen, L. H. | Betting against beta | J. Financial Economics | BAB | candidate | high |
| L010 | 2012 | Ang, A. | Asset Management: A Systematic Approach to Factor Investing | Oxford University Press | Факторы, режимы | candidate | high |
| L011 | 2012 | Moskowitz, T. J., Ooi, Y. H., Pedersen, L. H. | Time series momentum | J. Financial Economics | TSMOM | candidate | high |
| L012 | 2012 | Asness, C. S., Moskowitz, T. J., Pedersen, L. H. | Value and momentum everywhere | J. Finance | Кросс-активные факторы | candidate | high |
| L013 | 2012 | Moreira, A., Muir, T. | Volatility-managed portfolios | J. Finance | Масштабирование по волатильности | candidate | high |
| L014 | 2011 | Cochrane, J. H. | Presidential address: discount rates | J. Finance | Дисконт-рейты и предсказуемость | candidate | high |
| L015 | 2008 | Avellaneda, M., Lee, J.-H. | Statistical arbitrage in the US equities market | Quantitative Finance | PCA-остатки; s-score | candidate | high |
| L016 | 2006 | Gatev, E., Goetzmann, W. N., Rouwenhorst, K. G. | Pairs trading: performance of a relative-value arbitrage rule | Review of Financial Studies | Пары на расстоянии | candidate | high |
| L017 | 1999 | Grinold, R. C., Kahn, R. N. | Active Portfolio Management | McGraw-Hill | IC, breadth, фундаментальный закон | anchor_used | high |
| L018 | 1993 | Fama, E. F., French, K. R. | Common risk factors in the returns on stocks and bonds | J. Financial Economics | Трёхфакторная модель | anchor_used | high |

## 9. Опционы и IV

_options-iv: улыбка, RND, безмодельные моменты._

| id | год | авторы | работа | где | зачем проекту | status | уверенность |
|---|---|---|---|---|---|---|---|
| L131 | 2017 | Martin, I. | What is the expected return on the market? | Quarterly J. Economics | SVIX: нижняя граница из лесенки | candidate | high |
| L132 | 2015 | Ross, S. A. | The recovery theorem | J. Finance | Восстановление физической меры | candidate | high |
| L133 | 2014 | Gatheral, J., Jacquier, A. | Arbitrage-free SVI volatility surfaces | Quantitative Finance | Без-арбитражная параметризация | candidate | high |
| L134 | 2013 | Andries, M., Eisenbach, T., Schmalz, M. C. | The term structure of the price of variance risk | J. Financial Economics | VRP по срокам | candidate | high |
| L135 | 2009 | Fengler, M. R. | Arbitrage-free smoothing of the implied volatility surface | Quantitative Finance | Сглаживание с ограничениями | candidate | high |
| L136 | 2009 | Carr, P., Wu, L. | Variance risk premiums | Review of Financial Studies | VRP как отдельная премия | anchor_used | high |
| L137 | 2005 | Gatheral, J. | The Volatility Surface | Wiley | SVI, калибровка улыбки | anchor_used | high |
| L138 | 2005 | Carr, P., Madan, D. | A note on sufficient conditions for no arbitrage | Finance Research Letters | Условия отсутствия арбитража в лесенке | anchor_used | high |
| L139 | 2004 | Jiang, G. J., Tian, Y. S. | The model-free implied volatility and its information content | Review of Financial Studies | MFIV | candidate | high |
| L140 | 2003 | Bakshi, G., Kapadia, N., Madan, D. | Stock return characteristics, skew laws, and the differential pricing of individual equity options | Review of Financial Studies | Безмодельные моменты | candidate | high |
| L141 | 2001 | Britten-Jones, M., Neuberger, A. | Option prices implied by the risk-neutral density | Review of Financial Studies | Связь цены и плотности | anchor_used | high |
| L142 | 1996 | Bates, D. S. | Jumps and stochastic volatility | Review of Financial Studies | SVJ | candidate | high |
| L143 | 1994 | Dupire, B. | Pricing with a smile | Risk | Локальная волатильность | candidate | high |
| L144 | 1993 | Heston, S. L. | A closed-form solution for options with stochastic volatility | Review of Financial Studies | SV с закрытой формой | candidate | high |
| L145 | 1978 | Breeden, D. T., Litzenberger, R. H. | Prices of state-contingent claims implicit in option prices | J. Business | RND = вторая производная по страйку | anchor_used | high |
| L146 | 1973 | Black, F., Scholes, M. | The pricing of options and corporate liabilities | J. Political Economy | База; бинарный контракт как предел | anchor_used | high |

## 10. Крипторынок

_crypto: перпы, фандинг, базис, сегментация площадок._

| id | год | авторы | работа | где | зачем проекту | status | уверенность |
|---|---|---|---|---|---|---|---|
| L033 | 2024 | Ackerer, D., Hugonnier, J., Jermann, U. | Perpetual futures pricing | Mathematical Finance | Теория перпов | candidate | high |
| L034 | 2022 | Lehar, A., Parlour, C. A. | Decentralized exchanges | Working paper | AMM и MEV | candidate | high |
| L035 | 2021 | Alexander, C., Deng, J., Zou, B. | Hedging with automatic liquidation and leverage selection on Bitcoin futures | European J. Operational Research | Ликвидации и плечо | candidate | high |
| L036 | 2021 | Liu, Y., Tsyvinski, A., Wu, X. | Common risk factors in cryptocurrency | J. Finance | Три фактора крипты | candidate | high |
| L037 | 2021 | Cong, L. W., Li, X., Tang, K., Yang, Y. | Crypto wash trading | Management Science | Фиктивный объём: осторожно с volume | candidate | high |
| L038 | 2020 | Makarov, I., Schoar, A. | Trading and arbitrage in cryptocurrency markets | J. Financial Economics | Сегментация и арбитражные границы | candidate | high |
| L039 | 2020 | Alexander, C., Heck, D. F. | Price discovery in Bitcoin: the impact of unregulated markets | J. Financial Stability | Ценовая находка между площадками | candidate | high |
| L040 | 2020 | Grobys, K., Sapkota, N. | Cryptocurrencies and momentum | Economics Letters | Моментум в крипте | candidate | high |
| L041 | 2019 | Brandvold, M., Molnar, P., Vagstad, K., Valstad, O. C. A. | Price discovery on Bitcoin exchanges | J. International Financial Markets | Ведущая биржа | candidate | high |
| L042 | 2019 | Liu, Y., Tsyvinski, A. | Risks and returns of cryptocurrency | Review of Financial Studies | Предсказуемость крипты | candidate | high |
| L043 | 2019 | Ardia, D., Bluteau, K., Ruede, M. | Regime changes in Bitcoin GARCH volatility dynamics | Finance Research Letters | Режимы волатильности BTC | candidate | high |
| L044 | 2018 | Corbet, S., Lucey, B., Urquhart, A., Yarovaya, L. | Cryptocurrencies as a financial asset: a systematic analysis | Int. Review of Financial Analysis | Обзор рынка | candidate | high |

## 11. Портфель и риск

_portfolio-risk: размер позиции, Келли, просадка._

| id | год | авторы | работа | где | зачем проекту | status | уверенность |
|---|---|---|---|---|---|---|---|
| L147 | 2016 | Bailey, D. H., Lopez de Prado, M. | The Sharpe ratio efficient frontier | J. Risk | PSR: значимость Шарпа | candidate | high |
| L148 | 2013 | Roncalli, T. | Introduction to Risk Parity and Budgeting | CRC Press | Практика риск-парити | candidate | high |
| L149 | 2006 | Thorp, E. O. | The Kelly criterion in blackjack, sports betting, and the stock market | Handbook of Asset and Liability Management | Дробный Келли; переоценка | anchor_used | high |
| L150 | 2005 | Maillard, S., Roncalli, T., Teiletche, J. | The properties of equally weighted risk contribution portfolios | J. Portfolio Management | Risk parity | candidate | high |
| L151 | 1992 | Rockafellar, R. T., Uryasev, S. | Optimization of conditional value-at-risk | J. Risk | CVaR | candidate | high |
| L152 | 1956 | Kelly, J. L. | A new interpretation of information rate | Bell System Technical Journal | Критерий Келли | anchor_used | high |
| L153 | 1952 | Markowitz, H. | Portfolio selection | J. Finance | Среднее-дисперсия | anchor_used | high |

## 12. Эконометрический инструментарий

_econometrics-tools: тесты, бутстрэп, HAC._

| id | год | авторы | работа | где | зачем проекту | status | уверенность |
|---|---|---|---|---|---|---|---|
| L045 | 2018 | Athey, S., Imbens, G. W. | Design-based analysis in Difference-in-Differences settings with staggered adoption | Working paper | Причинный вывод для A/B стратегий | candidate | medium |
| L046 | 2016 | Hastie, T., Tibshirani, R., Wainwright, M. | Statistical Learning with Sparsity | CRC Press | L1/L2, отбор признаков | anchor_used | high |
| L047 | 2013 | Efron, B., Hastie, T. | Computer Age Statistical Inference | Cambridge University Press | Бутстрэп и тесты в современном виде | candidate | high |
| L048 | 2011 | Clark, T. E., West, K. D. | Nested forecast model comparisons | J. Econometrics | Тест для вложенных моделей | anchor_used | high |
| L049 | 2005 | Hansen, P. R., Lunde, A., Nason, J. M. | The model confidence set | Econometrica | MCS: какие модели статистически неразличимы | anchor_used | high |
| L050 | 2004 | Ghysels, E., Santa-Clara, P., Valkanov, R. | The MIDAS touch: mixed data sampling regression models | Working paper | MIDAS: разночастотные регрессии | candidate | high |
| L051 | 2000 | Hamilton, J. D. | Time Series Analysis | Princeton University Press | Базовый учебник | anchor_used | high |
| L052 | 1998 | Bai, J., Perron, P. | Estimating and testing linear models with multiple structural changes | Econometrica | Множественные структурные сдвиги | candidate | high |
| L053 | 1995 | Diebold, F. X., Mariano, R. S. | Comparing predictive accuracy | J. Business & Economic Statistics | DM-тест | anchor_used | high |
| L054 | 1994 | Politis, D. N., Romano, J. P. | The stationary bootstrap | JASA | Блочный бутстрэп для зависимых рядов | anchor_used | high |
| L055 | 1993 | Kunsch, H. R. | The jackknife and the bootstrap for general stationary observations | Annals of Statistics | Блочный бутстрэп | anchor_used | high |
| L056 | 1987 | Newey, W. K., West, K. D. | A simple, positive semi-definite, heteroskedasticity and autocorrelation consistent covariance matrix | Econometrica | HAC-ошибки | anchor_used | high |
| L057 | 1979 | Dickey, D. A., Fuller, W. A. | Distribution of the estimators for autoregressive time series with a unit root | JASA | ADF | candidate | high |
| L058 | 1978 | Ljung, G. M., Box, G. E. P. | On a measure of lack of fit in time series models | Biometrika | Автокорреляция остатков | candidate | high |
| L059 | 1960 | Kalman, R. E. | A new approach to linear filtering and prediction problems | J. Basic Engineering | Фильтр Калмана | candidate | high |

## 13. Воспроизводимость

_infra-repro: чтобы через полгода всё считалось тем же._

| id | год | авторы | работа | где | зачем проекту | status | уверенность |
|---|---|---|---|---|---|---|---|
| L080 | 2019 | Bouthillier, X. et al. | Unreproducible research is reproducible | ICML | Как фиксировать сиды и окружение | candidate | high |
| L081 | 2018 | Sandve, G. K., Nekrutenko, A., Taylor, J., Hovig, E. | Ten simple rules for reproducible computational research | PLOS Computational Biology | Правила воспроизводимости | anchor_used | high |
| L082 | 2017 | Sculley, D. et al. | Hidden technical debt in machine learning systems | NeurIPS | Технический долг ML-систем | candidate | high |
| L083 | 2016 | Gentleman, R., Temple Lang, D. | Statistical analyses and reproducible research | J. Computational and Graphical Statistics | Воспроизводимость расчётов | candidate | high |
| L084 | 2015 | Breck, E., Cai, S., Nielsen, E., Salib, M., Sculley, D. | The ML test score: a rubric for ML production readiness | IEEE Big Data | Чек-лист готовности | candidate | high |
| L085 | 2013 | Peng, R. D. | Reproducible research in computational science | Science | Культура воспроизводимости | candidate | high |

---

## Что читать дальше

Отсортировано по близости к деньгам: сначала то, под что уже есть данные (`needs_data` и `candidate`), новые — выше.

| id | год | авторы | работа | status |
|---|---|---|---|---|
| L186 | 2012 | Hansen, P. R., Huang, Z., Shek, H. H. | Realized GARCH: a joint model of returns and realized measures of volatility | needs_data |
| L033 | 2024 | Ackerer, D., Hugonnier, J., Jermann, U. | Perpetual futures pricing | candidate |
| L177 | 2023 | Christensen, K., Siggaard, M., Veliyev, B. | A machine learning approach to volatility forecasting | candidate |
| L034 | 2022 | Lehar, A., Parlour, C. A. | Decentralized exchanges | candidate |
| L097 | 2022 | Grinsztajn, L., Oyallon, E., Varoquaux, G. | Why do tree-based models still outperform deep learning on typical tabular data? | candidate |
| L098 | 2022 | Shwartz-Ziv, R., Armon, A. | Tabular data: deep learning is not all you need | candidate |
| L099 | 2022 | Hoffmann, J. et al. | Training compute-optimal large language models | candidate |
| L178 | 2022 | Bennedsen, M., Lunde, A., Pakkanen, M. S. | Decoupling the short- and long-term behavior of stochastic volatility | candidate |
| L035 | 2021 | Alexander, C., Deng, J., Zou, B. | Hedging with automatic liquidation and leverage selection on Bitcoin futures | candidate |
| L036 | 2021 | Liu, Y., Tsyvinski, A., Wu, X. | Common risk factors in cryptocurrency | candidate |
| L037 | 2021 | Cong, L. W., Li, X., Tang, K., Yang, Y. | Crypto wash trading | candidate |
| L100 | 2021 | Gu, S., Kelly, B., Xiu, D. | Autoencoder asset pricing models | candidate |
| L101 | 2021 | Gorishniy, Y., Rubachev, I., Khrulkov, V., Babenko, A. | Revisiting deep learning models for tabular data | candidate |
| L102 | 2021 | Dosovitskiy, A. et al. | An image is worth 16x16 words: transformers for image recognition at scale | candidate |
| L103 | 2021 | Bouthillier, X. et al. | Accounting for variance in machine learning benchmarks | candidate |
| L001 | 2020 | Bianchi, D., Buchner, M., Tamoni, A. | Bond risk premia with machine learning | candidate |
| L038 | 2020 | Makarov, I., Schoar, A. | Trading and arbitrage in cryptocurrency markets | candidate |
| L039 | 2020 | Alexander, C., Heck, D. F. | Price discovery in Bitcoin: the impact of unregulated markets | candidate |
| L040 | 2020 | Grobys, K., Sapkota, N. | Cryptocurrencies and momentum | candidate |
| L105 | 2020 | Feng, G., Giglio, S., Xiu, D. | Taming the factor zoo: a test of new factors | candidate |
| L106 | 2020 | Kaplan, J. et al. | Scaling laws for neural language models | candidate |
| L179 | 2020 | Bucci, A. | Realized volatility forecasting with neural networks | candidate |
| L020 | 2019 | Harvey, C. R., Liu, Y. | False (and missed) discoveries in financial economics | candidate |
| L021 | 2019 | Fabozzi, F. J., Lopez de Prado, M. | Being honest in backtest reporting: a template for disclosing multiple tests | candidate |
| L041 | 2019 | Brandvold, M., Molnar, P., Vagstad, K., Valstad, O. C. A. | Price discovery on Bitcoin exchanges | candidate |
| L042 | 2019 | Liu, Y., Tsyvinski, A. | Risks and returns of cryptocurrency | candidate |
| L043 | 2019 | Ardia, D., Bluteau, K., Ruede, M. | Regime changes in Bitcoin GARCH volatility dynamics | candidate |
| L060 | 2019 | Eisler, Z., Bouchaud, J.-P., Kockelkoren, J. | The impact of order book events: market orders, limit orders and cancellations | candidate |
| L080 | 2019 | Bouthillier, X. et al. | Unreproducible research is reproducible | candidate |
| L107 | 2019 | Kelly, B. T., Pruitt, S., Su, Y. | Characteristics are covariances: a unified model of risk and return | candidate |

## Как расширить до тысяч работ

```bash
# 1. скачать метаданные настоящих работ (на вашей машине, где есть интернет)
curl -s 'https://api.openalex.org/works?search=realized%20volatility&per-page=200' -o data/lit/openalex_vol.json
curl -s 'https://api.openalex.org/works?search=prediction%20markets&per-page=200' -o data/lit/openalex_pm.json
# 2. импортировать: дедуп по названию и DOI, домен по ключевым словам, статус imported
python -m src.lit.import_openalex data/lit/openalex_vol.json data/lit/openalex_pm.json
# 3. пересобрать энциклопедию
python -m src.lit.build_encyclopedia
```

Импортированные строки попадают в конец (status `imported`, confidence `unverified`) и не считаются прочитанными, пока у них не появится `why` и `status=candidate`. Это разделение специально: иначе «энциклопедия на 5 000 источников» превращается в свалку, где никто не помнит, что из неё следует для стратегии.

---

_Счёт по десятилетиям: 1950s 3, 1960s 2, 1970s 7, 1980s 5, 1990s 15, 2000s 46, 2010s 99, 2020s 24._
