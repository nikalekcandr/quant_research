# Selling cheap wings on the Kalshi hourly ladder (`studies/wings_ladder.py`, 29.10.2026)

Data: 1-minute book candles of the hourly ladder, 2 782 hours (01.07–28.10.2026), 33 740 strikes, 856 297 minutes in
which a strike's cheap side bid 0.5–20¢. Sell the cheap side at its bid (taker), hold to settlement, Kalshi fee.
Clusters = hours. Grid of 435 rules (band × minutes left × model filter × strikes per hour), chosen on Jul–Aug,
read on Sep–Oct.

* **W01-like** (2–6¢, ≥ 15 min, no filter, every strike): 22 908 trades, −0.94¢ per trade, t −4.4 (hours), hit 96%.
  Selling every cheap wing loses: the rare loss (−94…−98¢) outweighs 25 wins.
* **The model filter matters.** The best region: cheap side 10–20¢, 5–15 minutes left, sell only if the bid exceeds
  our fair value + fee, one strike per hour:

| Rule | Jul–Aug (train) | Sep–Oct (test) |
|---|---|---|
| 10–20¢, 5–15 min, edge ≥ 1¢, 1/hour (**W03**) | 218 h, +2.2¢, t 1.0 | 427 h, **+3.7¢, t 2.5**, hit 90% |
| 10–20¢, 5–15 min, edge ≥ 0, 1/hour | 268 h, +3.0¢, t 1.6 | 516 h, +3.6¢, t 2.7 |
| 10–20¢, 5–15 min, edge ≥ 2¢, 1/hour | 171 h, +4.9¢, t 2.2 | 347 h, +2.6¢, t 1.5 |
| 6–10¢, 15–30 min, t4 fair, 1/hour (**W04**) | 685 h, +1.2¢, t 1.3 | 980 h, **+1.8¢, t 2.5**, hit 95% |
| same rules, all strikes per hour | | ≈ 0 (correlated strikes) |

* Reality check against data mining: 2.3% of all 435 rules have t > 2 on the test half, 30% of the ten best train
  rules do; mean test result of all rules −0.24¢, of the top ten +1.6¢. The selection carries information, but one
  rule at t 2.5 out of 435 would not survive Holm. W03 and W04 are frozen for a live test (one trade per hour,
  ≈ 10–20 trades a day): a verdict needs about a month.
