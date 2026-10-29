# Funding extremes → forward BTC drift (contrarian)

_Generated 2026-10-29 13:02 in 0.0 min._
Ticks: 734 rows, 2026-10-28 → 2026-10-29; price points 734.
Threshold: |funding| > 90% rolling quantile; horizons 1h/4h/8h; HAC t clustered by day.
Interpretation: contrarian = -sign(funding) * fwd return. If funding squeezes are contrarian-profitable, contrarian mean > 0 with t > 2.

## How to use

- If contrarian 8h mean ≈ +15–30 bps net of 10 bps fees (entry+exit) and hit > 52% → add D04 variant sized by funding z-score.
- Else: keep F01 (inter-venue funding spread, R11) as the only funding trade; D04 fade is noise.