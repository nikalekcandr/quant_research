import unittest
import numpy as np
import pandas as pd

from studies.wings_regime_ml import features,minute_data


class RegimeMLTests(unittest.TestCase):
    def test_future_prices_do_not_change_a_wing_prediction_input(self):
        dates = pd.date_range("2026-07-01",periods=12*1440,freq="min",tz="UTC")
        rng = np.random.default_rng(42)
        close = pd.Series(60000*np.exp(np.cumsum(rng.normal(0,.0003,len(dates)))),index=dates)
        now = dates[-1000]
        quotes = pd.DataFrame({"ts":[now],"end":[now+pd.Timedelta(minutes=20)],"bid":[.08],"ask":[.09],
                               "t4":[.06],"cheap_yes":[True]})
        def row(prices):
            stats,hist = minute_data(prices)
            return features(quotes,stats,hist,pd.Series(False,index=stats.index)).iloc[0]
        original = row(close)
        close.loc[now:] *= 1.2  # candle opening at the decision time is still in progress
        changed = row(close)
        for col in ["direction","downside","ratio","fhs","fhs_all"]:
            self.assertTrue(np.isfinite(original[col]))
            self.assertAlmostEqual(original[col],changed[col])
