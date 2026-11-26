import unittest

import pandas as pd

from studies.wing_consensus import forecasts, robustness
from studies.wing_capacity_ml import replay


class ConsensusTest(unittest.TestCase):
    def test_repricing_respects_original_limit_and_rerounds_fees(self):
        ts = pd.Timestamp("2026-11-23T12:00Z")
        row = dict(ts=ts, end=ts+pd.Timedelta(minutes=20), later_ts=ts+pd.Timedelta(seconds=15),
                   market="a", cheap_yes=True, t4=.04, y=0., bid=.10, ask=.11, bid_size=10,
                   ask_size=10, yes_depth=None, later_yes_depth='[[0.10,5],[0.11,5]]')
        trades = pd.DataFrame(replay(pd.DataFrame([row]), "t4", "test", stress=True))
        self.assertEqual(trades.delayed_n.item(), 10)
        self.assertEqual(trades.stress_n.item(), 5)
        self.assertAlmostEqual(trades.stress_edge.item(), .26)
        summary = robustness(trades, "test", "t4", [row["end"]])
        self.assertEqual(summary[1]["Нетто без лучшей прибыли, $"], 0)

    def test_prior_quote_must_be_fresh_and_from_the_same_wing(self):
        q = pd.DataFrame({"market":["a"] * 5, "cheap_yes":[True,True,True,False,False],
                          "t4":[.12,.06,.08,.08,.10], "curve":[.08,.04,.08,.88,.90],
                          "ts":pd.to_datetime([0,90,181,182,185], unit="s", utc=True)})
        out = forecasts(q.iloc[::-1])
        self.assertAlmostEqual(out.loc[1,"persistent"], .10)
        self.assertTrue(out.loc[[0,2,3],"persistent"].isna().all())
        self.assertAlmostEqual(out.loc[4,"persistent"], .10)
        self.assertAlmostEqual(out.loc[3,"conservative"], .11)
