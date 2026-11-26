"""Regression checks for the research-only paired forecasts and the calibration study."""
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from lab import wing_calibration as wc
from lab.engine import Ctx
from studies.wings_calibration import evaluate, fit_offset


class CalibrationTests(unittest.TestCase):
    def test_log_odds_pool_preserves_endpoints_and_weekend_offset(self):
        for share, expected in [(0.,.08),(1.,.15)]:
            self.assertAlmostEqual(wc.pooled_probability(.15,.08,False,[0.,-.5,share]),expected)
        self.assertLess(wc.pooled_probability(.15,.08,True,[0.,-.5,.3]),
                        wc.pooled_probability(.15,.08,False,[0.,-.5,.3]))

    def test_weighting_ignores_repeated_snapshots(self):
        end = pd.Timestamp("2026-11-21 11:00Z")
        d = pd.DataFrame({"end":[end]*4,"ts":pd.date_range(end-pd.Timedelta(minutes=29),periods=4,freq="min"),
                          "left":[29,28,24,23],"market":["A","A","A","B"],"y":[1,1,1,0],"t4":[.1]*4})
        s = wc.sampled(d)
        self.assertEqual(len(s),3)
        self.assertEqual(s.groupby("market").weight.sum().to_dict(),{"A":.5,"B":.5})
        self.assertGreater(fit_offset(s,"t4"),0)

    def test_validation_does_not_remove_unresolved_signals_before_entry(self):
        end = pd.Timestamp("2026-11-21 11:00Z")
        base = dict(end=end,market="unresolved",ts=end-pd.Timedelta(minutes=25),left=25,book=1,
                    bid=.08,ask=.09,bid_size=10,ask_size=10,cheap_yes=True,mid=.085,y=np.nan,
                    fair_t4=.01,t4=.01,normal=.01,t4_cal=.01,market_cal=.01,hybrid=.01,depth=.01)
        later = dict(base,market="resolved",ts=end-pd.Timedelta(minutes=22),y=0,left=22)
        result = evaluate(pd.DataFrame([base,later]),pd.DatetimeIndex([end]),"test")
        self.assertEqual(result["unresolved_markets"],1)
        self.assertEqual(result["trade_hours"],0)

    def test_collector_records_both_sides_before_outcome(self):
        now = pd.Timestamp("2026-11-21 10:40Z")
        end = now.ceil("h")
        row = dict(market="upper",kind="kx_ladder_1h",venue="kalshi",book=1,end=end,
                   fair_t4=.10,bid=.08,ask=.10,bid_size=6,ask_size=8)
        q = pd.DataFrame([row,dict(row,market="lower",bid=.90,ask=.92,fair_t4=.90),
                          dict(row,market="listing",book=0)])
        ctx = Ctx(now=now,tick={},ticks=None,minutes=None,markets=q,book=None,ml=None,cache={})
        def blend(c,r):
            c.cache["diagnostic"] = True
            return .11 if r.market=="upper" else .89
        with tempfile.TemporaryDirectory() as tmp, patch.object(wc,"LAB",Path(tmp)), patch("lab.wings._fair_blend",blend):
            with patch.object(wc,"utcnow",return_value=now+pd.Timedelta(seconds=8)):
                wc.record_forecasts(ctx)
            with patch.object(wc,"utcnow",return_value=end):
                wc.record_forecasts(ctx)
            saved = pd.read_csv(next(Path(tmp).rglob("*.csv")))
        self.assertEqual(set(saved.market),{"upper","lower"})
        self.assertEqual(len(saved),2)
        self.assertTrue(np.allclose(saved.w06,.11))
        self.assertTrue(np.allclose(saved.t4,.1))
        self.assertTrue(np.allclose(saved.t4_cal,wc.odds_adjust(.1,wc.PARAMS["t4"])))
        self.assertNotIn("diagnostic",ctx.cache)
        saved["end"] = pd.to_datetime(saved.end,utc=True)
        depth = wc.depth_hybrid(saved,wc.PARAMS)
        self.assertTrue((depth[saved.cheap_yes] < saved.hybrid[saved.cheap_yes]).all())
        self.assertTrue((depth[~saved.cheap_yes] > saved.hybrid[~saved.cheap_yes]).all())
        self.assertTrue(np.allclose(wc.depth_hybrid(saved.assign(ask_size=np.nan),wc.PARAMS),saved.hybrid))


if __name__=="__main__":
    unittest.main()
