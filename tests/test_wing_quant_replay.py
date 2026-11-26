import unittest
import pandas as pd

from studies.wing_quant_replay import density_ok,observations


class QuantReplayTests(unittest.TestCase):
    def test_flow_uses_only_previous_snapshot_of_same_market(self):
        start=pd.Timestamp("2026-11-01T00:00Z")
        q=pd.DataFrame({"market":["A-T100","B-T100","A-T100","A-T100"],"end":[start.ceil("h")]*4,
                        "ts":[start,start,start+pd.Timedelta(seconds=60),start+pd.Timedelta(seconds=180)],
                        "bid":[.08,.08,.08,.08],"ask":[.09]*4,"bid_size":[10,500,15,35],"ask_size":[10]*4})
        result=observations(q)
        self.assertTrue(pd.isna(result.loc[0,"ofi"]))
        self.assertAlmostEqual(result.loc[2,"ofi"],5/21)
        self.assertTrue(pd.isna(result.loc[3,"ofi"]))

    def test_flat_variance_has_nonnegative_density(self):
        self.assertTrue(density_ok([.25,0,0,0,.1],-.3,.3,1/365))
