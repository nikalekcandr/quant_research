import unittest
import numpy as np
import pandas as pd
from studies.wing_shapes import curves,NAMES


class LadderShapesTest(unittest.TestCase):
    def test_target_price_excluded_and_no_edge_extrapolation(self):
        p=np.linspace(.85,.15,9)
        f=pd.DataFrame({"market":[f"BTC-T{100+i}" for i in range(9)],"ts":pd.Timestamp("2026-10-01T12:00Z"),
                        "end":pd.Timestamp("2026-10-01T13:00Z"),"bid":p-.01,"ask":p+.01})
        first=curves(f)
        f.loc[4,["bid","ask"]]=[.70,.72]
        second=curves(f)
        self.assertTrue(first.loc[[0,8],NAMES].isna().all().all())
        x=first.loc[4,NAMES].to_numpy(dtype=float)
        self.assertTrue(np.isfinite(x).all())
        np.testing.assert_allclose(x,second.loc[4,NAMES].to_numpy(dtype=float),atol=1e-7)
