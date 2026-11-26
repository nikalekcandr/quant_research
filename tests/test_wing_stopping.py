import unittest
import pandas as pd
from studies.wing_stopping import policies,BASE


class EntryTimingTest(unittest.TestCase):
    def test_only_earlier_contiguous_history_counts(self):
        start=pd.Timestamp("2026-11-14T12:00Z")
        q=pd.DataFrame({"ts":[start+pd.Timedelta(seconds=i) for i in [0,60,120,240]],
                        "market":"a","cheap_yes":True,"left":[24,23,22,20],
                        "bid":.09,"ask":.10,BASE:[.05,.05,.05,.04]})
        truncated=q.iloc[:2].copy()
        names=policies(truncated)
        policies(q)
        pd.testing.assert_frame_equal(truncated[names],q.iloc[:2][names])
        self.assertTrue(pd.notna(q.loc[2,"Три подтверждения"]))
        self.assertTrue(pd.isna(q.loc[3,"Три подтверждения"]))
