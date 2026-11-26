import unittest
import pandas as pd
from studies.wings_neighbors import enrich,replay


class NeighborTests(unittest.TestCase):
    def test_curve_ignores_target_price_and_other_times(self):
        ts = pd.Timestamp("2026-11-02 10:30:00Z")
        q = pd.DataFrame(dict(ts=[ts]*3,end=[ts.ceil("h")]*3,market=["E-T100","E-T110","E-T120"],
                              bid=[.15,.09,.05],ask=[.17,.10,.07]))
        p = enrich(q).curve.iloc[1]
        self.assertTrue(.06 < p < .16)
        q.loc[1,["bid","ask"]] = [.3,.32]
        self.assertEqual(enrich(q).curve.iloc[1],p)
        q.loc[2,"ts"] += pd.Timedelta(seconds=1)
        self.assertTrue(pd.isna(enrich(q).curve.iloc[1]))

    def test_retry_waits_until_failure_and_counts_only_one_fill(self):
        ts = pd.Timestamp("2026-11-02 10:30:00Z")
        end = ts.ceil("h")
        q = pd.DataFrame(dict(ts=[ts+pd.Timedelta(seconds=s) for s in (0,30,60,120)],
                              end=end,market=["E-T100","E-T110","E-T100","E-T100"],
                              bid=[.09,.10,.07,.08],ask=[.10,.11,.08,.09],cheap_yes=True,
                              y=[0.,1.,0.,0.],t4=.01,normal=.01))
        rows = {r["Правило"]:r for r in replay(enrich(q),pd.DatetimeIndex([end]),"test")}
        self.assertEqual(rows["База t₄"]["Исполнимо"],0)
        retry = rows["База t₄ · повторная попытка"]
        self.assertEqual((retry["Попытки"],retry["Исполнимо"],retry["С задержкой, $"]),(2,1,.07))
