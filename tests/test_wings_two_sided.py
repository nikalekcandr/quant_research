import unittest
import pandas as pd

from studies.wings_two_sided import compare


class TwoSidedTests(unittest.TestCase):
    def test_opposite_side_pays_spread_and_baseline_wins_simultaneous_signal(self):
        end = pd.Timestamp("2026-11-19 10:00Z")
        row = dict(end=end,ts=end-pd.Timedelta(minutes=20),market="a",bid=.08,ask=.10,
                   bid_size=4,ask_size=2,cheap_yes=True,t4=.25,normal=.25,y=1.)
        quotes = pd.DataFrame([row,dict(row,market="b",t4=.01,normal=.01,y=0.)])
        hours = pd.DatetimeIndex([end,end+pd.Timedelta(hours=1)])
        result = {r["Правило"]:r for r in compare(quotes,hours,"test")}
        self.assertEqual(result["Покупка · согласие"]["Нетто, $"],.89)
        self.assertEqual(result["Обратная сделка · после анализа"]["Нетто, $"],-.93)
        self.assertEqual(result["Продажа t₄ или обратная"]["Нетто, $"],.07)
        self.assertEqual(result["Продажа t₄ или обратная"]["¢/час"],3.5)
