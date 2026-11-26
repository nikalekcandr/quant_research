import unittest

import numpy as np
import pandas as pd

from studies.wing_panel import fee, fee_per_contract, settlement_level
from studies.wings_scale import kelly_size, pnl_at


class FeeTest(unittest.TestCase):
    def test_order_fee_rounds_up_to_a_cent(self):
        self.assertAlmostEqual(fee(1, 0.10), 0.01)
        self.assertAlmostEqual(fee(100, 0.10), 0.63)
        self.assertAlmostEqual(fee(0, 0.10), 0.00)

    def test_per_contract_fee_falls_with_size_and_then_flat(self):
        small = float(fee_per_contract(1, 0.10))
        big = float(fee_per_contract(100, 0.10))
        huge = float(fee_per_contract(500, 0.10))
        self.assertGreater(small, big)
        self.assertAlmostEqual(big, huge, places=6)
        self.assertLess(big, 0.007)

    def test_fee_is_symmetric_in_price(self):
        self.assertAlmostEqual(float(fee(10, 0.3)), float(fee(10, 0.7)))


class PnlTest(unittest.TestCase):
    def _trades(self):
        return pd.DataFrame({"wing_bid": [0.10, 0.06, 0.20], "won": [0.0, 0.0, 1.0],
                             "price": [0.90, 0.94, 0.80]})

    def test_fee_is_charged_per_order_not_per_contract(self):
        t = self._trades()
        one = pnl_at(t, 1.0)
        hundred = pnl_at(t, 100.0)
        # the price term scales exactly, the fee is (almost) the same for the whole order
        self.assertTrue(np.allclose(hundred - one * 100.0, -(fee(100.0, t.price.to_numpy())
                                                            - 100 * fee(1.0, t.price.to_numpy()))))

    def test_bigger_orders_never_turn_a_profit_into_a_loss(self):
        t = self._trades()
        for n in (1, 10, 100, 500):
            with self.subTest(n=n):
                self.assertGreaterEqual(pnl_at(t, n)[0], pnl_at(t, 1)[0] * n - 1e-9)

    def test_winning_trade_loses_the_payoff(self):
        t = pd.DataFrame({"wing_bid": [0.08], "won": [1.0], "price": [0.92]})
        self.assertLess(pnl_at(t, 1)[0], 0)


class KellyTest(unittest.TestCase):
    def test_bounds_and_monotone_in_edge(self):
        t = pd.DataFrame({"fair": [0.02, 0.05, 0.05, 0.05], "price": [0.97, 0.94, 0.94, 0.94]})
        n = kelly_size(t)
        self.assertEqual(len(n), 4)
        self.assertTrue((n >= 1).all() and (n <= 200).all())
        bigger = kelly_size(pd.DataFrame({"fair": [0.02], "price": [0.94]}))
        self.assertGreaterEqual(int(bigger.iloc[0]), int(n.iloc[1]))

    def test_no_edge_still_one_contract(self):
        t = pd.DataFrame({"fair": [0.5], "price": [0.5]})
        self.assertEqual(int(kelly_size(t).iloc[0]), 1)


class SettlementLevelTest(unittest.TestCase):
    def test_uses_past_hours_only(self):
        hours = pd.date_range("2026-11-10", periods=40, freq="h", tz="UTC")
        rng = np.random.default_rng(0)
        value = 60_000 * np.exp(np.cumsum(rng.normal(0, 0.002, len(hours))))
        d = pd.DataFrame({"end": hours, "expiration_value": value})
        level = settlement_level(d)
        # the level for an hour must not move when the value of that same hour is replaced
        changed = d.copy()
        changed.loc[changed.end == hours[30], "expiration_value"] = value[30] * 1.5
        level2 = settlement_level(changed)
        self.assertAlmostEqual(float(level.loc[hours[30]]), float(level2.loc[hours[30]]))
        # but it does move for the next hour, because the move is now in the past
        self.assertNotAlmostEqual(float(level.loc[hours[31]]), float(level2.loc[hours[31]]))
        self.assertTrue(level.dropna().gt(0).all())


if __name__ == "__main__":
    unittest.main()
