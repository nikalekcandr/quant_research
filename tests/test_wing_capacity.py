import unittest
from studies.wing_capacity_ml import sweep
from lab.markets import _levels


class CapacityTests(unittest.TestCase):
    def test_unit_and_batch_have_different_edge(self):
        self.assertEqual(sweep([(.92,1)],.937)[0],0)
        n,cost,fee,last=sweep([(.92,100),(.94,500)],.937)
        self.assertEqual((n,round(cost,2),fee,last),(100,92.,.52,.92))

    def test_delayed_execution_obeys_both_caps(self):
        n,cost,fees,last=sweep([(.89,11),(.90,20),(.92,100)],.99,limit=.90,cap=20)
        self.assertEqual((n,round(cost,2),last),(20,17.89,.90))

    def test_cent_and_dollar_levels_are_distinct(self):
        self.assertEqual(_levels([[1,18],[99,3]],False),[(.01,18.),(.99,3.)])
        self.assertEqual(_levels([["0.0100","18.00"]],True),[(.01,18.)])
