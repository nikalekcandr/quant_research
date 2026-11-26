import math
import unittest
from scipy.special import stdtr

from trader.executor import refresh_fair


class RefreshTests(unittest.TestCase):
    def test_both_sides_match_direct_t4_repricing(self):
        ticker = "KXBTCD-26NOV2107-T58099.99"
        s0,s1,k,vol = 57858.77,57955.,58099.99,.0028
        for side,sign in [("yes",1),("no",-1)]:
            p0 = stdtr(4,sign*math.sqrt(2)*math.log(s0/k)/vol)
            expected = stdtr(4,sign*math.sqrt(2)*math.log(s1/k)/vol)
            self.assertAlmostEqual(refresh_fair(p0,side,ticker,s0,s1,t4=True),expected)
