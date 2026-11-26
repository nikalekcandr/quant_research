import unittest

import pandas as pd

from studies.wing_sleep_replay import overlap_seconds


class SleepTest(unittest.TestCase):
    def test_overlapping_and_outside_events_do_not_inflate_sleep(self):
        start = pd.Timestamp("2026-11-15T12:00Z")
        minute = pd.Timedelta(minutes=1)
        periods = pd.DataFrame([(start+a*minute,start+b*minute) for a,b in
                               [(-20,-10),(-5,4),(1,4),(3,10),(12,12),(13,20)]],
                               columns=["sleep","wake"])
        self.assertEqual(overlap_seconds(start,start+15*minute,periods), 12*60)
