import unittest
from unittest import mock

import numpy as np
import pandas as pd

from src import runner
from studies.wings_ten import acf, boot, runs_test, utc_index


class StatsHelpersTest(unittest.TestCase):
    def test_acf_of_white_noise_is_near_zero(self):
        rng = np.random.default_rng(0)
        x = pd.Series(rng.normal(0, 1, 4000))
        for rho in acf(x, 3):
            self.assertLess(abs(rho), 0.08)

    def test_acf_of_a_cycle_is_positive_at_the_period(self):
        t = np.arange(2000)
        x = pd.Series(np.sin(2 * np.pi * t / 24) + 0.1 * np.random.default_rng(1).normal(size=t.size))
        self.assertGreater(acf(x, 1)[0], 0.5)

    def test_runs_test_flags_streaks_and_alternation(self):
        streak = pd.Series(np.where((np.arange(400) // 50) % 2 == 0, 1.0, -1.0))   # 8 long blocks
        self.assertLess(runs_test(streak)["z"], -5)
        zigzag = pd.Series(np.where(np.arange(400) % 2 == 0, 1.0, -1.0))           # alternates every hour
        self.assertGreater(runs_test(zigzag)["z"], 5)

    def test_runs_test_needs_a_sample(self):
        self.assertTrue(np.isnan(runs_test(pd.Series([1.0, -1.0, 1.0]))["z"]))

    def test_bootstrap_interval_covers_the_mean(self):
        rng = np.random.default_rng(2)
        x = pd.Series(rng.normal(2.0, 6.0, 700), index=pd.date_range("2026-07-01", periods=700, freq="h", tz="UTC"))
        out = boot(x, block=24, draws=500)
        self.assertLess(out["lo"], out["mean"])
        self.assertGreater(out["hi"], out["mean"])
        self.assertGreater(out["p_pos"], 0.95)

    def test_bootstrap_refuses_a_short_series(self):
        x = pd.Series(np.ones(10), index=pd.date_range("2026-07-01", periods=10, freq="h", tz="UTC"))
        self.assertTrue(np.isnan(boot(x, block=24)["mean"]))

    def test_utc_index_makes_naive_and_aware_the_same(self):
        naive = pd.to_datetime(["2026-07-01 12:00", "2026-07-01 13:00"])
        aware = naive.tz_localize("UTC")
        self.assertEqual(list(utc_index(naive)), list(utc_index(aware)))
        self.assertIn("UTC", str(utc_index(naive).tz))


class KeepAwakeTest(unittest.TestCase):
    def tearDown(self):
        runner.keep_awake_off()

    def test_returns_a_bool_and_never_raises(self):
        self.assertIsInstance(runner.keep_awake(), bool)

    def test_missing_inhibitor_is_not_fatal(self):
        with mock.patch("shutil.which", return_value=None), mock.patch("sys.platform", "linux"):
            runner.keep_awake_off()
            self.assertFalse(runner.keep_awake())

    def test_a_broken_inhibitor_is_not_fatal(self):
        with mock.patch("subprocess.Popen", side_effect=OSError("no dbus")), \
                mock.patch("shutil.which", return_value="/usr/bin/systemd-inhibit"), \
                mock.patch("sys.platform", "linux"):
            runner.keep_awake_off()
            self.assertFalse(runner.keep_awake())


if __name__ == "__main__":
    unittest.main()
