import unittest

import numpy as np

from studies.wings_capacity import calib_slope, n_params


class CalibrationSlopeTest(unittest.TestCase):
    def test_calibrated_probilities_give_a_slope_near_one(self):
        rng = np.random.default_rng(0)
        p = rng.uniform(0.02, 0.3, 20000)
        y = (rng.uniform(size=p.size) < p).astype(int)
        self.assertAlmostEqual(calib_slope(y, p), 1.0, delta=0.15)

    def test_a_constant_forecast_has_no_slope(self):
        rng = np.random.default_rng(1)
        y = (rng.uniform(size=5000) < 0.1).astype(int)
        self.assertLess(abs(calib_slope(y, np.full(y.size, 0.1))), 0.05)

    def test_an_inverted_forecast_has_a_negative_slope(self):
        rng = np.random.default_rng(2)
        p = rng.uniform(0.02, 0.3, 20000)
        y = (rng.uniform(size=p.size) < 1 - p).astype(int)
        self.assertLess(calib_slope(y, p), -0.5)


class ParameterCountTest(unittest.TestCase):
    def test_mlp_parameter_counts(self):
        self.assertEqual(n_params("mlp 16"), 12 * 16 + 16 + 16 + 1)
        self.assertEqual(n_params("mlp 64-32"), (12 * 64 + 64) + (64 * 32 + 32) + (32 * 1 + 1))
        self.assertEqual(n_params("logit"), 13)

    def test_trees_report_no_count(self):
        self.assertTrue(np.isnan(n_params("hgb big")))


if __name__ == "__main__":
    unittest.main()
