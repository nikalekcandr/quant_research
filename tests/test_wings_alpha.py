import unittest

import numpy as np
import pandas as pd

from studies.wings_alpha import deflated_sharpe, pbo


def series(mean, sd, n=500, seed=0):
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(mean, sd, n))


class DeflatedSharpeTest(unittest.TestCase):
    def test_strong_edge_survives_many_trials(self):
        r = series(0.15, 1.0)
        trials = np.array([0.05, 0.01, -0.02, 0.0, 0.03])
        out = deflated_sharpe(r, trials)
        self.assertGreater(out["sharpe"], out["sr0"])
        self.assertGreater(out["dsr"], 0.9)

    def test_noise_does_not_survive(self):
        r = series(0.0, 1.0, seed=3)
        trials = np.array([0.2, 0.18, 0.15, 0.12, 0.1, 0.08, 0.05, 0.02])
        out = deflated_sharpe(r, trials)
        self.assertLess(out["dsr"], 0.6)

    def test_the_threshold_grows_with_the_number_of_trials(self):
        r = series(0.02, 1.0)
        few = deflated_sharpe(r, np.linspace(-0.05, 0.1, 5))
        many = deflated_sharpe(r, np.linspace(-0.05, 0.1, 200))
        self.assertGreater(many["sr0"], few["sr0"])

    def test_degenerate_inputs_do_not_crash(self):
        self.assertTrue(np.isnan(deflated_sharpe(pd.Series([0.1, 0.1]), np.array([0.1]))["dsr"]))
        flat = pd.Series(np.zeros(100))
        self.assertTrue(np.isnan(deflated_sharpe(flat, np.array([0.1, 0.2]))["sharpe"]))


class PboTest(unittest.TestCase):
    def test_a_dominant_variant_has_no_overfitting(self):
        rng = np.random.default_rng(1)
        good = rng.normal(0.3, 1.0, 600)
        px = pd.DataFrame({"good": good, "a": rng.normal(0.0, 1.0, 600), "b": rng.normal(0.0, 1.0, 600)})
        self.assertLess(pbo(px)["pbo"], 0.1)

    def test_pure_noise_is_near_a_coin_flip(self):
        rng = np.random.default_rng(2)
        px = pd.DataFrame(rng.normal(0.0, 1.0, (600, 6)), columns=list("abcdef"))
        out = pbo(px)
        self.assertGreater(out["pbo"], 0.35)
        self.assertLess(out["pbo"], 0.65)
        self.assertEqual(out["combos"], 252)

    def test_too_short_a_history_returns_nan(self):
        px = pd.DataFrame({"a": np.arange(10), "b": np.arange(10)})
        self.assertTrue(np.isnan(pbo(px)["pbo"]))


if __name__ == "__main__":
    unittest.main()
