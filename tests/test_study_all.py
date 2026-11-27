"""The one-button study runner: registry sane, filters work, failures classified instead of guessed."""
from __future__ import annotations

import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import study_all as sa  # noqa: E402


class RegistryTest(unittest.TestCase):
    def test_every_registered_study_exists_on_disk(self):
        missing = [n for n, *_ in sa.REGISTRY if not (ROOT / "studies" / f"{n}.py").exists()]
        self.assertEqual(missing, [])

    def test_registry_covers_the_studies_folder(self):
        on_disk = {p.stem for p in (ROOT / "studies").glob("*.py") if "__main__" in p.read_text()}
        known = {n for n, *_ in sa.REGISTRY} | sa.NOT_STUDIES
        untracked = sorted(on_disk - known)
        self.assertEqual(untracked, [], f"в scripts/study_all.py нет: {untracked}")

    def test_descriptions_and_estimates_are_sane(self):
        for name, group, what, est, heavy in sa.REGISTRY:
            self.assertTrue(what and len(what) > 8, name)
            self.assertGreaterEqual(est, 0, name)
            self.assertIsInstance(heavy, bool, name)
            self.assertIn(group, {"панель", "крылья", "волатильность", "опционы", "Kalshi 15m", "отчёты"}, name)

    def test_no_downloaders_are_run_as_studies(self):
        self.assertEqual(sa.NOT_STUDIES, {"wing_gap_download", "wing_forecast_settlements", "quant_inputs"})
        self.assertFalse({n for n, *_ in sa.REGISTRY} & sa.NOT_STUDIES)


class PlanTest(unittest.TestCase):
    def test_default_plan_starts_with_the_panel(self):
        rows = sa.plan()
        self.assertEqual(rows[0]["name"], "wing_panel")
        self.assertGreater(len(rows), 40)

    def test_only_skip_and_no_heavy(self):
        self.assertEqual([r["name"] for r in sa.plan(only=["wings_sleep", "wings_alpha"])],
                         ["wings_sleep", "wings_alpha"])
        names = {r["name"] for r in sa.plan(skip=["wings_ml"])}
        self.assertNotIn("wings_ml", names)
        self.assertTrue(all(not r["heavy"] for r in sa.plan(heavy=False)))
        self.assertIn("wings_ml", {r["name"] for r in sa.plan()})

    def test_estimate_is_a_number_of_seconds(self):
        total = sum(r["est"] for r in sa.plan())
        self.assertGreater(total, 600)          # the full set is tens of minutes, not seconds


class ClassifyTest(unittest.TestCase):
    def test_missing_file(self):
        status, why = sa.classify(1, "Traceback...\nFileNotFoundError: [Errno 2] No such file or directory: "
                                     "'/x/data/raw/kx15/trades.csv'")
        self.assertEqual(status, "nodata")
        self.assertIn("kx15/trades.csv", why)

    def test_missing_package(self):
        self.assertEqual(sa.classify(1, "ModuleNotFoundError: No module named 'plotly'"), ("nopkg", "нет пакета: plotly"))
        self.assertEqual(sa.classify(1, "ImportError: `Import tabulate` failed.")[0], "nopkg")

    def test_out_of_memory(self):
        self.assertEqual(sa.classify(-9, "")[0], "killed")

    def test_empty_selection_is_data_not_a_crash(self):
        self.assertEqual(sa.classify(1, "ValueError: No objects to concatenate")[0], "nodata")

    def test_real_error_keeps_its_text(self):
        status, why = sa.classify(1, "Traceback (most recent call last):\nKeyError: 'rule'")
        self.assertEqual(status, "error")
        self.assertIn("KeyError: 'rule'", why)


class RunTest(unittest.TestCase):
    def test_failure_is_isolated_and_summary_lists_it(self):
        rows = [{"name": "alpha", "group": "тест", "what": "ok", "est": 1, "heavy": False},
                {"name": "beta", "group": "тест", "what": "нет данных", "est": 1, "heavy": False}]
        old = sa.run_study

        def fake(row, work, index, total):
            ok = row["name"] == "alpha"
            return {**row, "status": "ok" if ok else "nodata", "why": "" if ok else "нет данных: data/raw/x",
                    "seconds": 2.0, "log": str(Path(work) / f"{row['name']}.log"), "tail": ["строка"]}

        sa.run_study = fake
        try:
            with tempfile.TemporaryDirectory() as d, redirect_stdout(io.StringIO()):
                state = sa.run(rows, work=Path(d))
                summary = (Path(d) / "summary.md").read_text(encoding="utf-8")
                saved = json.loads((Path(d) / "state.json").read_text(encoding="utf-8"))
        finally:
            sa.run_study = old
        self.assertEqual([s["status"] for s in state["studies"]], ["ok", "nodata"])
        self.assertIn("beta", summary)
        self.assertIn("нет данных: data/raw/x", summary)
        self.assertIn("--only beta", summary)
        self.assertEqual(len(saved["studies"]), 2)

    def test_status_reports_the_last_run(self):
        with tempfile.TemporaryDirectory() as d:
            work = Path(d) / "studies-20260101-000000"
            work.mkdir(parents=True)
            (work / "state.json").write_text(json.dumps({"seconds": 12, "studies": [
                {"name": "alpha", "status": "ok", "seconds": 12, "why": ""}], "artifacts": {}}), encoding="utf-8")
            old = sa.ROOT
            sa.ROOT = Path(d).parent
            try:
                out = io.StringIO()
                with redirect_stdout(out):
                    rc = sa.status(work)
            finally:
                sa.ROOT = old
            self.assertEqual(rc, 0)
            self.assertIn("1/1 прошло", out.getvalue())

    def test_selftest_runs(self):
        out = io.StringIO()
        with redirect_stdout(out):
            self.assertEqual(sa.selftest(), 0)
        self.assertIn("selftest ok", out.getvalue())


if __name__ == "__main__":
    unittest.main()
