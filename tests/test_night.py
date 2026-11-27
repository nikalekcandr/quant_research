"""scripts/night.py: plan, estimate, per-source isolation, the report files, --status, detached spawn."""
import importlib.util
import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("night", ROOT / "scripts/night.py")
night = importlib.util.module_from_spec(spec)
spec.loader.exec_module(night)


def args(**kw):
    base = dict(since=None, until="2026-11-27", only=None, workers=2, min_interval=0.0, build=False)
    base.update(kw)
    return SimpleNamespace(**base)


class PlanTest(unittest.TestCase):
    def test_kalshi_window_is_the_repository_history(self):
        with TemporaryDirectory() as d:
            root = Path(d)
            (root / "data" / "raw" / "kalshi").mkdir(parents=True)
            (root / "data" / "raw" / "kalshi" / "2025-12-01.csv.gz").write_bytes(b"")
            rows = night.plan(root, "2025-12-04", None, ["kalshi"])
            self.assertEqual(rows[0]["source"], "kalshi")
            self.assertEqual(rows[0]["since"], "2025-12-01")     # not 2024-12-12 from starts.json
            self.assertEqual(rows[0]["missing"], 2)              # 12-02 and 12-03

    def test_estimate_scales_with_missing_days(self):
        self.assertAlmostEqual(night.estimate_hours([{"missing": 60}]), 1.0)
        self.assertEqual(night.estimate_hours([{"source": "binance"}]), 0.0)

    def test_only_filters_the_plan(self):
        rows = night.plan(ROOT, "2026-11-27", None, ["binance", "macro"])
        self.assertEqual([r["source"] for r in rows], ["binance", "macro"])


class RunTest(unittest.TestCase):
    def _sources(self, root, fail=()):
        def ok(name):
            def fn(since, until, http):
                (root / f"{name}.txt").write_text(f"{since}|{until}", encoding="utf-8")
            return fn

        def boom(since, until, http):
            raise RuntimeError("venue said no")

        return [(n, f"source {n}", boom if n in fail else ok(n)) for n in ("alpha", "beta", "gamma")]

    def test_success_writes_state_summary_and_builds_nothing(self):
        with TemporaryDirectory() as d:
            root = Path(d)
            work = root / "logs" / "night-test"
            state = night.run(args(only=["alpha", "beta"]), sources=self._sources(root), http=FakeHttp(),
                              root=root, work=work)
            self.assertEqual([s["status"] for s in state["sources"]], ["ok", "ok"])
            self.assertTrue((root / "alpha.txt").exists())
            self.assertEqual(json.loads((work / "state.json").read_text())["finished"], state["finished"])
            summary = (work / "summary.md").read_text(encoding="utf-8")
            self.assertIn("| alpha |", summary)
            self.assertIn("Всё скачалось", summary)

    def test_one_failed_source_does_not_stop_the_rest(self):
        with TemporaryDirectory() as d:
            root = Path(d)
            work = root / "logs" / "night-test"
            state = night.run(args(), sources=self._sources(root, fail=("beta",)), http=FakeHttp(),
                              root=root, work=work)
            self.assertEqual([s["status"] for s in state["sources"]], ["ok", "failed", "ok"])
            self.assertTrue((root / "gamma.txt").exists())              # the last one still ran
            summary = (work / "summary.md").read_text(encoding="utf-8")
            self.assertIn("Не всё скачалось", summary)
            self.assertIn("beta", summary)

    def test_build_flag_rebuilds_registered_sources(self):
        from src.download import update as upd
        called = []
        old = dict(upd.BUILDERS)
        upd.BUILDERS["alpha"] = lambda: called.append("alpha")
        try:
            with TemporaryDirectory() as d:
                root = Path(d)
                night.run(args(only=["alpha"], build=True), sources=self._sources(root), http=FakeHttp(),
                          root=root, work=root / "logs" / "night-test")
                self.assertEqual(called, ["alpha"])
        finally:
            upd.BUILDERS.clear()
            upd.BUILDERS.update(old)


class FakeHttp:
    def __init__(self):
        self.stats = {"requests": 0, "429": 0, "errors": 0}
        self.min_interval = 0.1


class StatusTest(unittest.TestCase):
    def test_status_reads_the_newest_run(self):
        import contextlib
        import io
        with TemporaryDirectory() as d:
            root = Path(d)
            night.run(args(only=["alpha"]), sources=[("alpha", "a", lambda s, u, h: None)],
                      http=FakeHttp(), root=root, work=root / "logs" / "night-20260101-000000")
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                self.assertEqual(night.status(root), 0)
            text = out.getvalue()
            self.assertIn("night-20260101-000000", text)
            self.assertIn("alpha", text)

    def test_status_without_runs_is_a_clean_miss(self):
        import contextlib
        import io
        with TemporaryDirectory() as d:
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                self.assertEqual(night.status(Path(d)), 1)
            self.assertIn("прогонов ещё не было", out.getvalue())


class SpawnTest(unittest.TestCase):
    def test_detached_command_says_no_build_when_asked(self):
        import contextlib
        import io
        seen = {}

        class FakePopen:
            def __init__(self, cmd, **kw):
                seen["cmd"] = cmd
                self.pid = 1

        old, old_root = night.subprocess.Popen, night.ROOT
        night.subprocess.Popen = FakePopen
        try:
            with TemporaryDirectory() as d:
                night.ROOT = Path(d)
                with contextlib.redirect_stdout(io.StringIO()):
                    night.spawn(args(build=False))
                self.assertIn("--no-build", seen["cmd"])
        finally:
            night.subprocess.Popen, night.ROOT = old, old_root

    def test_detached_command_carries_the_flags(self):
        import contextlib
        import io
        seen = {}

        class FakePopen:
            def __init__(self, cmd, **kw):
                seen["cmd"], seen["kw"] = cmd, kw
                self.pid = 4242

        old_popen, old_root = night.subprocess.Popen, night.ROOT
        night.subprocess.Popen = FakePopen
        try:
            with TemporaryDirectory() as d:
                night.ROOT = Path(d)
                out = io.StringIO()
                with contextlib.redirect_stdout(out):
                    self.assertEqual(night.spawn(args(since="2026-07-01", only=["kalshi"], workers=4,
                                                      min_interval=0.05, build=True)), 0)
                cmd = seen["cmd"]
                self.assertIn("--foreground", cmd)
                self.assertIn("--work-dir", cmd)
                self.assertEqual(cmd[cmd.index("--since") + 1], "2026-07-01")
                self.assertEqual(cmd[cmd.index("--only") + 1], "kalshi")
                self.assertEqual(cmd[cmd.index("--workers") + 1], "4")
                self.assertIn("--build", cmd)
                self.assertNotIn("--no-build", cmd)
                self.assertIn("4242", out.getvalue())
                self.assertTrue((Path(d) / "logs").exists())
        finally:
            night.subprocess.Popen, night.ROOT = old_popen, old_root


if __name__ == "__main__":
    unittest.main()
