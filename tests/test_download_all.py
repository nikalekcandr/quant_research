"""scripts/download_all.py: plan, estimate, per-source isolation, report files, --status, detached spawn."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("download_all", ROOT / "scripts/download_all.py")
dl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dl)


def args(**kw):
    base = dict(since=None, until="2026-11-27", only=None, kalshi_since="2026-01-01", kalshi_minutes=60,
                one_second=None, workers=2, min_interval=0.0, build=False)
    base.update(kw)
    return SimpleNamespace(**base)


class FakeHttp:
    def __init__(self):
        self.stats = {"requests": 0, "429": 0, "errors": 0}
        self.min_interval = 0.1


class PlanTest(unittest.TestCase):
    def test_kalshi_plan_counts_missing_days_of_the_60_minute_ladder(self):
        with TemporaryDirectory() as d:
            root = Path(d)
            (root / "data" / "raw" / "kalshi_kxbtcd_60m").mkdir(parents=True)
            (root / "data" / "raw" / "kalshi_kxbtcd_60m" / "2026-01-01.csv.gz").write_bytes(b"")
            rows = dl.plan(root, "2026-01-05", None, ["kalshi"])
            self.assertEqual(rows[0]["since"], "2026-01-01")
            self.assertEqual(rows[0]["missing"], 3)               # 01-02 .. 01-04

    def test_one_second_is_opt_in(self):
        self.assertNotIn("binance_1s", [r["source"] for r in dl.plan(ROOT, "2026-11-27", None, None)])
        rows = dl.plan(ROOT, "2026-11-27", None, None, one_second="2026-11-01:2026-11-26")
        self.assertIn("binance_1s", [r["source"] for r in rows])

    def test_estimate_counts_only_kalshi(self):
        self.assertAlmostEqual(dl.estimate_hours([{"missing": 60}]), 1.0)
        self.assertEqual(dl.estimate_hours([{"source": "binance"}]), 0.0)

    def test_steps_put_cheap_sources_first_and_kalshi_last(self):
        names = [t[0] for t in dl.steps(args())]
        self.assertEqual(names[0], "binance")
        self.assertEqual(names[-1], "kalshi")
        self.assertIn("perp", names)                                  # never downloaded before
        self.assertNotIn("binance_1s", names)
        self.assertIn("binance_1s", [t[0] for t in dl.steps(args(one_second="2026-11-01"))])


class RunTest(unittest.TestCase):
    def _sources(self, root, fail=()):
        def ok(name):
            def fn(since, until, http):
                (root / f"{name}.txt").write_text(f"{since}|{until}", encoding="utf-8")
            return fn

        def boom(since, until, http):
            raise RuntimeError("venue said no")

        return [(n, f"source {n}", boom if n in fail else ok(n)) for n in ("alpha", "beta", "gamma")]

    def test_success_writes_state_and_summary(self):
        with TemporaryDirectory() as d:
            root = Path(d)
            work = root / "logs" / "download-test"
            state = dl.run(args(only=["alpha", "beta"]), registry=self._sources(root), http=FakeHttp(),
                           root=root, work=work)
            self.assertEqual([s["status"] for s in state["sources"]], ["ok", "ok"])
            self.assertTrue((root / "alpha.txt").exists())
            self.assertEqual(json.loads((work / "state.json").read_text())["finished"], state["finished"])
            summary = (work / "summary.md").read_text(encoding="utf-8")
            self.assertIn("| alpha |", summary)
            self.assertIn("Всё скачалось", summary)

    def test_a_failed_source_does_not_stop_the_rest(self):
        with TemporaryDirectory() as d:
            root = Path(d)
            work = root / "logs" / "download-test"
            state = dl.run(args(), registry=self._sources(root, fail=("beta",)), http=FakeHttp(),
                           root=root, work=work)
            self.assertEqual([s["status"] for s in state["sources"]], ["ok", "failed", "ok"])
            self.assertTrue((root / "gamma.txt").exists())
            self.assertIn("Не всё скачалось", (work / "summary.md").read_text(encoding="utf-8"))

    def test_build_step_runs_for_registered_sources(self):
        from src.download import update as upd
        called = []
        old = dict(upd.BUILDERS)
        upd.BUILDERS["alpha"] = lambda: called.append("alpha")
        try:
            with TemporaryDirectory() as d:
                root = Path(d)
                dl.run(args(only=["alpha"], build=True), registry=self._sources(root), http=FakeHttp(),
                       root=root, work=root / "logs" / "download-test")
                self.assertEqual(called, ["alpha"])
        finally:
            upd.BUILDERS.clear()
            upd.BUILDERS.update(old)


class StatusTest(unittest.TestCase):
    def test_status_reads_the_newest_run(self):
        with TemporaryDirectory() as d:
            root = Path(d)
            dl.run(args(only=["alpha"]), registry=[("alpha", "a", lambda s, u, h: None)],
                   http=FakeHttp(), root=root, work=root / "logs" / "download-20260101-000000")
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                self.assertEqual(dl.status(root), 0)
            text = out.getvalue()
            self.assertIn("download-20260101-000000", text)
            self.assertIn("alpha", text)

    def test_no_runs_is_a_clean_miss(self):
        with TemporaryDirectory() as d:
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                self.assertEqual(dl.status(Path(d)), 1)
            self.assertIn("прогонов ещё не было", out.getvalue())


class SpawnTest(unittest.TestCase):
    def _capture(self, **kw):
        seen = {}

        class FakePopen:
            def __init__(self, cmd, **kw_):
                seen["cmd"], seen["kw"] = cmd, kw_
                self.pid = 4242

        old_popen, old_root = dl.subprocess.Popen, dl.ROOT
        dl.subprocess.Popen = FakePopen
        try:
            with TemporaryDirectory() as d:
                dl.ROOT = Path(d)
                out = io.StringIO()
                with contextlib.redirect_stdout(out):
                    dl.spawn(args(**kw))
                self.assertTrue((Path(d) / "logs").exists())
                return seen, out.getvalue()
        finally:
            dl.subprocess.Popen, dl.ROOT = old_popen, old_root

    def test_flags_reach_the_child(self):
        seen, out = self._capture(since="2026-07-01", only=["kalshi"], workers=4, min_interval=0.05,
                                  build=True, one_second="2026-11-01")
        cmd = seen["cmd"]
        self.assertIn("--foreground", cmd)
        self.assertIn("--work-dir", cmd)
        self.assertEqual(cmd[cmd.index("--since") + 1], "2026-07-01")
        self.assertEqual(cmd[cmd.index("--only") + 1], "kalshi")
        self.assertEqual(cmd[cmd.index("--kalshi-minutes") + 1], "60")
        self.assertEqual(cmd[cmd.index("--one-second") + 1], "2026-11-01")
        self.assertEqual(cmd[cmd.index("--workers") + 1], "4")
        self.assertIn("--build", cmd)
        self.assertIn("4242", out)

    def test_no_build_is_propagated(self):
        seen, _ = self._capture(build=False)
        self.assertIn("--no-build", seen["cmd"])
        self.assertNotIn("--build", seen["cmd"])


if __name__ == "__main__":
    unittest.main()
