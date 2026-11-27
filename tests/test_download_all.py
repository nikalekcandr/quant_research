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


def fake_audit(root=None):
    import pandas as pd
    return pd.DataFrame([{"source": "spot 1m", "path": "data/raw/binance", "last": pd.Timestamp("2026-11-26"),
                          "gap days": 1.0, "command": "python -m src.download.update --only binance", "note": ""},
                         {"source": "November books", "path": "data/raw/wing_gap_candles",
                          "last": pd.Timestamp("2026-11-25"), "gap days": 2.0, "command": "(live logger)", "note": ""}])


class RunTest(unittest.TestCase):
    def setUp(self):
        self._old_audit = dl.audit_frame
        dl.audit_frame = fake_audit

    def tearDown(self):
        dl.audit_frame = self._old_audit
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
            self.assertIn("Все источники скачались", summary)
            self.assertIn("всё свежее", summary)
            self.assertIn("## Проверка", summary)
            self.assertIn("wing_panel.parquet", summary)      # the next steps point at the panel cache

    def test_a_failed_source_does_not_stop_the_rest(self):
        with TemporaryDirectory() as d:
            root = Path(d)
            work = root / "logs" / "download-test"
            state = dl.run(args(), registry=self._sources(root, fail=("beta",)), http=FakeHttp(),
                           root=root, work=work)
            self.assertEqual([s["status"] for s in state["sources"]], ["ok", "failed", "ok"])
            self.assertTrue((root / "gamma.txt").exists())
            text = (work / "summary.md").read_text(encoding="utf-8")
            self.assertIn("Не всё скачалось", text)
            self.assertIn("beta", text)

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


class CliTest(unittest.TestCase):
    """The flags the wrapper scripts and spawn() rely on must parse — a typo here kills a whole run."""

    def test_build_and_no_build_parse(self):
        self.assertEqual(dl.main(["--dry-run", "--build"]), 0)
        self.assertEqual(dl.main(["--dry-run", "--no-build"]), 0)

    def test_selftest_runs_the_whole_machinery(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertEqual(dl.main(["--selftest"]), 0)
        self.assertIn("selftest ok", out.getvalue())

    def test_verify_uses_the_audit(self):
        old = dl.audit_frame
        dl.audit_frame = fake_audit
        try:
            out = io.StringIO()
            result = dl.verify(ROOT, out=out.write)
            self.assertTrue(result["ok"])
            self.assertIn("spot 1m", out.getvalue())

            import pandas as pd

            def stale_audit(root=None):
                return pd.DataFrame([{"source": "Kalshi ladder", "path": "x", "last": pd.Timestamp("2026-10-27"),
                                      "gap days": 31.0, "command": "python -m src.download.update --only kalshi",
                                      "note": ""}])

            dl.audit_frame = stale_audit
            result = dl.verify(ROOT, out=io.StringIO().write)
            self.assertFalse(result["ok"])
            self.assertEqual(result["stale"], ["Kalshi ladder"])
        finally:
            dl.audit_frame = old


class IsOursTest(unittest.TestCase):
    """Убивать можно только настоящий загрузчик: иначе снесём свою же оболочку (проверено на практике)."""

    def test_only_our_loader_matches(self):
        cases = {
            "python scripts/download_all.py --foreground": True,
            "python -m src.download.update --only binance": True,
            "python -m src.download.binance_1s --since 2026-01-01": True,
            'C:\\Python314\\python.exe scripts\\backfill.py': True,
            'bash -c "cd /repo && python scripts/download_all.py"': False,
            'cmd.exe /c "C:\\x\\download_all.bat"': False,
            "python scripts/live.py --run": False,
            "python scripts/study_wing_gap.py --run": False,
            "python -m pytest -q": False,
        }
        wrong = {cmd: want for cmd, want in cases.items() if dl._is_ours(cmd) is not want}
        self.assertEqual(wrong, {})


class PreflightTest(unittest.TestCase):
    def test_found_processes_are_listed_and_killed(self):
        procs = [{"pid": 111, "cmd": "python scripts/download_all.py"}, {"pid": 222, "cmd": "python -m src.download.update"}]
        killed = []
        old_list, old_kill = dl.list_download_processes, dl.kill_processes
        dl.list_download_processes = lambda: procs
        dl.kill_processes = lambda p, grace=3.0: killed.extend(x["pid"] for x in p) or [x["pid"] for x in p]
        try:
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                self.assertEqual(dl.preflight(), 2)
            self.assertEqual(killed, [111, 222])
            self.assertIn("111", out.getvalue())
        finally:
            dl.list_download_processes, dl.kill_processes = old_list, old_kill

    def test_no_kill_leaves_them_alone(self):
        old_list, old_kill = dl.list_download_processes, dl.kill_processes
        dl.list_download_processes = lambda: [{"pid": 5, "cmd": "python -m src.download.kalshi"}]
        dl.kill_processes = lambda p, grace=3.0: (_ for _ in ()).throw(AssertionError("must not kill"))
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(dl.preflight(kill=False), 1)
        finally:
            dl.list_download_processes, dl.kill_processes = old_list, old_kill

    def test_empty_list_is_a_no_op(self):
        old = dl.list_download_processes
        dl.list_download_processes = lambda: []
        try:
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                self.assertEqual(dl.preflight(), 0)
            self.assertIn("чисто", out.getvalue())
        finally:
            dl.list_download_processes = old


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
