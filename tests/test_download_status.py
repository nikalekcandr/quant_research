"""Monitoring: parsing the progress out of the log and rendering the status screen."""
import contextlib
import importlib.util
import io
import json
from datetime import datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("download_all", ROOT / "scripts/download_all.py")
dl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dl)

LOG = """[12:00:00] закачка: 7 источник(ов), окно до 2026-11-27
[12:00:01] === binance: Binance 1-minute spot candles (default .. 2026-11-27)
[12:03:05] === binance: ok за 3.0 мин
[12:03:05] === kalshi: Kalshi часовые лесенки (2026-01-01 .. 2026-11-27)
[12:54:36] kalshi: 2025-01-01 (515 candle rows)
[13:02:11] kalshi: 83/211 days done, 58.1 min elapsed, 31400 requests (9.0/s), 1.4 days/min, ETA 1.5 h, 2 failed
"""


def make_run(root: Path, log=LOG, finished=None, heartbeat=None):
    work = root / "logs" / "download-20261127-120000"
    work.mkdir(parents=True)
    state = {"started": (datetime.now() - timedelta(minutes=62)).isoformat(timespec="seconds"), "pid": 1,
             "sources": [{"source": "binance", "what": "spot", "status": "ok", "seconds": 180.0},
                         {"source": "perp", "what": "perp", "status": "failed", "seconds": 12.0,
                          "error": "RuntimeError: нет сети"}],
             "requests": {"requests": 31400, "429": 12, "errors": 0}, "final_interval": 0.1}
    if finished:
        state["finished"] = finished
    state["heartbeat"] = (heartbeat or datetime.now()).isoformat(timespec="seconds")
    (work / "state.json").write_text(json.dumps(state), encoding="utf-8")
    (work / "run.log").write_text(log, encoding="utf-8")
    return work


class ParseTest(unittest.TestCase):
    def test_last_progress_line_wins(self):
        text = LOG + ("[13:03:11] kalshi: 120/211 days done, 65.0 min elapsed, 44000 requests (11.3/s), "
                      "1.8 days/min, ETA 50 min, 2 failed\n")
        p = dl.parse_progress(text)
        self.assertEqual((p["done"], p["total"], p["failed"]), (120, 211, 2))
        self.assertEqual(p["eta"], "50 min")
        self.assertAlmostEqual(p["rps"], 11.3)

    def test_no_progress_yet(self):
        self.assertIsNone(dl.parse_progress(LOG.splitlines()[3]))

    def test_running_step_is_the_open_one(self):
        self.assertEqual(dl.current_step(LOG), ("kalshi", "12:03:05"))
        closed = LOG + "[14:00:00] === kalshi: ok за 117.0 мин\n"
        self.assertIsNone(dl.current_step(closed))


class RenderTest(unittest.TestCase):
    def render(self, root):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = dl.render_status(root)
        return code, out.getvalue()

    def test_live_run_shows_progress_percent_and_eta(self):
        with TemporaryDirectory() as d:
            make_run(Path(d))
            code, text = self.render(Path(d))
            self.assertEqual(code, 0)
            self.assertIn("идёт", text)
            self.assertIn("сейчас: kalshi", text)
            self.assertIn("83/211 суток (39%)", text)
            self.assertIn("осталось ~1.5 h", text)
            self.assertIn("429: 12", text)
            self.assertIn("perp", text)                      # the failed source is listed

    def test_finished_run(self):
        with TemporaryDirectory() as d:
            make_run(Path(d), finished=datetime.now().isoformat(timespec="seconds"))
            code, text = self.render(Path(d))
            self.assertEqual(code, 0)
            self.assertIn("завершён", text)

    def test_dead_run_without_heartbeat(self):
        with TemporaryDirectory() as d:
            make_run(Path(d), heartbeat=datetime.now() - timedelta(minutes=10))
            code, text = self.render(Path(d))
            self.assertEqual(code, 2)                        # nothing finished, nothing beating
            self.assertIn("остановлен", text)

    def test_no_runs(self):
        with TemporaryDirectory() as d:
            code, text = self.render(Path(d))
            self.assertEqual(code, 1)
            self.assertIn("прогонов ещё не было", text)


if __name__ == "__main__":
    unittest.main()
