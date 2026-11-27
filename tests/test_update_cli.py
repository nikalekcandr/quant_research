import contextlib
import io
import subprocess
import sys
import unittest
from pathlib import Path

from src.download import update

ROOT = Path(__file__).resolve().parents[1]


class CliTest(unittest.TestCase):
    def run_main(self, argv):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = update.main(argv)
        return code, out.getvalue()

    def test_no_arguments_only_audits_and_never_downloads(self):
        code, text = self.run_main([])
        self.assertEqual(code, 0)
        self.assertIn("source", text)            # the audit table header
        self.assertIn("Ничего не скачано", text)  # the explicit hint
        self.assertIn("--all", text)

    def test_audit_flag_is_the_same_action(self):
        code, text = self.run_main(["--audit"])
        self.assertEqual(code, 0)
        self.assertIn("source", text)

    def test_plan_keeps_download_order(self):
        self.assertEqual(update.plan("kalshi,binance"), ["binance", "kalshi"])
        self.assertEqual(update.plan(None), update.ORDER)

    def test_unknown_source_is_a_clean_exit(self):
        with self.assertRaises(SystemExit):
            update.plan("astrology")

    def test_script_invocation_works_from_another_directory(self):
        """The bug behind `No module named 'config'`: running the file directly must import the project."""
        for cwd in (str(ROOT), str(ROOT.parent)):          # repository root and anywhere else
            done = subprocess.run([sys.executable, str(ROOT / "src/download/update.py"), "--audit"],
                                  cwd=cwd, capture_output=True, text=True, timeout=180)
            self.assertEqual(done.returncode, 0, done.stderr[-800:])
            self.assertIn("source", done.stdout)
            self.assertNotIn("ModuleNotFoundError", done.stderr)

    def test_module_invocation_still_works(self):
        done = subprocess.run([sys.executable, "-m", "src.download.update", "--audit"],
                              cwd=str(ROOT), capture_output=True, text=True, timeout=180)
        self.assertEqual(done.returncode, 0, done.stderr[-800:])


if __name__ == "__main__":
    unittest.main()
