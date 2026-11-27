"""scripts/get_code.py: which files count as code, how changes are detected, and a full git-backend cycle."""
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("get_code", ROOT / "scripts/get_code.py")
get_code = importlib.util.module_from_spec(spec)
spec.loader.exec_module(get_code)


class FilterTest(unittest.TestCase):
    def test_code_files_pass(self):
        for p in ("src/download/update.py", "studies/wings_ten.py", "tests/test_lit.py", "scripts/get_code.py",
                  "docs/ENCYCLOPEDIA.md", "data/lit/bibliography.csv", "results/studies/wings_capacity.md",
                  "README.md", "config.py", "requirements.txt",
                  "update_code.bat", "update_code.sh"):
            self.assertTrue(get_code.is_code_path(p), p)

    def test_data_and_binaries_do_not(self):
        for p in ("data/raw/kalshi/2026-10-01.csv.gz", "data/interim/panel.parquet",
                  "results/forecasts_kx1h.parquet", "results/studies/o02_audit_rows.csv",
                  "reports/wing_atlas.html", "data/lab/markets/2026-10-29.csv",
                  "results_backup_1015/scores.parquet", "scores.parquet", "../etc/passwd", "src/x.so"):
            self.assertFalse(get_code.is_code_path(p), p)

    def test_size_cap(self):
        self.assertTrue(get_code.is_code_path("data/lit/bibliography.csv", 60_000))
        self.assertFalse(get_code.is_code_path("data/lit/bibliography.csv", 5_000_000))

    def test_blob_sha_matches_git(self):
        data = b"hello\n"
        self.assertEqual(get_code.blob_sha(data), "ce013625030ba8dba906f756967f9e9ca394464a")
        with TemporaryDirectory() as d:
            f = Path(d) / "a.txt"
            f.write_bytes(data)
            out = subprocess.run(["git", "hash-object", str(f)], capture_output=True, text=True)
            if out.returncode == 0:
                self.assertEqual(get_code.blob_sha(data), out.stdout.strip())

    def test_changed_paths_uses_content(self):
        with TemporaryDirectory() as d:
            root = Path(d)
            (root / "same.py").write_bytes(b"x = 1\n")
            (root / "different.py").write_bytes(b"x = 2\n")
            remote = {"same.py": get_code.blob_sha(b"x = 1\n"),
                      "different.py": get_code.blob_sha(b"x = 3\n"),
                      "new.py": get_code.blob_sha(b"x = 4\n")}
            write, _ = get_code.changed_paths(root, remote)
            self.assertEqual(write, ["different.py", "new.py"])

    def test_parse_remote(self):
        self.assertEqual(get_code._parse_remote("https://github.com/nikalekcandr/quant_research.git"),
                         ("nikalekcandr", "quant_research"))
        self.assertEqual(get_code._parse_remote("git@github.com:o/r"), ("o", "r"))
        self.assertIsNone(get_code._parse_remote("https://example.com/o/r"))


class GitBackendTest(unittest.TestCase):
    """A local bare-ish remote: the whole cycle without the network."""

    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.remote = Path(self.tmp.name) / "remote"
        self.work = Path(self.tmp.name) / "work"
        self.remote.mkdir()
        git = ["git", "-C", str(self.remote)]
        subprocess.run(git + ["init", "-q"], check=True)
        subprocess.run(git + ["config", "user.email", "t@t"], check=True)
        subprocess.run(git + ["config", "user.name", "t"], check=True)
        (self.remote / "src").mkdir()
        (self.remote / "src" / "app.py").write_text("print('v1')\n")
        (self.remote / "data" / "raw").mkdir(parents=True)
        (self.remote / "data" / "raw" / "big.bin").write_bytes(b"0" * 100_000)
        (self.remote / "README.md").write_text("# r\n")
        subprocess.run(git + ["add", "-A"], check=True)
        subprocess.run(git + ["commit", "-qm", "v1"], check=True)
        self.branch = subprocess.run(git + ["rev-parse", "--abbrev-ref", "HEAD"],
                                     capture_output=True, text=True).stdout.strip()

    def tearDown(self):
        self.tmp.cleanup()

    def run_sync(self, *extra):
        return get_code.main(["--root", str(self.work), "--remote", str(self.remote),
                              "--branch", self.branch, *extra])

    def test_cycle(self):
        self.assertEqual(self.run_sync(), 0)
        self.assertEqual((self.work / "src" / "app.py").read_text(), "print('v1')\n")
        self.assertTrue((self.work / "README.md").exists())
        self.assertFalse((self.work / "data" / "raw" / "big.bin").exists())   # data are never fetched

        self.assertEqual(self.run_sync(), 0)                                   # idempotent
        self.assertEqual((self.work / "src" / "app.py").read_text(), "print('v1')\n")

        # an update upstream is picked up, an untracked data file is left alone
        (self.remote / "src" / "app.py").write_text("print('v2')\n")
        (self.work / "data").mkdir(exist_ok=True)
        (self.work / "data" / "raw.bin").write_bytes(b"local data")
        subprocess.run(["git", "-C", str(self.remote), "commit", "-qam", "v2"], check=True)
        self.assertEqual(self.run_sync(), 0)
        self.assertEqual((self.work / "src" / "app.py").read_text(), "print('v2')\n")
        self.assertEqual((self.work / "data" / "raw.bin").read_bytes(), b"local data")

    def test_dry_run_writes_nothing(self):
        self.assertEqual(self.run_sync("--dry-run"), 0)
        self.assertFalse((self.work / "src" / "app.py").exists())
        self.assertEqual(self.run_sync(), 0)

    def test_local_edit_is_not_overwritten_without_force(self):
        self.assertEqual(self.run_sync(), 0)
        (self.work / "src" / "app.py").write_text("print('mine')\n")
        (self.remote / "src" / "app.py").write_text("print('v2')\n")
        subprocess.run(["git", "-C", str(self.remote), "commit", "-qam", "v2"], check=True)
        self.assertEqual(self.run_sync(), 2)
        self.assertEqual((self.work / "src" / "app.py").read_text(), "print('mine')\n")
        self.assertEqual(self.run_sync("--force"), 0)
        self.assertEqual((self.work / "src" / "app.py").read_text(), "print('v2')\n")


class HttpBackendTest(unittest.TestCase):
    def test_backend_writes_verified_files(self):
        payload = b"print('from api')\n"
        sha = get_code.blob_sha(payload)

        def fake_get(url, token=None):
            if "/git/trees/" in url:
                return json.dumps({"sha": "c0ffee", "tree": [
                    {"path": "src/app.py", "type": "blob", "sha": sha},
                    {"path": "data/raw/big.bin", "type": "blob", "sha": "x" * 40},
                ]}).encode()
            if "raw.githubusercontent.com" in url:
                raise RuntimeError("cdn blocked in this test")
            return json.dumps({"content": __import__("base64").b64encode(payload).decode()}).encode()

        old = get_code._get
        get_code._get = fake_get
        try:
            with TemporaryDirectory() as d:
                root = Path(d)
                self.assertEqual(get_code.http_backend(root, "https://github.com/o/r.git", "main", False, None), 0)
                self.assertEqual((root / "src" / "app.py").read_bytes(), payload)
                self.assertFalse((root / "data").exists())
                sync = json.loads((root / ".code_sync.json").read_text())
                self.assertEqual(sync["commit"], "c0ffee")
        finally:
            get_code._get = old


if __name__ == "__main__":
    unittest.main()
