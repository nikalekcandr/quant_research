"""Pack the whole project (code, data, results, reports, live forecasts) into zip parts of <= 90 MB for upload to
GitHub. Skips caches, virtual environments, logs and the export folder itself.   -> _export/part_NN.zip, manifest.csv"""
from __future__ import annotations

import csv
import os
import sys
import zipfile
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "_export"
SKIP_DIRS = {".venv", "venv", "__pycache__", ".git", "_export", "logs", ".pytest_cache", ".ipynb_checkpoints"}
PART_LIMIT = 90e6


def say(m):
    print(f"[{datetime.now():%H:%M:%S}] {m}", flush=True)


def main():
    OUT.mkdir(exist_ok=True)
    for f in OUT.glob("part_*.zip"):
        f.unlink()
    files = []
    for dp, dns, fns in os.walk(ROOT):
        dns[:] = [d for d in dns if d not in SKIP_DIRS]
        for fn in fns:
            if fn.endswith(".pyc"):
                continue
            p = Path(dp) / fn
            files.append(p)
    files.sort()
    say(f"{len(files)} files, {sum(p.stat().st_size for p in files) / 1e6:.0f} MB raw")
    part, zf, written, rows = 0, None, 0.0, []
    for i, p in enumerate(files):
        if zf is None or written > PART_LIMIT:
            if zf:
                zf.close()
            part += 1
            zf = zipfile.ZipFile(OUT / f"part_{part:02d}.zip", "w", zipfile.ZIP_DEFLATED, compresslevel=6)
            written = 0.0
        rel = p.relative_to(ROOT).as_posix()
        try:
            zf.write(p, rel)
        except Exception as e:                    # a file still being written by the lab: skip, log
            say(f"skip {rel}: {e}")
            continue
        info = zf.getinfo(rel)
        written += info.compress_size
        rows.append((rel, p.stat().st_size, part))
        if i % 500 == 0:
            say(f"{i}/{len(files)} files, part {part}")
    zf.close()
    with open(OUT / "manifest.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["path", "bytes", "part"])
        w.writerows(rows)
    for f in sorted(OUT.glob("part_*.zip")):
        say(f"{f.name}: {f.stat().st_size / 1e6:.1f} MB")
    say("done")


if __name__ == "__main__":
    main()
