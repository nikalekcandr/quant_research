"""One-off background jobs on this computer, started by the lab daemon.

Drop a script into jobs/ (project root). The daemon starts every script that has no jobs/done/<name>.json yet,
one at a time, in its own process (so a restart of the daemon does not stop it), and the wrapper writes
jobs/done/<name>.json (ok / error + traceback) when the script ends. Output goes to logs/jobs/<name>.log.
A job counts as running while its log was written in the last 20 minutes, so a job should log at least that often.
To re-run a job, give the script a new name.

    python -m lab.jobs run jobs/x.py      (what the daemon starts; can also be run by hand)
"""
from __future__ import annotations

import json
import runpy
import subprocess
import sys
import time
import traceback
from datetime import datetime

import config

JOBS = config.ROOT / "jobs"
DONE = JOBS / "done"
LOGS = config.ROOT / "logs" / "jobs"
ALIVE_SEC = 20 * 60
_started: dict = {}


def _running(name: str) -> bool:
    p = _started.get(name)
    if p is not None and p.poll() is None:
        return True
    f = LOGS / f"{name}.log"
    return f.exists() and time.time() - f.stat().st_mtime < ALIVE_SEC and not (DONE / f"{name}.json").exists()


def poll():
    """Start the first pending job if none is running."""
    if not JOBS.exists():
        return
    DONE.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    pending = [f for f in sorted(JOBS.glob("*.py")) if not (DONE / f"{f.stem}.json").exists()]
    # names ending in _now run at once, next to whatever else is running; the others one at a time
    for f in [f for f in pending if f.stem.endswith("_now") and not _running(f.stem)]:
        _start(f)
    queue = [f for f in pending if not f.stem.endswith("_now")]
    if queue and not any(_running(f.stem) for f in queue):
        _start(queue[0])


def _start(f):
    out = open(LOGS / f"{f.stem}.log", "a", encoding="utf-8")
    out.write(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] start {f.name}\n")
    out.flush()
    flags = 0
    if sys.platform == "win32":
        flags = subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP
    _started[f.stem] = subprocess.Popen([sys.executable, "-u", "-m", "lab.jobs", "run", str(f)], cwd=str(config.ROOT),
                                        stdout=out, stderr=subprocess.STDOUT, creationflags=flags)


def run(path: str):
    from pathlib import Path
    f = Path(path)
    DONE.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    rec = {"job": f.name, "started": datetime.now().isoformat(timespec="seconds")}
    try:
        sys.argv = [str(f)]
        runpy.run_path(str(f), run_name="__main__")
        rec["status"] = "ok"
    except SystemExit as e:
        rec["status"] = "ok" if not e.code else f"exit {e.code}"
    except BaseException:
        rec["status"] = "error"
        rec["traceback"] = traceback.format_exc()[-4000:]
        print(rec["traceback"], flush=True)
    rec["minutes"] = round((time.time() - t0) / 60, 1)
    (DONE / f"{f.stem}.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] end {f.name}: {rec['status']} ({rec['minutes']} min)", flush=True)


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "run":
        run(sys.argv[2])
