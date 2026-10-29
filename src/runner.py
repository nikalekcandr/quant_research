"""Shared driver for backfill.py and daily_update.py: downloads that never stop on one failing source, the
calculation steps of run_all.py, and an optional "staging" mode that builds interim tables and results in
side folders and swaps them in only when everything succeeded.

Why staging: live_once.py reads data/interim (minutes, panels) every hour to make live forecasts. If the nightly
rebuild overwrote those files in place, a forecast made during the rebuild could read a half-written table.
"""
from __future__ import annotations

import os
import shutil
import sys
import time
import traceback
from contextlib import contextmanager
from pathlib import Path

import pandas as pd

import config
from src.utils import ensure, log, utcnow

CALC_STEPS = ["interim", "panels", "models", "evaluate", "extensions", "alpha", "figures", "report"]


# ------------------------------------------------------------------ logging to a file and the console
class Tee:
    def __init__(self, path: Path, also_console: bool = True):
        ensure(path.parent)
        self.f = open(path, "a", encoding="utf-8", buffering=1)
        self.console = sys.__stdout__ if (also_console and sys.__stdout__ is not None) else None

    def write(self, s):
        self.f.write(s)
        if self.console is not None:
            try:
                self.console.write(s)
            except Exception:
                pass

    def flush(self):
        self.f.flush()
        if self.console is not None:
            try:
                self.console.flush()
            except Exception:
                pass


def log_to_file(name: str, also_console: bool = True) -> Path:
    path = config.ROOT / "logs" / f"{name}_{pd.Timestamp.now():%Y%m%d}.log"
    sys.stdout = sys.stderr = Tee(path, also_console)
    return path


def keep_awake():
    """Windows: no idle sleep while this process runs (long downloads), no admin rights needed."""
    if sys.platform != "win32":
        return
    try:
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
    except Exception:
        pass


# ------------------------------------------------------------------ a lock file shared by the long jobs
@contextmanager
def lock(name: str, stale_hours: float = 6):
    """Only one backfill/daily update at a time. A lock older than `stale_hours` (crashed run) is ignored."""
    path = config.DATA / f".{name}.lock"
    ensure(path.parent)
    if path.exists() and time.time() - path.stat().st_mtime < stale_hours * 3600:
        raise RuntimeError(f"{path.name} exists: another run is in progress (delete it if that run crashed)")
    path.write_text(f"{os.getpid()} {pd.Timestamp(utcnow())}", encoding="utf-8")
    try:
        yield
    finally:
        try:
            path.unlink()
        except FileNotFoundError:
            pass


def busy(*names: str) -> bool:
    return any((config.DATA / f".{n}.lock").exists() for n in names)


# ------------------------------------------------------------------ downloads
def download_all(kalshi: bool = True, kalshi_first: str | None = None) -> list[str]:
    """Every source, each in its own try block; the newest data first. Returns the sources that failed."""
    from src.download import binance, deribit, kalshi as kx, polymarket
    jobs = [("binance", "Binance 1-minute candles", lambda: binance.download(config.BINANCE_START)),
            ("dvol", "Deribit DVOL", lambda: deribit.download_dvol(config.DVOL_START)),
            ("polymarket", "Polymarket ladders", lambda: polymarket.download_history(config.POLYMARKET_START)),
            ("deribit", "Deribit option trades", lambda: deribit.download_trades(config.DERIBIT_START))]
    if kalshi:
        # new data first: the out-of-sample period must be complete even if the long backfill is interrupted
        first = kalshi_first or config.NEW_DATA_START
        jobs.append(("kalshi_new", f"Kalshi hourly ladders from {first}", lambda: kx.download_history(first)))
        if kalshi_first is None and pd.Timestamp(config.KALSHI_START) < pd.Timestamp(first):   # backfill only
            jobs.append(("kalshi_old", f"Kalshi hourly ladders {config.KALSHI_START} .. {first}, newest month first",
                         lambda: kalshi_backwards(config.KALSHI_START, first)))
    failed = []
    for key, label, fn in jobs:
        log(f"--- downloading {label}")
        t0 = time.time()
        try:
            fn()
            log(f"--- {label}: ok ({(time.time() - t0) / 60:.1f} min)")
        except Exception as e:
            failed.append(key)
            log(f"!!! {label} failed: {e!r}")
            traceback.print_exc()
    if failed:
        log(f"!!! failed sources: {', '.join(failed)} — the rest continues; re-run later to resume them")
    return failed


def kalshi_backwards(start: str, end: str):
    """Kalshi history month by month from `end` back to `start`: whatever is finished is contiguous with the
    data we already have, so an interrupted backfill still gives a clean (shorter) sample."""
    from src.download import kalshi as kx
    lo, hi = pd.Timestamp(start), pd.Timestamp(end) - pd.Timedelta(days=1)
    m_end = hi
    while m_end >= lo:
        m_start = max(lo, m_end.to_period("M").to_timestamp())
        kx.download_history(f"{m_start:%Y-%m-%d}", f"{m_end:%Y-%m-%d}")
        m_end = m_start - pd.Timedelta(days=1)


# ------------------------------------------------------------------ calculation steps (same as run_all.py)
def run_steps(steps=CALC_STEPS):
    from src import figures, pipeline, report
    m = None
    for step in steps:
        log(f"===== {step} =====")
        t0 = time.time()
        if step == "interim":
            pipeline.interim()
        elif step == "panels":
            m = pipeline.panels()
        elif step == "models":
            pipeline.run_models(m)
        elif step == "evaluate":
            pipeline.run_evaluation()
        elif step == "extensions":
            from src import extensions
            extensions.run()
        elif step == "alpha":
            from src import alpha
            alpha.run()
        elif step == "figures":
            figures.make_all()
        elif step == "report":
            report.write()
        log(f"===== {step}: {time.time() - t0:.0f} s")


# ------------------------------------------------------------------ staging
LIVE_DIRS = {"INTERIM": "interim", "RESULTS": "results", "REPORTS": "reports"}


def stage_paths():
    """Point interim/results/reports at *_next folders (raw data and live forecasts stay shared)."""
    orig = {"INTERIM": config.INTERIM, "RESULTS": config.RESULTS, "TABLES": config.TABLES,
            "FIGURES": config.FIGURES, "REPORTS": config.REPORTS}
    config.INTERIM = orig["INTERIM"].with_name(orig["INTERIM"].name + "_next")
    config.RESULTS = orig["RESULTS"].with_name(orig["RESULTS"].name + "_next")
    config.TABLES = config.RESULTS / "tables"
    config.FIGURES = config.RESULTS / "figures"
    config.REPORTS = (config.RESULTS if orig["REPORTS"] == orig["RESULTS"]
                      else orig["REPORTS"].with_name(orig["REPORTS"].name + "_next"))
    for k in _swap_keys(orig):
        p = getattr(config, k)
        if p.exists():
            shutil.rmtree(p)
        ensure(p)
    # the manual inputs and files the steps do not rebuild are carried over
    return orig


def _swap_keys(orig: dict) -> list[str]:
    keys = ["INTERIM", "RESULTS"]
    if orig["REPORTS"] != orig["RESULTS"]:
        keys.append("REPORTS")
    return keys


def swap_in(orig: dict, retries: int = 30):
    """Replace interim/results/reports by their *_next versions. A file held open by a running live forecast
    makes the rename fail on Windows, so retry for up to a minute."""
    for k in _swap_keys(orig):
        new, live = getattr(config, k), orig[k]
        old = live.with_name(live.name + "_prev")
        for attempt in range(retries):
            try:
                if old.exists():
                    shutil.rmtree(old)
                if live.exists():
                    live.rename(old)
                new.rename(live)
                break
            except OSError as e:
                if attempt == retries - 1:
                    raise
                log(f"swap {live.name}: {e!r}; retrying")
                time.sleep(2)
        shutil.rmtree(old, ignore_errors=True)
    for k, v in orig.items():
        setattr(config, k, v)
    log("swapped the new interim tables and results in")
