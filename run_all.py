"""Run the whole project.

    python run_all.py                    download everything, then build, estimate, evaluate, plot, write report
    python run_all.py --skip-download    reuse data/raw (e.g. after an interrupted run or a code change)
    python run_all.py --from models      start from a later step: download, interim, panels, models, evaluate, extensions, alpha, figures, report
    python run_all.py --synthetic        generate fake data in data_synthetic/ and run every step (pipeline test)
    python run_all.py --no-kalshi        skip the Kalshi download (the slowest source)
"""
from __future__ import annotations

import argparse
import time

import config

STEPS = ["download", "interim", "panels", "models", "evaluate", "extensions", "alpha", "figures", "report"]


def use_synthetic_paths():
    """Point every path at data_synthetic/ and results_synthetic/ so fake data never mixes with real data."""
    config.DATA = config.ROOT / "data_synthetic"
    config.RAW = config.DATA / "raw"
    config.INTERIM = config.DATA / "interim"
    config.LIVE = config.DATA / "live"
    config.RESULTS = config.ROOT / "results_synthetic"
    config.TABLES = config.RESULTS / "tables"
    config.FIGURES = config.RESULTS / "figures"
    config.REPORTS = config.ROOT / "results_synthetic"
    config.FORECASTS_LIVE = config.DATA / "forecasts_live"
    config.KALSHI_START = "2026-06-01"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-download", action="store_true")
    ap.add_argument("--from", dest="start", choices=STEPS, default="download")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--no-kalshi", action="store_true")
    args = ap.parse_args()
    if args.synthetic:
        use_synthetic_paths()
    from src import pipeline, report, figures  # imported after the paths are set
    from src.utils import log

    steps = STEPS[STEPS.index(args.start):]
    if args.skip_download and "download" in steps:
        steps.remove("download")
    t0 = time.time()
    m = None
    for step in steps:
        log(f"===== {step} =====")
        if step == "download":
            if args.synthetic:
                from src import synthetic
                synthetic.generate()
            else:
                pipeline.download(skip=("kalshi",) if args.no_kalshi else ())
        elif step == "interim":
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
            report.write(synthetic=args.synthetic)
    log(f"done in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
