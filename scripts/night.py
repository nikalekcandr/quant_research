"""One command for the night: bring the data up to date, then write a report you can read in the morning.

    python scripts/night.py                # запустить в фоне и пойти спать
    python scripts/night.py --dry-run      # только показать план: что и за сколько будет качаться
    python scripts/night.py --foreground   # то же самое, но в этом окне (видно каждую строку)
    python scripts/night.py --status       # утром: что скачалось, что упало, что осталось

Что он делает сам, без флагов:

* окно закачки. Для Kalshi история в репозитории начинается 2025-12-01, а `data/manual/starts.json`
  просит 2024-12-12 (это самая ранняя дата, которая есть у биржи, а не та, что нужна исследованиям).
  Скрипт берёт окно 2025-12-01 .. вчера и не тратит ночь на два лишних года; `--since` перекрывает это.
* темп. Один общий лимитер на все источники, адаптивный: на 429 все потоки ждут, пауза растёт, потом
  сама возвращается. Kalshi идёт в 6 потоков, потому что его узкое место — round-trip.
* порядок. Сначала дешёвые источники (минуты), последним Kalshi (часы): если ночь окажется короткой,
  нужное для панели всё равно будет на диске.
* устойчивость. Сбой одного источника не останавливает остальные, сбой одного дня Kalshi не роняет
  источник, а недокачанные дни просто остаются — следующий запуск их подберёт.
* отчёт. `logs/night-<дата>/`: state.json по ходу дела, summary.md в конце, run.log целиком.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Where each source starts when nobody says otherwise. None = whatever config.py / data/manual/starts.json say.
# Kalshi is the exception on purpose: nothing before 2025-12-01 was ever downloaded, and before 2026-03-18 the
# strike grid was $250 instead of $100 (a different contract), so pulling 2024-12 buys nothing for the studies.
WINDOW = {
    "kalshi": "2025-12-01",
}
ORDER = ["binance", "funding", "deribit", "perp", "polymarket", "macro", "kalshi"]   # cheap first, long last
KALSHI_SEC_PER_DAY = 60          # rough: 15 s for a live day, ~90 s for an archived one


def _iso(dt) -> str:
    return dt.strftime("%Y-%m-%d")


def today_utc() -> str:
    return _iso(datetime.utcnow())


def missing_days(root: Path, since: str, until: str) -> list[str]:
    """Calendar days in [since, until) that have no data/raw/kalshi/YYYY-MM-DD.csv.gz yet."""
    d = datetime.strptime(since, "%Y-%m-%d").date()
    stop = datetime.strptime(until, "%Y-%m-%d").date()
    have = {p.name[:10] for p in (root / "data" / "raw" / "kalshi").glob("*.csv.gz")}
    out = []
    while d < stop:
        if d.isoformat() not in have:
            out.append(d.isoformat())
        d += timedelta(days=1)
    return out


def plan(root: Path, until: str, since: str | None = None, only: list[str] | None = None) -> list[dict]:
    """What will be downloaded: source, its window and (for Kalshi) how many days are missing."""
    rows = []
    for name in ORDER:
        if only and name not in only:
            continue
        start = since or WINDOW.get(name)
        row = {"source": name, "since": start or "default", "until": until}
        if name == "kalshi":
            from config import KALSHI_START                                    # noqa: delayed: config needs root
            start = start or KALSHI_START
            row["since"] = start
            row["missing"] = len(missing_days(root, start, until))
        rows.append(row)
    return rows


def estimate_hours(rows: list[dict]) -> float:
    """Rough wall-clock guess: only Kalshi is predictable enough to quote."""
    days = sum(r.get("missing", 0) for r in rows)
    return days * KALSHI_SEC_PER_DAY / 3600


def _p(msg: str = "") -> None:
    print(msg, flush=True)


def run(args, sources=None, http=None, root: Path | None = None, work: Path | None = None) -> dict:
    """Do the work. `sources`, `http` and `work` are injectable so the tests never touch the network."""
    from src.download import update as upd
    from src.utils import Http

    root = Path(root or ROOT)
    upd.WORKERS = args.workers
    upd.MIN_INTERVAL = args.min_interval
    http = http or Http(min_interval=args.min_interval)

    if work is None:
        work = root / "logs" / f"night-{datetime.now():%Y%m%d-%H%M%S}"
    work.mkdir(parents=True, exist_ok=True)
    state = {"started": datetime.now().isoformat(timespec="seconds"), "pid": os.getpid(),
             "window": {"until": args.until or today_utc()}, "sources": [], "build": args.build}
    state_path, summary_path = work / "state.json", work / "summary.md"

    def save():
        state_path.write_text(json.dumps(state, ensure_ascii=False, indent=1), encoding="utf-8")

    registry = sources if sources is not None else [
        (n, upd.SOURCES[n][0], upd.SOURCES[n][1]) for n in ORDER if n in upd.SOURCES
    ]
    if args.only:                                     # --only keeps the order of ORDER, not of the flag
        registry = [t for t in registry if t[0] in args.only]
    rows = plan(root, state["window"]["until"], args.since, args.only)
    state["plan"] = rows
    save()

    _p(f"ночная закачка: {len(registry)} источник(ов), окно до {state['window']['until']}, "
       f"темп {1 / max(args.min_interval, 1e-3):.0f} запросов/с, {args.workers} потоков")
    for r in rows:
        if "missing" in r:
            _p(f"  Kalshi: {r['missing']} суток нет за {r['since']}..{r['until']} "
               f"≈ {estimate_hours([r]):.1f} ч")
    _p(f"  журнал: {work}")
    _p()

    try:
        state["audit_before"] = upd.audit_table().to_csv(index=False)
        (work / "audit_before.csv").write_text(state["audit_before"], encoding="utf-8")
    except Exception as e:                                    # an audit failure must not stop the download
        _p(f"  (аудит до закачки не получился: {e!r})")
    save()

    for name, what, fn in registry:
        row = next((r for r in rows if r["source"] == name), {"since": "default"})
        t0 = time.time()
        _p(f"=== {name}: {what} ({row['since']} .. {state['window']['until']})")
        try:
            fn(row["since"] if row["since"] != "default" else None, state["window"]["until"], http)
            status, err = "ok", ""
        except Exception as e:
            status, err = "failed", f"{type(e).__name__}: {e}"
            _p(f"!!! {name} упал: {err}")
            traceback.print_exc()
        dt = time.time() - t0
        state["sources"].append({"source": name, "what": what, "status": status, "seconds": round(dt, 1),
                                 "error": err[:800]})
        _p(f"=== {name}: {status} за {dt / 60:.1f} мин")
        save()

    built = []
    if args.build:
        for name, _what, _fn in registry:
            if name not in upd.BUILDERS:
                continue
            t0 = time.time()
            try:
                upd.BUILDERS[name]()
                built.append({"source": name, "status": "ok", "seconds": round(time.time() - t0, 1)})
            except Exception as e:
                built.append({"source": name, "status": "failed", "error": f"{type(e).__name__}: {e}"[:800]})
                _p(f"!!! пересборка {name} не удалась: {e!r}")
    state["builds"] = built

    try:
        state["audit_after"] = upd.audit_table().to_csv(index=False)
        (work / "audit_after.csv").write_text(state["audit_after"], encoding="utf-8")
    except Exception:
        pass
    state["finished"] = datetime.now().isoformat(timespec="seconds")
    state["requests"] = dict(http.stats)
    state["final_interval"] = round(http.min_interval, 3)
    save()
    summary_path.write_text(summary_markdown(state), encoding="utf-8")
    _p()
    _p(f"готово. Отчёт: {summary_path}")
    _p("утром: python scripts/night.py --status")
    return state


def summary_markdown(state: dict) -> str:
    lines = [f"# Ночная закачка {state.get('started', '')}", ""]
    lines.append(f"Окно: до {state.get('window', {}).get('until', '?')}. "
                 f"Темп после закачки: 1 запрос в {state.get('final_interval', 0):.2f} с.")
    if state.get("requests"):
        lines.append(f"Запросов: {state['requests'].get('requests', 0)}, "
                     f"из них 429: {state['requests'].get('429', 0)}.")
    plan = {r["source"]: r for r in state.get("plan", [])}
    for name, r in plan.items():
        if "missing" in r:
            lines.insert(3, f"Kalshi: не хватало {r['missing']} суток за {r['since']}..{r['until']}.")
    lines += ["", "| источник | что качали | статус | время |", "|---|---|---|---|"]
    for s in state.get("sources", []):
        lines.append(f"| {s['source']} | {s.get('what', '')} | {s['status']} | {s['seconds'] / 60:.1f} мин |")
    if state.get("builds"):
        lines += ["", "## Пересборка таблиц", ""]
        for b in state["builds"]:
            extra = f" — {b.get('error', '')}" if b.get("status") == "failed" else ""
            lines.append(f"- {b['source']}: {b['status']}{extra}")
    bad = [s for s in state.get("sources", []) if s["status"] != "ok"]
    lines += ["", "## Что делать утром", ""]
    if bad:
        lines.append("Не всё скачалось. Повторите ту же команду — докачает только недостающее.")
        for s in bad:
            lines.append(f"- **{s['source']}**: {s.get('error', '')[:300]}")
    else:
        lines.append("Всё скачалось. Проверьте: `python -m src.download.update --audit`.")
    lines.append("")
    lines.append("```bash\npython scripts/night.py --status\n```")
    return "\n".join(lines)


def status(root: Path | None = None) -> int:
    """Morning report: the newest run, its per-source result and the tail of its log."""
    root = Path(root or ROOT)
    runs = sorted((root / "logs").glob("night-*")) if (root / "logs").exists() else []
    if not runs:
        _p("прогонов ещё не было: python scripts/night.py")
        return 1
    work = runs[-1]
    state_file, log_file = work / "state.json", work / "run.log"
    _p(f"последний прогон: {work.name}")
    if state_file.exists():
        state = json.loads(state_file.read_text(encoding="utf-8"))
        if state.get("finished"):
            _p(f"  завершён: {state['finished']}")
        else:
            _p("  ещё идёт (или был прерван) — state.json без времени окончания")
        if state.get("plan"):
            for r in state["plan"]:
                extra = f", нет {r['missing']} суток" if "missing" in r else ""
                _p(f"  план: {r['source']} {r['since']}..{r['until']}{extra}")
        for s in state.get("sources", []):
            _p(f"  {s['source']:<12} {s['status']:<7} {s['seconds'] / 60:5.1f} мин"
               + (f"  {s.get('error', '')[:120]}" if s["status"] != "ok" else ""))
        if state.get("requests"):
            _p(f"  запросов: {state['requests'].get('requests', 0)}, 429: {state['requests'].get('429', 0)}, "
               f"пауза {state.get('final_interval', 0):.2f} с")
    if log_file.exists():
        tail = log_file.read_text(encoding="utf-8", errors="replace").splitlines()[-12:]
        _p("  последние строки журнала:")
        for line in tail:
            _p(f"    {line}")
    report = work / "summary.md"
    _p(f"  отчёт: {report}" if report.exists() else "  (отчёт ещё не написан)")
    return 0


def spawn(args) -> int:
    """Start the real work as a detached process and return immediately — the window can be closed."""
    work = ROOT / "logs" / f"night-{datetime.now():%Y%m%d-%H%M%S}"
    work.mkdir(parents=True, exist_ok=True)
    log = open(work / "run.log", "w", encoding="utf-8")                # noqa: SIM115 (lives in the child)
    cmd = [sys.executable, str(Path(__file__)), "--foreground", "--work-dir", str(work)]
    for flag, value in (("--since", args.since), ("--until", args.until),
                        ("--only", ",".join(args.only) if args.only else None),
                        ("--workers", str(args.workers)), ("--min-interval", str(args.min_interval))):
        if value:
            cmd += [flag, value]
    cmd.append("--build" if args.build else "--no-build")     # the child must not re-decide this
    kwargs = {"cwd": str(ROOT), "stdout": log, "stderr": subprocess.STDOUT, "stdin": subprocess.DEVNULL}
    if os.name == "nt":
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP | 0x00000008   # DETACHED_PROCESS
    else:
        kwargs["start_new_session"] = True
    proc = subprocess.Popen(cmd, **kwargs)
    _p(f"запущено в фоне (pid {proc.pid}).")
    _p(f"  журнал: {work / 'run.log'}")
    _p(f"  утром:  python scripts/night.py --status")
    _p("  окно можно закрывать, машину — оставить включённой (экран гасить можно).")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="одна команда: докачать данные за ночь и утром прочитать отчёт")
    ap.add_argument("--since", default=None, help="начало окна, YYYY-MM-DD (по умолчанию: 2025-12-01 для Kalshi)")
    ap.add_argument("--until", default=None, help="конец окна, YYYY-MM-DD (по умолчанию: сегодня)")
    ap.add_argument("--only", default=None, help="только эти источники, через запятую")
    ap.add_argument("--workers", type=int, default=6, help="потоков для Kalshi (по умолчанию 6)")
    ap.add_argument("--min-interval", type=float, default=0.10,
                    help="секунд между запросами, общий лимит (по умолчанию 0.10; растёт сам при 429)")
    ap.add_argument("--no-build", dest="build", action="store_false", help="не пересобирать таблицы в конце")
    ap.add_argument("--dry-run", action="store_true", help="только показать план")
    ap.add_argument("--foreground", action="store_true", help="работать в этом окне, не уходить в фон")
    ap.add_argument("--status", action="store_true", help="отчёт по последнему прогону")
    ap.add_argument("--work-dir", default=None, help=argparse.SUPPRESS)
    ap.set_defaults(build=True)
    a = ap.parse_args(argv)
    a.only = [s.strip() for s in a.only.split(",") if s.strip()] if a.only else None

    if a.status:
        return status()
    until = a.until or today_utc()
    if a.dry_run:
        rows = plan(ROOT, until, a.since, a.only)
        _p(f"план на {until}:")
        for r in rows:
            extra = f", нет {r['missing']} суток ≈ {estimate_hours([r]):.1f} ч" if "missing" in r else ""
            _p(f"  {r['source']:<12} с {r['since']}{extra}")
        _p(f"\nвсего закачки ≈ {estimate_hours(rows):.1f} ч (грубо, только Kalshi поддаётся оценке)")
        return 0
    if not a.foreground:
        return spawn(a)

    from src.runner import keep_awake
    if keep_awake():
        _p("сон машины заблокирован на время закачки; экран можно гасить")
    work = Path(a.work_dir) if a.work_dir else None      # the detached child writes into its own folder
    run(a, root=ROOT, work=work)
    return 0


if __name__ == "__main__":
    sys.exit(main())
