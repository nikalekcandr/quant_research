"""One command: bring every useful dataset up to date, then write a report you can read later.

    python scripts/download_all.py --dry-run     # план: что качаем, чего не хватает, сколько это часов
    python scripts/download_all.py               # в фон и заниматься своими делами
    python scripts/download_all.py --foreground  # то же, но в этом окне (видно каждую строку)
    python scripts/download_all.py --status      # что получилось

Что качаем и почему именно это:

| источник | гранулярность | что даёт |
|---|---|---|
| binance | 1 минута, 2023 → сейчас | цена, на которой сеттлятся лесенки; источник реализованной волатильности |
| funding | 8 часов | стоимость удержания, режимы рынка |
| perp | 1 минута + базис | то, чего до сих пор не было: перп, базис, связь со спотом |
| deribit | DVOL 1 час + сделки в окне снимка | подразумеваемая волатильность, то, с чем сравниваем лесенку |
| polymarket | дневные лесенки | второй рынок того же вопроса, для сверки |
| macro | календарь | часы релизов — гейт для W03/W04 |
| kalshi | свечи за ВЕСЬ час (было 10 минут) | 6x больше строк на то же число запросов: видно, что происходит до экспирации |
| binance_1s | 1 секунда (по флагу) | сеттлмент (среднее 60 печатей BRTI в последнюю минуту) и исполнение |

Почему у Kalshi окно с 2026-01-01: до 18.03.2026 у биржи была другая сетка страйков ($250 вместо $100) — это
другой контракт. Всё, что раньше, качается отдельной командой `--kalshi-since 2024-12-12` (≈10 часов).

Что скрипт делает сам: общий адаптивный лимитер на 429, порядок «дешёвое сначала», изоляцию сбоев (падает
источник — остальные качаются; падает день — остальные дни качаются), пересборку таблиц в конце и отчёт
`logs/download-<дата>/summary.md` со статусами и ошибками.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import threading
import time
import traceback
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ORDER = ["binance", "funding", "perp", "deribit", "polymarket", "macro", "kalshi", "binance_1s"]
KALSHI_SINCE = "2026-01-01"      # strikes are $100 apart from 18.03.2026; earlier is a different contract
KALSHI_SEC_PER_DAY = 60          # rough: ~15 s for a live day, ~90 s for an archived one


def today_utc() -> str:
    return datetime.utcnow().strftime("%Y-%m-%d")


def missing_days(root: Path, since: str, until: str, subdir: str = "kalshi_kxbtcd_60m") -> list[str]:
    """Calendar days in [since, until) with no file yet in data/raw/<subdir>."""
    d = datetime.strptime(since, "%Y-%m-%d").date()
    stop = datetime.strptime(until, "%Y-%m-%d").date()
    have = {p.name[:10] for p in (Path(root) / "data" / "raw" / subdir).glob("*.csv.gz")}
    out = []
    while d < stop:
        if d.isoformat() not in have:
            out.append(d.isoformat())
        d += timedelta(days=1)
    return out


def plan(root: Path, until: str, since: str | None = None, only: list[str] | None = None,
         kalshi_since: str = KALSHI_SINCE, one_second: str | None = None) -> list[dict]:
    rows = []
    for name in ORDER:
        if only and name not in only:
            continue
        if name == "binance_1s" and not one_second:
            continue
        row = {"source": name, "since": since or "default", "until": until}
        if name == "kalshi":
            start = since or kalshi_since
            row["since"] = start
            row["missing"] = len(missing_days(root, start, until))
        if name == "binance_1s":
            row["since"] = one_second.partition(":")[0] or one_second
        rows.append(row)
    return rows


def estimate_hours(rows: list[dict]) -> float:
    return sum(r.get("missing", 0) for r in rows) * KALSHI_SEC_PER_DAY / 3600


def steps(args):
    """(name, what, runner) for every source this run should touch, cheap ones first."""
    from src.download import binance_1s, kalshi
    from src.download import update as upd

    out = [(n, upd.SOURCES[n][0], upd.SOURCES[n][1])
           for n in ("binance", "funding", "perp", "deribit", "polymarket", "macro") if n in upd.SOURCES]
    out.append(("kalshi", f"Kalshi часовые лесенки, свечи за весь час (было 10 минут)",
                lambda s, u, h: kalshi.download_history(s or args.kalshi_since, u, series="KXBTCD",
                                                        minutes=args.kalshi_minutes, workers=args.workers,
                                                        min_interval=args.min_interval, http=h)))
    if args.one_second:
        since, _, until = args.one_second.partition(":")
        out.append(("binance_1s", "Binance секундные свечи",
                    lambda s, u, h: binance_1s.download_days(since or s, until or u, h)))
    if args.only:
        out = [t for t in out if t[0] in args.only]
    return out


def _p(msg: str = "") -> None:
    try:
        print(f"[{datetime.now():%H:%M:%S}] {msg}" if msg else "", flush=True)
    except BrokenPipeError:            # piping into `head`/`Select-Object -First` must not print a traceback
        pass


class _Tee:
    """Write to the screen and to logs/…/run.log at once, so --status works in both modes."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for st in self.streams:
            try:
                st.write(data)
            except Exception:
                pass

    def flush(self):
        for st in self.streams:
            try:
                st.flush()
            except Exception:
                pass


def run(args, registry=None, http=None, root: Path | None = None, work: Path | None = None) -> dict:
    """Run the download and mirror everything it prints into the run folder."""
    root = Path(root or ROOT)
    if work is None:
        work = root / "logs" / f"download-{datetime.now():%Y%m%d-%H%M%S}"
    work.mkdir(parents=True, exist_ok=True)
    handle = open(work / "run.log", "a", encoding="utf-8", buffering=1)
    original = sys.stdout
    sys.stdout = _Tee(original, handle)
    try:
        return _run_body(args, registry, http, root, work)
    finally:
        sys.stdout = original
        handle.close()


def _run_body(args, registry=None, http=None, root: Path | None = None, work: Path | None = None) -> dict:
    """Do the work. Everything is injectable so the tests never touch the network."""
    from src.download import update as upd
    from src.utils import Http

    root = Path(root or ROOT)
    upd.WORKERS = args.workers
    upd.MIN_INTERVAL = args.min_interval
    http = http or Http(min_interval=args.min_interval)
    registry = registry if registry is not None else steps(args)
    if args.only:
        registry = [t for t in registry if t[0] in args.only]

    work = Path(work)
    until = args.until or today_utc()
    state = {"started": datetime.now().isoformat(timespec="seconds"), "pid": os.getpid(),
             "window": {"until": until}, "sources": [], "build": args.build}
    state_path, summary_path = work / "state.json", work / "summary.md"

    save_lock = threading.Lock()

    def save():
        with save_lock:
            state["heartbeat"] = datetime.now().isoformat(timespec="seconds")
            state_path.write_text(json.dumps(state, ensure_ascii=False, indent=1), encoding="utf-8")

    stop_beat = threading.Event()

    def beat():
        while not stop_beat.wait(20):
            save()

    threading.Thread(target=beat, daemon=True).start()

    rows = plan(root, until, args.since, args.only, args.kalshi_since, args.one_second)
    state["plan"] = rows
    save()

    _p(f"закачка: {len(registry)} источник(ов), окно до {until}, темп {1 / max(args.min_interval, 1e-3):.0f} "
       f"запросов/с, {args.workers} потоков")
    for r in rows:
        if "missing" in r:
            _p(f"  Kalshi: не хватает {r['missing']} суток за {r['since']}..{r['until']} "
               f"≈ {estimate_hours([r]):.1f} ч")
    _p(f"  журнал и отчёт: {work}")
    _p()

    try:
        state["audit_before"] = upd.audit_table().to_csv(index=False)
        (work / "audit_before.csv").write_text(state["audit_before"], encoding="utf-8")
    except Exception as e:
        _p(f"  (аудит до закачки не получился: {e!r})")
    save()

    for name, what, fn in registry:
        row = next((r for r in rows if r["source"] == name), {"since": None})
        t0 = time.time()
        _p(f"=== {name}: {what} ({row.get('since') and str(row['since']) or 'default'} .. {until})")
        try:
            fn(None if row.get("since") in (None, "default") else row["since"], until, http)
            status, err = "ok", ""
        except Exception as e:
            status, err = "failed", f"{type(e).__name__}: {e}"
            _p(f"!!! {name} упал: {err}")
            traceback.print_exc()
        dt = time.time() - t0
        state["sources"].append({"source": name, "what": what, "status": status,
                                 "seconds": round(dt, 1), "error": err[:800]})
        _p(f"=== {name}: {status} за {dt / 60:.1f} мин")
        save()

    built = []
    if args.build:
        from src.download import binance_1s
        builders = dict(upd.BUILDERS)
        if args.one_second:
            builders["binance_1s"] = binance_1s.build_interim
        for name, _what, _fn in registry:
            if name not in builders:
                continue
            t0 = time.time()
            try:
                builders[name]()
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
    stop_beat.set()
    state["finished"] = datetime.now().isoformat(timespec="seconds")
    state["requests"] = dict(http.stats)
    state["final_interval"] = round(http.min_interval, 3)
    save()
    summary_path.write_text(summary_markdown(state), encoding="utf-8")
    _p()
    _p(f"готово. Отчёт: {summary_path}")
    _p("итог: python scripts/download_all.py --status")
    return state


def summary_markdown(state: dict) -> str:
    lines = [f"# Закачка {state.get('started', '')}", ""]
    lines.append(f"Окно: до {state.get('window', {}).get('until', '?')}. "
                 f"Пауза между запросами в конце: {state.get('final_interval', 0):.2f} с.")
    if state.get("requests"):
        lines.append(f"Запросов: {state['requests'].get('requests', 0)}, из них 429: "
                     f"{state['requests'].get('429', 0)}.")
    for r in state.get("plan", []):
        if "missing" in r:
            lines.append(f"Kalshi: не хватало {r['missing']} суток за {r['since']}..{r['until']}.")
    lines += ["", "| источник | что качали | статус | время |", "|---|---|---|---|"]
    for s in state.get("sources", []):
        lines.append(f"| {s['source']} | {s.get('what', '')} | {s['status']} | {s['seconds'] / 60:.1f} мин |")
    if state.get("builds"):
        lines += ["", "## Пересборка таблиц", ""]
        for b in state["builds"]:
            extra = f" — {b.get('error', '')}" if b.get("status") == "failed" else ""
            lines.append(f"- {b['source']}: {b['status']}{extra}")
    bad = [s for s in state.get("sources", []) if s["status"] != "ok"]
    lines += ["", "## Что дальше", ""]
    if bad:
        lines.append("Не всё скачалось. Повторите ту же команду — докачает только недостающее.")
        for s in bad:
            lines.append(f"- **{s['source']}**: {s.get('error', '')[:300]}")
    else:
        lines.append("Всё скачалось. Проверьте: `python -m src.download.update --audit`.")
    lines += ["", "```bash\npython scripts/download_all.py --status\n```"]
    return "\n".join(lines)


PROGRESS_RE = re.compile(
    r"kalshi: (\d+)/(\d+) days done, ([\d.]+) min elapsed, (\d+) requests "
    r"\(([\d.]+)/s\), ([\d.]+) days/min, ETA ([^,]+), (\d+) failed")
STEP_RE = re.compile(r"\[(\d\d:\d\d:\d\d)\] === ([^\s:]+): ([^\n]*)")


def parse_progress(log_text: str) -> dict | None:
    """The last Kalshi progress line turned into numbers (it prints one every 30 s)."""
    hits = PROGRESS_RE.findall(log_text)
    if not hits:
        return None
    done, total, minutes, requests, rps, per_min, eta, failed = hits[-1]
    return {"done": int(done), "total": int(total), "minutes": float(minutes), "requests": int(requests),
            "rps": float(rps), "days_per_min": float(per_min), "eta": eta.strip(), "failed": int(failed)}


def current_step(log_text: str) -> tuple[str, str] | None:
    """The source that started and has not reported back yet, plus the time it started."""
    running = None
    for stamp, name, rest in STEP_RE.findall(log_text):
        if " за " in rest or rest.startswith(("ok ", "failed ")):
            running = None                       # "=== name: ok за ..." closes the step
        else:
            running = (name, stamp)
    return running


def _age(state: dict) -> str:
    try:
        started = datetime.fromisoformat(state["started"])
    except (KeyError, ValueError):
        return "?"
    end = datetime.fromisoformat(state["finished"]) if state.get("finished") else datetime.now()
    secs = (end - started).total_seconds()
    return f"{secs / 3600:.1f} ч" if secs > 5400 else f"{secs / 60:.0f} мин"


def render_status(root: Path) -> int:
    """One screen: is it running, which source is being downloaded, how far, how long is left."""
    runs = sorted((root / "logs").glob("download-*")) if (root / "logs").exists() else []
    if not runs:
        _p("прогонов ещё не было: python scripts/download_all.py")
        return 1
    work = runs[-1]
    state_file, log_file = work / "state.json", work / "run.log"
    log_text = log_file.read_text(encoding="utf-8", errors="replace") if log_file.exists() else ""
    state = json.loads(state_file.read_text(encoding="utf-8")) if state_file.exists() else {}

    alive = False
    if state and not state.get("finished"):
        try:                                     # a heartbeat every 20 s: older means the process is gone
            beat = datetime.fromisoformat(state.get("heartbeat") or state.get("started"))
            alive = (datetime.now() - beat).total_seconds() < 90
        except (TypeError, ValueError):
            alive = False

    head = f"{work.name} — "
    head += ("идёт" if alive else ("завершён" if state.get("finished") else "остановлен/прерван"))
    head += f", {_age(state)}"
    _p(head)
    step = current_step(log_text)
    if step and alive:
        _p(f"  сейчас: {step[0]} (с {step[1]})")
    prog = parse_progress(log_text)
    if prog:
        pct = 100 * prog["done"] / max(prog["total"], 1)
        _p(f"  Kalshi: {prog['done']}/{prog['total']} суток ({pct:.0f}%), {prog['requests']} запросов "
           f"({prog['rps']:.1f}/с), осталось ~{prog['eta']}, упало дней: {prog['failed']}")
    elif step and step[0] == "kalshi" and alive:
        _p("  Kalshi: первая строка прогресса появится примерно через 30 секунд после старта")
    done_sources = state.get("sources", [])
    if done_sources:
        _p("  готово: " + ", ".join(f"{s['source']} {s['seconds'] / 60:.1f} мин" for s in done_sources))
    for s in done_sources:
        if s["status"] != "ok":
            _p(f"  !!! {s['source']}: {s.get('error', '')[:140]}")
    if state.get("requests"):
        _p(f"  запросов всего: {state['requests'].get('requests', 0)}, 429: {state['requests'].get('429', 0)}")
    _p(f"  журнал: {log_file}")
    tail = log_text.splitlines()[-6:]
    if tail:
        _p("  последние строки:")
        for line in tail:
            _p(f"    {line}")
    report = work / "summary.md"
    _p(f"  отчёт: {report}" if report.exists() else "")
    return 0 if (alive or state.get("finished")) else 2


def status(root: Path | None = None, watch: float | None = None) -> int:
    """`--status` once, or `--watch N` — the same screen refreshed every N seconds until the run ends."""
    root = Path(root or ROOT)
    if not watch:
        return render_status(root)
    try:
        while True:
            if sys.stdout.isatty():
                print("\033[2J\033[H", end="")       # clear screen; plain repeated blocks if not a terminal
            print(f"обновление каждые {watch:.0f} с, Ctrl+C — выход\n")
            code = render_status(root)
            if code != 0:
                return code
            time.sleep(watch)
    except KeyboardInterrupt:
        _p("\nвыход из --watch, закачка продолжается в фоне")
        return 0


def spawn(args) -> int:
    work = ROOT / "logs" / f"download-{datetime.now():%Y%m%d-%H%M%S}"
    work.mkdir(parents=True, exist_ok=True)
    log = open(work / "spawn.log", "w", encoding="utf-8")        # noqa: SIM115 (the child keeps it open)
    cmd = [sys.executable, str(Path(__file__)), "--foreground", "--work-dir", str(work)]
    for flag, value in (("--since", args.since), ("--until", args.until),
                        ("--only", ",".join(args.only) if args.only else None),
                        ("--kalshi-since", args.kalshi_since),
                        ("--kalshi-minutes", str(args.kalshi_minutes)),
                        ("--one-second", args.one_second),
                        ("--workers", str(args.workers)), ("--min-interval", str(args.min_interval))):
        if value:
            cmd += [flag, value]
    cmd.append("--build" if args.build else "--no-build")
    kwargs = {"cwd": str(ROOT), "stdout": log, "stderr": subprocess.STDOUT, "stdin": subprocess.DEVNULL}
    if os.name == "nt":
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP | 0x00000008    # DETACHED_PROCESS
    else:
        kwargs["start_new_session"] = True
    proc = subprocess.Popen(cmd, **kwargs)
    _p(f"запущено в фоне (pid {proc.pid}).")
    _p(f"  журнал: {work / 'run.log'}")
    _p("  итог:   python scripts/download_all.py --status")
    _p("  окно можно закрывать; экран гасить можно; машина не уснёт сама (инхибитор сна).")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="одна команда: докачать все нужные данные и отчитаться")
    ap.add_argument("--since", default=None, help="начало окна для всех источников, YYYY-MM-DD")
    ap.add_argument("--until", default=None, help="конец окна, YYYY-MM-DD (по умолчанию: сегодня)")
    ap.add_argument("--only", default=None, help="только эти источники, через запятую")
    ap.add_argument("--kalshi-since", default=KALSHI_SINCE, help=f"начало окна Kalshi (по умолчанию {KALSHI_SINCE})")
    ap.add_argument("--kalshi-minutes", type=int, default=60,
                    help="минут свечей на событие: 60 = весь час (по умолчанию), 10 = как в старом пайплайне")
    ap.add_argument("--one-second", default=None, metavar="SINCE[:UNTIL]",
                    help="ещё и секундные свечи Binance (тяжёлые: ~2 МБ на сутки)")
    ap.add_argument("--workers", type=int, default=6, help="потоков для Kalshi (по умолчанию 6)")
    ap.add_argument("--min-interval", type=float, default=0.10,
                    help="секунд между запросами, общий лимит (по умолчанию 0.10; сам растёт при 429)")
    ap.add_argument("--no-build", dest="build", action="store_false", help="не пересобирать таблицы в конце")
    ap.add_argument("--dry-run", action="store_true", help="только показать план")
    ap.add_argument("--foreground", action="store_true", help="работать в этом окне, не уходить в фон")
    ap.add_argument("--status", action="store_true", help="итог последнего прогона")
    ap.add_argument("--watch", type=float, nargs="?", const=15.0, default=None, metavar="SECONDS",
                    help="обновлять экран статуса каждые N секунд (по умолчанию 15), пока закачка идёт")
    ap.add_argument("--work-dir", default=None, help=argparse.SUPPRESS)
    ap.set_defaults(build=True)
    a = ap.parse_args(argv)
    a.only = [s.strip() for s in a.only.split(",") if s.strip()] if a.only else None

    if a.watch is not None:
        return status(watch=a.watch)
    if a.status:
        return status()
    until = a.until or today_utc()
    if a.dry_run:
        rows = plan(ROOT, until, a.since, a.only, a.kalshi_since, a.one_second)
        _p(f"план на {until}:")
        for r in rows:
            extra = f", не хватает {r['missing']} суток ≈ {estimate_hours([r]):.1f} ч" if "missing" in r else ""
            _p(f"  {r['source']:<12} с {r['since']}{extra}")
        _p(f"\nвсего по Kalshi ≈ {estimate_hours(rows):.1f} ч (остальные источники — минуты)")
        return 0
    if not a.foreground:
        return spawn(a)

    from src.runner import keep_awake
    if keep_awake():
        _p("сон машины заблокирован; экран можно гасить")
    run(a, root=ROOT, work=Path(a.work_dir) if a.work_dir else None)
    return 0


if __name__ == "__main__":
    sys.exit(main())
