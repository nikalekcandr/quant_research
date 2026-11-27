"""Один скрипт: остановить забытые закачки, докачать всё нужное, показать прогресс и проверить результат.

    scripts\\download_all.bat          (двойной клик)  =  python scripts/download_all.py
    python scripts/download_all.py             живая закачка в этом окне, прогресс построчно
    python scripts/download_all.py --dry-run   только план: чего не хватает и сколько это часов
    python scripts/download_all.py --background  уйти в фон (окно можно закрыть)
    python scripts/download_all.py --status      итог последнего прогона, без закачки
    python scripts/download_all.py --watch       то же, обновляясь каждые 15 секунд
    python scripts/download_all.py --verify      только проверить, что уже лежит на диске

Порядок работы (и почему именно такой):

1. **Найти забытые закачки и остановить их.** Два процесса, пишущих одни и те же файлы, — это и мусор в
   данных, и лишний нагрев. Скрипт показывает, что нашёл, и убивает; `--no-kill` это отключает.
2. **Проверить, чего не хватает** — таблица по всем датасетам, прежде чем что-то качать.
3. **Докачать** дешёвое сначала, Kalshi последним. Всё пишется в окно и в `logs/download-<дата>/run.log`.
4. **Пересобрать таблицы** (это единственный шаг, который реально греет процессор, — он объявляется заранее).
5. **Проверить результат** — снова таблица: дата последних данных по каждому источнику и вердикт.

Что качаем:

| источник | гранулярность | зачем |
|---|---|---|
| binance | 1 минута, 2023 → | цена, на которой сеттлятся лесенки; реализованная волатильность |
| funding | 8 часов | стоимость удержания, режимы рынка |
| perp | 1 минута + базис | перп и базис — раньше не качались ни разу |
| deribit | DVOL 1 час + сделки у снимка | подразумеваемая волатильность для сравнения |
| polymarket | дневные лесенки | второй рынок того же вопроса |
| macro | календарь | часы релизов — гейт для W03/W04 |
| kalshi | свечи за ВЕСЬ час события | 6x строк на те же запросы: путь цены до экспирации |
| binance_1s | 1 секунда (по флагу) | сеттлмент и исполнение (тяжёлый: ~2 МБ на сутки) |
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import threading
import time
import traceback
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ORDER = ["binance", "funding", "perp", "deribit", "polymarket", "macro", "kalshi", "binance_1s"]
KALSHI_SINCE = "2026-01-01"      # до 18.03.2026 сетка страйков $250 вместо $100 — другой контракт
KALSHI_SEC_PER_DAY = 60
MARKERS = ("download_all.py", "night.py", "src.download.update", "src.download.kalshi", "src.download.binance",
           "src.download.binance_1s", "src.download.deribit", "src.download.polymarket", "src.download.perp",
           "src.download.funding", "src.download.macro", "src.download.gaps --run", "backfill.py")


def today_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _p(msg: str = "") -> None:
    try:
        print(f"[{datetime.now():%H:%M:%S}] {msg}" if msg else "", flush=True)
    except BrokenPipeError:
        pass


# ------------------------------------------------------------------ plan
def missing_days(root: Path, since: str, until: str, subdir: str = "kalshi_kxbtcd_60m") -> list[str]:
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
            row["since"] = since or kalshi_since
            row["missing"] = len(missing_days(root, row["since"], until))
        if name == "binance_1s":
            row["since"] = one_second.partition(":")[0] or one_second
        rows.append(row)
    return rows


def estimate_hours(rows: list[dict]) -> float:
    return sum(r.get("missing", 0) for r in rows) * KALSHI_SEC_PER_DAY / 3600


def steps(args):
    """(name, what, runner) в порядке выполнения: дешёвое сначала, Kalshi последним."""
    from src.download import binance_1s, kalshi
    from src.download import update as upd

    out = [(n, upd.SOURCES[n][0], upd.SOURCES[n][1])
           for n in ("binance", "funding", "perp", "deribit", "polymarket", "macro") if n in upd.SOURCES]
    out.append(("kalshi", "Kalshi часовые лесенки, все страйки события одним запросом",
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


# ------------------------------------------------------------------ stale processes
def _cmdline_posix(pid: str) -> str:
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError:
        return ""
    return raw.replace(b"\x00", b" ").decode("utf-8", "replace").strip()


def _cmdline_windows() -> dict[int, str]:
    """{pid: command line} через CIM; пусто, если PowerShell недоступен."""
    ps = ("Get-CimInstance Win32_Process | Select-Object ProcessId,CommandLine | "
          "ForEach-Object { \"$($_.ProcessId)`t$($_.CommandLine)\" }")
    try:
        done = subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True,
                              text=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return {}
    out = {}
    for line in done.stdout.splitlines():
        pid, _, cmd = line.partition("\t")
        if pid.strip().isdigit() and cmd.strip():
            out[int(pid)] = cmd.strip()
    return out


def _is_ours(cmd: str) -> bool:
    """Это python, который запускает наш загрузчик? Смотрим только на саму команду.

    Проверять подстроку во всей строке нельзя: `bash -c "... download_all.py ..."` (или cmd.exe,
    который запустил .bat) тоже содержит имя скрипта — так можно убить самого себя или свою оболочку.
    """
    try:
        tokens = shlex.split(cmd, posix=os.name != "nt")
    except ValueError:
        tokens = cmd.split()
    tokens = [t.strip('"\'') for t in tokens]
    if not tokens or "python" not in Path(tokens[0]).name.lower() and Path(tokens[0]).name.lower() not in ("py", "py.exe"):
        return False
    for token in tokens[1:]:
        if token.startswith("-"):
            continue
        return any(m in token for m in MARKERS)      # первый не-ключ = скрипт или модуль
    return False


def list_download_processes() -> list[dict]:
    """Забытые закачки, запущенные раньше: (pid, command line). Себя, родителя и live.py не трогаем."""
    skip = {os.getpid(), os.getppid()}
    found = []
    if os.name == "nt":
        for pid, cmd in _cmdline_windows().items():
            if pid not in skip and _is_ours(cmd):
                found.append({"pid": pid, "cmd": cmd})
    else:
        for entry in Path("/proc").iterdir():
            if not entry.name.isdigit():
                continue
            pid = int(entry.name)
            cmd = _cmdline_posix(entry.name)
            if pid not in skip and cmd and _is_ours(cmd):
                found.append({"pid": pid, "cmd": cmd})
    return sorted(found, key=lambda r: r["pid"])


def kill_processes(procs: list[dict], grace: float = 3.0) -> list[int]:
    """SIGTERM, потом SIGKILL; на Windows — taskkill /F /T."""
    killed = []
    for proc in procs:
        pid = proc["pid"]
        try:
            if os.name == "nt":
                subprocess.run(["taskkill", "/PID", str(pid), "/F", "/T"], capture_output=True, timeout=30)
            else:
                os.kill(pid, signal.SIGTERM)
            killed.append(pid)
        except (OSError, subprocess.SubprocessError) as e:
            _p(f"  не смог остановить pid {pid}: {e}")
    if os.name != "nt" and killed:
        time.sleep(grace)
        for pid in killed:
            try:
                os.kill(pid, signal.SIGKILL)
            except OSError:
                pass
    return killed


def preflight(kill: bool = True) -> int:
    """Найти и остановить забытые закачки: два писателя в одни файлы — это мусор и лишний нагрев."""
    procs = list_download_processes()
    if not procs:
        _p("зависших закачек нет — чисто")
        return 0
    _p(f"нашёл {len(procs)} процесс(ов) от прошлых запусков:")
    for proc in procs:
        _p(f"  pid {proc['pid']}: {proc['cmd'][:150]}")
    if not kill:
        _p("  (оставляю, --no-kill)")
        return len(procs)
    killed = kill_processes(procs)
    _p(f"остановил: {', '.join(map(str, killed))}")
    return len(killed)


# ------------------------------------------------------------------ verify
def audit_frame(root: Path | None = None):
    from src.download import gaps
    return gaps.audit()


def _fmt_last(value) -> str:
    if value is None:
        return "нет данных"
    if isinstance(value, str):
        return value[:16]
    try:
        if value != value:                      # NaT
            return "нет данных"
        return f"{value:%Y-%m-%d %H:%M}"
    except (TypeError, ValueError):
        return str(value)[:16]


def verify(root: Path | None = None, out=None) -> dict:
    """Что лежит на диске: последняя дата по каждому источнику, отставание, вердикт."""
    sink = out or _p
    try:
        df = audit_frame(root)
    except Exception as e:
        sink(f"проверка не удалась: {e!r}")
        return {"ok": False, "stale": [], "error": str(e)}
    sink("")
    sink(f"{'источник':<22}{'последние данные':<20}{'отставание, дн':>15}")
    for rec in df.to_dict("records"):
        gap = rec.get("gap days")
        gap_s = "" if gap is None or gap != gap else f"{float(gap):.0f}"
        sink(f"{str(rec.get('source', '?')):<22}{_fmt_last(rec.get('last')):<20}{gap_s:>15}")
    downloadable = df[df["command"].astype(str).str.startswith("python")]
    stale = downloadable[downloadable["gap days"].fillna(0) > 2]
    names = [str(r.source) for r in stale.itertuples()]
    if names:
        sink(f"\nотстаёт: {', '.join(names)} — запустите скрипт ещё раз, он докачает только это")
    else:
        sink("\nвердикт: всё скачиваемое — свежее")
    return {"ok": not names, "stale": names}


# ------------------------------------------------------------------ run
class _Tee:
    """Пишет и на экран, и в logs/…/run.log, поэтому --status работает в обоих режимах."""

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
    """Закачка + зеркало всего вывода в папку прогона."""
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
    from src.download import update as upd
    from src.utils import Blocked, Http

    root = Path(root or ROOT)
    work = Path(work)
    upd.WORKERS = args.workers
    upd.MIN_INTERVAL = args.min_interval
    http = http or Http(min_interval=args.min_interval)
    registry = registry if registry is not None else steps(args)
    if args.only:
        registry = [t for t in registry if t[0] in args.only]

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
    _p(f"  журнал: {work / 'run.log'}")
    _p()

    try:
        state["audit_before"] = audit_frame(root).to_csv(index=False)
        (work / "audit_before.csv").write_text(state["audit_before"], encoding="utf-8")
    except Exception as e:
        _p(f"  (проверка до закачки не получилась: {e!r})")
    save()

    for name, what, fn in registry:
        row = next((r for r in rows if r["source"] == name), {"since": None})
        t0 = time.time()
        since = None if row.get("since") in (None, "default") else row["since"]
        _p(f"=== {name}: {what} ({since or 'по умолчанию'} .. {until})")
        try:
            fn(since, until, http)
            status_, err = "ok", ""
        except Blocked as e:
            status_, err = "skipped", str(e)
            _p(f"--- {name}: сервер не пускает эту сеть ({err})")
            _p("    это не потеря данных: календарь релизов уже лежит в data/manual/us_macro_calendar.csv")
        except Exception as e:
            status_, err = "failed", f"{type(e).__name__}: {e}"
            _p(f"!!! {name} упал: {err}")
            traceback.print_exc()
        dt = time.time() - t0
        state["sources"].append({"source": name, "what": what, "status": status_,
                                 "seconds": round(dt, 1), "error": err[:800]})
        _p(f"=== {name}: {status_} за {dt / 60:.1f} мин")
        save()

    built = []
    if args.build and registry:
        from src.download import binance_1s
        builders = dict(upd.BUILDERS)
        if args.one_second:
            builders["binance_1s"] = binance_1s.build_interim
        todo = [n for n, _w, _f in registry if n in builders]
        if todo:
            _p()
            _p(f"=== пересборка таблиц ({', '.join(todo)}) — это единственный шаг, который греет процессор")
        for name in todo:
            t0 = time.time()
            try:
                builders[name]()
                built.append({"source": name, "status": "ok", "seconds": round(time.time() - t0, 1)})
            except Exception as e:
                built.append({"source": name, "status": "failed", "error": f"{type(e).__name__}: {e}"[:800]})
                _p(f"!!! пересборка {name} не удалась: {e!r}")
    state["builds"] = built

    _p()
    _p("=== проверка: что теперь лежит на диске")
    state["verify"] = verify(root)
    try:
        state["audit_after"] = audit_frame(root).to_csv(index=False)
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
    _p("итог без повторной закачки: python scripts/download_all.py --status")
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
    ver = state.get("verify") or {}
    lines += ["", "## Проверка", ""]
    lines.append("всё свежее" if ver.get("ok") else f"отстаёт: {', '.join(ver.get('stale', []))}")
    bad = [s for s in state.get("sources", []) if s["status"] not in ("ok", "skipped")]
    skipped = [s for s in state.get("sources", []) if s["status"] == "skipped"]
    if skipped:
        lines += ["", "Пропущено (сервер не пускает сеть, данные уже есть): "
                  + ", ".join(s["source"] for s in skipped)]
    lines += ["", "## Что дальше", ""]
    if bad:
        lines.append("Не всё скачалось — повторите ту же команду, она докачает только недостающее.")
        for s in bad:
            lines.append(f"- **{s['source']}**: {s.get('error', '')[:300]}")
    else:
        lines.append("Все источники скачались без ошибок.")
    lines += ["", "```bash\npython scripts/download_all.py --status\n```"]
    return "\n".join(lines)


# ------------------------------------------------------------------ status
PROGRESS_RE = re.compile(
    r"kalshi: (\d+)/(\d+) days done, ([\d.]+) min elapsed, (\d+) requests \(([\d.]+)/s\)"
    r"(?:, gap ([\d.]+) s)?(?:, 429: (\d+))?, ETA ([^,]+), (\d+) failed")
STEP_RE = re.compile(r"\[(\d\d:\d\d:\d\d)\] === ([^\s:]+): ([^\n]*)")


def parse_progress(log_text: str) -> dict | None:
    hits = PROGRESS_RE.findall(log_text)
    if not hits:
        return None
    done, total, minutes, requests, rps, gap, n429, eta, failed = hits[-1]
    return {"done": int(done), "total": int(total), "minutes": float(minutes), "requests": int(requests),
            "rps": float(rps), "gap": float(gap) if gap else None, "n429": int(n429) if n429 else 0,
            "eta": eta.strip(), "failed": int(failed)}


def current_step(log_text: str) -> tuple[str, str] | None:
    running = None
    for stamp, name, rest in STEP_RE.findall(log_text):
        if " за " in rest or rest.startswith(("ok ", "failed ")):
            running = None
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
        try:
            beat = datetime.fromisoformat(state.get("heartbeat") or state.get("started"))
            alive = (datetime.now() - beat).total_seconds() < 90
        except (TypeError, ValueError):
            alive = False

    head = f"{work.name} — "
    head += "идёт" if alive else ("завершён" if state.get("finished") else "остановлен/прерван")
    head += f", {_age(state)}"
    _p(head)
    step = current_step(log_text)
    if step and alive:
        _p(f"  сейчас: {step[0]} (с {step[1]})")
    prog = parse_progress(log_text)
    if prog:
        pct = 100 * prog["done"] / max(prog["total"], 1)
        throttle = ""
        if prog.get("gap") or prog.get("n429"):
            throttle = (f", пауза {prog['gap']:.2f} с" if prog.get("gap") else "") + \
                       (f", 429: {prog['n429']}" if prog.get("n429") else "")
        _p(f"  Kalshi: {prog['done']}/{prog['total']} суток ({pct:.0f}%), {prog['requests']} запросов "
           f"({prog['rps']:.1f}/с){throttle}, осталось ~{prog['eta']}, упало дней: {prog['failed']}")
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
    ver = state.get("verify")
    if ver:
        _p("  проверка: " + ("всё свежее" if ver.get("ok") else "отстаёт: " + ", ".join(ver.get("stale", []))))
    _p(f"  журнал: {log_file}")
    tail = log_text.splitlines()[-6:]
    if tail:
        _p("  последние строки:")
        for line in tail:
            _p(f"    {line}")
    report = work / "summary.md"
    if report.exists():
        _p(f"  отчёт: {report}")
    return 0 if (alive or state.get("finished")) else 2


def status(root: Path | None = None, watch: float | None = None) -> int:
    root = Path(root or ROOT)
    if not watch:
        return render_status(root)
    try:
        while True:
            if sys.stdout.isatty():
                print("\033[2J\033[H", end="")
            print(f"обновление каждые {watch:.0f} с, Ctrl+C — выход\n")
            if render_status(root) != 0:
                return 0
            time.sleep(watch)
    except KeyboardInterrupt:
        _p("\nвыход из --watch")
        return 0


# ------------------------------------------------------------------ background
def spawn(args) -> int:
    work = ROOT / "logs" / f"download-{datetime.now():%Y%m%d-%H%M%S}"
    work.mkdir(parents=True, exist_ok=True)
    log = open(work / "spawn.log", "w", encoding="utf-8")      # noqa: SIM115 (the child keeps it open)
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
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP | 0x00000008
    else:
        kwargs["start_new_session"] = True
    proc = subprocess.Popen(cmd, **kwargs)
    _p(f"запущено в фоне (pid {proc.pid}).")
    _p(f"  журнал: {work / 'run.log'}")
    _p("  смотреть: python scripts/download_all.py --watch")
    _p("  окно можно закрывать; экран гасить можно.")
    return 0


def selftest() -> int:
    """Прогон всей машинерии без сети: два фейковых источника, один падает."""
    from types import SimpleNamespace

    a = SimpleNamespace(since=None, until="2026-11-27", only=None, kalshi_since=KALSHI_SINCE,
                        kalshi_minutes=60, one_second=None, workers=2, min_interval=0.0, build=False)
    calls = []

    def ok(name):
        def fn(since, until, http):
            calls.append(name)
        return fn

    def boom(since, until, http):
        raise RuntimeError("faked failure")

    class FakeHttp:
        stats = {"requests": 0, "429": 0, "errors": 0}
        min_interval = 0.0

    import tempfile
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        state = run(a, registry=[("alpha", "ok source", ok("alpha")),
                                 ("beta", "broken source", boom)], http=FakeHttp(), root=root,
                    work=root / "logs" / "download-selftest")
        assert state["sources"][0]["status"] == "ok" and state["sources"][1]["status"] == "failed", state
        assert (root / "logs" / "download-selftest" / "summary.md").exists()
    _p("selftest ok: машинерия закачки, отчёта и изоляции сбоев работает")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="одна команда: докачать все нужные данные, показать прогресс и проверить")
    ap.add_argument("--since", default=None, help="начало окна для всех источников, YYYY-MM-DD")
    ap.add_argument("--until", default=None, help="конец окна, YYYY-MM-DD (по умолчанию: сегодня)")
    ap.add_argument("--only", default=None, help="только эти источники, через запятую")
    ap.add_argument("--kalshi-since", default=KALSHI_SINCE, help=f"начало окна Kalshi (по умолчанию {KALSHI_SINCE})")
    ap.add_argument("--kalshi-minutes", type=int, default=60,
                    help="минут свечей на событие: 60 = весь час (по умолчанию), 10 = как раньше")
    ap.add_argument("--one-second", default=None, metavar="SINCE[:UNTIL]",
                    help="ещё и секундные свечи Binance (~2 МБ на сутки)")
    ap.add_argument("--workers", type=int, default=6, help="потоков для Kalshi (по умолчанию 6)")
    ap.add_argument("--min-interval", type=float, default=0.10,
                    help="секунд между запросами, общий лимит (по умолчанию 0.10; сам растёт при 429)")
    ap.add_argument("--build", dest="build", action="store_true", help="пересобрать таблицы в конце (так по умолчанию)")
    ap.add_argument("--no-build", dest="build", action="store_false", help="не пересобирать таблицы")
    ap.add_argument("--no-kill", dest="kill", action="store_false", help="не останавливать забытые закачки")
    ap.add_argument("--dry-run", action="store_true", help="только план")
    ap.add_argument("--foreground", action="store_true", help="работать в этом окне (так по умолчанию)")
    ap.add_argument("--background", action="store_true", help="уйти в фон")
    ap.add_argument("--status", action="store_true", help="итог последнего прогона")
    ap.add_argument("--watch", type=float, nargs="?", const=15.0, default=None, metavar="SECONDS",
                    help="обновлять экран статуса каждые N секунд (по умолчанию 15)")
    ap.add_argument("--verify", action="store_true", help="только проверить, что лежит на диске")
    ap.add_argument("--selftest", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--work-dir", default=None, help=argparse.SUPPRESS)
    ap.set_defaults(build=True, kill=True)
    a = ap.parse_args(argv)
    a.only = [s.strip() for s in a.only.split(",") if s.strip()] if a.only else None

    if a.selftest:
        return selftest()
    if a.watch is not None:
        return status(watch=a.watch)
    if a.status:
        return status()
    if a.verify:
        return 0 if verify(ROOT)["ok"] else 2
    until = a.until or today_utc()
    if a.dry_run:
        rows = plan(ROOT, until, a.since, a.only, a.kalshi_since, a.one_second)
        _p(f"план на {until}:")
        for r in rows:
            extra = f", не хватает {r['missing']} суток ≈ {estimate_hours([r]):.1f} ч" if "missing" in r else ""
            _p(f"  {r['source']:<12} с {r['since']}{extra}")
        _p(f"\nвсего по Kalshi ≈ {estimate_hours(rows):.1f} ч (остальные источники — минуты)")
        return 0
    if a.background:
        preflight(a.kill)
        return spawn(a)

    _p("=== шаг 1: остановить забытые закачки")
    preflight(a.kill)
    from src.runner import keep_awake
    if keep_awake():
        _p("сон машины заблокирован; экран гасить можно")
    return 0 if run(a, root=ROOT, work=Path(a.work_dir) if a.work_dir else None) else 0


if __name__ == "__main__":
    sys.exit(main())
