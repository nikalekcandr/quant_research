"""Один запуск — все исследования, какие позволяют данные: прогресс построчно, сбои изолированы, итог в один файл.

    scripts\\study_all.bat                 (двойной клик)  =  python scripts/study_all.py
    python scripts/study_all.py --dry-run        что будет запущено и сколько это примерно минут
    python scripts/study_all.py --no-heavy       без тяжёлых (машинное обучение, ёмкость, опционные прогоны)
    python scripts/study_all.py --only wings_sleep,wings_alpha    только эти
    python scripts/study_all.py --skip wings_ml,test_100_ideas    всё, кроме этих
    python scripts/study_all.py --list           список исследований с оценками
    python scripts/study_all.py --status         итог последнего запуска
    python scripts/study_all.py --group крылья   только одна группа
    python scripts/study_all.py --rerun          пересчитать всё, даже если данные и код не менялись
    python scripts/study_all.py --no-check       без проверки импорта в начале (экономит ~30 с)

Повторный запуск не повторяет работу: если ни данные (`data/raw`, `data/lab`, `data/lit`), ни код
(`studies/`, `src/`, `scripts/`) не менялись с прошлого успешного прогона, исследование пропускается с
пометкой «пропущено (нет изменений)». Оценки времени берутся из фактических прогонов
(`data/interim/study_times.json`), а не из констант в реестре. Перед стартом все исследования
проверяются на импорт — сломанный пакет видно за 15 секунд, а не через 80 минут.

Как это устроено:

1. **Панель.** Если кэш панели крыльев старше самых свежих сырых свечей, он удаляется: иначе исследования
   молча прочитают вчерашние данные. Об этом печатается строка, чтобы не было сюрпризов.
2. **Каждое исследование — отдельный процесс.** Падение одного не трогает остальные; вывод идёт в окно и в
   `logs/studies-<дата>/<исследование>.log`. Таймаут на одно исследование — 45 минут.
3. **Итог.** `logs/studies-<дата>/summary.md`: что прошло, что нет и почему, сколько заняло, какие файлы в
   `results/` и `reports/` обновились, и последние строки вывода каждого исследования (обычно там вывод).

Почему исследования могут не пройти — и что скрипт об этом пишет:

| подпись в итоге | что это значит |
|---|---|
| `нет данных: <путь>` | исследованию нужен датасет, которого на диске нет (например, печатные ленты KXBTC15M) |
| `нет пакета: <имя>` | не хватает библиотеки: `pip install <имя>` |
| `убит (память)` | процесс убит системой: не хватило памяти, закройте браузер и запустите `--only <это>` |
| `нет данных в окне` | данные есть, но не в том окне, которое нужно этому исследованию |
| `ошибка: ...` | настоящая ошибка в коде — её видно целиком в логе исследования |

Ничего не выду мывается: если данных нет, исследование помечается пропущенным, а не «пройденным».
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PANEL_CACHE = ROOT / "data" / "interim" / "wing_panel.parquet"
TIMES_FILE = ROOT / "data" / "interim" / "study_times.json"      # фактические длительности прогонов
STATE_FILE = ROOT / "data" / "interim" / "study_state.json"      # что уже посчитано на этих данных и коде
INPUT_DIRS = ("data/raw", "data/lab", "data/lit")                # входы: меняются — пересчитываем
CODE_DIRS = ("studies", "src", "scripts")
LADDER_DIR = ROOT / "data" / "raw" / "kalshi_kxbtcd_60m"
PER_STUDY_TIMEOUT = 45 * 60

# Ребёнку — UTF-8 и предупреждения «по разу». Windows с русской локалью отдаёт дочернему процессу cp1251,
# если тот пишет в pipe: на символе ¢ (в названиях колонок — «¢/час») print падал с UnicodeEncodeError,
# и это выглядело как ошибка исследования. PYTHONWARNINGS=once убирает и заливку одинаковыми
# предупреждениями pandas (Pandas4Warning про dayofweek повторялся больше пятнадцати минут подряд).
CHILD_ENV = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8", "PYTHONWARNINGS": "once"}

# (исследование, группа, что показывает, оценка в секундах, тяжёлое)
REGISTRY = [
    # --- панель: то, из чего считаются крылья
    ("wing_panel", "панель", "собрать панель крыльев из свечей (кэш для остальных)", 180, True),
    # --- крылья: основная линия работы
    ("wings_ladder", "крылья", "продажа дешёвых крыльев на часовой лесенке", 95, True),
    ("wings_calibration", "крылья", "калибровка дешёвой стороны крыла по архивным исходам", 60, False),
    ("wings_sleep", "крылья", "сон, пропуски и залипание книги: смещает ли это результат", 25, False),
    ("wings_alpha", "крылья", "относительная стоимость внутри лесенки, DSR/PBO", 30, False),
    ("wings_ten", "крылья", "десять профессиональных проверок на записанных лентах", 60, False),
    ("wings_scale", "крылья", "семь способов заработать больше: комиссии, выходы, спреды", 30, False),
    ("wings_accuracy", "крылья", "насколько точна вероятность крыла и платит ли точность", 60, False),
    ("wings_capacity", "крылья", "лестница ёмкости: сколько модель выдерживает данных", 200, True),
    ("wings_ml", "крылья", "добавляют ли градиентный бустинг и нейросеть что-то к логиту", 180, True),
    ("wings_neighbors", "крылья", "цена по соседним страйкам и чувствительность к исполнению", 40, False),
    ("wings_execution", "крылья", "как исполнять: тейкером, мейкером или ждать", 25, False),
    ("wings_maker", "крылья", "стоять в книге и ждать покупателя лотереи", 1500, True),
    ("wings_regime", "крылья", "пауза по волатильности и аудит реальных входов W04/W06", 25, False),
    ("wings_regime_ml", "крылья", "хвосты по режимам против прямой классификации крыла", 25, False),
    ("wings_two_sided", "крылья", "согласие и противоположные стороны, хронологический реплей", 30, False),
    ("wings_old_replay", "крылья", "старые двусторонние правила на архивных котировках", 25, False),
    ("wings_varfix", "крылья", "какая модель дисперсии соответствует живой лаборатории", 30, False),
    ("wings_odds_pool", "крылья", "пул лог-шансов, замороженный до сентября", 30, False),
    ("wings_scale_in", "крылья", "ступенчатые входы на сигналах W04 и снимках книги", 25, False),
    ("wings_weekend2", "крылья", "выходные: одна сторона против одной на час (W11 против W08)", 25, False),
    ("wings_edge_timing", "крылья", "политики остановки edge на полной часовой лесенке", 30, False),
    ("wing_consensus", "крылья", "консервативные прогнозы на одинаковых возможностях", 30, False),
    ("wing_filters", "крылья", "фиксированные гейты на модели соседей", 40, False),
    ("wing_shapes", "крылья", "десять кривых вероятности leave-one-strike-out", 40, False),
    ("wing_sizing", "крылья", "десять политик размера на одних и тех же сигналах", 30, False),
    ("wing_stopping", "крылья", "десять причинных правил остановки", 60, False),
    ("wing_ten", "крылья", "десять альтернатив на замороженных ноябрьских свечах", 60, False),
    ("wing_capacity_ml", "крылья", "ёмкость показанных уровней, метки исполнения, HMM", 90, True),
    ("wing_ledger_capacity", "крылья", "ёмкость верхней строки книги на реальных сигналах W01–W12", 40, False),
    ("wing_quant_replay", "крылья", "CAViaR, цифровые опционы SVI, OFI/очередь, два транша", 60, True),
    ("wing_quant_archive", "крылья", "независимый тест CAViaR на наблюдаемых свечах архива", 45, False),
    ("wings_flow", "крылья", "кто платит премию крыла: печатные ленты Kalshi", 40, False),
    ("wing_sleep_replay", "крылья", "октябрьский отсеч и ноябрьский реплей по минутам", 45, False),
    ("w04_entry_stress", "крылья", "стресс продолжения на одну минуту для входов W04", 30, False),
    # --- волатильность и общий контур
    ("vol_horserace", "волатильность", "гонка моделей волатильности на минутах Binance", 90, True),
    ("invariance_vol", "волатильность", "прогнозы волатильности по микроструктуре", 60, True),
    ("gbm_kx1h", "волатильность", "градиентный бустинг на трёх годах минут Binance", 120, True),
    ("short_lags", "волатильность", "дисперсия последнего часа и короткие лаги", 40, False),
    ("test_100_ideas", "волатильность", "сто идей и моделей прогноза волатильности", 200, True),
    ("microstructure_leadlag", "волатильность", "ведёт ли спот короткие бинарные контракты", 30, False),
    ("brti_proxy", "волатильность", "насколько наш спот близок к CF BRTI", 30, False),
    ("funding_carry", "волатильность", "фандинг и базис: предсказывает ли экстремум дрейф", 20, False),
    ("carry_portfolio", "волатильность", "керри как второй рукав портфеля", 20, False),
    # --- опционы Deribit
    ("gamma_intraday", "опционы", "внутридневная гамма на дневных опционах Deribit", 200, True),
    ("o04_straddle", "опционы", "бэктест ATM-стрэддла ближайшей дневной экспирации", 200, True),
    ("o04_report", "опционы", "таблицы для текста O04", 30, False),
    ("options_arb_audit", "опционы", "аудит и строгий реплей «арбитражей» O01", 60, False),
    # --- Kalshi 15 минут
    ("kx15_study", "Kalshi 15m", "KXBTC15M: печатные ленты, кто платит тейкеру", 60, False),
    ("kx15_flow_deep", "Kalshi 15m", "глубже по 15-минутным печатям", 40, False),
    ("kx15_makers", "Kalshi 15m", "кто зарабатывает на 15-минутных рынках", 40, False),
    ("kx_15m_vs_hourly", "Kalshi 15m", "15-минутный рынок против часового на одном сеттлменте", 90, True),
    ("kx_pm_15m_arb", "Kalshi 15m", "Kalshi 15m против Polymarket 15m в одном квартале", 40, False),
    ("tape_strategies", "Kalshi 15m", "бэктесты стратегий по печатям KXBTC15M", 40, False),
    # --- отчёты
    ("wing_atlas", "отчёты", "интерактивные графики ноябрьского исследования", 60, False),
    ("wing_public_report", "отчёты", "публичный снимок Alpha Lab из агрегатов", 30, False),
    ("gen_ideas100_doc", "отчёты", "docs/IDEAS_100.md из results/ideas100", 10, False),
]

# Загрузчики: это не исследования, им место в scripts/download_all.py
NOT_STUDIES = {"wing_gap_download", "wing_forecast_settlements", "quant_inputs"}


def _p(msg: str = "") -> None:
    try:
        print(f"[{datetime.now():%H:%M:%S}] {msg}" if msg else "", flush=True)
    except BrokenPipeError:
        pass


def _mmss(sec: float) -> str:
    if sec < 1:
        return "мгновенно"
    if sec < 90:
        return f"{sec:.0f} с"
    return f"{sec / 60:.1f} мин"


def code_stamp() -> str:
    """Короткая метка версии кода: commit, если есть .git, иначе хеш файлов.

    Нужна ровно для одного вопроса — «та ли это версия, что я обновил»: она печатается в первой строке
    прогона и в итоговом файле, поэтому «запустил старый код» видно сразу, а не через час.
    """
    try:
        r = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=str(ROOT), capture_output=True,
                           text=True, timeout=10)
        if r.returncode == 0 and r.stdout.strip():
            return r.stdout.strip()
    except Exception:                                     # noqa: BLE001 — git не обязателен
        pass
    import hashlib

    h = hashlib.sha256()
    for d in CODE_DIRS:
        for f in sorted((ROOT / d).rglob("*.py")):
            h.update(f"{f.relative_to(ROOT)}:{f.stat().st_mtime_ns}".encode())
    return "b" + h.hexdigest()[:8]


def input_stamp() -> str:
    """Отпечаток входных данных: путь+размер+mtime всех файлов в data/raw, data/lab, data/lit.

    Дороже хеширования содержимого (данные — 375 МБ), но честно отвечает на вопрос «данные с прошлого раза
    изменились?», не читая их целиком.
    """
    import hashlib

    h = hashlib.sha256()
    for d in INPUT_DIRS:
        base = ROOT / d
        if not base.exists():
            continue
        for f in sorted(base.rglob("*")):
            if f.is_file() and not f.name.startswith("."):
                st = f.stat()
                h.update(f"{f.relative_to(ROOT)}:{st.st_size}:{st.st_mtime_ns}".encode())
    return h.hexdigest()[:16]


def watermark() -> str:
    """Код + данные. Пока отпечаток тот же, успешные исследования повторять незачем."""
    return f"{code_stamp()}/{input_stamp()}"


def load_json(path: Path) -> dict:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:                                     # noqa: BLE001 — файла нет или он битый
        return {}


def _history(entry) -> list[dict]:
    """История прогонов в одном формате: [{"s": секунды, "ok": был ли прогон успешным}]."""
    out = []
    for h in entry or []:
        if isinstance(h, dict):
            out.append({"s": float(h.get("s", 0)), "ok": bool(h.get("ok", True))})
        elif isinstance(h, (int, float)):
            out.append({"s": float(h), "ok": True})         # старый формат — просто число
    return out


def record_time(name: str, seconds: float, ok: bool = True) -> None:
    """Запомнить фактическую длительность: оценки в реестре стареют, факт — нет."""
    times = load_json(TIMES_FILE)
    times[name] = [{"s": round(seconds, 1), "ok": bool(ok)}, *_history(times.get(name))[:4]]
    TIMES_FILE.parent.mkdir(parents=True, exist_ok=True)
    TIMES_FILE.write_text(json.dumps(times, ensure_ascii=False, indent=1), encoding="utf-8")


def learned_estimate(row: dict, times: dict) -> tuple[float, bool]:
    """(секунды, из_факта). Медиана последних успешных прогонов, иначе константа из реестра.

    Именно успешных: упавшее за секунду исследование не должно обещать, что в следующий раз оно быстрое.
    """
    hist = _history(times.get(row["name"])) if times else []
    ok = sorted(h["s"] for h in hist if h["ok"])[:3] or sorted(h["s"] for h in hist)[:3]
    if not ok:
        return float(row["est"]), False
    return float(ok[len(ok) // 2]), True


def apply_estimates(rows: list[dict]) -> list[dict]:
    times = load_json(TIMES_FILE)
    for r in rows:
        r["est"], r["est_fact"] = learned_estimate(r, times)
    return rows


def pending(rows: list[dict], stamp: str, rerun: bool) -> tuple[list[dict], list[dict]]:
    """Разделить на (надо считать, можно пропустить) по отпечатку прошлого успешного прогона."""
    if rerun:
        return rows, []
    state = load_json(STATE_FILE)
    todo, skip = [], []
    for r in rows:
        done = state.get(r["name"], {})
        if done.get("status") == "ok" and done.get("stamp") == stamp:
            skip.append(r)
        else:
            todo.append(r)
    return todo, skip


def remember(row: dict) -> None:
    """Отметить прогон под текущим отпечатком (атомарно: переживёт сбой посреди записи)."""
    state = load_json(STATE_FILE)
    entry = state.get(row["name"], {})
    entry.update({"status": row["status"], "stamp": row.get("stamp", ""),
                  "when": datetime.now().isoformat(timespec="seconds"),
                  "seconds": row.get("seconds", 0)})
    state[row["name"]] = entry
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATE_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(STATE_FILE)


def import_check(names: list[str], workers: int = 8) -> dict[str, str]:
    """Импортируется ли каждое исследование — параллельно, за секунды вместо 80-минутного прогона.

    Ловит «нет пакета», синтаксис и несовместимость с этой версией pandas до старта: такие исследования не
    запускаются вовсе, а сразу попадают в итог с причиной.
    """
    from concurrent.futures import ThreadPoolExecutor

    def one(name: str) -> tuple[str, str]:
        try:
            r = subprocess.run([sys.executable, "-c", f"import studies.{name}"], cwd=str(ROOT),
                               capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=300)
        except Exception as exc:                          # noqa: BLE001
            return name, f"{type(exc).__name__}: {exc}"
        if r.returncode == 0:
            return name, ""
        tail = [ln for ln in (r.stderr or "").strip().splitlines() if ln.strip()]
        return name, (tail[-1].strip()[:160] if tail else "не импортируется")

    with ThreadPoolExecutor(max_workers=workers) as pool:
        return {n: why for n, why in pool.map(one, names) if why}


def plan(only=None, skip=None, heavy=True) -> list[dict]:
    seen, out = set(), []
    for name, group, what, est, is_heavy in REGISTRY:
        if name in seen or name in NOT_STUDIES:
            continue
        if not (ROOT / "studies" / f"{name}.py").exists():
            continue
        seen.add(name)
        out.append({"name": name, "group": group, "what": what, "est": est, "heavy": is_heavy})
    if only:
        out = [r for r in out if r["name"] in only]
    if skip:
        out = [r for r in out if r["name"] not in skip]
    if not heavy:
        out = [r for r in out if not r["heavy"]]
    return out


def panel_is_stale() -> bool:
    """Кэш панели старше самых свежих сырых свечей — значит его надо пересобрать, иначе данные вчерашние."""
    if not PANEL_CACHE.exists():
        return False
    newest = max((f.stat().st_mtime for f in LADDER_DIR.glob("*.csv.gz")), default=0)
    return newest > PANEL_CACHE.stat().st_mtime


def migrate_dense_markers(raw_dir: Path | None = None) -> int:
    """Old re-download markers were named `2026-01-19.csv.dense` — i.e. they matched the readers' `*.csv*` glob and
    were parsed as data ("Usecols do not match columns" killed twenty studies). Rename them to `<date>.dense`, which
    `src/download/kalshi.py` uses now and `_sparse()` still recognises. Returns how many were renamed."""
    n = 0
    for marker in (Path(raw_dir) if raw_dir else LADDER_DIR).glob("*.csv.dense"):
        target = marker.with_suffix("")           # 2026-01-19.csv.dense -> 2026-01-19.csv ... keep only the flag
        target = Path(str(target)[:-len(".csv")] + ".dense") if str(target).endswith(".csv") else target
        if target.exists():
            marker.unlink()
        else:
            marker.rename(target)
        n += 1
    return n


def _snapshot() -> dict[str, float]:
    out = {}
    for base in ("results", "reports", "docs"):
        for f in (ROOT / base).rglob("*"):
            if f.is_file():
                out[str(f.relative_to(ROOT))] = f.stat().st_mtime
    return out


def classify(rc, log_text: str) -> tuple[str, str]:
    """Сбой исследования → понятная причина. Ничего не додумываем: путь и текст ошибки — из самого лога."""
    if rc == -9 or rc == 137:
        return "killed", "убит (память)"
    missing = re.search(r"FileNotFoundError: .*?'([^']+)'", log_text)
    pkg = re.search(r"(?:ModuleNotFoundError: No module named '([^']+)'|ImportError: .*?`Import ([^`]+)` failed)", log_text)
    if pkg:
        return "nopkg", f"нет пакета: {pkg.group(1) or pkg.group(2)}"
    if missing:
        return "nodata", f"нет данных: {missing.group(1)}"
    window = re.findall(r"ValueError: нет данных в окне: ([^\n]+)", log_text)
    if window:                                    # последняя строка traceback'а — сама причина, не исходник
        return "nodata", f"нет данных в окне: {window[-1].strip()}"
    if re.search(r"No objects to concatenate|Empty DataFrame|No .* in the study window|not enough|empty", log_text,
                 re.IGNORECASE):
        return "nodata", "нет данных в окне"
    # Сначала ищем настоящую ошибку: без этого в итог попадал последний прогресс-принт
    # («ошибка: 5665 hours, 50272 strikes …» вместо UnicodeEncodeError).
    named = [ln.strip() for ln in log_text.splitlines() if re.search(r"\w*(?:Error|Exception)\b", ln)]
    err = named[-1][:160] if named else ""
    if not err:
        for line in reversed(log_text.strip().splitlines()):
            if line.strip() and not line.startswith(" "):
                err = line.strip()[:160]
                break
    return "error", f"ошибка: {err}" if err else "ошибка (см. лог)"


def run_study(row: dict, work: Path, index: int, total: int) -> dict:
    name, what = row["name"], row["what"]
    _p(f"=== [{index}/{total}] {name} — {what} (оценка {_mmss(row['est'])})")
    log_path = work / f"{name}.log"
    t0 = time.time()
    lines: list[str] = []
    try:
        with open(log_path, "w", encoding="utf-8", errors="replace") as log:
            proc = subprocess.Popen([sys.executable, "-m", f"studies.{name}"], cwd=str(ROOT),
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                    encoding="utf-8", errors="replace", bufsize=1, env=CHILD_ENV)
            deadline = t0 + PER_STUDY_TIMEOUT
            last, repeats = None, 0
            for line in proc.stdout:
                log.write(line)
                text_line = line.rstrip()
                lines.append(text_line)
                if text_line == last:
                    repeats += 1
                else:
                    if repeats:
                        _p(f"  | … то же сообщение ещё {repeats} раз (в логе целиком)")
                    last, repeats = text_line, 0
                    _p(f"  | {text_line[:200]}")
                if time.time() > deadline:
                    proc.kill()
                    break
            rc = proc.wait()
    except Exception as e:                      # запуск не удался — это тоже изолируется
        rc = -1
        lines.append(f"{type(e).__name__}: {e}")
    dt = time.time() - t0
    text = "\n".join(lines)
    if rc == 0:
        status, why = "ok", ""
    elif time.time() - t0 >= PER_STUDY_TIMEOUT:
        status, why = "timeout", f"таймаут {PER_STUDY_TIMEOUT // 60} мин"
    else:
        status, why = classify(rc, text)
    _p(f"=== [{index}/{total}] {name}: {status} за {_mmss(dt)}" + (f" — {why}" if why else ""))
    return {**row, "status": status, "why": why, "seconds": round(dt, 1), "log": str(log_path),
            "tail": [ln for ln in lines if ln.strip()][-3:]}


def write_summary(state: dict, work: Path) -> Path:
    rows = state["studies"]
    ok = [r for r in rows if r["status"] == "ok"]
    done = [r for r in rows if r["status"] != "skipped"]
    bad = [r for r in done if r["status"] != "ok"]
    L = [f"# Исследования {state['started']}", ""]
    L.append(f"Версия кода: `{state.get('version', '?')}` · Python {sys.version.split()[0]} · "
             f"отпечаток данных `{state.get('stamp', '?')}`")
    L.append("")
    L.append(f"Прошло **{len(ok)}** из **{len(rows) - len([r for r in rows if r['status'] == 'skipped'])}** "
             f"запущенных, общее время {_mmss(state['seconds'])}."
             + (f" Пропущено без изменений: **{len([r for r in rows if r['status'] == 'skipped'])}**."
                if any(r["status"] == "skipped" for r in rows) else ""))
    if bad:
        L.append("")
        L.append("## Что не прошло и почему")
        L += ["", "| исследование | причина |", "|---|---|"]
        L += [f"| {r['name']} | {r['why']} |" for r in bad]
        by_reason = {}
        for r in bad:
            key = r["why"].split(":")[0]
            by_reason[key] = by_reason.get(key, 0) + 1
        L += ["", "Сводка причин: " + ", ".join(f"{k} — {v}" for k, v in sorted(by_reason.items(), key=lambda x: -x[1]))]
    L += ["", "## Что прошло", "", "| исследование | группа | время | последние строки |", "|---|---|---|---|"]
    for r in ok:
        tail = " / ".join(t.replace("|", "\\|")[:90] for t in r["tail"][-2:])
        L.append(f"| {r['name']} | {r['group']} | {_mmss(r['seconds'])} | {tail} |")
    if state["artifacts"]:
        L += ["", f"## Обновились {len(state['artifacts'])} файлов результатов", "",
              "| файл | когда |", "|---|---|"]
        L += [f"| {p} | {'новый' if m else 'обновлён'} |" for p, m in state["artifacts"].items()]
    L += ["", "## Где смотреть", "",
          f"- полный вывод каждого исследования: `{work}`",
          "- таблицы и тексты: `results/`, снимки: `reports/`",
          "- повторить только упавшие: `python scripts/study_all.py --only "
          + ",".join(r["name"] for r in bad) + "`" if bad else "- всё прошло: `python scripts/study_all.py --dry-run` для оценок"]
    path = work / "summary.md"
    path.write_text("\n".join(L), encoding="utf-8")
    return path


def run(rows: list[dict], work: Path | None = None, *, rerun: bool = False, check: bool = True) -> dict:
    work = Path(work or ROOT / "logs" / f"studies-{datetime.now():%Y%m%d-%H%M%S}")
    work.mkdir(parents=True, exist_ok=True)
    stamp = watermark()
    rows, skipped = pending(rows, stamp, rerun)
    for r in skipped:
        r.update({"status": "skipped", "why": "пропущено (нет изменений в данных и коде)", "seconds": 0.0,
                  "log": "", "tail": [], "stamp": stamp})
    total_est = sum(r["est"] for r in rows)
    state = {"started": datetime.now().isoformat(timespec="seconds"), "studies": [], "artifacts": {},
             "version": code_stamp(), "stamp": stamp, "skipped": len(skipped), "work": str(work)}
    t0 = time.time()

    renamed = migrate_dense_markers()
    if renamed:
        _p(f"старые файлы-маркеры *.csv.dense переименованы в *.dense: {renamed} (они не данные, изучения их больше не читают)")
    if panel_is_stale():
        _p(f"кэш панели старше свежих свечей — пересобираю ({PANEL_CACHE.name})")
        PANEL_CACHE.unlink(missing_ok=True)
    n_raw = len(list(LADDER_DIR.glob("*.csv.gz")))
    n_fact = sum(1 for r in rows if r.get("est_fact"))
    _p(f"версия кода: {state['version']}, Python {sys.version.split()[0]}")
    _p(f"к запуску: {len(rows)} исследований, пропущено (нет изменений): {len(skipped)}, "
       f"сырых часовых лесенок: {n_raw}, оценка {_mmss(total_est)}"
       + (f" (по {n_fact} фактическим прогонам)" if n_fact else ""))
    if n_raw == 0:
        _p("!!! нет ни одного файла в data/raw/kalshi_kxbtcd_60m — сначала scripts\\download_all.bat")
    _p(f"журналы: {work}")

    broken: dict[str, str] = {}
    if check and rows:
        _p(f"проверяю импорт {len(rows)} исследований…")
        broken = import_check([r["name"] for r in rows])
        for name, why in broken.items():
            _p(f"!!! не импортируется и не будет запущено: {name} — {why}")
    _p()

    before = _snapshot()
    done_rows = [r for r in rows if r["name"] not in broken]
    for i, row in enumerate(done_rows, 1):
        row["stamp"] = stamp
        result = run_study(row, work, i, len(done_rows))
        log_file = Path(result["log"])
        result["log"] = str(log_file.relative_to(ROOT) if log_file.is_relative_to(ROOT) else log_file)
        state["studies"].append(result)
        record_time(row["name"], result["seconds"], result["status"] == "ok")
        remember({**result, "stamp": stamp})
    for name, why in broken.items():
        row = next(r for r in rows if r["name"] == name)
        state["studies"].append({**row, "status": "nopkg", "why": f"не импортируется: {why}",
                                 "seconds": 0.0, "log": "", "tail": [], "stamp": stamp})
    state["studies"].extend(skipped)
    after = _snapshot()
    state["artifacts"] = {p: "" for p in after
                          if p.startswith(("results/", "reports/")) and after.get(p) != before.get(p)}
    state["seconds"] = round(time.time() - t0, 1)
    state["finished"] = datetime.now().isoformat(timespec="seconds")
    (work / "state.json").write_text(json.dumps(state, ensure_ascii=False, indent=1), encoding="utf-8")
    path = write_summary(state, work)

    ok = sum(1 for r in state["studies"] if r["status"] == "ok")
    n_skip = sum(1 for r in state["studies"] if r["status"] == "skipped")
    failed = [r for r in state["studies"] if r["status"] not in ("ok", "skipped")]
    _p()
    if ok or failed:
        _p(f"готово: {ok} из {len(state['studies']) - n_skip} запущенных прошло, {_mmss(state['seconds'])}, "
           f"обновлено файлов: {len(state['artifacts'])}")
    if failed:
        _p(f"не прошло: {len(failed)} — причины в итоговом файле")
    if n_skip:
        _p(f"пропущено без изменений: {n_skip}" + ("" if ok or failed else " — всё уже посчитано на этих "
                                                   "данных и коде; пересчитать заново: --rerun"))
    _p(f"итог: {path}")
    return state


def status(work: Path | None = None) -> int:
    if work is None:
        runs = sorted((ROOT / "logs").glob("studies-*")) if (ROOT / "logs").exists() else []
        if not runs:
            _p("запусков ещё не было: scripts\\study_all.bat")
            return 1
        work = runs[-1]
    work = Path(work)
    state_file = work / "state.json"
    if not state_file.exists():
        _p(f"{work.name}: итога нет (запуск не дошёл до конца)")
        return 2
    state = json.loads(state_file.read_text(encoding="utf-8"))
    rows = state["studies"]
    ok = [r for r in rows if r["status"] == "ok"]
    _p(f"{work.name}: {len(ok)}/{len(rows)} прошло, {_mmss(state.get('seconds', 0))}")
    for r in rows:
        mark = "ok  " if r["status"] == "ok" else r["status"]
        _p(f"  {mark:10s} {r['name']:24s} {_mmss(r['seconds']):>9s} {r.get('why', '')}")
    _p(f"  обновлено файлов: {len(state.get('artifacts', {}))}, итог: {work / 'summary.md'}")
    return 0


def selftest() -> int:
    """Проверка самой машинерии: изоляция сбоев, классификация причин, итог — без запуска настоящих студий."""
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        work = Path(d)
        fake = [
            {"name": "alpha", "group": "тест", "what": "ok", "est": 1, "heavy": False},
            {"name": "beta", "group": "тест", "what": "падает", "est": 1, "heavy": False},
        ]
        old = globals()["run_study"]

        def fake_run(row, work, index, total):
            status_ = "ok" if row["name"] == "alpha" else "nodata"
            return {**row, "status": status_, "why": "" if status_ == "ok" else "нет данных: data/raw/nope",
                    "seconds": 1.0, "log": "x.log", "tail": ["готово"]}

        globals()["run_study"] = fake_run
        try:
            state = run(fake, work=work, check=False)
        finally:
            globals()["run_study"] = old
        assert (work / "summary.md").exists() and (work / "state.json").exists()
        assert [r["status"] for r in state["studies"]] == ["ok", "nodata"]
    _p("selftest ok: изоляция, классификация и итог работают")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="один запуск — все исследования, какие позволяют данные")
    ap.add_argument("--only", default=None, help="только эти исследования, через запятую")
    ap.add_argument("--skip", default=None, help="всё, кроме этих")
    ap.add_argument("--group", default=None, help="только эта группа: крылья, волатильность, опционы, Kalshi 15m, отчёты")
    ap.add_argument("--rerun", action="store_true", help="пересчитать всё, даже если данные и код не менялись")
    ap.add_argument("--no-check", dest="check", action="store_false", help="не проверять импорт перед стартом")
    ap.add_argument("--no-heavy", dest="heavy", action="store_false", help="без тяжёлых (ML, ёмкость, опционы)")
    ap.add_argument("--dry-run", action="store_true", help="что будет запущено и сколько это минут")
    ap.add_argument("--list", action="store_true", help="список исследований")
    ap.add_argument("--status", action="store_true", help="итог последнего запуска")
    ap.add_argument("--selftest", action="store_true", help=argparse.SUPPRESS)
    ap.set_defaults(heavy=True, check=True)
    a = ap.parse_args(argv)
    split = lambda s: [x.strip() for x in s.split(",") if x.strip()] if s else None  # noqa: E731

    if a.selftest:
        return selftest()
    if a.status:
        return status()
    rows = apply_estimates(plan(split(a.only), split(a.skip), a.heavy))
    if a.group:
        known = sorted({r["group"] for r in rows})
        rows = [r for r in rows if r["group"].lower() == a.group.lower()]
        if not rows:
            _p(f"такой группы нет: {a.group}. Есть: {', '.join(known)}")
            return 1
    if a.list or a.dry_run:
        todo, skipped = pending(rows, watermark(), a.rerun)
        _p(f"версия кода: {code_stamp()}, Python {sys.version.split()[0]}")
        _p(f"{'исследование':26s}{'группа':16s}{'оценка':>10s}  что показывает")
        for r in rows:
            mark = "" if r in todo else "  ← пропустится (нет изменений)"
            _p(f"{r['name']:26s}{r['group']:16s}{_mmss(r['est']):>10s}  {r['what']}{mark}")
        _p(f"\nвсего {len(rows)} исследований, к запуску {len(todo)}, "
           f"оценка {_mmss(sum(r['est'] for r in todo))} (по факту прошлых прогонов, где он есть)")
        if skipped:
            _p(f"пропустится без изменений: {len(skipped)} — пересчитать всё: --rerun")
        _p(f"не входят (это загрузчики, им место в download_all): {', '.join(sorted(NOT_STUDIES))}")
        return 0
    if not rows:
        _p("нечего запускать: проверьте --only/--skip/--group")
        return 1
    return 0 if run(rows, rerun=a.rerun, check=a.check) else 0


if __name__ == "__main__":
    sys.exit(main())
