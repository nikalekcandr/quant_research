"""Один запуск — все исследования, какие позволяют данные: прогресс построчно, сбои изолированы, итог в один файл.

    scripts\\study_all.bat                 (двойной клик)  =  python scripts/study_all.py
    python scripts/study_all.py --dry-run        что будет запущено и сколько это примерно минут
    python scripts/study_all.py --no-heavy       без тяжёлых (машинное обучение, ёмкость, опционные прогоны)
    python scripts/study_all.py --only wings_sleep,wings_alpha    только эти
    python scripts/study_all.py --skip wings_ml,test_100_ideas    всё, кроме этих
    python scripts/study_all.py --list           список исследований с оценками
    python scripts/study_all.py --status         итог последнего запуска

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
LADDER_DIR = ROOT / "data" / "raw" / "kalshi_kxbtcd_60m"
PER_STUDY_TIMEOUT = 45 * 60

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
    ("wings_maker", "крылья", "стоять в книге и ждать покупателя лотереи", 25, False),
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
    if re.search(r"No objects to concatenate|Empty DataFrame|No .* in the study window|not enough|empty", log_text,
                 re.IGNORECASE):
        return "nodata", "нет данных в окне"
    err = ""
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
                                    encoding="utf-8", errors="replace", bufsize=1)
            deadline = t0 + PER_STUDY_TIMEOUT
            for line in proc.stdout:
                log.write(line)
                lines.append(line.rstrip())
                _p(f"  | {line.rstrip()[:200]}")
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
    bad = [r for r in rows if r["status"] != "ok"]
    L = [f"# Исследования {state['started']}", ""]
    L.append(f"Прошло **{len(ok)}** из **{len(rows)}**, общее время {_mmss(state['seconds'])}.")
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


def run(rows: list[dict], work: Path | None = None) -> dict:
    work = Path(work or ROOT / "logs" / f"studies-{datetime.now():%Y%m%d-%H%M%S}")
    work.mkdir(parents=True, exist_ok=True)
    total_est = sum(r["est"] for r in rows)
    state = {"started": datetime.now().isoformat(timespec="seconds"), "studies": [], "artifacts": {}}
    t0 = time.time()

    if panel_is_stale():
        _p(f"кэш панели старше свежих свечей — пересобираю ({PANEL_CACHE.name})")
        PANEL_CACHE.unlink(missing_ok=True)
    n_raw = len(list(LADDER_DIR.glob("*.csv.gz")))
    _p(f"исследований: {len(rows)}, сырых часовых лесенок на диске: {n_raw}, оценка {_mmss(total_est)}")
    if n_raw == 0:
        _p("!!! нет ни одного файла в data/raw/kalshi_kxbtcd_60m — сначала scripts\\download_all.bat")
    _p(f"журналы: {work}")
    _p()

    before = _snapshot()
    for i, row in enumerate(rows, 1):
        state["studies"].append(run_study(row, work, i, len(rows)))
        log_file = Path(state["studies"][-1]["log"])
        state["studies"][-1]["log"] = str(log_file.relative_to(ROOT) if log_file.is_relative_to(ROOT) else log_file)
    after = _snapshot()
    state["artifacts"] = {p: "" for p in after
                          if p.startswith(("results/", "reports/")) and after.get(p) != before.get(p)}
    state["seconds"] = round(time.time() - t0, 1)
    state["finished"] = datetime.now().isoformat(timespec="seconds")
    (work / "state.json").write_text(json.dumps(state, ensure_ascii=False, indent=1), encoding="utf-8")
    path = write_summary(state, work)

    ok = sum(1 for r in state["studies"] if r["status"] == "ok")
    _p()
    _p(f"готово: {ok}/{len(state['studies'])} прошло, {_mmss(state['seconds'])}, обновлено файлов: "
       f"{len(state['artifacts'])}")
    if ok < len(state["studies"]):
        _p("что не прошло и почему — в итоговом файле; там же причина каждого сбоя")
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
            state = run(fake, work=work)
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
    ap.add_argument("--no-heavy", dest="heavy", action="store_false", help="без тяжёлых (ML, ёмкость, опционы)")
    ap.add_argument("--dry-run", action="store_true", help="что будет запущено и сколько это минут")
    ap.add_argument("--list", action="store_true", help="список исследований")
    ap.add_argument("--status", action="store_true", help="итог последнего запуска")
    ap.add_argument("--selftest", action="store_true", help=argparse.SUPPRESS)
    ap.set_defaults(heavy=True)
    a = ap.parse_args(argv)
    split = lambda s: [x.strip() for x in s.split(",") if x.strip()] if s else None  # noqa: E731

    if a.selftest:
        return selftest()
    if a.status:
        return status()
    rows = plan(split(a.only), split(a.skip), a.heavy)
    if a.list or a.dry_run:
        _p(f"{'исследование':26s}{'группа':16s}{'оценка':>10s}  что показывает")
        for r in rows:
            _p(f"{r['name']:26s}{r['group']:16s}{_mmss(r['est']):>10s}  {r['what']}")
        _p(f"\nвсего {len(rows)} исследований, оценка {_mmss(sum(r['est'] for r in rows))}")
        _p(f"не входят (это загрузчики, им место в download_all): {', '.join(sorted(NOT_STUDIES))}")
        return 0
    if not rows:
        _p("нечего запускать: проверьте --only/--skip")
        return 1
    return 0 if run(rows) else 0


if __name__ == "__main__":
    sys.exit(main())
