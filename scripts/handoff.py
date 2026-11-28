"""Одна команда: проверить окружение, досчитать всё несчитанное, собрать отчёт и проверить его числа.

    python scripts/handoff.py                 всё подряд (проверка → расчёт → отчёт → аудит → HANDOFF.md)
    python scripts/handoff.py --check         только окружение и импорт 57 исследований (~40 с)
    python scripts/handoff.py --compute       только расчёт (study_all с пропуском уже посчитанного)
    python scripts/handoff.py --audit         только проверка чисел в готовом отчёте
    python scripts/handoff.py --no-heavy      без ML, ёмкости и опционных прогонов
    python scripts/handoff.py --rerun         пересчитать все исследования, даже неизменные

Зачем это существует. Отчёт `reports/final_report.md` собирается из таблиц, но его заголовочные числа когда-то
были вписаны руками («0 из 53», «2 399 сделок») и разошлись с данными — отчёт начал противоречить своим же
таблицам, и любая модель, которой поручают «поправить отчёт», натыкается на это первой. Здесь:

1. **check** — версия кода, Python, pandas, импорт каждого исследования, наличие обязательных файлов данных.
2. **compute** — `study_all` с пропуском неизменного: считаются только те исследования, чьи данные или код
   изменились; затем пересобирается отчёт генератором репозитория (если он есть).
3. **audit** — числа отчёта сверяются с фактами (`scripts/report_facts.py`, считается из таблиц): каждое
   заголовочное число обязано быть в тексте и совпадать; неподставленные маркеры и расхождения попадают в
   список проблем, а не в красивый отчёт.
4. **handoff → `HANDOFF.md` + `handoff.json`** — машинно-читаемая передача: что посчитано, что нет и почему,
   какие числа проверены, где лежат результаты, что делать дальше. Следующей модели не нужно угадывать по
   простыне HTML.

Скрипт идемпотентен: повторный запуск не пересчитывает неизменное, `--rerun` пересчитывает всё.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts", ROOT / "reports"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

REPORT_MD = ROOT / "reports" / "final_report.md"
REPORT_HTML = ROOT / "reports" / "final_report.html"
BUILDER = ROOT / "reports" / "build_final_report.py"
HANDOFF_MD = ROOT / "HANDOFF.md"
HANDOFF_JSON = ROOT / "handoff.json"

# файлы, без которых половина исследований честно скажет «нет данных»
REQUIRED = [
    ("data/raw/kalshi_kxbtcd_60m", "часовые лесенки Kalshi", "dir"),
    ("data/raw/wing_gap_candles", "ноябрьские снимки книги", "dir"),
    ("data/lab/markets", "журнал книги лаборатории", "dir"),
    ("data/interim/minutes.parquet", "минуты Binance", "file"),
    ("STRATEGIES.md", "каталог правил", "file"),
]


def _p(msg: str = "") -> None:
    print(f"[{datetime.now():%H:%M:%S}] {msg}" if msg else "", flush=True)


# --------------------------------------------------------------------------- 1. окружение
def check() -> dict:
    import pandas as pd

    import study_all as sa

    out = {"version": sa.code_stamp(), "python": sys.version.split()[0], "pandas": pd.__version__,
           "imports": {}, "inputs": {}, "ok": True}
    names = [r["name"] for r in sa.plan()]
    out["imports"] = sa.import_check(names)
    for rel, what, kind in REQUIRED:
        path = ROOT / rel
        exists = path.is_dir() if kind == "dir" else path.is_file()
        n = len([f for f in path.rglob("*") if f.is_file()]) if exists and kind == "dir" else None
        out["inputs"][rel] = {"what": what, "exists": exists, "files": n}
    out["ok"] = not out["imports"] and all(v["exists"] for v in out["inputs"].values())
    out["coverage"] = coverage()
    return out


def coverage() -> dict:
    """Что из реестра исследований уже имеет результат в results/studies, а что нет."""
    import study_all as sa

    rows = sa.plan(None, None, True)
    out_dir = ROOT / "results" / "studies"
    have, missing = [], []
    for r in rows:
        f = out_dir / f"{r['name']}.md"
        if f.exists() and f.stat().st_size > 0:
            have.append({"name": r["name"], "group": r["group"], "what": r["what"],
                         "when": datetime.fromtimestamp(f.stat().st_mtime).isoformat(timespec="minutes")})
        else:
            missing.append({"name": r["name"], "group": r["group"], "what": r["what"], "est_min": r["est"]})
    return {"total": len(rows), "with_result": len(have), "missing": missing,
            "missing_min": round(sum(m["est_min"] for m in missing) / 60, 1),
            "have": have}


# --------------------------------------------------------------------------- 2. расчёт
def compute(heavy: bool = True, rerun: bool = False, only: list[str] | None = None) -> dict:
    import study_all as sa

    rows = sa.apply_estimates(sa.plan(only, None, heavy))
    t0 = time.time()
    state = sa.run(rows, rerun=rerun)
    res = {"studies": len(state["studies"]),
           "ok": sum(1 for r in state["studies"] if r["status"] == "ok"),
           "skipped": sum(1 for r in state["studies"] if r["status"] == "skipped"),
           "failed": [{"name": r["name"], "why": r["why"]} for r in state["studies"]
                      if r["status"] not in ("ok", "skipped")],
           "seconds": round(time.time() - t0, 1),
           "summary": str(Path(state.get("work", "")).relative_to(ROOT)) if state.get("work") else ""}
    return res


def rebuild_report() -> dict:
    """Пересобрать отчёт генератором репозитория. Ошибка сборки — результат, а не исключение."""
    if not BUILDER.exists():
        return {"ran": False, "why": f"нет генератора {BUILDER.relative_to(ROOT)}"}
    t0 = time.time()
    r = subprocess.run([sys.executable, str(BUILDER)], cwd=str(ROOT), capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    tail = [ln for ln in (r.stdout or "").strip().splitlines() + (r.stderr or "").strip().splitlines() if ln.strip()]
    return {"ran": True, "rc": r.returncode, "seconds": round(time.time() - t0, 1), "tail": tail[-6:]}


# --------------------------------------------------------------------------- 3. аудит чисел
def audit(md: Path = REPORT_MD, html: Path = REPORT_HTML, facts: dict | None = None) -> dict:
    """Числа отчёта против фактов из таблиц.

    Проверяется то, что действительно можно проверить механически:
      * каждый факт присутствует в тексте отчёта в том виде, в каком его подставили;
      * ни одного неподставленного маркера `__N_...__` не осталось;
      * в заголовочных местах HTML нет круглых чисел, которых нет в фактах.
    """
    import report_facts as rf

    facts = facts if facts is not None else rf.compute()
    issues: list[str] = []
    if not md.exists():
        return {"facts": facts, "issues": ["нет reports/final_report.md — сначала сборка отчёта"],
                "checked": 0, "ok": False}
    text = md.read_text(encoding="utf-8")
    # часть чисел живёт только в HTML-таблицах (инвентарь источников), поэтому ищем в обоих файлах
    hay = text + (html.read_text(encoding="utf-8") if html.exists() else "")
    left = sorted(set(re.findall(r"__[A-Z_]+__", hay)))
    if left:
        issues.append(f"в отчёте остались маркеры: {left}")
    checked = 0
    for token, name in rf.TOKENS.items():
        value = facts.get(name)
        if value is None:
            continue
        checked += 1
        if rf.fmt(value) not in hay:
            issues.append(f"{name} = {rf.fmt(value)} не найдено в отчёте")
    html_issues = _html_headline_audit(facts, html) if html.exists() else []
    issues.extend(html_issues)
    return {"facts": facts, "issues": issues, "checked": checked, "ok": not issues}


def _html_headline_audit(facts: dict, html_path: Path = REPORT_HTML) -> list[str]:
    """Заголовочные числа KPI в HTML: если рядом с «из N» стоит число, которого нет в фактах, — это литерал."""
    html = html_path.read_text(encoding="utf-8")
    issues = []
    for m in re.finditer(r'<div class="v">([\d\s]+)<small>\s*из\s*([\d\s]+)</small>', html):
        have, total = (int(x.replace(" ", "")) for x in m.groups())
        if facts.get("n_hypotheses") is not None and total != int(facts["n_hypotheses"]):
            issues.append(f"KPI «{have} из {total}»: в данных {facts['n_hypotheses']} гипотез")
        if facts.get("n_holm_positive") is not None and have != int(facts["n_holm_positive"]):
            issues.append(f"KPI «{have} из {total}»: прошли Холма с плюсом {facts['n_holm_positive']}")
    return issues


# --------------------------------------------------------------------------- 4. передача
def _load_prev() -> dict:
    if not HANDOFF_JSON.exists():
        return {}
    try:
        return json.loads(HANDOFF_JSON.read_text(encoding="utf-8"))
    except Exception:                                         # noqa: BLE001
        return {}


def write_handoff(check_res: dict | None, compute_res: dict | None, report_res: dict | None,
                  audit_res: dict | None) -> Path:
    """Собрать HANDOFF.md и handoff.json. Стадии, не запущенные сейчас, берутся из прошлого прогона:
    `--check` утром и `--compute` вечером дают один общий файл, а не два половинчатых."""
    ts = datetime.now().isoformat(timespec="seconds")
    prev = _load_prev()
    stages = {"check": check_res, "compute": compute_res, "report": report_res, "audit": audit_res}
    kept = [k for k, v in stages.items() if v is None and prev.get(k)]
    check_res = check_res or prev.get("check")
    compute_res = compute_res or prev.get("compute")
    report_res = report_res or prev.get("report")
    audit_res = audit_res or prev.get("audit")
    lines = [f"# Передача: что посчитано и что делать дальше", "",
             f"Сформировано `{ts}` · версия кода `{(check_res or {}).get('version', '?')}` · "
             f"Python {(check_res or {}).get('python', '?')} · pandas {(check_res or {}).get('pandas', '?')}"
             + (f" · стадии из прошлого запуска: {', '.join(kept)}" if kept else ""), ""]

    if audit_res:
        import report_facts as rf

        f = audit_res.get("facts", {})
        lines += ["## Числа отчёта — проверены механически", "",
                  "| факт | значение | откуда |", "|---|---|---|",
                  f"| гипотез в таблице лаборатории | {rf.fmt(f.get('n_hypotheses'))} | `reports/final/tables/lab.csv` |",
                  f"| из них с ≥5 сделками | {rf.fmt(f.get('n_with_trades'))} | там же |",
                  f"| прошли Холма с положительным P&L | {rf.fmt(f.get('n_holm_positive'))} | там же |",
                  f"| ID в каталоге | {rf.fmt(f.get('n_catalog_ids'))} | `STRATEGIES.md` |",
                  f"| из них с исполненными сделками | {rf.fmt(f.get('n_catalog_active'))} | `lab.csv` |",
                  f"| открытий в журнале | {rf.fmt(f.get('n_open_trades'))} | `data/lab/ledger.csv` |",
                  f"| закрытых сделок в журнале | {rf.fmt(f.get('n_closed_trades'))} | там же |", ""]
        if audit_res["ok"]:
            lines += [f"Аудит: **пройден** ({audit_res['checked']} чисел сверено с таблицами).", ""]
        else:
            lines += ["Аудит: **есть расхождения** — их нельзя «поправить» руками в тексте, надо пересобрать:",
                      ""] + [f"- {x}" for x in audit_res["issues"]] + [""]
            lines += ["Важно: числа в отчёте подставляются из `scripts/report_facts.py`. Правка цифры в тексте "
                      "или в шаблоне — это гарантированное расхождение на следующей сборке.", ""]

    if check_res:
        imp = check_res.get("imports") or {}
        lines += ["## Окружение", ""]
        lines.append(f"- импорт исследований: {'все 57 импортируются' if not imp else f'не импортируются: {sorted(imp)}'}")
        for rel, info in (check_res.get("inputs") or {}).items():
            mark = "есть" if info["exists"] else "**нет**"
            n = f", файлов {info['files']}" if info.get("files") else ""
            lines.append(f"- `{rel}` — {info['what']}: {mark}{n}")
        lines.append("")

    cov = (check_res or {}).get("coverage") or {}
    if cov:
        lines += [f"## Что ещё не посчитано: {len(cov.get('missing') or [])} из {cov['total']} исследований", ""]
        if cov.get("missing"):
            lines += [f"Оценка машинного времени по реестру — {cov['missing_min']} мин (первый прогон дороже: "
                      f"панель и кэши считаются заново). {cov['with_result']} уже имеют результат в "
                      f"`results/studies/`.", "",
                      "| исследование | группа | что даёт | ~мин |", "|---|---|---|---|"]
            lines += [f"| {m['name']} | {m['group']} | {m['what']} | {m['est_min']} |" for m in cov["missing"]]
            heavy = max(cov["missing"], key=lambda m: m["est_min"])
            lines += ["", f"Самое дорогое — `{heavy['name']}` ({heavy['est_min']} мин по реестру); в `--no-heavy` "
                          "оно пропускается.", "",
                      "Считать именно их (пропуская всё, что уже посчитано): `python scripts/handoff.py` "
                      "или только перечисленные — `python scripts/study_all.py --only <имя> --only <имя>`.", ""]
            if cov.get("have"):
                lines += ["Файл результата уже есть и без `--rerun` не пересчитывается "
                          "(время — метка файла, не обязательно время расчёта): "
                          + ", ".join(f"`{h['name']}` ({h['when'].replace('T', ' ')})" for h in cov["have"]), ""]
        else:
            lines += ["Все исследования реестра имеют результат. Пересчёт не нужен; `--rerun` только если "
                      "поменялась методика.", ""]

    if compute_res:
        lines += ["## Расчёт", "",
                  f"- исследований: {compute_res['studies']}, прошло {compute_res['ok']}, "
                  f"пропущено без изменений {compute_res['skipped']}, время {compute_res['seconds'] / 60:.1f} мин"]
        if compute_res["failed"]:
            lines += ["", "| не прошло | причина |", "|---|---|"]
            lines += [f"| {f['name']} | {f['why']} |" for f in compute_res["failed"]]
        lines.append("")

    if report_res:
        lines += ["## Отчёт", ""]
        if not report_res.get("ran"):
            lines.append(f"- не собирался: {report_res.get('why')}")
        elif report_res.get("rc") == 0:
            lines.append(f"- собран за {report_res.get('seconds')} с; последние строки: "
                         + " / ".join(report_res.get("tail") or [])[:400])
        else:
            lines.append(f"- **сборка упала** (код {report_res.get('rc')}): "
                         + " / ".join(report_res.get("tail") or [])[:600])
        lines.append(f"- `reports/final_report.md`, `reports/final_report.html`")
        lines.append("")

    lines += ["## Что делать дальше", "",
              "1. Если аудит выше показывает расхождения — не править цифры руками, а пересобрать: "
              "`python scripts/handoff.py --audit` после `python reports/build_final_report.py`.",
              "2. Если какое-то исследование не прошло — его причина в таблице «Расчёт» и в "
              "`logs/studies-<дата>/summary.md`; повторить только его: `python scripts/study_all.py --only <имя>`.",
              "3. Считать заново всё: `python scripts/handoff.py --rerun` (или `study_all.bat --rerun`).", ""]

    HANDOFF_MD.write_text("\n".join(lines), encoding="utf-8")
    merged = {"generated": ts, "stages_run": [k for k, v in (("check", check_res), ("compute", compute_res),
                                                             ("report", report_res), ("audit", audit_res)) if v],
              "stages_kept_from_previous_run": kept,
              "check": check_res, "compute": compute_res, "report": report_res, "audit": audit_res}
    HANDOFF_JSON.write_text(json.dumps(merged, ensure_ascii=False, indent=1), encoding="utf-8")
    return HANDOFF_MD


# --------------------------------------------------------------------------- main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="досчитать всё несчитанное и проверить числа отчёта")
    ap.add_argument("--check", action="store_true", help="только окружение и импорт исследований")
    ap.add_argument("--compute", action="store_true", help="только расчёт (study_all)")
    ap.add_argument("--audit", action="store_true", help="только аудит чисел в готовом отчёте")
    ap.add_argument("--no-report", action="store_true", help="не пересобирать отчёт")
    ap.add_argument("--no-heavy", dest="heavy", action="store_false", help="без ML, ёмкости, опционов")
    ap.add_argument("--rerun", action="store_true", help="пересчитать все исследования, даже неизменные")
    ap.add_argument("--only", action="append", metavar="ИМЯ", help="только эти исследования (можно несколько)")
    ap.set_defaults(heavy=True)
    a = ap.parse_args(argv)
    only = a.check or a.compute or a.audit
    do_check, do_compute, do_audit = (not only or a.check), (not only or a.compute), (not only or a.audit)

    check_res = compute_res = report_res = audit_res = None
    _p(f"handoff: версия кода {__import__('study_all').code_stamp()}")

    if do_check:
        _p("проверяю окружение и импорт исследований…")
        check_res = check()
        if check_res["imports"]:
            _p(f"  не импортируются: {sorted(check_res['imports'])}")
        cov = check_res.get("coverage") or {}
        _p(f"  результаты есть у {cov.get('with_result')} из {cov.get('total')} исследований; "
           f"не посчитано {len(cov.get('missing') or [])} (≈{cov.get('missing_min')} мин по реестру)")
        missing = [rel for rel, info in check_res["inputs"].items() if not info["exists"]]
        if missing:
            _p(f"  нет на диске: {', '.join(missing)} — эти исследования честно скажут «нет данных»")

    if do_compute:
        _p("считаю всё, что изменилось…")
        compute_res = compute(heavy=a.heavy, rerun=a.rerun, only=a.only)
        if not a.no_report:
            _p("пересобираю отчёт…")
            report_res = rebuild_report()
            if report_res.get("ran") and report_res.get("rc") != 0:
                _p(f"  сборка отчёта упала (код {report_res['rc']}): {report_res['tail'][-1:] }")

    if do_audit:
        _p("проверяю числа отчёта против таблиц…")
        audit_res = audit()
        if audit_res.get("issues"):
            for x in audit_res["issues"]:
                _p(f"  ! {x}")
        else:
            _p(f"  аудит пройден: {audit_res.get('checked', 0)} чисел сверено с таблицами")

    path = write_handoff(check_res, compute_res, report_res, audit_res)
    _p(f"итог для следующей модели: {path.relative_to(ROOT)} и {HANDOFF_JSON.name}")
    ok = (audit_res is None or audit_res["ok"]) and (compute_res is None or not compute_res["failed"])
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
