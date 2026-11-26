"""Generate `docs/MEGA_ENCYCLOPEDIA.md` from `data/lit/bibliography.csv`.

    python -m src.lit.build_encyclopedia            validate, then write the encyclopedia
    python -m src.lit.build_encyclopedia --dry-run  validate and print the counts, write nothing
    python -m src.lit.build_encyclopedia --top 40   how many "read next" entries to print

The file is meant to be regenerated, never edited: everything in it comes from the CSV, so adding a paper is a
one-line change in one place. Errors stop the build (a paper without a year or a domain is a bug); warnings are
printed and also appear at the bottom of the generated file, so they cannot be silently ignored.
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from datetime import datetime, timezone

from src.lit import bibliography as bib

READ_NEXT_ORDER = ["anchor_tested", "needs_data", "candidate", "imported"]


def _cell(x) -> str:
    return str(x).replace("|", "/").replace("\n", " ").strip()


def _table(rows: list[dict], columns: list[str], headers: list[str] | None = None) -> str:
    headers = headers or columns
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(columns)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(_cell(r.get(c, "")) for c in columns) + " |")
    return "\n".join(out)


def counts(rows: list[dict]) -> dict:
    return {"total": len(rows), "domains": len({r["domain"] for r in rows}),
            "by_domain": Counter(r["domain"] for r in rows),
            "by_status": Counter(r["status"] for r in rows),
            "by_decade": Counter((int(r["year"]) // 10) * 10 for r in rows),
            "verified": sum(1 for r in rows if r.get("confidence") == "high"),
            "with_doi": sum(1 for r in rows if str(r.get("doi", "") or "").strip())}


def read_next(rows: list[dict], top: int = 30) -> list[dict]:
    """Candidate papers ordered by 'how close to money' rather than by year: needs_data first among candidates."""
    def key(r):
        return (READ_NEXT_ORDER.index(r["status"]) if r["status"] in READ_NEXT_ORDER else 9,
                -int(r["year"] or 0))
    pool = [r for r in rows if r["status"] in ("needs_data", "candidate")]
    return sorted(pool, key=key)[:top]


def build(rows: list[dict], warnings: list[str], top: int = 30) -> str:
    c = counts(rows)
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        "# Мега-энциклопедия литературы проекта",
        "",
        f"_Сгенерировано {now} из `data/lit/bibliography.csv` (`src/lit/build_encyclopedia.py`). "
        "Руками этот файл не правится: правится CSV._",
        "",
        "## Что это и чего здесь нет",
        "",
        f"В хранилище **{c['total']}** строк в {c['domains']} доменах. Из них {c['with_doi']} с DOI, "
        f"{c['verified']} с высокой уверенностью в выходных данных. Это **курируемый корпус**, а не выгрузка "
        "базы: каждая строка прочитана и связана с конкретным местом проекта (колонка `why`).",
        "",
        "Здесь **намеренно нет** ссылок, которых я не могу подтвердить: в песочнице нет доступа к arXiv и "
        "издателям, поэтому DOI и страницы не придумываются. Строки без DOI помечены `confidence=medium` "
        "или `unverified` — их надо открыть и сверить. Массовое расширение до тысяч работ делается импортом "
        "настоящих метаданных (`src/lit/import_openalex.py`), а не генерацией текста.",
        "",
        "## Как читать",
        "",
        "`status` — это стадия работы с источником, а не качество статьи:",
        "",
        "| status | значение |",
        "|---|---|",
    ]
    for s, text in bib.STATUS_TEXT.items():
        lines.append(f"| `{s}` | {text} |")
    lines += [
        "",
        "## Счёт по доменам",
        "",
        _table([{"domain": d, "what": bib.DOMAINS[d][0], "why": bib.DOMAINS[d][1],
                 "n": c["by_domain"][d], "used": sum(1 for r in rows if r["domain"] == d and r["status"] == "anchor_used")}
                for d in bib.DOMAINS if c["by_domain"][d]],
               ["domain", "what", "why", "n", "used"], ["домен", "что", "зачем проекту", "работ", "уже в коде"]),
        "",
        f"По статусам: " + ", ".join(f"`{k}` {v}" for k, v in c["by_status"].most_common()) + ".",
        "",
        "## Оглавление",
        "",
    ]
    for i, d in enumerate([d for d in bib.DOMAINS if c["by_domain"][d]], start=1):
        lines.append(f"{i}. [{bib.DOMAINS[d][0]}](#{i}-{d}) — {c['by_domain'][d]} работ")
    lines += ["", "---", ""]
    for i, d in enumerate([d for d in bib.DOMAINS if c["by_domain"][d]], start=1):
        name, why = bib.DOMAINS[d]
        part = sorted([r for r in rows if r["domain"] == d], key=lambda r: (-int(r["year"] or 0), r["id"]))
        lines += [f"## {i}. {name}", "", f"_{d}: {why}._", "",
                  _table(part, ["id", "year", "authors", "title", "venue", "why", "status", "confidence"],
                         ["id", "год", "авторы", "работа", "где", "зачем проекту", "status", "уверенность"]),
                  ""]
    lines += ["---", "", "## Что читать дальше", "",
              "Отсортировано по близости к деньгам: сначала то, под что уже есть данные (`needs_data` и "
              "`candidate`), новые — выше.", "",
              _table(read_next(rows, top), ["id", "year", "authors", "title", "status"],
                     ["id", "год", "авторы", "работа", "status"]), ""]
    lines += ["## Как расширить до тысяч работ", "",
              "```bash",
              "# 1. скачать метаданные настоящих работ (на вашей машине, где есть интернет)",
              "curl -s 'https://api.openalex.org/works?search=realized%20volatility&per-page=200' -o data/lit/openalex_vol.json",
              "curl -s 'https://api.openalex.org/works?search=prediction%20markets&per-page=200' -o data/lit/openalex_pm.json",
              "# 2. импортировать: дедуп по названию и DOI, домен по ключевым словам, статус imported",
              "python -m src.lit.import_openalex data/lit/openalex_vol.json data/lit/openalex_pm.json",
              "# 3. пересобрать энциклопедию",
              "python -m src.lit.build_encyclopedia",
              "```",
              "",
              "Импортированные строки попадают в конец (status `imported`, confidence `unverified`) и не "
              "считаются прочитанными, пока у них не появится `why` и `status=candidate`. Это разделение "
              "специально: иначе «энциклопедия на 5 000 источников» превращается в свалку, где никто не "
              "помнит, что из неё следует для стратегии.", ""]
    if warnings:
        lines += ["## Предупреждения проверки", ""] + [f"* {w}" for w in warnings[:200]] + [""]
    lines += ["---", "",
              f"_Счёт по десятилетиям: "
              + ", ".join(f"{k}s {v}" for k, v in sorted(c["by_decade"].items())) + "._", ""]
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="validate the bibliography and regenerate the encyclopedia")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--top", type=int, default=30)
    a = ap.parse_args(argv)

    rows = bib.load()
    errors, warnings = bib.validate(rows)
    for w in warnings:
        print(f"warning: {w}")
    if errors:
        for e in errors:
            print(f"ERROR: {e}")
        return 1
    c = counts(rows)
    print(f"{c['total']} works, {c['domains']} domains, {c['with_doi']} with DOI, "
          f"{len(warnings)} warnings")
    if a.dry_run:
        return 0
    path = bib.ENSA
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(build(rows, warnings, a.top), encoding="utf-8")
    print(f"written {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
