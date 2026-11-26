"""Import real metadata (OpenAlex or Semantic Scholar JSON) into `data/lit/bibliography.csv`.

    python -m src.lit.import_openalex data/lit/openalex_vol.json data/lit/openalex_pm.json
    python -m src.lit.import_openalex --dry-run data/lit/*.json

Accepts three shapes, all of which the APIs return as-is:
  * OpenAlex search result  {"results": [{"title": ..., "publication_year": ..., "authorships": [...],
                                           "doi": ..., "primary_location": {"source": {"display_name": ...}}}]}
  * a bare list of such work objects
  * JSONL, one work object per line

Nothing is invented: a record without a title or a year is skipped, and every imported row is marked
`status=imported`, `confidence=unverified` with `why` = the venue plus the OpenAlex concepts, so a reviewer can
tell at a glance that nobody has read it yet. Duplicates are dropped by DOI first, then by normalised title.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from src.lit import bibliography as bib


def _iter_records(path: Path):
    text = path.read_text(encoding="utf-8", errors="ignore").strip()
    if not text:
        return
    if text.lstrip().startswith("{"):
        try:
            blob = json.loads(text)
        except json.JSONDecodeError:                 # JSONL
            for line in text.splitlines():
                line = line.strip()
                if line:
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        continue
            return
        if isinstance(blob, dict) and "results" in blob:
            yield from blob["results"]
        elif isinstance(blob, dict):
            yield blob
        else:
            yield from blob
    else:
        for line in text.splitlines():
            line = line.strip()
            if line:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


def _authors(rec: dict) -> str:
    out = []
    for a in rec.get("authorships") or rec.get("authors") or []:
        if isinstance(a, str):
            out.append(a.strip())
            continue
        person = a.get("author") or a
        name = person.get("display_name") or person.get("name") or ""
        if not name:
            continue
        parts = name.split()
        out.append(f"{parts[-1]}, {' '.join(p[0] + '.' for p in parts[:-1])}".strip().rstrip(","))
    return "; ".join(out[:8])


def _venue(rec: dict) -> str:
    source = ((rec.get("primary_location") or {}).get("source") or {})
    for value in (source.get("display_name"), rec.get("venue"), rec.get("host_venue")):
        if value:
            return str(value)
    pv = rec.get("publicationVenue")
    if isinstance(pv, dict) and pv.get("name"):
        return str(pv["name"])
    return ""


def _doi(rec: dict) -> str:
    doi = rec.get("doi")
    if not doi and isinstance(rec.get("externalIds"), dict):
        doi = rec["externalIds"].get("DOI")
    if not doi:
        return ""
    return re.sub(r"^https?://(dx\.)?doi\.org/", "", str(doi)).strip().lower()


def _concepts(rec: dict) -> str:
    items = rec.get("concepts") or rec.get("concepts_scores") or []
    names = []
    for c in items[:3]:
        if isinstance(c, dict) and c.get("display_name"):
            names.append(c["display_name"])
        elif isinstance(c, str):
            names.append(c)
    if not names and isinstance(rec.get("fieldsOfStudy"), list):
        names = [str(x) for x in rec["fieldsOfStudy"][:3]]
    return ", ".join(names)


def to_row(rec: dict) -> dict | None:
    title = (rec.get("title") or rec.get("display_name") or "").strip()
    year = rec.get("publication_year") or rec.get("year")
    if not title or not year:
        return None
    try:
        year = int(year)
    except (TypeError, ValueError):
        return None
    if not 1900 <= year <= 2100:
        return None
    authors = _authors(rec) or "(без авторов в метаданных)"
    venue = _venue(rec)
    concepts = _concepts(rec)
    why = "импортировано: " + " / ".join(x for x in [venue, concepts] if x)
    return {"id": "", "domain": bib.guess_domain(title, concepts), "year": year, "authors": authors,
            "title": title, "venue": venue, "why": why.strip(), "status": "imported",
            "confidence": "unverified", "doi": _doi(rec)}


def merge(existing: list[dict], incoming: list[dict]) -> tuple[list[dict], int, int]:
    """Add rows that are new by DOI or by normalised title. Returns (rows, added, skipped)."""
    dois = {str(r.get("doi", "") or "").lower() for r in existing if r.get("doi")}
    titles = {bib.normalize_title(r["title"]) for r in existing}
    added = skipped = 0
    out = list(existing)
    for r in incoming:
        key = bib.normalize_title(r["title"])
        doi = str(r.get("doi", "") or "").lower()
        if (doi and doi in dois) or key in titles:
            skipped += 1
            continue
        out.append(r)
        titles.add(key)
        if doi:
            dois.add(doi)
        added += 1
    return out, added, skipped


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="import OpenAlex / Semantic Scholar JSON into the bibliography")
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)

    incoming, empty = [], 0
    for p in map(Path, a.paths):
        if not p.exists():
            print(f"!! {p} not found")
            continue
        for rec in _iter_records(p):
            row = to_row(rec)
            if row is None:
                empty += 1
            else:
                incoming.append(row)
    if not incoming:
        print("nothing to import (no record had both a title and a year)")
        return 1

    existing = bib.load()
    rows, added, skipped = merge(existing, incoming)
    print(f"records read {len(incoming)}, skipped without title/year {empty}, "
          f"new {added}, duplicates {skipped}, total {len(rows)}")
    if a.dry_run:
        return 0
    bib.save(rows)
    print(f"written {bib.STORE}; now run: python -m src.lit.build_encyclopedia")
    return 0


if __name__ == "__main__":
    sys.exit(main())
