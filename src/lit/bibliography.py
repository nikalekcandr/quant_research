"""Bibliography of the project: one CSV is the store, the markdown encyclopedia is generated from it.

    data/lit/bibliography.csv      curated rows (hand-written, one line per work)
    docs/MEGA_ENCYCLOPEDIA.md      generated: counts, per-domain tables, reading order

The store grows in two ways: by hand (the CSV is plain text) and by importing a real metadata dump
(`src/lit/import_openalex.py` reads an OpenAlex or Semantic Scholar JSON, which is how the list can reach
thousands of entries without anybody typing them). Imported rows are marked `status=imported` and
`confidence=unverified` until somebody reads the paper.
"""
from __future__ import annotations

import csv
import re
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
STORE = ROOT / "data/lit/bibliography.csv"
ENSA = ROOT / "docs/MEGA_ENCYCLOPEDIA.md"

COLUMNS = ["id", "domain", "year", "authors", "title", "venue", "why", "status", "confidence", "doi"]

# domain -> (Russian title, one line on why the domain matters for this project)
DOMAINS = {
    "vol-forecasting": ("Прогноз волатильности", "дисперсия до расчёта — единственный вход модельной вероятности"),
    "hf-econometrics": ("Высокочастотная эконометрика", "что вообще можно оценить по минутным данным"),
    "jumps-tails": ("Скачки и хвосты", "крыло продаёт хвост, значит хвост надо моделировать отдельно"),
    "microstructure-theory": ("Микроструктура: теория", "спред, adverse selection, информированная торговля"),
    "prediction-markets": ("Рынки предсказаний", "почему бинарная цена систематически смещена"),
    "probability-scoring": ("Оценка вероятностных прогнозов", "Brier, лог-лосс, калибровка, пулинг"),
    "backtest-overfitting": ("Переподгонка и множественные тесты", "сколько испытаний выдержала гипотеза"),
    "ml-methods": ("Методы машинного обучения", "что реально работает на таблицах и сколько данных нужно"),
    "asset-pricing": ("Эмпирические факторы", "откуда берутся премии и как их не спутать с шумом"),
    "options-iv": ("Опционы и IV", "улыбка, RND, безмодельные моменты"),
    "crypto": ("Крипторынок", "перпы, фандинг, базис, сегментация площадок"),
    "portfolio-risk": ("Портфель и риск", "размер позиции, Келли, просадка"),
    "econometrics-tools": ("Эконометрический инструментарий", "тесты, бутстрэп, HAC"),
    "infra-repro": ("Воспроизводимость", "чтобы через полгода всё считалось тем же"),
}

STATUS_ORDER = ["anchor_used", "anchor_tested", "needs_data", "candidate", "imported", "rejected"]
STATUS_TEXT = {
    "anchor_used": "использовано в коде проекта",
    "anchor_tested": "проверено и описано в `results/studies`",
    "needs_data": "нужны данные, которых пока нет",
    "candidate": "кандидат: не пробовали",
    "imported": "импортировано автоматически, не читано",
    "rejected": "проверено и отклонено",
}

# keyword rules for the importer: first match wins
DOMAIN_RULES = [
    ("backtest-overfitting", ["backtest", "overfit", "multiple testing", "data snooping", "false discovery",
                              "deflated sharpe", "reality check", "multiple comparis"]),
    ("prediction-markets", ["prediction market", "betting", "parimutuel", "polymarket", "kalshi", "lottery"]),
    ("probability-scoring", ["calibration", "scoring rule", "brier", "crps", "proper scoring", "probabilistic forecast"]),
    ("options-iv", ["option pric", "implied volatilit", "volatility surface", "smile", "risk-neutral densit",
                    "variance swap", "variance risk premium"]),
    ("crypto", ["bitcoin", "cryptocurrenc", "blockchain", "perpetual future", "stablecoin", "ethereum"]),
    ("jumps-tails", ["jump", "extreme value", "tail risk", "heavy-tail", "heavy tail", "regime switch"]),
    ("hf-econometrics", ["high-frequency", "high frequency", "microstructure noise", "realized kernel",
                         "pre-averag", "two-scale", "order book", "bid-ask", "limit order"]),
    ("vol-forecasting", ["realized volatility", "garch", "stochastic volatility", "volatility forecast",
                         "rough volatilit", "har model", "volatility model"]),
    ("portfolio-risk", ["portfolio", "kelly", "value-at-risk", "drawdown", "risk parity", "position siz"]),
    ("asset-pricing", ["asset pricing", "factor model", "cross-section of", "momentum", "carry", "anomal"]),
    ("ml-methods", ["neural network", "deep learning", "machine learning", "gradient boosting", "transformer",
                    "random forest", "xgboost", "regulariz", "generaliz"]),
    ("econometrics-tools", ["bootstrap", "unit root", "cointegration", "instrumental variable", "heteroskedastic",
                            "autocorrelat", "panel data"]),
]


def load(path: Path | str = STORE) -> list[dict]:
    with Path(path).open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["year"] = int(float(r["year"])) if str(r.get("year", "")).strip() else None
    return rows


def save(rows: list[dict], path: Path | str = STORE) -> None:
    rows = sorted(rows, key=lambda r: (r.get("domain", ""), -(r.get("year") or 0)))
    for i, r in enumerate(rows, start=1):
        r["id"] = f"L{i:03d}"
    with Path(path).open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def normalize_title(t: str) -> str:
    """Lower case, no diacritics, no punctuation, single spaces — for duplicate detection."""
    t = unicodedata.normalize("NFKD", str(t)).encode("ascii", "ignore").decode("ascii").lower()
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]+", " ", t)).strip()


def guess_domain(title: str, abstract: str = "") -> str:
    text = f"{title} {abstract}".lower()
    for domain, keys in DOMAIN_RULES:
        if any(k in text for k in keys):
            return domain
    return "ml-methods"


def validate(rows: list[dict]) -> tuple[list[str], list[str]]:
    """(errors, warnings). Errors break the build; warnings are printed."""
    errors, warnings = [], []
    seen_title, seen_doi = {}, {}
    for r in rows:
        rid = r.get("id", "?")
        for c in COLUMNS:
            if c not in r:
                errors.append(f"{rid}: нет колонки {c}")
        if not str(r.get("title", "")).strip():
            errors.append(f"{rid}: пустой title")
        if not str(r.get("authors", "")).strip():
            errors.append(f"{rid}: пустые authors")
        year = r.get("year")
        if year is None or not (1900 <= int(year) <= 2100):
            errors.append(f"{rid}: неверный год {year!r}")
        if r.get("domain") not in DOMAINS:
            errors.append(f"{rid}: неизвестный домен {r.get('domain')!r}")
        if r.get("status") not in STATUS_TEXT:
            errors.append(f"{rid}: неизвестный status {r.get('status')!r}")
        if r.get("confidence") not in ("high", "medium", "unverified"):
            warnings.append(f"{rid}: непривычная confidence {r.get('confidence')!r}")
        key = normalize_title(r.get("title", ""))
        if key in seen_title:
            warnings.append(f"{rid}: дубликат названия с {seen_title[key]}")
        seen_title.setdefault(key, rid)
        doi = str(r.get("doi", "") or "").strip().lower()
        if doi:
            if doi in seen_doi:
                warnings.append(f"{rid}: дубликат DOI с {seen_doi[doi]}")
            seen_doi.setdefault(doi, rid)
        if r.get("status") == "imported" and not doi:
            warnings.append(f"{rid}: импортировано без DOI — проверить вручную")
    return errors, warnings
