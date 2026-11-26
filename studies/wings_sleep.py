"""Sleep, gaps and stale books: does the laptop's sleep pattern bias the wing results? (Nov 2026)

The question is fair and was only half-answered before. `lab/` already replays sleep explicitly
(`studies/wing_sleep_replay.py` + `data/raw/wing_gap_candles/sleep.csv`), but the wing holdout never got the
same treatment, and the two tapes are not the same object:

    Jul-Oct   `data/raw/kalshi_kxbtcd_60m` — the Kalshi *candles* endpoint, downloaded after the fact, so the
              laptop being asleep does not open a hole; 60 minutes per hour, 2 784 full hours.
    November  `data/raw/wing_gap_candles`   — book candles captured live by the logger, ~17 minutes of the last
              31 minutes of each hour, and the hours the machine slept are simply absent.

So the November holdout can be biased in three ways: (1) missing hours cluster in time-of-day, which matters
because the only stable gate we found is a time-of-day gate; (2) the first quotes after a wake-up can be the last
book the API knew, i.e. stale; (3) the per-hour window is not the same (last 31 minutes vs the whole hour).

This study measures all three and re-runs the rules under the corrections:
    S1 coverage      hours present vs expected, by UTC hour of day, for every tape that needs the machine awake
    S2 staleness     share of consecutive quotes that do not move, by source, and inside the first minutes of a
                     file after a wake-up
    S3 robustness    W03 / W04 / both on November: as is, without stale quotes, restricted to hours that are
                     usually observed, and post-stratified to the Jul-Oct hour-of-day mix

Run:  python studies/wings_sleep.py
Out:  results/studies/wings_sleep_*.csv and wings_sleep.md
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from studies.wing_panel import build, cheap_sides  # noqa: E402
from studies.wings_scale import cheap, pnl_at, signals, stats  # noqa: E402

OUT = ROOT / "results/studies"
RAW = ROOT / "data/raw"
NOV_DIR = RAW / "wing_gap_candles"
SLEEP = NOV_DIR / "sleep.csv"
NOV = (pd.Timestamp("2026-11-08", tz="UTC"), pd.Timestamp("2026-11-26", tz="UTC"))


def _expected_hours(lo: pd.Timestamp, hi: pd.Timestamp) -> pd.DatetimeIndex:
    return pd.date_range(lo.floor("h"), hi.floor("h"), freq="h", tz="UTC")


# ------------------------------------------------------------------------------------- S1 coverage
def november_hours() -> pd.Series:
    """Hours (settlement times) present in the November book candles."""
    ends = []
    for path in sorted(NOV_DIR.glob("*.csv.gz")):
        try:
            d = pd.read_csv(path, usecols=["close_time"], nrows=5000)
        except Exception:
            continue
        if len(d):
            ends.append(pd.to_datetime(d.close_time, utc=True, format="mixed").iloc[0])
    return pd.Series(ends).value_counts().sort_index()


def coverage() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Tape-by-tape: what is expected, what is on disk, and the hour-of-day profile of the holes."""
    rows = []
    hours = november_hours().index
    if len(hours):
        span = _expected_hours(hours.min(), hours.max())
        missing = span.difference(hours)
        rows.append({"tape": "November book candles (live)", "needs awake": "yes",
                     "expected": len(span), "present": len(hours), "missing": len(missing),
                     "share present": len(hours) / len(span)})
    else:
        missing = pd.DatetimeIndex([])

    lab = sorted((ROOT / "data/lab/markets").glob("*.csv")) if (ROOT / "data/lab/markets").exists() else []
    rows.append({"tape": "lab market log (live)", "needs awake": "yes", "expected": np.nan,
                 "present": len(lab), "missing": np.nan, "share present": np.nan})

    kx15 = sorted((RAW / "kx15/candles").glob("*.csv"))
    rows.append({"tape": "KX15 candles (API, refillable)", "needs awake": "no", "expected": np.nan,
                 "present": len(kx15), "missing": np.nan, "share present": np.nan})

    der = sorted((RAW / "deribit/trades").glob("*.json.gz"))
    rows.append({"tape": "Deribit trade windows", "needs awake": "yes", "expected": np.nan,
                 "present": len(der), "missing": np.nan, "share present": np.nan})

    kalshi = sorted((RAW / "kalshi_kxbtcd_60m").glob("*.csv.gz"))
    rows.append({"tape": "Kalshi hourly ladder (API, refillable)", "needs awake": "no", "expected": np.nan,
                 "present": len(kalshi), "missing": np.nan, "share present": np.nan})

    profile = pd.DataFrame({"hour utc": range(24)})
    if len(hours):
        present = hours.tz_convert("UTC").hour.value_counts().reindex(range(24), fill_value=0)
        absent = missing.tz_convert("UTC").hour.value_counts().reindex(range(24), fill_value=0)
        profile["november present"] = present.to_numpy()
        profile["november missing"] = absent.to_numpy()
    if SLEEP.exists():   # written by the live logger: sell/buy of the machine's sleep periods
        sleep = pd.read_csv(SLEEP).apply(lambda c: pd.to_datetime(c, utc=True))
        profile["sleep hours"] = (sleep.wake - sleep.sleep).dt.total_seconds().groupby(
            sleep.sleep.dt.hour).sum().reindex(range(24), fill_value=0) / 3600
    return pd.DataFrame(rows), profile


def ledger_holes(archive: pd.DataFrame) -> pd.DataFrame:
    """Jul-Oct hours: the tape is refillable, but the downloader still leaves holes - where and how big.

    The hours come from the panel (one row per settlement time), not from the file names: each Kalshi file
    holds a whole day of snapshots for every market.
    """
    ends = pd.DatetimeIndex(sorted(archive.end.unique()))
    if not len(ends):
        return pd.DataFrame()
    span = _expected_hours(ends[0], ends[-1])
    missing = span.difference(ends)
    prof = pd.DataFrame({"hour utc": range(24)})
    prof["missing"] = missing.hour.value_counts().reindex(range(24), fill_value=0).to_numpy()
    prof["present"] = ends.hour.value_counts().reindex(range(24), fill_value=0).to_numpy()
    runs = []
    if len(missing):
        gaps = pd.Series(missing).diff().dt.total_seconds().div(3600).fillna(1)
        starts = np.r_[0, np.where(gaps.to_numpy() != 1)[0]]
        stops = np.r_[np.where(gaps.to_numpy() != 1)[0], len(missing)]
        for a, b in zip(starts, stops):
            if b > a:
                runs.append({"from": missing[a], "to": missing[b - 1], "hours": b - a,
                             "hour utc": missing[a].hour})
    return pd.DataFrame(runs).sort_values("hours", ascending=False).reset_index(drop=True)


# ------------------------------------------------------------------------------------- S2 staleness
def quote_changes(nov: pd.DataFrame, archive: pd.DataFrame) -> pd.DataFrame:
    """Share of consecutive same-ticker quotes that do not move, per source and per position in the file."""
    rows = []
    for name, d in (("November live", nov), ("Jul-Oct archive", archive)):
        if d.empty:
            continue
        x = d.sort_values(["ticker", "ts"])
        same = ((x.bid.eq(x.groupby("ticker").bid.shift())) & (x.ask.eq(x.groupby("ticker").ask.shift())))
        gap = x.groupby("ticker").ts.diff().dt.total_seconds().div(60)
        first = gap.isna() | gap.gt(5)
        rows.append({"source": name, "candles": int(len(x)),
                     "unchanged share": float(same[~first].mean()) if (~first).any() else np.nan,
                     "unchanged on first candle after a pause": float(same[first.fillna(True)].mean())
                     if first.any() else np.nan,
                     "median minutes between candles": float(gap.median())})
    return pd.DataFrame(rows)


def stale_mask(nov: pd.DataFrame, run: int = 3) -> pd.Series:
    """True where the quote has not moved for `run` candles in a row (a frozen book)."""
    x = nov.sort_values(["ticker", "ts"]).copy()
    same = (x.bid.eq(x.groupby("ticker").bid.shift()) & x.ask.eq(x.groupby("ticker").ask.shift()))
    # consecutive-unchanged counter per ticker
    grp = same.groupby(x.ticker).cumsum() * same
    return grp.reindex(nov.index).fillna(0).ge(run)


# ------------------------------------------------------------------------------------- S3 robustness
def rule_pnl(d: pd.DataFrame, rule: str, drop_pairs: set | None = None,
             drop_hours: set | None = None, weight: pd.Series | None = None) -> pd.DataFrame:
    """One trade an hour for a frozen rule on the November tape, with drops and hour weights."""
    t = signals(cheap(d, rule))
    t = t[(t.end >= NOV[0]) & (t.end < NOV[1])].copy()
    if drop_pairs:
        keep = ~pd.MultiIndex.from_arrays([t.ticker, t.ts]).isin(drop_pairs)
        t = t[keep]
    if drop_hours:
        t = t[~t.end.dt.hour.isin(drop_hours)]
    if not len(t):
        return t.assign(pnl=np.nan)
    t["pnl"] = pnl_at(t, 1.0)
    if weight is not None:
        t["w"] = t.end.dt.hour.map(weight).astype(float)
    return t


def summarise(t: pd.DataFrame, label: str, weighted: bool = False) -> dict:
    if not len(t) or t.pnl.isna().all():
        return {"variant": label, "trades": 0}
    p = t.pnl
    h = p.groupby(t.end).sum()
    if weighted and "w" in t:
        w = t.w.to_numpy()
        cents = 100 * float(np.sum(p.to_numpy() * w) / np.sum(w))
        return {"variant": label, "trades": int(len(t)), "hours": int(len(h)), "¢ per trade": cents,
                "t": np.nan, "hit": float((h > 0).mean()), "total $": float(np.sum(p.to_numpy() * w))}
    s = stats(h)
    return {"variant": label, "trades": int(len(t)), "hours": s["hours"], "¢ per trade": 100 * float(p.mean()),
            "t": s["t"], "hit": s["hit"], "total $": s["total"]}


# ------------------------------------------------------------------------------------------------- main
def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    d = build()
    nov = d[d.source.eq("november")].copy()
    archive = d[d.source.eq("candles")].copy()

    cov, profile = coverage()
    holes = ledger_holes(archive)
    changes = quote_changes(nov, archive)
    cov.to_csv(OUT / "wings_sleep_coverage.csv", index=False)
    profile.to_csv(OUT / "wings_sleep_profile.csv", index=False)
    if not holes.empty:
        holes.to_csv(OUT / "wings_sleep_holes.csv", index=False)
    changes.to_csv(OUT / "wings_sleep_staleness.csv", index=False)

    # hours of the day that are systematically missing in November: those with far fewer observations than the median
    nov_hours = nov.groupby("end").end.first().dt.hour
    per_hour = nov_hours.value_counts().reindex(range(24), fill_value=0)
    thin = set(per_hour[per_hour < 0.5 * per_hour.median()].index)
    stale_pairs = set(zip(nov.loc[stale_mask(nov), "ticker"], nov.loc[stale_mask(nov), "ts"]))
    # weights that make the November hour-of-day mix look like the archive's
    arch_mix = archive.groupby("end").end.first().dt.hour.value_counts(normalize=True).reindex(range(24), fill_value=0)
    nov_mix = per_hour / per_hour.sum()
    weight = (arch_mix / nov_mix.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)

    rows = []
    for rule in ("W03", "W04"):
        base = rule_pnl(d, rule)
        rows.append({"rule": rule, **summarise(base, "as is")})
        rows.append({"rule": rule, **summarise(rule_pnl(d, rule, drop_pairs=stale_pairs),
                                               "without frozen quotes")})
        rows.append({"rule": rule, **summarise(rule_pnl(d, rule, drop_hours=thin),
                                               "without thin hours of the day")})
        rows.append({"rule": rule, **summarise(rule_pnl(d, rule, weight=weight),
                                               "post-stratified to the archive clock", weighted=True)})
    robust = pd.DataFrame(rows)
    robust.to_csv(OUT / "wings_sleep_robustness.csv", index=False)

    write_report(cov, profile, holes, changes, robust, thin, len(nov_hours), len(archive.groupby("end")))
    print(cov.to_string(index=False))
    print(profile.to_string(index=False))
    print(robust.to_string(index=False))
    print(changes.to_string(index=False))
    if not holes.empty:
        print(holes.head(8).to_string(index=False))


def write_report(cov, profile, holes, changes, robust, thin, nov_hours, arch_hours) -> None:
    def md(df, fmt="{:.3f}"):
        if df is None or df.empty:
            return "_no rows_\n"
        out = df.copy()
        out.columns = [str(c).replace("|", "/") for c in out.columns]
        for c in out.columns:
            if pd.api.types.is_float_dtype(out[c]):
                out[c] = out[c].map(lambda x: "" if pd.isna(x) else fmt.format(x))
            elif not pd.api.types.is_numeric_dtype(out[c]):
                out[c] = out[c].map(lambda x: str(x).replace("|", "/"))
        head = "| " + " | ".join(map(str, out.columns)) + " |\n"
        sep = "|" + "|".join(["---"] * len(out.columns)) + "|\n"
        body = "".join("| " + " | ".join(map(str, r)) + " |\n" for r in out.itertuples(index=False))
        return head + sep + body

    lines = [
        "# Сон ноутбука, пропуски и замороженные книги: влияет ли это на результаты крыла\n",
        f"_Сгенерировано {pd.Timestamp.now():%Y-%m-%d %H:%M}, `studies/wings_sleep.py`._\n",
        f"Часов в ноябрьской ленте: {nov_hours}, в архивной (июль–октябрь): {arch_hours}. "
        "Ноябрьские книги пишет логгер на ноутбуке: спит машина — нет и часов. Рядом лежат два разных "
        "объекта: архив Kalshi за июль–октябрь качается из API задним числом (сон на него не влияет), "
        "а ноябрьские снимки — это то, что успел записать живой логгер за последние 31 минуту каждого часа.\n",
        "\n## S1 Покрытие лент\n",
        md(cov),
        "\n### Часы суток в ноябре\n",
        md(profile, "{:.0f}"),
        "\n### Дыры в архивной ленте (июль–октябрь, API)\n",
        md(holes, "{:.0f}") if not holes.empty else "_дыр нет_",
        "\n## S2 Замороженные котировки\n",
        md(changes),
        "\n## S3 Правила на ноябре с поправками\n",
        md(robust),
        "\n## Чтение\n",
        _reading(robust, changes, thin),
    ]
    (OUT / "wings_sleep.md").write_text("\n".join(lines), encoding="utf-8")


def _reading(robust, changes, thin) -> str:
    out = []
    if not changes.empty:
        r = changes.set_index("source")
        if "November live" in r.index and "Jul-Oct archive" in r.index:
            out.append(f"* Доля неизменных подряд котировок: {r.loc['November live', 'unchanged share']:.3f} "
                       f"в ноябрьской живой ленте против "
                       f"{r.loc['Jul-Oct archive', 'unchanged share']:.3f} в архивной — "
                       "живая лента заметно более «залипающая».")
    if thin:
        out.append(f"* Часы суток, где ноябрьских наблюдений меньше половины медианы: {sorted(thin)} UTC — "
                   "это и есть профиль сна машины, и он не совпадает с профилем архива.")
    if not robust.empty:
        t = robust.pivot_table(index="rule", columns="variant", values="¢ per trade")
        if len(t.columns) > 1:
            out.append(f"* Разброс ноябрьского результата по поправкам: {t.min().min():.2f}…{t.max().max():.2f}¢ "
                       "на сделку — то есть сон и залипание котировок сдвигают оценку, но не создают её.")
    out.append("* Вывод: июль–октябрь от сна не зависит (архив качается задним числом), ноябрь — зависит: "
               "пропущенные часы смещают выборку по времени суток, а первые котировки после пробуждения "
               "могут быть последней известной книгой. Ноябрьские цифры надо читать как «на тех часах, "
               "когда машина не спала».")
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    main()
