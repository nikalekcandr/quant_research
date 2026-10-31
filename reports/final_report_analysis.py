"""Build every table and figure of the final report.

The script reads only files that are in the repository (no network).  It writes

    reports/final/tables/*.csv     derived numbers, one file per table
    reports/final/figures/*.png    figures used in reports/final_report.html

Run from the repository root:  python reports/final_report_analysis.py
"""
from __future__ import annotations

import json
import re
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import FancyBboxPatch, Rectangle

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "final"
FIG = OUT / "figures"
TAB = OUT / "tables"
for p in (FIG, TAB):
    p.mkdir(parents=True, exist_ok=True)

SEC_YEAR = 365 * 24 * 3600

# --------------------------------------------------------------------------- style
INK = "#0b1220"
MUTED = "#5b6472"
GRID = "#e6e9ef"
PANEL = "#f8fafc"
BS = "#2563eb"       # blue - the model with the ladder
VIOLET = "#7c3aed"   # GBM
RED = "#e11d48"      # market ladder
AMBER = "#f59e0b"    # Deribit
TEAL = "#0d9488"     # HAR
SLATE = "#94a3b8"
PINK = "#f9a8d4"
GREEN = "#16a34a"

MODEL_STYLE = {
    "naive": ("Наивный (прошлое окно)", "#cbd5e1", "--"),
    "har": ("HAR", TEAL, "-"),
    "harx": ("HAR-X", "#38bdf8", "-"),
    "harx_d": ("HAR-X + Deribit", AMBER, "-"),
    "har_m": ("HAR + лесенка", "#f472b6", "-"),
    "harx_m": ("HAR-X + лесенка", "#fb7185", "-"),
    "harx_all": ("HAR-X + Deribit + лесенка", BS, "-"),
    "iv_raw": ("Deribit IV (как есть)", "#fbbf24", ":"),
    "iv_mz": ("Deribit IV (калибр.)", AMBER, "--"),
    "mkt_raw": ("Лесенка BJN (как есть)", "#fb7185", ":"),
    "mkt_mz": ("Лесенка BJN (калибр.)", RED, "-"),
    "mkt_ln_raw": ("Лесенка: центр (логнорм.)", "#be123c", "-"),
    "mkt_t_raw": ("Лесенка Student-t", PINK, ":"),
    "combo": ("Смесь равных весов", "#8b5cf6", "-"),
    "gbm_all": ("Бустинг gbm_all", VIOLET, "-"),
    "garch": ("GARCH(1,1)-t", "#a3a3a3", "-"),
}


def style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "figure.dpi": 150,
        "savefig.dpi": 170,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#cbd5e1",
        "axes.labelcolor": MUTED,
        "axes.titlecolor": INK,
        "axes.titlesize": 11.5,
        "axes.titleweight": "bold",
        "axes.labelsize": 9.5,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "lines.linewidth": 1.7,
        "lines.solid_capstyle": "round",
    })


style()


def save(fig, name: str, pad=0.25):
    fig.savefig(FIG / name, bbox_inches="tight", pad_inches=pad, facecolor="white")
    plt.close(fig)
    print("  figure", name)


def _naive(x):
    s = pd.to_datetime(x)
    return s.dt.tz_localize(None) if s.dt.tz is not None else s


def _model_series(d: pd.DataFrame, model: str, freq: str) -> pd.DataFrame:
    """Mean variance per calendar bucket, then annualised volatility."""
    x = d[d["model"] == model].copy()
    if x.empty:
        return pd.DataFrame(columns=["bucket", "act", "pred", "n"])
    x["bucket"] = _naive(x["T"]).dt.to_period(freq).dt.start_time
    g = x.groupby("bucket").agg(act_var=("av", "mean"), pred_var=("pv", "mean"), n=("av", "size"))
    g["act"] = np.sqrt(g["act_var"])
    g["pred"] = np.sqrt(g["pred_var"])
    return g.reset_index()


def _actual_series(d: pd.DataFrame, freq: str, models=("har",)) -> pd.DataFrame:
    x = d[d["model"].isin(models)].copy()
    x["bucket"] = _naive(x["T"]).dt.to_period(freq).dt.start_time
    g = x.groupby("bucket").agg(act_var=("av", "mean"), n=("av", "size"))
    g["act"] = np.sqrt(g["act_var"])
    return g.reset_index()


# --------------------------------------------------------------------------- data
def load_forecasts(h: str) -> pd.DataFrame:
    d = pd.read_parquet(ROOT / f"forecasts_{h}.parquet")
    d["t"] = pd.to_datetime(d["t"])
    d["T"] = pd.to_datetime(d["T"])
    d["dur"] = (d["T"] - d["t"]).dt.total_seconds() - (40.0 if h == "kx1h" else 0.0)
    for k in (1, 2, 3):
        for c, fill in (("w", 0.0), ("m", 0.0), ("s", 0.0)):
            d[f"{c}{k}"] = pd.to_numeric(d.get(f"{c}{k}"), errors="coerce").fillna(fill)
    wsum = d[["w1", "w2", "w3"]].sum(axis=1).replace(0, np.nan)
    d["ev"] = sum(d[f"w{k}"] * np.exp(d[f"m{k}"] + 0.5 * d[f"s{k}"] ** 2) for k in (1, 2, 3)) / wsum
    d["av"] = d["rv"] * SEC_YEAR / d["dur"]   # annualised variance, realised
    d["pv"] = d["ev"] * SEC_YEAR / d["dur"]   # annualised variance, forecast (E[RV])
    return d.dropna(subset=["av", "pv"])


def load_panel(h: str) -> pd.DataFrame:
    p = pd.read_parquet(ROOT / "data" / "interim" / f"panel_{h}.parquet")
    p["t"] = pd.to_datetime(p["t"])
    p["T"] = pd.to_datetime(p["T"])
    p["dur"] = (p["T"] - p["t"]).dt.total_seconds() - (40.0 if h == "kx1h" else 0.0)
    p["act"] = np.sqrt(p["rv"] * SEC_YEAR / p["dur"])
    if "mkt_mfiv" in p:
        p["ladder"] = np.sqrt(p["mkt_mfiv"] * SEC_YEAR / p["dur"])
    if "l_iv_var" in p and p["l_iv_var"].notna().any():
        p["deribit"] = np.sqrt(np.exp(p["l_iv_var"]) * SEC_YEAR / p["dur"])
    if "dvol" in p:
        p["dvol_vol"] = p["dvol"] / 100.0
    return p


# --------------------------------------------------------------------------- tables
def table_data_inventory():
    rows = [
        ("Binance BTC/USDT, 1 мин", "2023-01-01", "2026-10-29", "2 011 811 минут",
         "доходности, реализованная дисперсия, активность", "data/raw/binance, data/interim/minutes.parquet"),
        ("Deribit DVOL (30 дней)", "2023-02-01", "2026-10-29", "32 763 снимка",
         "рыночный прогноз волатильности №1", "data/raw/deribit/dvol.csv.gz"),
        ("Deribit, сделки опционов", "2025-06-01", "2026-10-29", "516 дневных файлов",
         "ATM IV и полная дисперсия на 24 ч / 6 дн", "data/raw/deribit/trades"),
        ("Polymarket, дневные лесенки", "2025-08-12", "2026-10-30", "406 / 412 снимков (24 ч / 6 дн)",
         "рыночный прогноз волатильности №2", "data/raw/polymarket, data/interim/pm_ladder.parquet"),
        ("Kalshi, часовые лесенки KXBTCD", "2025-12-01", "2026-10-28", "6 891 час, 1.03 млн строк свечей",
         "главный горизонт 55 минут", "data/raw/kalshi*, data/interim/kalshi.parquet"),
        ("Kalshi, 15-минутные рынки KXBTC15M", "2025-12-01", "2026-10-28", "30 141 рынок",
         "проверка вероятностей страйков", "data/raw/kx15/markets.csv"),
        ("Polymarket 5m / 15m Up-Down", "2026-10-18", "2026-10-29", "живые стаканы каждые 15 с",
         "лаборатория альфы", "data/lab/markets_fast"),
        ("Лаборатория (7 площадок)", "2026-10-28 23:37", "2026-10-29 12:00", "2 122 открытия, 2 399 закрытий",
         "51 замороженная гипотеза", "data/lab/ledger.csv"),
        ("Фандинг перпетуалов", "2023-01-01", "2026-10-29", "дневные ставки",
         "cash-and-carry, R11", "data/raw/funding"),
    ]
    df = pd.DataFrame(rows, columns=["Источник", "От", "До", "Объём", "Что даёт", "Где лежит"])
    df.to_csv(TAB / "data_inventory.csv", index=False)
    return df


def table_scoreboard():
    sc = pd.read_csv(ROOT / "results" / "tables" / "scores_common_sample.csv")
    keep = ["naive", "har", "harx", "harx_d", "har_m", "harx_m", "harx_all", "gbm_all",
            "mkt_ln_raw", "mkt_mz", "mkt_raw", "mkt_t_raw", "iv_raw", "iv_mz", "combo", "garch"]
    out = []
    for h in ["kx1h", "pm24h", "pm6d"]:
        x = sc[(sc.horizon == h) & (sc.model.isin(keep))].copy()
        x = x.sort_values("crps")
        x["horizon"] = h
        x["model_label"] = x.model.map(lambda m: MODEL_STYLE.get(m, (m,))[0])
        out.append(x)
    df = pd.concat(out, ignore_index=True)
    df.to_csv(TAB / "scoreboard.csv", index=False)
    return df


def table_recent_oos():
    """Common sample of the last out-of-sample stretch, recomputed from scores.parquet."""
    sc = pd.read_parquet(ROOT / "scores.parquet")
    sc["t"] = pd.to_datetime(sc["t"])
    models = ["har", "harx", "harx_d", "har_m", "harx_m", "harx_all", "gbm_all",
              "mkt_ln_raw", "mkt_mz", "mkt_raw", "iv_raw", "iv_mz", "combo", "naive"]
    rows = []
    for h in ["kx1h", "pm24h", "pm6d"]:
        d = sc[(sc.horizon == h) & (sc.model.isin(models)) & (sc.t >= pd.Timestamp("2026-10-13", tz="UTC"))]
        if d.empty:
            continue
        common = set.intersection(*[set(d[d.model == m].t) for m in models if (d.model == m).any()])
        d = d[d.t.isin(common)]
        for m, g in d.groupby("model"):
            rows.append({"horizon": h, "model": m, "n": len(g), "crps": g.crps.mean(),
                         "qlike": g.qlike.mean(), "logscore": g.logscore.mean(),
                         "cov50": g.in50.mean(), "cov90": g.in90.mean()})
    df = pd.DataFrame(rows)
    for h in df.horizon.unique():
        base = df[(df.horizon == h) & (df.model == "har")].crps
        if len(base):
            df.loc[df.horizon == h, "crps_vs_har_pct"] = 100 * (df[df.horizon == h].crps / base.iloc[0] - 1)
    df["model_label"] = df.model.map(lambda m: MODEL_STYLE.get(m, (m,))[0])
    df.to_csv(TAB / "recent_oos.csv", index=False)
    return df


def table_dm_key():
    dm = pd.read_csv(ROOT / "results" / "tables" / "diebold_mariano.csv")
    # keep the encompassing chain and the key tests, CRPS + QLIKE only
    want = [
        "HAR vs naive", "time-series extensions", "adding Deribit",
        "adding the ladder on top of Deribit (key test)", "adding the ladder alone",
        "ladder vs Deribit as standalone forecasts", "combination vs best regression",
        "gradient-boosted trees vs the same features in OLS",
    ]
    x = dm[dm.comparison.isin(want) & dm.loss.isin(["qlike", "crps"])].copy()
    x["A_label"] = x.A.map(lambda m: MODEL_STYLE.get(m, (m,))[0])
    x["B_label"] = x.B.map(lambda m: MODEL_STYLE.get(m, (m,))[0])
    x.to_csv(TAB / "dm_key.csv", index=False)
    return x


def table_horserace():
    df = pd.read_csv(ROOT / "results" / "studies" / "vol_horserace.csv")
    df.to_csv(TAB / "horserace.csv", index=False)
    return df


def table_implied_bias():
    df = pd.read_csv(ROOT / "results" / "tables" / "implied_bias.csv")
    df.to_csv(TAB / "implied_bias.csv", index=False)
    return df


def table_brier():
    lad = pd.read_csv(ROOT / "results" / "tables" / "ladder_scores.csv")
    dm = pd.read_csv(ROOT / "results" / "tables" / "ladder_dm.csv")
    lad.to_csv(TAB / "brier.csv", index=False)
    dm.to_csv(TAB / "brier_dm.csv", index=False)
    return lad, dm


def table_regimes():
    txt = (ROOT / "reports" / "research.md").read_text(encoding="utf-8", errors="ignore")
    rows = re.findall(r"\|\s*(20\d\d-\d\d-\d\d)\s*\|\s*(20\d\d-\d\d-\d\d)\s*\|\s*(\d+)\s*\|\s*([\d.]+)\s*\|", txt)
    df = pd.DataFrame(rows, columns=["from", "to", "weeks", "ann_vol"])
    df["from"] = pd.to_datetime(df["from"])
    df["to"] = pd.to_datetime(df["to"])
    for c in ("weeks", "ann_vol"):
        df[c] = pd.to_numeric(df[c])
    df.to_csv(TAB / "regimes.csv", index=False)
    return df


def table_research():
    txt = (ROOT / "reports" / "research.md").read_text(encoding="utf-8", errors="ignore")
    block = txt.split("## 2. Strategies")[0]
    lines = [l for l in block.splitlines() if re.match(r"^\|\s*R\d", l)]
    rows = []
    for l in lines:
        cells = [c.strip() for c in l.strip().strip("|").split("|")]
        if len(cells) < 13:
            continue
        rows.append({
            "id": cells[0], "hypothesis": cells[1], "effect": cells[2], "unit": cells[3],
            "t": cells[4], "p": cells[5], "holm": cells[6], "bh": cells[7], "n": cells[8],
            "stable": cells[9], "verdict": cells[10],
        })
    df = pd.DataFrame(rows)
    for c in ("effect", "t", "p", "holm", "bh", "n"):
        df[c] = pd.to_numeric(df[c].str.replace(",", ""), errors="coerce")
    df.to_csv(TAB / "research.csv", index=False)
    return df


def table_lab():
    txt = (ROOT / "reports" / "lab.md").read_text(encoding="utf-8", errors="ignore")
    lines = [l for l in txt.splitlines() if re.match(r"^\|\s*[A-Z]\d", l)]
    rows = []
    for l in lines:
        cells = [c.strip() for c in l.strip().strip("|").split("|")]
        if len(cells) < 15:
            continue
        rows.append({
            "id": cells[0], "family": cells[1], "hypothesis": cells[2], "unit": cells[3],
            "trades": cells[4], "clusters": cells[5], "open": cells[6], "mean_net": cells[7],
            "t": cells[8], "holm": cells[9], "hit": cells[10], "total_net": cells[11],
            "mean_gross": cells[12], "dsr": cells[13], "verdict": cells[14],
        })
    df = pd.DataFrame(rows)
    for c in ("trades", "clusters", "open"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in ("mean_net", "t", "holm", "hit", "total_net"):
        df[c] = pd.to_numeric(df[c].replace({"—": np.nan, "": np.nan}), errors="coerce")
    df.to_csv(TAB / "lab.csv", index=False)
    return df


def table_seasonality():
    p = load_panel("kx1h")
    q = p[p["t"] >= pd.Timestamp("2025-12-01", tz="UTC")].copy()
    q["hour"] = q["t"].dt.hour
    g = q.groupby("hour").agg(n=("act", "size"), realized=("act", "median"),
                              ladder=("ladder", "median"), dvol=("dvol_vol", "median"))
    g[["realized", "ladder", "dvol"]] *= 100
    g.to_csv(TAB / "seasonality.csv", index=True)
    return g.reset_index()


# --------------------------------------------------------------------------- figures
def fig_data_timeline():
    rows = [
        ("Binance 1m · 2.01 млн минут", "2023-01-01", "2026-10-29", "#0f172a", "2 011 811"),
        ("Deribit DVOL", "2023-02-01", "2026-10-29", AMBER, "32 763"),
        ("Фандинг перпетуалов", "2023-01-01", "2026-10-29", "#a3a3a3", "дневные"),
        ("Deribit · сделки опционов", "2025-06-01", "2026-10-29", "#fbbf24", "516 файлов"),
        ("Polymarket · дневные лесенки", "2025-08-12", "2026-10-30", RED, "406 / 412"),
        ("Kalshi · 15-минутные рынки", "2025-12-01", "2026-10-28", "#be123c", "30 141 рынок"),
        ("Kalshi · часовые лесенки", "2025-12-01", "2026-10-28", BS, "6 891 час"),
        ("Polymarket 5m / 15m · живые", "2026-10-28", "2026-10-29", VIOLET, "15-сек стаканы"),
        ("Лаборатория · 7 площадок", "2026-10-28", "2026-10-29", TEAL, "51 гипотеза"),
    ]
    fig, ax = plt.subplots(figsize=(11.4, 4.6))
    for i, (name, a, b, color, volume) in enumerate(rows):
        y = len(rows) - 1 - i
        start = pd.Timestamp(a)
        end = pd.Timestamp(b)
        # a minimum visible width for one-day sources
        end_plot = max(end, start + pd.Timedelta(days=9))
        ax.barh(y, (end_plot - start).days, left=start, height=0.52, color=color, alpha=0.9,
                edgecolor="white", linewidth=0.8)
        ax.text(start - pd.Timedelta(days=12), y, name, ha="right", va="center", fontsize=9, color=INK)
        ax.text(end_plot + pd.Timedelta(days=12), y, volume, ha="left", va="center", fontsize=8, color=MUTED)
    ax.set_yticks([])
    ax.set_xlim(pd.Timestamp("2022-06-01"), pd.Timestamp("2027-03-01"))
    ax.set_ylim(-1.5, len(rows) - 0.3)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
    ax.grid(axis="x", color=GRID)
    ax.grid(axis="y", visible=False)
    ax.set_title("Покрытие данных: три года минут, восемь месяцев лесенок, двое суток лаборатории",
                 loc="left", pad=12)
    ax.set_xlabel("")
    ax.axvspan(pd.Timestamp("2025-12-01"), pd.Timestamp("2026-10-30"), color=BS, alpha=0.04, zorder=0)
    ax.text(pd.Timestamp("2026-05-15"), -1.05, "эпоха лесенок", ha="center", va="center",
            fontsize=8.5, color=BS, style="italic")
    for s in ("left", "bottom"):
        ax.spines[s].set_visible(False)
    save(fig, "fig01_data_timeline.png")


def fig_vol_full():
    d = load_forecasts("kx1h")
    monthly = _actual_series(d, "M", models=("har",))
    models = [("har", "HAR"), ("gbm_all", "бустинг"), ("iv_mz", "DVOL (Deribit)")]
    fig, ax = plt.subplots(figsize=(11.6, 4.6))
    ax.fill_between(monthly.bucket, 0, monthly.act * 100, color="#e2e8f0", alpha=0.9, linewidth=0)
    ax.plot(monthly.bucket, monthly.act * 100, color=INK, lw=1.8, label="реализованная (месяц)")
    for m, lab in models:
        s = _model_series(d, m, "M")
        ax.plot(s.bucket, s["pred"] * 100, color=MODEL_STYLE[m][1], lw=1.6, alpha=0.95, label=lab)
    ax.set_ylabel("годовая волатильность, %")
    ax.set_ylim(0, 78)
    ax.set_title("Kalshi, горизонт 55 минут: что предсказывали модели и что случилось, 2023–2026",
                 loc="left", pad=10)
    ax.legend(loc="upper center", ncol=4, bbox_to_anchor=(0.5, -0.13))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    save(fig, "fig02_vol_full_history.png")


def fig_vol_2026():
    d = load_forecasts("kx1h")
    start = pd.Timestamp("2026-01-01", tz="UTC")
    d = d[d["T"] >= start]
    weekly = _actual_series(d, "W", models=("har",))
    methods = [("har", "HAR"), ("gbm_all", "Бустинг"), ("harx_all", "HAR-X + Deribit + лесенка"),
               ("mkt_ln_raw", "Лесенка: центр (логнорм.)"), ("iv_mz", "Deribit / DVOL")]
    fig, ax = plt.subplots(figsize=(11.6, 4.6))
    ax.fill_between(weekly.bucket, 0, weekly.act * 100, color="#eef2f7", alpha=1, linewidth=0,
                    label="реализованная (неделя)")
    ax.plot(weekly.bucket, weekly.act * 100, color=INK, lw=1.6, label="реализованная")
    for m, lab in methods:
        s = _model_series(d, m, "W")
        ax.plot(s.bucket, s["pred"] * 100, color=MODEL_STYLE[m][1], lw=1.8, label=lab)
    ax.set_ylabel("годовая волатильность, %")
    ax.set_ylim(0, 125)
    ax.set_title("2026 год: прогнозы волатильности на час вперёд, недельные средние", loc="left", pad=10)
    ax.legend(loc="upper center", ncol=5, bbox_to_anchor=(0.5, -0.12))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    # annotate the last out-of-sample stretch
    ax.axvspan(pd.Timestamp("2026-10-13"), pd.Timestamp("2026-10-30"), color=BS, alpha=0.06, zorder=0)
    ax.annotate("новая выборка\n13–30.10", xy=(pd.Timestamp("2026-10-21"), 100), ha="center",
                fontsize=8, color=BS)
    save(fig, "fig03_vol_2026.png")


def fig_vol_last14():
    d = load_forecasts("kx1h")
    end = d["T"].max()
    start = end - pd.Timedelta(days=14)
    d = d[d["T"] >= start]
    methods = [("harx_all", "HAR-X + Deribit + лесенка"), ("mkt_ln_raw", "Лесенка: центр (логнорм.)"),
               ("iv_mz", "Deribit / DVOL")]
    base = d[d.model == "har"].sort_values("T")[["T", "av"]].dropna()
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(11.6, 6.4), sharex=True,
                                  gridspec_kw={"height_ratios": [2.3, 1]})
    xb = _naive(base["T"]).values
    ab = base.av.pow(0.5).values * 100
    ax.fill_between(xb, 0, ab, color="#eef2f7", linewidth=0)
    ax.plot(xb, ab, color=INK, lw=1.0, alpha=0.5)
    ax.scatter(xb, ab, s=9, color=INK, alpha=0.45, zorder=3, label="реализованная за час")
    for m, lab in methods:
        s = d[d.model == m].sort_values("T").dropna(subset=["pv"])
        ax.plot(_naive(s["T"]).values, s.pv.pow(0.5).values * 100, color=MODEL_STYLE[m][1],
                lw=1.9, label=lab)
        mg = s[["T", "pv"]].merge(base, on="T", how="inner").sort_values("T")
        err = (mg.pv.pow(0.5) / mg.av.pow(0.5) - 1) * 100
        ax2.plot(_naive(mg["T"]).values, err.rolling(6, min_periods=2, center=True).mean(),
                 color=MODEL_STYLE[m][1], lw=1.8)
    ax.set_ylabel("годовая волатильность, %")
    ax.set_ylim(0, 150)
    ax.set_title("Последние две недели, каждый час: реализованная волатильность против прогнозов",
                 loc="left", pad=10)
    ax2.axhline(0, color=INK, lw=1)
    ax2.axhspan(-15, 15, color=GREEN, alpha=0.07)
    ax2.set_ylabel("ошибка прогноза, %")
    ax2.set_ylim(-100, 240)
    ax2.set_xlabel("")
    ax2.xaxis.set_major_locator(mdates.DayLocator(interval=2))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%d.%m"))
    ax2.text(0.01, 0.04, "6-часовое среднее (прогноз / факт − 1); зелёная полоса ±15%", transform=ax2.transAxes,
             ha="left", fontsize=8, color=MUTED)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    save(fig, "fig04_vol_last14.png")


def fig_vol_pm():
    fig, axes = plt.subplots(2, 1, figsize=(11.6, 7.2), sharex=True)
    cfg = {
        "pm24h": [("har", "HAR"), ("harx_d", "HAR-X + Deribit"), ("harx_all", "HAR-X + Deribit + лесенка"),
                  ("mkt_ln_raw", "Лесенка: центр (логнорм.)"), ("iv_mz", "Deribit ATM (калибр.)")],
        "pm6d": [("har", "HAR"), ("harx_d", "HAR-X + Deribit"), ("harx_all", "HAR-X + Deribit + лесенка"),
                 ("mkt_ln_raw", "Лесенка: центр (логнорм.)"), ("iv_mz", "Deribit ATM (калибр.)")],
    }
    titles = {"pm24h": "Polymarket, 24 часа", "pm6d": "Polymarket, 6 дней"}
    for ax, h in zip(axes, ["pm24h", "pm6d"]):
        d = load_forecasts(h)
        d = d[d["T"] >= pd.Timestamp("2024-06-01", tz="UTC")]
        weekly = _actual_series(d, "W", models=("har",))
        ax.fill_between(weekly.bucket, 0, weekly.act * 100, color="#eef2f7", linewidth=0)
        ax.plot(weekly.bucket, weekly.act * 100, color=INK, lw=1.6, label="реализованная (неделя)")
        for m, lab in cfg[h]:
            s = _model_series(d, m, "W")
            ax.plot(s.bucket, s["pred"] * 100, color=MODEL_STYLE[m][1], lw=1.7, label=lab)
        ax.set_title(titles[h], loc="left")
        ax.set_ylabel("годовая волатильность, %")
        ax.set_ylim(0, 150)
    axes[0].set_title("Polymarket, 24 часа: рынок, опционы и модели", loc="left", pad=10)
    axes[1].set_title("Polymarket, 6 дней: на длинном горизонте все методы сходятся", loc="left", pad=10)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    axes[1].xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%m.%y"))
    save(fig, "fig05_vol_pm.png")


def fig_scatter():
    d = load_forecasts("kx1h")
    models = [("har", "HAR"), ("harx_all", "HAR-X + Deribit + лесенка"),
              ("mkt_ln_raw", "Лесенка: центр (логнорм.)"), ("iv_mz", "Deribit / DVOL")]
    # common sample so the panels are comparable
    common = set.intersection(*[set(d[d.model == m].t) for m, _ in models])
    dd = d[d.t.isin(common)]
    sc_common = pd.read_csv(ROOT / "results" / "tables" / "scores_common_sample.csv")
    sc_common = sc_common[sc_common.horizon == "kx1h"]
    fig, axes = plt.subplots(2, 2, figsize=(9.6, 8.2))
    for ax, (m, lab) in zip(axes.ravel(), models):
        s = dd[dd.model == m]
        x, y = s.pv.pow(0.5) * 100, s.av.pow(0.5) * 100
        hb = ax.hexbin(x, y, gridsize=42, bins="log", cmap="Blues", mincnt=1, linewidth=0.2,
                       extent=(0, 160, 0, 160))
        ax.plot([0, 160], [0, 160], color=RED, lw=1.2, ls="--")
        r2 = np.corrcoef(np.log(x), np.log(y))[0, 1] ** 2
        ax.set_xlim(0, 150)
        ax.set_ylim(0, 150)
        ax.set_title(lab, loc="left", fontsize=10.5)
        ql = sc_common[(sc_common.model == m)]
        qtxt = f"QLIKE {ql.qlike.iloc[0]:.2f} · CRPS {ql.crps.iloc[0]:.2f}" if len(ql) else ""
        ax.text(0.04, 0.95, f"R² (log) = {r2:.2f}\n{qtxt}", transform=ax.transAxes, va="top", fontsize=8.4,
                color=INK, bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#e2e8f0"))
        ax.text(0.96, 0.06, "1:1", transform=ax.transAxes, ha="right", fontsize=8, color=RED)
        ax.set_xlabel("прогноз, % годовых")
        ax.set_ylabel("факт, % годовых")
        cb = fig.colorbar(hb, ax=ax, pad=0.01, fraction=0.046)
        cb.ax.tick_params(labelsize=7)
        cb.set_label("часов", fontsize=7.5)
    fig.suptitle("Прогноз против факта, часовые окна Kalshi (одна и та же выборка, n=%d)" % len(common),
                 x=0.02, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    save(fig, "fig06_scatter.png")


def fig_bias_heatmap():
    d = load_forecasts("kx1h")
    d = d[(d["T"] >= pd.Timestamp("2026-01-01", tz="UTC"))]
    models = ["naive", "har", "harx_d", "harx_all", "gbm_all", "mkt_mz", "mkt_ln_raw", "iv_mz"]
    months = pd.period_range("2026-01", "2026-10", freq="M")
    M = np.full((len(models), len(months)), np.nan)
    for i, m in enumerate(models):
        s = d[d.model == m].copy()
        s["mon"] = _naive(s["T"]).dt.to_period("M")
        g = s.groupby("mon").apply(lambda x: np.log(x.av.mean() / x.pv.mean()), include_groups=False)
        for j, mo in enumerate(months):
            if mo in g.index:
                M[i, j] = g.loc[mo]
    fig, ax = plt.subplots(figsize=(10.8, 4.4))
    cmap = LinearSegmentedColormap.from_list("bias", ["#1d4ed8", "#93c5fd", "#ffffff", "#fca5a5", "#b91c1c"])
    norm = TwoSlopeNorm(vmin=-0.55, vcenter=0, vmax=0.55)
    im = ax.imshow(M, aspect="auto", cmap=cmap, norm=norm)
    ax.set_xticks(range(len(months)), [m.strftime("%b") for m in months])
    ax.set_yticks(range(len(models)), [MODEL_STYLE[m][0] for m in models])
    for i in range(len(models)):
        for j in range(len(months)):
            if np.isfinite(M[i, j]):
                ax.text(j, i, f"{M[i, j]:+.2f}", ha="center", va="center", fontsize=7.6,
                        color="white" if abs(M[i, j]) > 0.30 else INK)
    ax.set_title("Смещение прогноза: log(факт / прогноз), часовые окна 2026", loc="left", pad=10)
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    cb.set_label("> 0 — модель занизила волатильность", fontsize=8)
    ax.set_xlabel("месяц 2026")
    save(fig, "fig07_bias_heatmap.png")


def fig_seasonality():
    g = table_seasonality()
    fig, ax = plt.subplots(figsize=(10.8, 4.0))
    ax.fill_between(g.hour, 0, g.realized, color="#e2e8f0", alpha=0.7, linewidth=0)
    ax.plot(g.hour, g.realized, color=INK, lw=2.2, marker="o", ms=3.5, label="реализованная (медиана)")
    ax.plot(g.hour, g.ladder, color=RED, lw=2.2, marker="o", ms=3.5, label="лесенка Kalshi")
    ax.plot(g.hour, g.dvol, color=AMBER, lw=2.0, ls="--", label="Deribit DVOL (30 дней) — плоский")
    ax.set_xticks(range(0, 24, 2))
    ax.set_xlabel("час UTC")
    ax.set_ylabel("годовая волатильность, %")
    ax.set_title("Почему лесенка точнее на часе: она знает внутридневную сезонность, DVOL — нет",
                 loc="left", pad=10)
    ax.legend(loc="upper left", ncol=3)
    ax.axvspan(13, 16, color=AMBER, alpha=0.06, zorder=0)
    ax.annotate("открытие США", xy=(14.5, 63), ha="center", fontsize=8.5, color=AMBER)
    ax.annotate("азиатская ночь", xy=(6, 20), ha="center", fontsize=8.5, color=MUTED)
    ax.set_ylim(15, 70)
    save(fig, "fig08_seasonality.png")


def fig_scores_heatmap():
    sc = pd.read_csv(TAB / "scoreboard.csv")
    models = ["naive", "har", "harx", "harx_d", "harx_all", "gbm_all", "mkt_ln_raw", "mkt_mz", "iv_raw", "iv_mz", "combo"]
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.6))
    for ax, metric, title, better in [(axes[0], "crps", "CRPS плотности (ниже — лучше)", "min"),
                                      (axes[1], "qlike", "QLIKE точечного прогноза (ниже — лучше)", "min")]:
        M = np.full((len(models), 3), np.nan)
        for j, h in enumerate(["kx1h", "pm24h", "pm6d"]):
            s = sc[sc.horizon == h].set_index("model")
            for i, m in enumerate(models):
                if m in s.index:
                    M[i, j] = s.loc[m, metric]
        rel = M / np.nanmin(M, axis=0) - 1
        cmap = LinearSegmentedColormap.from_list("rank", ["#dcfce7", "#fef9c3", "#fecaca", "#ef4444"])
        im = ax.imshow(rel, aspect="auto", cmap=cmap, vmin=0, vmax=0.9)
        ax.set_xticks(range(3), ["55 мин", "24 часа", "6 дней"])
        ax.set_yticks(range(len(models)), [MODEL_STYLE[m][0] for m in models])
        for i in range(len(models)):
            for j in range(3):
                if np.isfinite(M[i, j]):
                    ax.text(j, i, f"{M[i, j]:.3f}", ha="center", va="center", fontsize=8.2, color=INK)
        ax.set_title(title, loc="left", fontsize=10.5)
        ax.grid(False)
    fig.suptitle("Кто точнее: сводная таблица потерь на общей выборке (лучшее в горизонте = 1.00)",
                 x=0.02, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    save(fig, "fig09_scores_heatmap.png")


def fig_dm_forest():
    dm = pd.read_csv(ROOT / "results" / "tables" / "oos_dm.csv")
    dm = dm[(dm["loss"] == "crps") & (dm["sample"] == "history")].copy()
    labels = {
        ("har", "naive"): "HAR против наивного",
        ("harx", "har"): "HAR-X против HAR",
        ("harx_d", "harx"): "+ Deribit к HAR-X",
        ("harx_all", "harx_d"): "+ лесенка к HAR-X + Deribit  ★",
        ("combo", "harx_all"): "смесь против лучшей модели",
        ("harx_all", "mkt_ln_raw"): "модель против центра лесенки",
        ("harx_d", "mkt_ln_raw"): "HAR-X + Deribit против центра лесенки",
    }
    hz_order = {"kx1h": "55 минут", "pm24h": "24 часа", "pm6d": "6 дней"}
    fig, ax = plt.subplots(figsize=(9.8, 5.8))
    ypos, ylabels = [], []
    y = 0
    for h in ["kx1h", "pm24h", "pm6d"]:
        sub = dm[dm.horizon == h]
        if sub.empty:
            continue
        for _, r in sub.iterrows():
            key = (r.A, r.B)
            if key not in labels:
                continue
            sig = r["p"] < 0.05
            color = BS if (r.dm < 0 and sig) else (RED if (r.dm > 0 and sig) else SLATE)
            ax.barh(y, r.dm, color=color, height=0.62, alpha=0.92)
            ax.text(r.dm + (0.5 if r.dm >= 0 else -0.5), y, f"{r.dm:+.1f}", va="center",
                    ha="left" if r.dm >= 0 else "right", fontsize=7.8, color=INK)
            ypos.append(y)
            ylabels.append(f"{hz_order[h]} · {labels[key]}")
            y += 1
        y += 0.9
    ax.set_yticks(ypos, ylabels, fontsize=8.2)
    ax.invert_yaxis()
    ax.axvline(0, color=INK, lw=1)
    for xx in (-2, 2):
        ax.axvline(xx, color=SLATE, lw=0.8, ls=":")
    ax.set_xlabel("статистика Диболда–Мариано по CRPS (слева — лучше)")
    ax.set_title("Значимость различий: что добавляет каждый блок модели", loc="left", pad=10)
    ax.set_xlim(-30, 20)
    ax.text(0.985, 0.02, "синий: значимо лучше (p < 0.05)\nкрасный: значимо хуже\nсерый: незначимо",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8, color=MUTED,
            bbox=dict(boxstyle="round,pad=0.4", fc="#f8fafc", ec="#e2e8f0"))
    save(fig, "fig10_dm_forest.png")


def fig_brier():
    lad = pd.read_csv(TAB / "brier.csv")
    dm = pd.read_csv(TAB / "brier_dm.csv")
    hz = ["kx1h", "pm24h", "pm6d"]
    names = {"harx": "HAR-X", "harx_d": "HAR-X + Deribit", "harx_all": "HAR-X + Deribit + лесенка"}
    colors = {"harx": "#38bdf8", "harx_d": AMBER, "harx_all": BS}
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11.4, 4.2), gridspec_kw={"width_ratios": [1, 1.7]})
    # left: absolute Brier
    src_names = {"market": "рынок", "harx": "HAR-X", "harx_d": "+ Deribit", "harx_all": "+ лесенка"}
    x = np.arange(len(hz))
    w = 0.2
    for k, src in enumerate(["market", "harx", "harx_d", "harx_all"]):
        vals = []
        for h in hz:
            v = lad[(lad.horizon == h) & (lad.source == src)].brier_per_strike.values
            vals.append(v[0] if len(v) else np.nan)
        ax.bar(x + (k - 1.5) * w, vals, w, label=src_names[src],
               color=INK if src == "market" else colors[src], alpha=0.95)
    ax.set_xticks(x, ["55 мин", "24 ч", "6 дн"])
    ax.set_ylabel("Brier на страйк")
    ax.set_title("Абсолютный уровень", loc="left", fontsize=10.5)
    ax.text(0.02, 0.96, "чёрный — рынок\nцвет — модель", transform=ax.transAxes, va="top",
            fontsize=7.6, color=MUTED)
    # right: delta vs market
    x2 = np.arange(len(hz))
    w2 = 0.24
    for k, src in enumerate(["harx", "harx_d", "harx_all"]):
        vals = []
        for h in hz:
            m = lad[(lad.horizon == h) & (lad.source == "market")].brier_per_strike.values
            v = lad[(lad.horizon == h) & (lad.source == src)].brier_per_strike.values
            vals.append((v[0] - m[0]) * 1e4 if len(v) and len(m) else np.nan)
        ax2.bar(x2 + (k - 1) * w2, vals, w2, color=colors[src], label=names[src])
        for xi, v, h in zip(x2 + (k - 1) * w2, vals, hz):
            t = dm[(dm.horizon == h) & (dm.model == src) & (dm.loss == "brier")]["dm"].values
            if np.isfinite(v):
                txt = f"{v:+.1f}"
                if len(t):
                    txt += f"\n(t {t[0]:+.1f})"
                ax2.text(xi, v + (0.2 if v >= 0 else -0.2), txt, ha="center",
                         va="bottom" if v >= 0 else "top", fontsize=6.8, color=INK)
    ax2.axhline(0, color=INK, lw=1)
    ax2.set_xticks(x2, ["55 мин", "24 часа", "6 дней"])
    ax2.set_ylabel("Δ Brier к рынку, ×10⁻⁴")
    ax2.set_title("Разница с рынком (положительное = модель хуже)", loc="left", fontsize=10.5)
    ax2.legend(fontsize=7.6, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    fig.suptitle("Вероятности страйков: на часе рынок лучше, на длинных горизонтах — ничья",
                 x=0.02, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    save(fig, "fig11_brier.png")


def fig_pit():
    sc = pd.read_parquet(ROOT / "scores.parquet")
    sc["t"] = pd.to_datetime(sc["t"])
    fig, axes = plt.subplots(3, 3, figsize=(9.6, 7.4), sharex=True, sharey=True)
    hz = ["kx1h", "pm24h", "pm6d"]
    models = [("har", "HAR"), ("harx_all", "HAR-X + Deribit + лесенка"), ("mkt_ln_raw", "Центр лесенки")]
    for i, h in enumerate(hz):
        for j, (m, lab) in enumerate(models):
            ax = axes[i, j]
            p = sc[(sc.horizon == h) & (sc.model == m)].pit.dropna()
            if len(p):
                ax.hist(p, bins=10, range=(0, 1), weights=np.full(len(p), 1 / len(p)),
                        color=BS if j == 1 else (RED if j == 2 else TEAL), alpha=0.85, edgecolor="white")
            ax.axhline(0.1, color=MUTED, lw=0.9, ls="--")
            ax.set_ylim(0, 0.2)
            if i == 0:
                ax.set_title(lab, fontsize=9.5)
            if j == 0:
                ax.set_ylabel(h, fontsize=9)
    fig.suptitle("PIT: равномерность = калибровка плотности (пунктир — идеал)", x=0.02, ha="left",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save(fig, "fig12_pit.png")


def fig_horserace():
    df = pd.read_csv(TAB / "horserace.csv")
    hz = ["15 minutes", "1 hour", "1 day"]
    hz_ru = {"15 minutes": "15 минут", "1 hour": "1 час", "1 day": "1 день"}
    models = ["HAR", "HAR deseasonalised", "HAR + activity", "GBM", "EWMA", "GARCH(1,1)"]
    colors = {"HAR": TEAL, "HAR deseasonalised": "#38bdf8", "HAR + activity": "#94a3b8",
              "GBM": VIOLET, "EWMA": "#fbbf24", "GARCH(1,1)": "#a3a3a3"}
    fig, ax = plt.subplots(figsize=(10.4, 5.0))
    x = np.arange(len(hz))
    w = 0.13
    for k, m in enumerate(models):
        vals = []
        for h in hz:
            v = df[(df.horizon == h) & (df.model == m)].QLIKE.values
            vals.append(v[0] if len(v) else np.nan)
        ax.bar(x + (k - 2.5) * w, vals, w, color=colors[m], label=m)
    ax.set_xticks(x, [hz_ru[h] for h in hz])
    ax.set_ylabel("QLIKE (ниже — лучше)")
    ax.set_ylim(0, 0.72)
    ax.set_title("Гонка моделей на трёх годах минут Binance, 2024–2026 вне выборки", loc="left", pad=10)
    for i, h in enumerate(hz):
        har = df[(df.horizon == h) & (df.model == "HAR")].QLIKE.values
        gbm = df[(df.horizon == h) & (df.model == "GBM")].QLIKE.values
        if len(har) and len(gbm):
            gain = (har[0] - gbm[0]) / har[0] * 100
            ax.text(i, 0.655, f"GBM {gain:+.0f}%", ha="center", fontsize=9, color=VIOLET,
                    fontweight="bold")
    ax.legend(ncol=3, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    fig.tight_layout()
    save(fig, "fig13_horserace.png")


def fig_research():
    df = pd.read_csv(TAB / "research.csv")
    df = df.sort_values("t")
    fig, ax = plt.subplots(figsize=(10.6, 5.6))
    colors = []
    for _, r in df.iterrows():
        if "CONFIRMED" in str(r.verdict):
            colors.append(GREEN)
        elif "likely" in str(r.verdict):
            colors.append(AMBER)
        else:
            colors.append(SLATE)
    y = np.arange(len(df))
    ax.hlines(y, 0, df.t, color=colors, lw=2.4, alpha=0.9)
    ax.scatter(df.t, y, color=colors, s=46, zorder=3)
    for yi, (_, r) in zip(y, df.iterrows()):
        ax.text(r.t + (0.6 if r.t >= 0 else -0.6), yi, f"{r.t:.1f}", va="center",
                ha="left" if r.t >= 0 else "right", fontsize=7.8, color=INK)
    short = {"R1": "сезонность волатильности", "R2": "доходность вне часов США",
             "R3": "внутридневной моментум", "R4": "недельный TSMOM", "R5": "понедельник",
             "R6": "дрейф перед FOMC", "R7": "vol-managed экспозиция", "R8a": "премия за риск дисперсии",
             "R8b": "VRP предсказывает доходность", "R9": "лесенка знает направление",
             "R10": "страховка от краха переоценена", "R11": "фандинг предсказывает доходность",
             "R12": "IV ведёт доходность", "R13": "всплеск в первую минуту часа",
             "R14": "CME-гэпы закрываются", "R15": "круглые уровни", "R16": "начало месяца",
             "R17": "плохая волатильность", "R18": "выигрыш лесенки по режимам",
             "R19": "HAR с режимами", "R20": "ML предсказывает доходность", "R21": "HAR бьёт GARCH"}
    ax.set_yticks(y, [f"{r.id} · {short.get(r.id, r.hypothesis[:34])}" for _, r in df.iterrows()], fontsize=8.2)
    ax.axvline(0, color=INK, lw=1)
    for x in (-2.9, 2.9):
        ax.axvline(x, color=MUTED, lw=0.8, ls=":")
    ax.set_xlabel("t-статистика (Newey–West); |t| > 2.9 ≈ порог после поправки Холма")
    ax.set_title("21 исследовательская гипотеза: что подтвердилось после поправки на множественность",
                 loc="left", pad=10)
    ax.set_xlim(-6, 44)
    for lab, c in [("подтверждено", GREEN), ("вероятно (BH 10%)", AMBER), ("не подтверждено", SLATE)]:
        ax.scatter([], [], color=c, label=lab, s=40)
    ax.legend(loc="upper center", ncol=3, bbox_to_anchor=(0.5, -0.10), scatterpoints=1)
    ax.invert_yaxis()
    save(fig, "fig14_research.png")


def fig_lab():
    df = pd.read_csv(TAB / "lab.csv")
    x = df[df.trades.fillna(0) >= 5].copy()
    x = x.sort_values("mean_net")
    fig, ax = plt.subplots(figsize=(9.4, 5.0))
    colors = [RED if v < 0 else (BS if v > 0 else SLATE) for v in x.mean_net.fillna(0)]
    y = np.arange(len(x))
    ax.barh(y, x.mean_net * 100, color=colors, alpha=0.9)
    for yi, (_, r) in zip(y, x.iterrows()):
        t = "" if pd.isna(r.t) else f"t {r.t:+.1f}"
        ax.text(r.mean_net * 100 + (0.12 if r.mean_net >= 0 else -0.12), yi, t, va="center",
                ha="left" if r.mean_net >= 0 else "right", fontsize=7.5, color=MUTED)
    short = {"P06": "Kalshi 15m vs fair value", "P02": "Polymarket 15m vs fair value",
             "P05": "Polymarket hourly ladders", "V04": "DVOL level", "P01": "Polymarket 5m",
             "V01": "lab blend", "P03": "Polymarket hourly Up/Down", "D16": "Kalshi ladder drift",
             "D07": "taker-flow momentum", "D06": "order-book imbalance", "V02": "Student-t(4) tails",
             "D01": "hourly reversal", "V05": "realised variance 1h", "V03": "Kalshi ladder variance",
             "M01": "maker on Kalshi 15m"}
    ax.set_yticks(y, [f"{r.id} · {short.get(r.id, r.hypothesis[:34])}" for _, r in x.iterrows()], fontsize=8.2)
    ax.axvline(0, color=INK, lw=1)
    ax.set_xlabel("средний чистый P&L на сделку, ¢ (после комиссий и спреда)")
    ax.set_title("Лаборатория: ни одна из 53 гипотез не прошла Холма с плюсом", loc="left", pad=26)
    ax.text(0.0, 1.012, "показаны 15 правил с ≥ 5 живыми сделками; P&L после комиссий и спреда",
            transform=ax.transAxes, fontsize=8, color=MUTED)
    ax.set_xlim(-25, 12)
    save(fig, "fig15_lab.png")


def fig_regimes():
    df = pd.read_csv(TAB / "regimes.csv")
    df["from"] = pd.to_datetime(df["from"])
    df["to"] = pd.to_datetime(df["to"])
    d = load_forecasts("kx1h")
    weekly = _actual_series(d, "W", models=("har",))
    fig, ax = plt.subplots(figsize=(11.2, 3.6))
    ax.fill_between(weekly.bucket, 0, weekly.act * 100, color="#e2e8f0", linewidth=0)
    ax.plot(weekly.bucket, weekly.act * 100, color=INK, lw=1.6)
    for _, r in df.iterrows():
        ax.axvspan(r["from"], r["to"], color=BS, alpha=0.035, zorder=0)
        mid = r["from"] + (r["to"] - r["from"]) / 2
        ax.text(mid, 91, f"{r.ann_vol:.0f}%", ha="center", fontsize=8.5, color=BS,
                bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.8))
    ax.set_ylim(0, 100)
    ax.set_ylabel("годовая волатильность, %")
    ax.set_title("Режимы волатильности: оптимальная разбивка недельного лог-RV (BIC)", loc="left", pad=10)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    save(fig, "fig16_regimes.png")


def fig_alpha_funnel():
    fig, ax = plt.subplots(figsize=(9.8, 2.9))
    ax.axis("off")
    steps = [
        ("53", "правила в каталоге\nзаморожены до первой сделки", "#e2e8f0", INK),
        ("19", "активны\nв каталоге", "#bfdbfe", INK),
        ("2 399", "закрытых сделок\nна реальных bid/ask", "#fde68a", INK),
        ("0", "прошли Холма\nс положительным P&L", "#fecaca", "#b91c1c"),
    ]
    w = 0.20
    for i, (big, small, bg, fg) in enumerate(steps):
        x0 = 0.03 + i * 0.245
        box = FancyBboxPatch((x0, 0.18), 0.205, 0.64, boxstyle="round,pad=0.012,rounding_size=0.02",
                             transform=ax.transAxes, facecolor=bg, edgecolor="white", linewidth=2)
        ax.add_patch(box)
        ax.text(x0 + 0.1025, 0.60, big, ha="center", va="center", fontsize=22, fontweight="bold",
                color=fg, transform=ax.transAxes)
        ax.text(x0 + 0.1025, 0.30, small, ha="center", va="center", fontsize=8.2, color=fg,
                transform=ax.transAxes)
        if i < len(steps) - 1:
            ax.annotate("", xy=(x0 + 0.235, 0.5), xytext=(x0 + 0.208, 0.5), xycoords=ax.transAxes,
                        arrowprops=dict(arrowstyle="-|>", color=SLATE, lw=1.4))
    fig.suptitle("Воронка поиска альфы: где исчезает прибыль", x=0.02, ha="left", fontsize=12,
                 fontweight="bold")
    save(fig, "fig17_alpha_funnel.png")


# --------------------------------------------------------------------------- main
def main():
    print("tables")
    table_data_inventory()
    table_scoreboard()
    table_recent_oos()
    table_dm_key()
    table_horserace()
    table_implied_bias()
    table_brier()
    table_regimes()
    table_research()
    table_lab()

    print("figures")
    fig_data_timeline()
    fig_vol_full()
    fig_vol_2026()
    fig_vol_last14()
    fig_vol_pm()
    fig_scatter()
    fig_bias_heatmap()
    fig_seasonality()
    fig_scores_heatmap()
    fig_dm_forest()
    fig_brier()
    fig_pit()
    fig_horserace()
    fig_research()
    fig_lab()
    fig_regimes()
    fig_alpha_funnel()
    print("done ->", OUT)


if __name__ == "__main__":
    main()
