"""Fragility checks of the frozen W03/W04 (04.11.2026): (1) how the one strike per hour is picked, (2) day-block
bootstrap on Jul–Aug, Sep–Oct and the whole sample, (3) leave-one-week-out."""
import math
import sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.argv = sys.argv[:1] + ["data/raw/usd_minutes"]
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from studies import brti_proxy as bp  # noqa: E402
from studies import wings_ladder as wl  # noqa: E402
wl.LAD, wl.MIN = Path("data/raw/kalshi_kxbtcd_60m"), Path("data/interim/minutes.parquet")
c = bp.wings_frame(bp.binance(), bp.settlements())
RULES = {"W03": (0.10, 0.20, 5, 15, "normal", 0.01), "W04": (0.06, 0.10, 15, 30, "t4", 0.0)}


def pick(rule, how):
    lo, hi, l0, l1, filt, mg = RULES[rule]
    g = c[c.c_bid.between(lo, hi) & c.left.between(l0, l1)]
    fair = g.c_fair_n if filt == "normal" else g.c_fair_t
    g = g.assign(edge=g.c_bid - fair - wl.fee(g.px))
    g = g[g.edge >= mg]
    if how == "frozen: first minute, dearest bid":
        g = g.sort_values(["T", "ts", "c_bid"], ascending=[True, True, False])
    elif how == "first minute, largest edge":
        g = g.sort_values(["T", "ts", "edge"], ascending=[True, True, False])
    elif how == "first minute, cheapest bid":
        g = g.sort_values(["T", "ts", "c_bid"], ascending=[True, True, True])
    elif how == "first minute, random strike":
        g = g.sample(frac=1, random_state=3).sort_values(["T", "ts"], kind="stable")
    elif how == "every signalling strike (first minute each)":
        return g.sort_values("ts").drop_duplicates(["T", "ticker"])
    return g.drop_duplicates(["T", "ticker"]).drop_duplicates("T")


def st(g):
    h = g.groupby("T").pnl.sum()
    return 100 * g.pnl.mean(), h.mean() / (h.std(ddof=1) / math.sqrt(len(h))), len(g)


rows = []
for rule in RULES:
    for how in ("frozen: first minute, dearest bid", "first minute, largest edge", "first minute, cheapest bid",
                "first minute, random strike", "every signalling strike (first minute each)"):
        g = pick(rule, how)
        rec = {"rule": rule, "one strike per hour picked by": how}
        for half, gg in (("Jul–Aug", g[g["T"] < wl.SPLIT]), ("Sep–Oct", g[g["T"] >= wl.SPLIT])):
            m, t, n = st(gg)
            rec[f"{half} ¢"], rec[f"{half} t"], rec[f"{half} n"] = m, t, n
        rows.append(rec)
A = pd.DataFrame(rows)
print("## 1. How the one strike per hour is picked\n")
print(A.round(2).to_markdown(index=False))

book = pd.concat([pick(r, "frozen: first minute, dearest bid") for r in RULES])
rng = np.random.default_rng(11)
print("\n## 2. Day-block bootstrap of the frozen W03+W04 book (1 contract per slot)\n")
out = []
for lab, b in (("Jul–Aug", book[book["T"] < wl.SPLIT]), ("Sep–Oct", book[book["T"] >= wl.SPLIT]), ("Jul–Oct", book)):
    d = b.groupby(b["T"].dt.floor("D")).pnl.sum()
    d = d.reindex(pd.date_range(d.index.min(), d.index.max(), freq="D"), fill_value=0.0)
    v = d.values
    blocks = [v[i:i + 3] for i in range(len(v) - 2)]
    nb = len(v) // 3
    bb = np.array([np.concatenate([blocks[j] for j in rng.integers(0, len(blocks), nb)]).mean() for _ in range(20000)])
    m30 = np.array([np.concatenate([blocks[j] for j in rng.integers(0, len(blocks), 10)]).sum() for _ in range(20000)])
    out.append({"period": lab, "days": len(v), "$/day": v.mean(), "worst day $": v.min(), "losing days %": 100 * (v < 0).mean(),
                "95% CI $/day": f"[{np.percentile(bb, 2.5):.2f}; {np.percentile(bb, 97.5):.2f}]",
                "P(mean ≤ 0) %": 100 * (bb <= 0).mean(), "P(30-day loss) %": 100 * (m30 < 0).mean(),
                "30-day 5% quantile $": np.percentile(m30, 5)})
print(pd.DataFrame(out).round(2).to_markdown(index=False))

print("\n## 3. Leave one week out (Jul–Oct, frozen book): ¢ per trade without that week\n")
wk = book["T"].dt.to_period("W")
lo = [{"week left out": str(w), "¢/trade": 100 * book[wk != w].pnl.mean(), "that week $": book[wk == w].pnl.sum()} for w in sorted(wk.unique())]
L = pd.DataFrame(lo)
print(f"min {L['¢/trade'].min():.2f}¢, max {L['¢/trade'].max():.2f}¢ (all weeks {100 * book.pnl.mean():.2f}¢); "
      f"best week ${L['that week $'].max():.2f}, worst ${L['that week $'].min():.2f}; "
      f"weeks positive {100 * (L['that week $'] > 0).mean():.0f}% of {len(L)}")

print("\n## 4. The frozen trade of the hour by how many strikes signalled in that hour (first minute each)\n")
rows = []
for rule in RULES:
    allg = pick(rule, "every signalling strike (first minute each)")
    nsig = allg.groupby("T").ticker.nunique()
    fz = pick(rule, "frozen: first minute, dearest bid").copy()
    fz["n_sig"] = fz["T"].map(nsig)
    allg = allg.assign(n_sig=allg["T"].map(nsig))
    for lab, (a, b) in {"1": (1, 1), "2": (2, 2), "3–4": (3, 4), "5+": (5, 999)}.items():
        for half, f, al in (("Jul–Aug", fz[fz["T"] < wl.SPLIT], allg[allg["T"] < wl.SPLIT]),
                            ("Sep–Oct", fz[fz["T"] >= wl.SPLIT], allg[allg["T"] >= wl.SPLIT])):
            ff, aa = f[f.n_sig.between(a, b)], al[al.n_sig.between(a, b)]
            rows.append({"rule": rule, "signalling strikes in the hour": lab, "half": half, "hours": len(ff),
                         "frozen trade ¢": 100 * ff.pnl.mean() if len(ff) else np.nan,
                         "frozen paid %": 100 * ff.c_won.mean() if len(ff) else np.nan,
                         "all signalling strikes ¢": 100 * aa.pnl.mean() if len(aa) else np.nan})
print(pd.DataFrame(rows).round(2).to_markdown(index=False))

print("\n## 5. No look-ahead: strikes with a model edge at the decision minute itself\n")
print("n_band = strikes in the rule's band with edge ≥ margin in that minute; n_wide = cheap sides 3–30¢ of the whole "
      "ladder with edge ≥ 0 in that minute (how widely the model disagrees with the book).\n")
rows = []
wide = c[c.c_bid.between(0.03, 0.30)]
for rule, (lo, hi, l0, l1, filt, mg) in RULES.items():
    fair_w = wide.c_fair_n if filt == "normal" else wide.c_fair_t
    nw = wide[(wide.c_bid - fair_w - wl.fee(wide.px)) >= 0].groupby(["T", "ts"]).ticker.nunique()
    g = c[c.c_bid.between(lo, hi) & c.left.between(l0, l1)]
    fair = g.c_fair_n if filt == "normal" else g.c_fair_t
    g = g[(g.c_bid - fair - wl.fee(g.px)) >= mg]
    nb = g.groupby(["T", "ts"]).ticker.nunique()
    fz = pick(rule, "frozen: first minute, dearest bid").copy()
    key = list(zip(fz["T"], fz["ts"]))
    fz["n_band"] = [nb.get(k, 0) for k in key]
    fz["n_wide"] = [nw.get(k, 0) for k in key]
    for col, cuts in (("n_band", {"1": (1, 1), "2": (2, 2), "3+": (3, 99)}),
                      ("n_wide", {"1": (1, 1), "2": (2, 2), "3–4": (3, 4), "5+": (5, 99)})):
        for lab, (a, b) in cuts.items():
            rec = {"rule": rule, "count": col, "value": lab}
            for half, f in (("Jul–Aug", fz[fz["T"] < wl.SPLIT]), ("Sep–Oct", fz[fz["T"] >= wl.SPLIT])):
                ff = f[f[col].between(a, b)]
                rec[f"{half} hours"] = len(ff)
                rec[f"{half} ¢"] = 100 * ff.pnl.mean() if len(ff) else np.nan
                rec[f"{half} t"] = ff.pnl.mean() / (ff.pnl.std(ddof=1) / math.sqrt(len(ff))) if len(ff) > 2 else np.nan
                rec[f"{half} paid %"] = 100 * ff.c_won.mean() if len(ff) else np.nan
            rows.append(rec)
print(pd.DataFrame(rows).round(2).to_markdown(index=False))
