"""Replay staged entries on W04 signals and actual Kalshi order-book snapshots.

Run from the project directory: python studies/wings_scale_in.py
Three-contract reference vs one now and up to two later on the same side/ticker.
No future extrema or settlement outcomes are used for entry decisions.
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lab.core import LAB, RESULTS, read_ledger
from lab.wing_report import prepare
from trader.executor import taker_fee

EXAMPLE = "KXBTCD-26NOV2015-T58799.99"
COLUMNS = ["ts", "market", "end", "book", "bid", "ask", "bid_size", "ask_size", "fair_t4"]


def quotes_for(books, side):
    """Executable cost and fair value of the favourite side that the rule buys."""
    b = books.copy()
    yes = side == "yes"
    b["price"] = (b.ask if yes else 1 - b.bid).round(6)
    b["size"] = b.ask_size if yes else b.bid_size
    b["fair"] = b.fair_t4 if yes else 1 - b.fair_t4
    b["valid"] = (b.book.eq(1) & b.bid.gt(0) & b.ask.lt(1) &
                  (b.ask - b.bid).round(6).between(0, .03) & b.fair.between(0, 1))
    return b.sort_values("ts").drop_duplicates("ts")


def replay(signal, path):
    start = signal.ts.floor("s")  # snapshot log truncates sub-second precision
    path = path[(path.ts >= start) & (path.ts <= signal.hour - pd.Timedelta(minutes=2))]
    entry = path[path.ts == start]
    if entry.empty:
        return {"status": "no matching entry"}
    q = entry.iloc[0]
    if not q.valid or abs(q.price - signal.price) > 1e-6 or not q["size"] >= 12:
        return {"status": "entry price or depth unavailable"}
    first = float(q.price)
    prices = [first]
    armed = None
    observed = 0
    for bar in path[path.ts > start].itertuples():
        if not bar.valid or not bar.size >= 4:
            armed = None
            continue
        observed += 1
        target = first - .05 * len(prices)
        qualifies = bar.price <= target + 1e-8 and bar.fair - bar.price - taker_fee(bar.price) >= -1e-8
        if qualifies and armed is not None and (bar.ts - armed).total_seconds() <= 90:
            prices.append(bar.price)
            armed = None
            if len(prices) == 3:
                break
        else:
            armed = bar.ts if qualifies else None
    payout = signal.exit_price
    base = 3 * (payout - first) - taker_fee(first, 3)
    staged = sum(payout - px - taker_fee(px) for px in prices)
    return {"status":"ok", "contracts":len(prices), "base":base, "staged":staged, "delta":staged-base,
            "additions_net":sum(payout-px-taker_fee(px) for px in prices[1:]),
            "avg_price":sum(prices)/len(prices), "price":first, "followup":observed > 0}


def main():
    closed, _ = prepare(read_ledger())
    signals = closed[closed.hyp.eq("W04") & closed.exit_price.isin([0, 1])].sort_values("ts")
    markets = set(signals.instrument) | {EXAMPLE}
    parts = []
    for file in sorted((LAB / "markets").glob("*.csv")):
        if not set(COLUMNS) <= set(pd.read_csv(file,nrows=0).columns):
            continue
        for chunk in pd.read_csv(file, usecols=COLUMNS, chunksize=100000):
            part = chunk[chunk.market.isin(markets)]
            if not part.empty:
                parts.append(part)
    books = pd.concat(parts,ignore_index=True) if parts else pd.DataFrame(columns=COLUMNS)
    books["ts"] = pd.to_datetime(books.ts,utc=True,format="mixed")
    paths = dict(tuple(books.groupby("market")))
    rows = []
    for s in signals.itertuples():
        b = paths.get(s.instrument)
        result = {"status":"no book"} if b is None else replay(s, quotes_for(b,s.side))
        rows.append({"id":s.id,"hour":s.hour,"market":s.instrument,**result})
    RESULTS.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows, columns=["id","hour","market","status","contracts","base","staged","delta","additions_net",
                                "avg_price","price","followup"]).to_csv(RESULTS/"scale_in.csv",index=False)
    example = quotes_for(books[books.market.eq(EXAMPLE)], "no")
    example = example[example.valid].copy()
    example["edge"] = example.fair-example.price-example.price.map(taker_fee)
    example[["ts","bid","fair_t4","edge"]].to_csv(RESULTS/"scale_in_example.csv",index=False)
    print(pd.DataFrame(rows).status.value_counts() if rows else "No W04 settlements")


if __name__ == "__main__":
    main()
