"""Executable candidates for a one-contract early-window Kalshi wing."""
import numpy as np
import pandas as pd

from lab.core import kalshi_fee

W12_MARGIN = .01
W12_FIRST_EXPIRY = pd.Timestamp("2026-11-20 21:00",tz="UTC")


def candidates(quotes):
    """Rank only simultaneous quotes. Each row's fair_t4 was computed at decision time."""
    q = quotes
    valid = (q.book.eq(1) & q.bid.gt(0) & q.ask.lt(1) & (q.ask-q.bid).round(6).between(0,.03)
             & q.fair_t4.between(0,1) & q.left.between(15,30))
    q = q.loc[valid].copy()
    cheap_yes = q.bid.lt(.5)
    q["side"] = np.where(cheap_yes, "no", "yes")
    q["wing"] = q.bid.where(cheap_yes, 1-q.ask).round(6)
    q["size"] = q.bid_size.where(cheap_yes, q.ask_size)
    q = q[q.wing.between(.06,.10) & q["size"].ge(1)].copy()
    q["price"] = 1-q.wing
    q["fair_side"] = q.fair_t4.where(q.side.eq("yes"), 1-q.fair_t4)
    q["fee"] = q.price.map(kalshi_fee)
    q["edge"] = q.fair_side-q.price-q.fee
    q = q[q.edge.ge(0)].sort_values(["end","ts","edge","market"],ascending=[True,True,False,True])
    return q.drop_duplicates(["end","ts"])

