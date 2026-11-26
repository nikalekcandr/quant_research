"""Probability calibration for the cheap leg of early hourly Kalshi binaries."""
from dataclasses import replace

from scipy.special import expit, logit
import numpy as np
import pandas as pd

from lab.core import LAB, append_rows, day_file, utcnow

# Fixed after the offline run; changing a fit requires a new forecast version.
VERSION = "2026-11-21.v1"
PARAMS = {"weekday":-.129230299442,"weekend":-.594571408634,"t4":-.435412122573,"weight":.601586514045}


def eligible(d):
    """Two-sided book and an early-window 6–10 cent bid. For candles, sizes are unavailable."""
    q = d[(d.bid > 0) & (d.ask < 1) & (d.ask-d.bid).round(6).between(0,.03)
          & d.left.between(15,30)].copy()
    q["cheap_yes"] = q.bid.lt(.5)
    wing = q.bid.where(q.cheap_yes,1-q.ask).round(6)
    q = q[wing.between(.06,.10)].copy()
    yes_mid = (q.bid+q.ask)/2
    q["mid"] = yes_mid.where(q.cheap_yes,1-yes_mid)
    q["weekend"] = q.end.dt.day_of_week >= 5
    return q


def sampled(d):
    """Balance expiries, then strikes, then observed 5-minute buckets within a strike."""
    x = d.dropna(subset=["y"]).sort_values("ts").copy()
    x["bucket"] = (x.left/5).astype(int)
    x = x.drop_duplicates(["end","market","bucket"])
    strikes = x.groupby("end").market.transform("nunique")
    buckets = x.groupby(["end","market"]).market.transform("size")
    x["weight"] = 1/(strikes*buckets)
    return x


def odds_adjust(probability, intercept):
    return expit(logit(np.clip(probability, 1e-6, 1-1e-6)) + intercept)


def predictions(t4, mid, weekend, params):
    market = odds_adjust(mid, np.where(weekend, params["weekend"], params["weekday"]))
    corrected = odds_adjust(t4, params["t4"])
    hybrid = params["weight"]*corrected + (1-params["weight"])*market
    return market, corrected, hybrid


def pooled_probability(t4, mid, weekend, coefficients):
    """Calendar intercept and convex pool in log odds, fitted on July–August."""
    a, b, share = coefficients
    market = logit(np.clip(mid,1e-6,1-1e-6))
    model = logit(np.clip(t4,1e-6,1-1e-6))
    return expit(market + share*(model-market) + np.where(weekend,b,a))


def depth_hybrid(q, params):
    """Replace the midpoint by a halfway move towards the top-of-book microprice."""
    bid, ask = q.bid_size, q.ask_size
    total = bid + ask
    valid = np.isfinite(total) & (total > 0) & (bid > 0) & (ask > 0)
    imbalance = ((bid - ask) / total.where(valid)).fillna(0)
    direction = np.where(q.cheap_yes, 1, -1)
    adjusted = q.mid + direction * (q.ask - q.bid) * imbalance / 4
    return predictions(q.t4, adjusted, q.end.dt.day_of_week >= 5, params)[2]


def record_forecasts(ctx):
    """Append paired forecasts after trading decisions, including non-signals."""
    from lab.wings import _fair_blend

    m = ctx.markets
    if m is None or m.empty or "book" not in m or "fair_t4" not in m:
        return
    q = m[m.kind.eq("kx_ladder_1h") & m.venue.eq("kalshi") & m.book.eq(1) & m.fair_t4.between(0,1)].copy()
    q["end"] = pd.to_datetime(q.end,utc=True)
    q["left"] = (q.end-ctx.now).dt.total_seconds()/60
    q = eligible(q)
    q = q[q.bid_size.where(q.cheap_yes,q.ask_size).ge(1)].copy()
    if q.empty:
        return
    # Computing diagnostic forecasts must not prime the trading models' state.
    research_ctx = replace(ctx,cache=ctx.cache.copy())
    blend = pd.to_numeric(q.apply(lambda r:_fair_blend(research_ctx,r),axis=1),errors="coerce")
    q["w06"] = blend.where(q.cheap_yes,1-blend)
    q["t4"] = q.fair_t4.where(q.cheap_yes,1-q.fair_t4)
    q["market_cal"],q["t4_cal"],q["hybrid"] = predictions(q.t4,q.mid,q.weekend,PARAMS)
    q["version"],q["ts"],q["recorded_ts"] = VERSION,ctx.now,utcnow()
    cols = ["version","ts","recorded_ts","end","market","cheap_yes","bid","ask","bid_size","ask_size",
            "mid","t4","w06","market_cal","t4_cal","hybrid"]
    q = q[q.recorded_ts < q.end]
    append_rows(day_file(LAB/"wing_forecasts",ctx.now),q[cols].to_dict("records"))

