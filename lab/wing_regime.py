"""Causal market-wide entry pause for wings; timestamps of inputs are minute opens."""
import json
import numpy as np
import pandas as pd

from lab.core import LAB, atomic_write

STATE = LAB/"wing_regime.json"
RANGE_LIMIT = 1.5
RV_LIMIT = 1.0
QUIET_MINUTES = 60


def regimes(close):
    prices = close.where(close.gt(0)&np.isfinite(close)).asfreq("min")
    # Move the timestamp to the first instant the minute close is observable.
    prices.index = prices.index + pd.Timedelta(minutes=1)
    span = 100*(prices.rolling(61).max()/prices.rolling(61).min()-1)
    rv = 100*np.sqrt(np.log(prices).diff().pow(2).rolling(30).sum())
    shock = span.ge(RANGE_LIMIT)|rv.ge(RV_LIMIT)
    known = span.notna()&rv.notna()
    quiet = known & ~shock
    blocked = quiet.rolling(QUIET_MINUTES,min_periods=QUIET_MINUTES).sum().ne(QUIET_MINUTES)
    return pd.DataFrame(dict(range_pct=span,rv_pct=rv,shock=shock,known=known,blocked=blocked))


def publish(close,now):
    end = now.floor("min")
    grid = pd.date_range(end-pd.Timedelta(minutes=125),end-pd.Timedelta(minutes=1),freq="min")
    d = regimes(close.reindex(grid))
    row = d.iloc[-1]
    status = {"minute":end.isoformat(),"blocked":bool(row.blocked),
              "range_pct":float(row.range_pct) if row.known else None,
              "rv_pct":float(row.rv_pct) if row.known else None,
              "reason":"данные неполны" if not row.known else "порог превышен" if row.shock else
                       "ожидание 60 спокойных минут" if row.blocked else "обычный режим"}
    atomic_write(STATE,json.dumps(status,ensure_ascii=False))
    return status


def read(now,path=None):
    try:
        status = json.loads((STATE if path is None else path).read_text(encoding="utf-8"))
        age = (now-pd.Timestamp(status["minute"])).total_seconds()
        if 0<=age<120 and isinstance(status["blocked"],bool):
            return status
    except (OSError,TypeError,ValueError,KeyError):
        pass
    return {"blocked":True,"reason":"нет свежей оценки режима"}
