"""Ten causal stopping rules for the same ladder forecast, 16–30 minutes before expiry."""
import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from lab.core import kalshi_fee
from studies.wing_shapes import BASE,main


def policies(q):
    s=q.sort_values(["market","ts"]).copy()
    key=[s.market,s.cheap_yes]
    gap=s.groupby(key).ts.diff().dt.total_seconds()
    key.append((gap.isna()|gap.gt(90)).cumsum())
    bid=s.bid.where(s.cheap_yes,1-s.ask)
    e=(bid-s[BASE]-(1-bid).map(kalshi_fee)).round(8)
    grp=e.groupby(key)
    prev=grp.shift()
    old=grp.shift(2)
    peak=grp.cummax().groupby(key).shift()
    trough=grp.cummin().groupby(key).shift()
    initial=e.groupby(key).transform("first")
    elapsed=(s.ts-s.ts.groupby(key).transform("first")).dt.total_seconds()
    fair_change=s[BASE]-s[BASE].groupby(key).shift()
    checks={
        "Повторное пересечение":prev.lt(.01)&e.ge(.01)&peak.ge(.01),
        "Три подтверждения":prev.ge(.01)&old.ge(.01),
        "Пробой максимума":e.ge(peak+.005-1e-9),
        "Откат после пика":peak.ge(.03)&e.le(peak-.01+1e-9),
        "Отскок от минимума":e.ge(trough+.015-1e-9)&prev.le(old)&e.gt(prev),
        "Улучшение от первой оценки":elapsed.ge(60)&e.ge(initial+.01-1e-9),
        "Стабильность edge":elapsed.ge(60)&(e-prev).abs().le(.005)&(prev-old).abs().le(.005),
        "Снижение модельного риска":fair_change.le(-.005+1e-9)&bid.ge(bid.groupby(key).shift()-1e-9),
        "Убывающий порог":e.ge(.01+.03*((s.left-16)/14).clip(0,1)),
        "Сильный вход или дедлайн":e.ge(.04)|s.left.le(18),
    }
    for name,mask in checks.items():
        q[name]=s[BASE].where(mask)
    return list(checks)


if __name__=="__main__":
    main(policies,"wing_stopping")
