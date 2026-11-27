"""One-contract chronological replay of agreement and opposite-side wing entries."""
import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lab.core import RESULTS, atomic_write, kalshi_fee, utc_index, utcnow
from studies.ladders import no_data
from studies.wings_calibration import books
from studies.wings_edge_timing import SPLIT, START, STOP, day_interval


def offers(q, buy, margin=.01):
    wing_bid = q.bid.where(q.cheap_yes, 1-q.ask)
    wing_ask = q.ask.where(q.cheap_yes, 1-q.bid)
    price = wing_ask if buy else 1-wing_bid
    size = q.ask_size.where(q.cheap_yes, q.bid_size) if buy else q.bid_size.where(q.cheap_yes, q.ask_size)
    fee = price.map(kalshi_fee)
    fair = q.t4 if buy else 1-q.t4
    other = q.normal if buy else 1-q.normal
    out = q.assign(price=price, fee=fee, edge=(fair-price-fee).round(10),
                   second=(other-price-fee).round(10), pay=q.y if buy else 1-q.y)
    return out[size.ge(1) & out.edge.ge(margin)]


def picks(q):
    return q.sort_values(["end","ts","edge","market"], ascending=[True,True,False,True]
                         ).drop_duplicates("end").set_index("end")


def compare(q, hours, period, momentum=False):
    hours = utc_index(hours)
    sell, buy = offers(q, False), offers(q, True)
    strict_sell, strict_buy = sell[sell.second.ge(.01)], buy[buy.second.ge(.01)]
    # Exploratory reversal after inspecting the long-wing replay. Rank strikes by
    # the original signal, but take the opposite book side and charge its fee.
    wing_bid = strict_buy.bid.where(strict_buy.cheap_yes, 1-strict_buy.ask)
    opposite = 1-wing_bid
    fade = strict_buy.assign(price=opposite, fee=opposite.map(kalshi_fee), pay=1-strict_buy.y)
    # If both signals occur in the first snapshot, keep the baseline trade.
    combined = pd.concat([sell,fade.merge(sell[["end","ts"]].drop_duplicates(),
                         on=["end","ts"],how="left",indicator=True).query("_merge == 'left_only'")])
    policies = {"Продажа · t₄": sell, "Продажа · согласие": strict_sell,
                "Покупка · согласие": strict_buy, "Обе стороны · согласие": pd.concat([strict_sell,strict_buy]),
                "Обратная сделка · после анализа":fade, "Продажа t₄ или обратная":combined}
    if momentum:
        away = sell.move5.where(sell.cheap_yes, -sell.move5).lt(0)
        policies["Продажа t₄ · BTC движется от крыла"] = sell[away]
    chosen = {name:picks(data) for name,data in policies.items()}
    pnl = pd.DataFrame(index=hours)
    for name, trades in chosen.items():
        pnl[name] = (trades.pay-trades.price-trades.fee).reindex(hours,fill_value=0.)
    pnl = pnl.dropna()
    rows = []
    for name, trades in chosen.items():
        known = trades[trades.index.isin(pnl.index)]
        profit = pnl[name]
        delta = profit-pnl.iloc[:,0]
        low,high = day_interval(delta)
        rows.append({"Период UTC":period,"Правило":name,"Часы с исходами / всего":f"{len(pnl)}/{len(hours)}",
                     "Входы":len(known),"Убытки":int((profit<0).sum()),"Нетто, $":round(profit.sum(),2),
                     "¢/час":round(100*profit.mean(),2),"Δ к продаже, ¢/час":round(100*delta.mean(),2),
                     "95% интервал Δ":f"{low:.2f} … {high:.2f}" if low is not None else "—"})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--until",default=None)
    args = parser.parse_args()
    last = utcnow().floor("h")-pd.Timedelta(hours=1)
    until = min(last,pd.to_datetime(args.until,utc=True)) if args.until else last
    data,hours = books(until)
    hours = utc_index(hours)
    rows = []
    for start,end in [(START,SPLIT),(SPLIT,STOP),(STOP,until+pd.Timedelta(seconds=1))]:
        sample = hours[(hours>=start)&(hours<end)]
        if len(sample):
            period = f"{sample.min():%d.%m %H:%M} – {sample.max():%d.%m %H:%M}"
            rows.extend(compare(data[data.end.isin(sample)],sample,period))
    if not rows:
        no_data(f"ни одного часа с котировками книги до {until:%d.%m %H:%M} — сравнивать нечего")
    frame = pd.DataFrame(rows)
    atomic_write(RESULTS/"wings_two_sided.csv",frame.to_csv(index=False))
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
