"""Offline regression checks for the wing dashboard and causal weekend replay."""
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
from types import SimpleNamespace
import json

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lab.core import LEDGER_COLS
from lab.wing_report import metrics, prepare, slot_results
from studies.wings_weekend2 import pick
from studies.wings_scale_in import replay
from trader.settings import Settings
from lab import intents
from lab.signals import w12, REGISTRY
from studies.wings_edge_timing import first_entries, resolve_ladder
from trader.monitor import demo_checks


class WingReportTests(unittest.TestCase):
    def test_w12_ranks_current_executable_quotes_and_respects_its_slot(self):
        end = pd.Timestamp("2026-11-20 21:00Z")
        market = dict(venue="kalshi",kind="kx_ladder_1h",end=end,event="e",market="first",book=1,
                      bid=.08,ask=.09,bid_size=4,ask_size=4,fair_t4=.065)
        booked = set()
        ctx = SimpleNamespace(now=end-pd.Timedelta(minutes=22),book=SimpleNamespace(has_key=booked.__contains__),
                              markets=pd.DataFrame([market]))
        self.assertFalse(w12(ctx))  # 0.5 cent after the fee
        ctx.markets = pd.DataFrame([market, dict(market,market="better",fair_t4=.045),
                                    dict(market,market="highest",fair_t4=.03),
                                    dict(market,market="empty",fair_t4=.00,bid_size=0),
                                    dict(market,market="listing",fair_t4=.00,book=0)])
        pick, = w12(ctx)
        self.assertEqual((pick.instrument,pick.side,pick.qty),("highest","no",1))
        self.assertAlmostEqual(pick.edge,.04)
        booked.add(pick.key)
        self.assertFalse(w12(ctx))
        self.assertTrue(any(h.id=="W12" and not h.retired for h in REGISTRY))
        self.assertNotIn("W12",intents.ENABLED)

    def test_first_threshold_crossing_is_causal(self):
        quotes = pd.DataFrame({"end":[10,10,10],"ts":[1,2,3],"edge":[.005,.02,.06],"minute":[29,28,27]})
        self.assertEqual(first_entries(quotes,.01).iloc[0].ts,2)
        self.assertEqual(first_entries(quotes,{29:.04,28:.03,27:.02}).iloc[0].ts,3)
        self.assertTrue(first_entries(quotes,.07).empty)

    def test_resolution_bounds_and_demo_diagnostics(self):
        known = {"kalshi|KXBTCD-TEST-T10":1, "kalshi|KXBTCD-TEST-T20":0}
        r = resolve_ladder(["KXBTCD-TEST-T9","KXBTCD-TEST-T12","KXBTCD-TEST-T21"],known)
        self.assertEqual(r["KXBTCD-TEST-T9"],1)
        self.assertEqual(r["KXBTCD-TEST-T21"],0)
        self.assertNotIn("KXBTCD-TEST-T12",r)
        events = [dict(intent="a",what="order_sent"),dict(intent="b",what="order_sent"),
                  dict(intent="a",what="order_result",filled=1),dict(intent="a",what="order_result",filled=1),
                  dict(intent="b",what="order_rejected"),dict(intent="b",what="settled",pnl=999)]
        self.assertEqual(list(demo_checks(events).values()),[2,1,1,1,0])

    def test_scale_in_rechecks_edge_and_does_not_trade_an_isolated_peak(self):
        start = pd.Timestamp("2026-11-01 10:30Z")
        signal = SimpleNamespace(ts=start, hour=start.ceil("h"), price=.92, exit_price=1)
        path = pd.DataFrame({"ts":pd.date_range(start,periods=6,freq="min"), "valid":True, "size":40,
                             "price":[.92,.86,.92,.86,.85,.80], "fair":[.96,.95,.96,.95,.95,.75]})
        result = replay(signal,path)
        self.assertEqual(result["contracts"],2)
        self.assertAlmostEqual(result["base"],.22)
        self.assertAlmostEqual(result["staged"],.21)
        path.loc[0,"size"] = 4
        self.assertNotEqual(replay(signal,path)["status"],"ok")

    def test_execution_admission_at_source_and_in_profiles(self):
        with TemporaryDirectory() as tmp:
            dest = Path(tmp)/"intents.jsonl"
            settings = Path(tmp)/"settings.json"
            settings.write_text('{"rules":["W04"]}',encoding="utf-8")
            for profile in ("demo_plumbing","live_1contract"):
                self.assertEqual(Settings.load(settings,profile).rules,["W04"])
            with patch.object(intents,"INTENTS",dest), patch.object(Settings,"load",return_value=Settings()), patch("lab.wing_regime.read",return_value={"blocked":False}):
                for rule in ("W03","W04","W05","W06"):
                    for kind in ("take","quote","cancel"):
                        intents.emit(kind, rule, "ticker","event","no",.9,pd.Timestamp("2026-11-01",tz="UTC"))
                events = [json.loads(line) for line in dest.read_text().splitlines()]
                self.assertEqual({r["hyp"] for r in events if r["kind"]!="cancel"},{"W04","W06"})
                self.assertEqual(sum(r["kind"]=="cancel" for r in events),4)

    def test_later_price_cannot_replace_earlier_entry(self):
        quotes = pd.DataFrame([
            (1, 1, .07, "A", "yes"), (1, 1, .08, "B", "yes"),
            (1, 2, .10, "C", "yes"), (1, 3, .09, "D", "no"),
        ], columns=["T", "ts", "c_bid", "ticker", "cheap"])
        one, two = pick(quotes)
        self.assertEqual(one.ticker.tolist(), ["B"])
        self.assertEqual(two.ticker.tolist(), ["B", "D"])

    def test_ledger_lifecycle_voids_size_and_delayed_settlement(self):
        base = dict(event="OPEN", id="a", hyp="W04", family="W", venue="kalshi", code="clean",
                    ts="2026-11-07 23:33:00Z", info='{"left_min":27}', qty=2, pnl=None, exit_ts=None)
        done = dict(base, event="CLOSE", pnl=-.92, exit_ts="2026-11-09 11:02:00Z")
        rows = [base, done, done, dict(base,id="b"), dict(base,id="c",code="960e7aaff561"), dict(done,id="c"),
                dict(base,id="d",family="X"), dict(done,id="e",pnl=None)]
        closed, opened = prepare(pd.DataFrame(rows,columns=LEDGER_COLS))
        self.assertEqual(opened.id.tolist(), ["b"])
        self.assertEqual(closed.hour.iloc[0], pd.Timestamp("2026-11-08 00:00Z"))
        self.assertEqual(metrics(closed)["Просадка, $"], "1,84")
        self.assertEqual(metrics(closed)["¢/контр."], "−92,00")
        self.assertEqual(slot_results(closed).iloc[0]["Нетто, $"], "−0,92")

    def test_empty_book(self):
        closed, opened = prepare(pd.DataFrame(columns=LEDGER_COLS))
        self.assertTrue(opened.empty)
        self.assertEqual(metrics(closed)["Часы"], 0)


if __name__ == "__main__":
    unittest.main()
