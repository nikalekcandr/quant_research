import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pandas as pd
from lab import intents,wing_regime
from trader.executor import Executor


class RegimeTests(unittest.TestCase):
    def test_shock_pause_quiet_period_and_missing_minute(self):
        times = pd.date_range("2026-11-21",periods=300,freq="min",tz="UTC")
        for factor in (1.02,.98):
            price = pd.Series(100.,index=times)
            price.iloc[150:] *= factor
            state = wing_regime.regimes(price)
            self.assertFalse(state.blocked.iloc[149])
            self.assertTrue(state.blocked.iloc[150])
            self.assertTrue(state.blocked.iloc[268])
            self.assertFalse(state.blocked.iloc[269])
            self.assertTrue(wing_regime.regimes(price.drop(times[275])).blocked.iloc[280])

    def test_missing_or_expired_state_blocks_entries_but_allows_cancel(self):
        now = pd.Timestamp.now(tz="UTC")
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch.object(wing_regime,"STATE",root/"wing_regime.json"),patch.object(intents,"INTENTS",root/"intents.jsonl"):
                self.assertTrue(wing_regime.read(now)["blocked"])
                wing_regime.STATE.write_text(json.dumps({"minute":str(now-pd.Timedelta(minutes=3)),"blocked":False}))
                self.assertTrue(wing_regime.read(now)["blocked"])
                for kind in ("take","quote","cancel"):
                    intents.emit(kind,"W04","ticker","event","no",.9,now)
                self.assertEqual([json.loads(s)["kind"] for s in intents.INTENTS.read_text().splitlines()],["cancel"])
                ex = Executor.__new__(Executor)
                ex.intents_path,ex.kill_file,ex.cfg = intents.INTENTS,root/"KILL",None
                self.assertIn("volatility pause",ex.refuse_reason({"event":"E","hyp":"W04"}))
