"""Demo, 04.11: every demo order on KXBTCD was rejected 'insufficient_shard_balance' — the play money sits on
shard 0, crypto markets trade on shard 2. Moves $900 of DEMO cash 0 -> 2 and prints the balance before/after.
The running demo executor picks the money up on its next order; nothing else is restarted."""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from trader.client import Kalshi  # noqa: E402
from trader.settings import credentials  # noqa: E402

kid, kf = credentials("demo")
v = Kalshi("demo", kid, kf)
v.server_clock_check()
print("before:", json.dumps(v.balance()["raw"])[:800], flush=True)
print("transfer:", v.shard_transfer(900.0, 0, 2), flush=True)
time.sleep(10)
print("after:", json.dumps(v.balance()["raw"])[:800], flush=True)
