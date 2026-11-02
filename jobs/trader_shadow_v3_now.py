"""Start the Kalshi executor in SHADOW mode on this computer: production order books, no keys, no orders.

Every wing intent of the lab (data/lab/intents.jsonl) is re-checked on the live production book a second after the
lab decided, and what the executor WOULD have bought (price, size, depth) is written to
data/trader/journal_shadow.jsonl. This measures, before any money moves, how often the lab's paper price is still
there, at what depth, and how much slippage a real order would have. Stop it with  python -m trader kill.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from trader.client import Kalshi  # noqa: E402
from trader.executor import Executor, Spot  # noqa: E402
from trader.settings import DATA, Settings  # noqa: E402

cfg = Settings.load()
cfg.mode = "shadow"
md = Kalshi("prod")
print("exchange status:", md.exchange_status(), flush=True)
ms = sorted(md.markets(series_ticker="KXBTCD", status="open"), key=lambda m: m.get("close_time", ""))
print(f"open KXBTCD markets: {len(ms)}", flush=True)
if ms:
    b = md.book(ms[0]["ticker"])
    print(f"sample {ms[0]['ticker']}: YES bid {b.yes_bid}, YES ask {b.ask('yes')}", flush=True)
(DATA / "KILL").unlink(missing_ok=True)
ex = Executor(cfg, None, md, Spot(cfg.spot_url))
last = 0.0
while not ex.kill_file.exists():
    ex.step()
    if time.time() - last > 300:
        print(f"{time.strftime('%H:%M:%S')} shadow alive: open {len(ex.st.positions)}, resting {len(ex.st.resting)}, "
              f"realised {ex.st.realised}", flush=True)
        last = time.time()
    time.sleep(cfg.poll_s)
print("kill switch: stopped", flush=True)
