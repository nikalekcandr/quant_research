"""Demo trading of the wing rules on this computer, started by the lab daemon.

Waits (up to 14 days) until the demo API key is in place:
    data/trader/keys/demo_key_id.txt   the key id (one line)
    data/trader/keys/demo.pem          the private key downloaded from demo.kalshi.co
then: clock and balance check, one post-only 1-contract order at 1¢ that is read back and cancelled (proves signing,
orders and cancels work), and the executor with the demo_plumbing profile: every wing decision of the lab is checked on
the production book and sent to the demo exchange as a 1-contract IOC order. Stop: python -m trader kill.
Journal: data/trader/journal_demo.jsonl · status: python -m trader status --mode demo
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from trader.client import Kalshi  # noqa: E402
from trader.executor import Executor, Spot  # noqa: E402
from trader.settings import DATA, Settings, credentials  # noqa: E402

t0 = time.time()
while True:
    kid, kf = credentials("demo")
    if kid and kf and Path(kf).exists():
        break
    if time.time() - t0 > 14 * 86400:
        sys.exit("no demo key after 14 days")
    print(time.strftime("%H:%M:%S"), "waiting for data/trader/keys/demo_key_id.txt and demo.pem", flush=True)
    time.sleep(600)

v = Kalshi("demo", kid, kf)
print("clock offset, s:", round(v.server_clock_check(), 2), flush=True)
print("demo exchange:", v.exchange_status().get("trading_active"), flush=True)
print("balance:", v.balance(), flush=True)
dm = sorted(v.markets(series_ticker="KXBTCD", status="open"), key=lambda m: m.get("close_time", ""))
print(f"open KXBTCD markets on demo: {len(dm)}", flush=True)
ms = dm or v.markets(status="open")[:50]
if ms:
    m = ms[-1]
    o = v.create_order(m["ticker"], "yes", 1, 0.01, tif="good_till_canceled", post_only=True,
                       expiration_ts=int(time.time()) + 300)
    print("test order posted:", {k: o.get(k) for k in ("order_id", "status", "yes_price_dollars")}, flush=True)
    print("read back:", {k: v.order(o["order_id"]).get(k) for k in ("status", "remaining_count", "fill_count")}, flush=True)
    print("cancelled:", v.cancel(o["order_id"]).get("order", {}).get("status"), flush=True)

cfg = Settings.load(profile="demo_plumbing")
(DATA / "KILL").unlink(missing_ok=True)
ex = Executor(cfg, v, Kalshi("prod"), Spot(cfg.spot_url))
print("demo executor running:", cfg.rules, flush=True)
last = 0.0
while not ex.kill_file.exists():
    ex.step()
    if time.time() - last > 300:
        print(time.strftime("%H:%M:%S"), f"demo alive: open {len(ex.st.positions)}, realised {ex.st.realised}", flush=True)
        last = time.time()
    time.sleep(cfg.poll_s)
n = v.cancel_all()
print(f"kill switch: cancelled {n} resting orders, stopped", flush=True)
