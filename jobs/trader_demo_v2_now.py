"""Demo trading of the wing rules on this computer, started by the lab daemon (v2: keys under any file name).

Put the two things you got from demo.kalshi.co into data/trader/keys/demo/ — names do not matter:
    the private key file you downloaded (its text contains 'PRIVATE KEY')
    the key id (paste it into any .txt file there, e.g. key_id.txt)
Checked every minute. Then: clock and balance check, one post-only 1-contract order at 1¢ that is read back and
cancelled (proves signing, orders and cancels), and the executor with the demo_plumbing profile. Stop:
python -m trader kill. Journal: data/trader/journal_demo.jsonl · python -m trader status --mode demo
"""
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from trader.client import Kalshi, KalshiError  # noqa: E402
from trader.executor import Executor, Spot  # noqa: E402
from trader.settings import DATA, KEYS, Settings, scan_key_dir  # noqa: E402

folder = KEYS / "demo"
folder.mkdir(parents=True, exist_ok=True)
t0, last = time.time(), None
while True:
    kid, kf = scan_key_dir(folder)
    state = f"private key: {'found ' + Path(kf).name if kf else 'not found'}; key id: {'found' if kid else 'not found'}"
    if state != last:
        print(time.strftime("%H:%M:%S"), f"{folder}: {state}", flush=True)
        last = state
    if kid and kf:
        break
    if time.time() - t0 > 14 * 86400:
        sys.exit("no demo key after 14 days")
    time.sleep(60)

try:
    v = Kalshi("demo", kid, kf)
    print("clock offset, s:", round(v.server_clock_check(), 2), flush=True)
    print("demo exchange trading:", v.exchange_status().get("trading_active"), flush=True)
    print("balance:", {k: x for k, x in v.balance().items() if k != "raw"}, flush=True)
except KalshiError as e:
    sys.exit(f"demo key rejected: {e} — check that the key id matches the private key and was created on demo.kalshi.co")
except Exception:
    traceback.print_exc()
    sys.exit("could not reach the demo exchange")

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
beat = 0.0
while not ex.kill_file.exists():
    ex.step()
    if time.time() - beat > 300:
        print(time.strftime("%H:%M:%S"), f"demo alive: open {len(ex.st.positions)}, realised {ex.st.realised}", flush=True)
        beat = time.time()
    time.sleep(cfg.poll_s)
print(f"kill switch: cancelled {v.cancel_all()} resting orders, stopped", flush=True)
