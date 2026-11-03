"""Pack the state the server needs to continue where this computer stops: the lab's ledger and positions, the vol
zoo, recent minutes, the interim tables (3 years of minutes for the boosted model), the executor's state and
journals. Keys are NOT included. Output: _export/server_data.tar.gz (copy it to the server, docs/SERVER.md §3)."""
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
out = ROOT / "_export" / "server_data.tar.gz"
out.parent.mkdir(exist_ok=True)
items = ["data/lab/ledger.csv", "data/lab/positions.json", "data/lab/resolutions.json", "data/lab/volzoo.csv",
         "data/lab/klines_recent.parquet", "data/lab/intents.jsonl", "data/lab/ml_predictions.csv",
         "data/interim", "data/trader/settings.json"]
items += [str(p.relative_to(ROOT)) for p in (ROOT / "data" / "trader").glob("*.json*") if "lock" not in p.name]
with tarfile.open(out, "w:gz") as tar:
    for it in items:
        p = ROOT / it
        if p.exists():
            tar.add(p, arcname=it, filter=lambda ti: None if "keys" in ti.name else ti)
            print("packed", it, flush=True)
print(f"{out}: {out.stat().st_size / 1e6:.1f} MB", flush=True)
