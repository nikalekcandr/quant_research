"""Pack the maker-test data downloaded so far (trade prints + Binance 1 s) -> _export/maker_partial.tar.gz."""
import tarfile
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
exp = ROOT / "_export"
exp.mkdir(exist_ok=True)
with tarfile.open(exp / "maker_partial.tar.gz", "w:gz") as tar:
    for sub, pat in (("kx_wing_trades", "*.csv"), ("binance_1s", "*.csv.gz")):
        for p in sorted((ROOT / "data/raw" / sub).glob(pat)):
            tar.add(p, arcname=f"{sub}/{p.name}")
print("packed", (exp / "maker_partial.tar.gz").stat().st_size, flush=True)
