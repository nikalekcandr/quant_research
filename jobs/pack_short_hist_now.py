"""Zip the short-binary history downloaded so far (data/raw/kx15, coinbase_1m, pm_updown) -> _export/short_hist.zip"""
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "_export"
OUT.mkdir(exist_ok=True)
with zipfile.ZipFile(OUT / "short_hist.zip", "w", zipfile.ZIP_DEFLATED) as z:
    n = 0
    for sub in ("kx15", "coinbase_1m", "pm_updown"):
        for p in sorted((ROOT / "data" / "raw" / sub).rglob("*")):
            if p.is_file():
                z.write(p, p.relative_to(ROOT).as_posix())
                n += 1
print(f"{n} files -> {(OUT / 'short_hist.zip').stat().st_size / 1e6:.1f} MB", flush=True)
