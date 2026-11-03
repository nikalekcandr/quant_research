"""Demo trading, v3: v2 stopped at 23:35 03.11 because the 'cryptography' package (key signing) was not installed.
Installs it, then runs the v2 job (key scan, clock/balance, 1¢ test order + cancel, demo executor)."""
import runpy
import subprocess
import sys
from pathlib import Path

try:
    import cryptography  # noqa: F401
except ImportError:
    print("installing cryptography ...", flush=True)
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", "cryptography>=42"])
    import cryptography  # noqa: F401
print("cryptography", cryptography.__version__, flush=True)
runpy.run_path(str(Path(__file__).with_name("trader_demo_v2_now.py")), run_name="__main__")
