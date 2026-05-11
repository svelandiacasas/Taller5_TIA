"""Pytest bootstrap: expose src/ on sys.path so tests can import heritage modules
(triqui, master_RL, ...) and the new package (`new.*`)."""
import sys
from pathlib import Path

ROOT = Path(__file__).parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
