"""Pytest bootstrap: expone `src/` en `sys.path` para que los tests importen
los módulos heredados (`triqui`, `master_RL`, ...) y el paquete nuevo (`new.*`).

Además propaga `PYTHONPATH` al entorno: los procesos hijos lanzados por
`multiprocessing.Pool` (Fase 5) usan `spawn` en Windows y NO heredan las
modificaciones de `sys.path` del padre — sí heredan variables de entorno.
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
existing = os.environ.get("PYTHONPATH", "")
if str(SRC) not in existing.split(os.pathsep):
    os.environ["PYTHONPATH"] = (
        f"{SRC}{os.pathsep}{existing}" if existing else str(SRC)
    )
