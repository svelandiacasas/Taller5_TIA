"""Utilidades de gestión de RNG.

`isolated_rng()`: context manager que aísla los RNGs de `random` y `numpy`
durante un bloque y los restaura al salir. Indispensable cuando una
evaluación intercalada en medio del entrenamiento no debe alterar la
trayectoria posterior del agente, o cuando un análisis ad-hoc no debe
contaminar el RNG global del proceso.
"""
from __future__ import annotations

import contextlib
import random

import numpy as np


@contextlib.contextmanager
def isolated_rng():
    """Guarda y restaura los estados de `random` y `np.random` al entrar/salir."""
    py_state = random.getstate()
    np_state = np.random.get_state()
    try:
        yield
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)
