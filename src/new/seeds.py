"""Control determinista de semillas para reproducibilidad."""
import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Fija las semillas de `random`, `numpy` y `torch` (incluida CUDA si aplica).

    Llamar al inicio de cada worker en `multi_seed.py` y al inicio de cada
    script de entrenamiento. Las semillas no se heredan a los procesos hijo
    automáticamente, así que esta función debe re-ejecutarse explícitamente.

    Parameters
    ----------
    seed : int
        Semilla entera no negativa.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
