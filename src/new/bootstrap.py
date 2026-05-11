"""Intervalos de confianza bootstrap y test pareado para comparación de medias.

Dos funciones públicas:

- `bootstrap_ci(values, statistic=np.mean, n_resamples=10000, confidence=0.95, seed=0)`
  → `(estadístico, low, high)` percentile-bootstrap.

- `paired_bootstrap_test(a, b, n_resamples=10000, seed=0)` → `p_value` para
  H₀: `mean(a) == mean(b)`. Usa el método de centrar diferencias bajo H₀.

Ambas usan `numpy.random.default_rng(seed)` interno para reproducibilidad.
La generación de muestras es vectorizada para velocidad (n_resamples × n
muestras de una sola pasada) y, cuando el `statistic` soporta `axis`, se
aplica vectorizadamente; si no, cae a un loop de Python.
"""
from __future__ import annotations

from typing import Callable

import numpy as np


def _vectorized_apply(statistic: Callable, samples: np.ndarray) -> np.ndarray:
    """Aplica `statistic` a cada fila de `samples`. Vectorizado si soporta
    `axis=1`, loop de Python si no."""
    try:
        out = statistic(samples, axis=1)
        return np.asarray(out)
    except TypeError:
        return np.array([statistic(s) for s in samples])


def bootstrap_ci(
    values: np.ndarray | list[float],
    statistic: Callable = np.mean,
    n_resamples: int = 10_000,
    confidence: float = 0.95,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Intervalo de confianza percentile-bootstrap para `statistic(values)`.

    Parameters
    ----------
    values : array-like
        Muestra observada.
    statistic : callable
        Función `f(values)` o `f(values, axis=...)` que devuelve un escalar.
        Default `np.mean`.
    n_resamples : int
        Número de remuestreos bootstrap (default 10 000).
    confidence : float
        Nivel de confianza, e.g. 0.95.
    seed : int
        Semilla de `np.random.default_rng` para reproducibilidad.

    Returns
    -------
    (estimate, low, high) : tuple[float, float, float]
        `estimate = statistic(values)`; `low`, `high` son los percentiles
        bootstrap del estadístico.
    """
    values = np.asarray(values, dtype=float)
    n = len(values)
    if n == 0:
        raise ValueError("values must be non-empty")
    if not 0 < confidence < 1:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")

    rng = np.random.default_rng(seed)
    indices = rng.integers(0, n, size=(n_resamples, n))
    samples = values[indices]
    boot_estimates = _vectorized_apply(statistic, samples)

    estimate = float(statistic(values))
    alpha = (1.0 - confidence) / 2.0
    low = float(np.percentile(boot_estimates, 100.0 * alpha))
    high = float(np.percentile(boot_estimates, 100.0 * (1.0 - alpha)))
    return estimate, low, high


def paired_bootstrap_test(
    a: np.ndarray | list[float],
    b: np.ndarray | list[float],
    n_resamples: int = 10_000,
    seed: int = 0,
) -> float:
    """Test bootstrap pareado para H₀: `mean(a) == mean(b)`.

    Asume que `a` y `b` son muestras pareadas (p.ej. dos métricas medidas
    sobre las mismas N semillas). Devuelve `p_value` bilateral.

    Implementación:
    1. Calcular `diffs = a - b`, `observed = mean(diffs)`.
    2. Centrar: `centered = diffs - observed` (bajo H₀, su media es 0).
    3. Generar `n_resamples` remuestreos con reposición de `centered` y
       calcular su media → distribución nula de la media de las diferencias.
    4. `p_value = P(|boot_mean| >= |observed|)`.

    Returns
    -------
    p_value : float in [0, 1]
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"shapes incompatibles: a={a.shape} b={b.shape}")
    n = len(a)
    if n == 0:
        raise ValueError("a, b must be non-empty")

    diffs = a - b
    observed = float(np.mean(diffs))
    centered = diffs - observed

    rng = np.random.default_rng(seed)
    indices = rng.integers(0, n, size=(n_resamples, n))
    boot_means = centered[indices].mean(axis=1)

    return float(np.mean(np.abs(boot_means) >= abs(observed)))
