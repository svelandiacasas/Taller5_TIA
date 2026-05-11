"""Canonicalización D₄ para estados de triqui (3×3).

El grupo diédrico D₄ tiene 8 elementos: 4 rotaciones (0°, 90°, 180°, 270°) y
4 reflexiones (horizontal, vertical, sobre la diagonal principal, sobre la
anti-diagonal). En triqui un estado y todos sus 7 conjugados bajo D₄ son
estratégicamente equivalentes: sus Q-values óptimos coinciden módulo la
permutación de acciones inducida por la transformación.

Reduce el espacio de estados de ~5 478 a ~700 estados canónicos. Esto
acelera el aprendizaje de `ImprovedQAgent` cuando `use_symmetries=True`
en aproximadamente 8× en datos por estado.

Convención de transformaciones (índice `t ∈ {0..7}`):

| t | Transformación |
|---|----------------|
| 0 | Identidad |
| 1 | Rotación 90° (CCW) |
| 2 | Rotación 180° |
| 3 | Rotación 270° (CCW) = 90° (CW) |
| 4 | Reflexión horizontal (`fliplr`) |
| 5 | Transposición (reflexión sobre diagonal principal) |
| 6 | Reflexión vertical (`flipud`) |
| 7 | Reflexión sobre anti-diagonal |

`canonical_state(s)` devuelve `(s_canon, t)` donde `s_canon` es el
representante lexicográficamente mínimo entre las 8 transformaciones
de `s`, y `t` es el índice de la transformación que produjo `s_canon`.
"""
from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------- #
# Tablas precomputadas
# ---------------------------------------------------------------------- #
def _apply_transform(arr: np.ndarray, t: int) -> np.ndarray:
    """Aplica la t-ésima transformación D₄ a un array 3×3."""
    if t < 4:
        return np.rot90(arr, k=t)
    return np.rot90(np.fliplr(arr), k=t - 4)


def _build_action_maps() -> np.ndarray:
    """`ACTION_MAPS[t, a]` = posición a la que va la celda `a` bajo la
    transformación `t`."""
    grid = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]], dtype=int)
    maps = np.zeros((8, 9), dtype=int)
    for t in range(8):
        transformed = _apply_transform(grid, t)
        # `transformed[i, j]` = índice ORIGINAL de la celda que ahora ocupa (i, j).
        # Para construir el mapa "original → nuevo", invertimos la relación.
        for i in range(3):
            for j in range(3):
                orig = int(transformed[i, j])
                maps[t, orig] = i * 3 + j
    return maps


ACTION_MAPS: np.ndarray = _build_action_maps()

# Inversa de cada transformación: aplicada después de la original, regresa al frame inicial.
# - Identidad e involuciones (rot180, flips) son auto-inversas.
# - rot90 ↔ rot270 son inversas mutuas.
INVERSE_MAP: np.ndarray = np.array([0, 3, 2, 1, 4, 5, 6, 7], dtype=int)


# ---------------------------------------------------------------------- #
# API pública
# ---------------------------------------------------------------------- #
def canonical_state(state: np.ndarray) -> tuple[np.ndarray, int]:
    """Devuelve `(s_canon, t)` con `s_canon` el representante D₄-canónico
    (lex-mínimo) y `t` el índice de la transformación que produjo `s_canon`
    a partir de `state`. En caso de empates, gana el `t` menor (determinista)."""
    state = np.asarray(state, dtype=int)
    best_state: np.ndarray | None = None
    best_tuple: tuple | None = None
    best_t = 0
    for t in range(8):
        transformed = _apply_transform(state, t)
        as_tuple = tuple(transformed.flatten().tolist())
        if best_tuple is None or as_tuple < best_tuple:
            best_tuple = as_tuple
            best_state = transformed
            best_t = t
    return best_state, best_t  # type: ignore[return-value]


def canonical_action(state: np.ndarray, action_int: int) -> int:
    """Mapea `action_int` (en el frame original de `state`) al frame canónico.

    Equivale a aplicar a `action_int` la misma transformación `t` que llevó
    `state` a su forma canónica.
    """
    _, t = canonical_state(state)
    return int(ACTION_MAPS[t, action_int])


def restore_action(state: np.ndarray, canonical_action_int: int) -> int:
    """Inversa de `canonical_action`: dada una acción en el frame canónico
    de `state`, devuelve la acción equivalente en el frame original.

    Usa `INVERSE_MAP[t]` donde `t` es la transformación que canonicalizó
    `state`.
    """
    _, t = canonical_state(state)
    t_inv = int(INVERSE_MAP[t])
    return int(ACTION_MAPS[t_inv, canonical_action_int])
