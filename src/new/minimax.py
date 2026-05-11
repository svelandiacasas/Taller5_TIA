"""Oráculo Minimax para triqui.

Triqui es un juego resuelto con ~5478 estados legales. Esta implementación
usa minimax exhaustivo con memoización (transposition table compartida a
nivel de clase). No usamos alpha-beta porque para este orden de magnitud
la memoización ya domina y los valores cacheados con cutoffs requieren
flags de bound que no aportan valor práctico aquí.

Convención: trabaja sobre la representación del `Game` del compañero
(IDs `1` y `2`, vacío `0`). El `value` se reporta desde la perspectiva
del jugador asignado al agente: `+1` victoria, `0` empate, `-1` derrota.
"""
from __future__ import annotations

from typing import Optional

import numpy as np


def _winner(state: np.ndarray) -> Optional[int]:
    """Devuelve `1` o `2` si hay tres en línea; `0` si tablero lleno sin
    ganador; `None` si la partida está en curso."""
    for p in (1, 2):
        if (np.any(np.all(state == p, axis=1)) or
                np.any(np.all(state == p, axis=0)) or
                np.all(np.diag(state) == p) or
                np.all(np.diag(np.fliplr(state)) == p)):
            return p
    if np.all(state != 0):
        return 0
    return None


def _legal_actions(state: np.ndarray) -> list[tuple[int, int]]:
    return [(r, c) for r in range(3) for c in range(3) if state[r, c] == 0]


def _other(player: int) -> int:
    return 2 if player == 1 else 1


class MinimaxAgent:
    """Agente óptimo basado en minimax + memoización."""

    # Cache compartido a nivel de clase: clave = (state.tobytes(), player_to_move, agent_player_id)
    _cache: dict[tuple[bytes, int, int], float] = {}
    # Contador de invocaciones de `_minimax` que NO se sirvieron del cache (es decir,
    # que ejecutaron cómputo real). Útil para tests deterministas de "el cache evitó
    # recomputar"; no usar wall-time, que es flaky.
    _node_visits: int = 0

    def __init__(self, player_id: int, cache: bool = True):
        if player_id not in (1, 2):
            raise ValueError(f"player_id debe ser 1 o 2, recibió {player_id}")
        self.player_id = player_id
        self.use_cache = cache

    @classmethod
    def clear_cache(cls) -> None:
        cls._cache.clear()

    @classmethod
    def reset_counters(cls) -> None:
        cls._node_visits = 0

    def value(self, state: np.ndarray, player_to_move: int) -> float:
        """Valor minimax del estado, desde la perspectiva de `self.player_id`."""
        if player_to_move not in (1, 2):
            raise ValueError(f"player_to_move debe ser 1 o 2, recibió {player_to_move}")
        return self._minimax(np.asarray(state, dtype=int), player_to_move)

    def optimal_actions(self, game) -> list[tuple[int, int]]:
        """Todas las jugadas con valor óptimo desde la perspectiva del jugador-a-mover.

        Si al agente le toca, devuelve las jugadas que MAXIMIZAN el valor para él;
        si le toca al rival, devuelve las que MINIMIZAN (modela un rival óptimo).
        """
        return self.optimal_actions_from_state(game.get_game_matrix(), int(game.current_player))

    def optimal_actions_from_state(self, state, player_to_move: int) -> list[tuple[int, int]]:
        """Igual que `optimal_actions` pero sobre `(state, player_to_move)` crudos.

        `state` y `player_to_move` deben venir en convención del `Game` (IDs `1` y `2`).
        Útil para `diagnostics.py`, que evalúa estados intermedios del entrenamiento
        sin necesidad de instanciar un `Game`.
        """
        if player_to_move not in (1, 2):
            raise ValueError(f"player_to_move debe ser 1 o 2, recibió {player_to_move}")
        state = np.asarray(state, dtype=int).copy()
        legal = _legal_actions(state)
        if not legal:
            return []
        next_player = _other(player_to_move)
        scored: list[tuple[tuple[int, int], float]] = []
        for (r, c) in legal:
            new_state = state.copy()
            new_state[r, c] = player_to_move
            w = _winner(new_state)
            if w is not None:
                v = self._terminal_value(w)
            else:
                v = self._minimax(new_state, next_player)
            scored.append(((r, c), v))
        target = max if player_to_move == self.player_id else min
        best = target(v for _, v in scored)
        return [a for a, v in scored if v == best]

    def select_action(self, game) -> tuple[int, int]:
        """Una jugada óptima (con tiebreak aleatorio entre las óptimas)."""
        opt = self.optimal_actions(game)
        if not opt:
            raise ValueError("No hay jugadas legales en el estado actual")
        idx = int(np.random.randint(len(opt)))
        return opt[idx]

    # ------------------------------------------------------------------ #
    # Núcleo recursivo
    # ------------------------------------------------------------------ #
    def _terminal_value(self, winner: int) -> float:
        if winner == self.player_id:
            return 1.0
        if winner == 0:
            return 0.0
        return -1.0

    def _minimax(self, state: np.ndarray, player_to_move: int) -> float:
        w = _winner(state)
        if w is not None:
            return self._terminal_value(w)

        key = (state.tobytes(), player_to_move, self.player_id)
        if self.use_cache:
            cached = MinimaxAgent._cache.get(key)
            if cached is not None:
                return cached

        MinimaxAgent._node_visits += 1
        legal = _legal_actions(state)
        next_player = _other(player_to_move)
        target = max if player_to_move == self.player_id else min

        best = target(
            self._value_after(state, r, c, player_to_move, next_player)
            for (r, c) in legal
        )

        if self.use_cache:
            MinimaxAgent._cache[key] = best
        return best

    def _value_after(self, state: np.ndarray, r: int, c: int,
                     mover: int, next_player: int) -> float:
        new_state = state.copy()
        new_state[r, c] = mover
        return self._minimax(new_state, next_player)
