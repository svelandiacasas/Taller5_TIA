"""Agente Q-Learning tabular fiel al notebook del profesor.

`BaseQAgent` replica bit a bit el código de `docs/referencia_profesor.md`:

- Convención `X = 1`, `O = -1`, `EMPTY = 0`.
- `Q` única compartida entre X y O (defecto intencional).
- `alpha = 0.1`, `gamma = 0.9`, `epsilon = 0.2` (constante, sin decay).
- Recompensas `{+1, 0, -1}` solo terminales.
- 10 000 episodios por defecto.

Las patologías son intencionales — este agente existe como strawman para que
`diagnostics.py` (Fase 3) las exhiba empíricamente; **no** debe "mejorarse".

La traducción a la convención del compañero (`Game` con IDs `1` y `2`) vive
exclusivamente en wrappers (`select_action_eval` y `TabularToTorchAdapter`),
no en el núcleo del agente, para mantener la fidelidad línea a línea con el
notebook del profesor.
"""
from __future__ import annotations

import random
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from new.seeds import set_seed


# Convenciones del profesor (no tocar)
EMPTY = 0
X = 1
O = -1


# ---------------------------------------------------------------------------- #
# Helpers (réplica literal del notebook)
# ---------------------------------------------------------------------------- #
def _initial_state() -> np.ndarray:
    return np.zeros((3, 3), dtype=int)


def _available_actions(state: np.ndarray) -> list[tuple[int, int]]:
    return [(i, j) for i in range(3) for j in range(3) if state[i, j] == EMPTY]


def _check_winner(state: np.ndarray):
    """Devuelve `+1` si gana X, `-1` si gana O, `0` empate, `None` en curso.

    Réplica literal del `check_winner` del profesor: usa `sum`/`abs`/`np.sign`
    sobre filas, columnas y diagonales. Equivalente a la lógica del `Game` del
    compañero, pero opera en la convención `±1` (no en `1/2`).
    """
    for i in range(3):
        if abs(sum(state[i, :])) == 3:
            return int(np.sign(sum(state[i, :])))
        if abs(sum(state[:, i])) == 3:
            return int(np.sign(sum(state[:, i])))
    diag1 = state[0, 0] + state[1, 1] + state[2, 2]
    diag2 = state[0, 2] + state[1, 1] + state[2, 0]
    if abs(diag1) == 3:
        return int(np.sign(diag1))
    if abs(diag2) == 3:
        return int(np.sign(diag2))
    if not _available_actions(state):
        return 0
    return None


def _state_to_tuple(state: np.ndarray) -> tuple:
    return tuple(state.flatten())


# ---------------------------------------------------------------------------- #
# Agente Q-Learning tabular (núcleo fiel al profesor)
# ---------------------------------------------------------------------------- #
class BaseQAgent:
    """Q-Learning tabular tal como en el notebook del profesor.

    Atributos
    ---------
    Q : dict
        `dict[(state_tuple, (i, j)), float]`. Defectos intencionales: una sola
        tabla compartida entre X y O.
    alpha, gamma, epsilon : float
        Hiperparámetros del notebook. `epsilon` es constante (sin decay).
    """

    def __init__(self, alpha: float = 0.1, gamma: float = 0.9, epsilon: float = 0.2):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.Q: dict[tuple[tuple, tuple[int, int]], float] = {}

    # ------------------------------------------------------------------ #
    # Núcleo (réplica literal del profesor)
    # ------------------------------------------------------------------ #
    def get_Q(self, state_tuple: tuple, action: tuple[int, int]) -> float:
        return self.Q.get((state_tuple, action), 0.0)

    def choose_action(self, state_tuple: tuple, training: bool = True) -> tuple[int, int]:
        """Política ε-greedy. En `training=False` usa ε=0 (evaluación)."""
        actions = _available_actions(np.array(state_tuple).reshape(3, 3))
        eps = self.epsilon if training else 0.0
        if random.random() < eps:
            return random.choice(actions)
        qs = [self.get_Q(state_tuple, a) for a in actions]
        max_q = max(qs)
        return actions[qs.index(max_q)]

    def update_Q(self, state_tuple: tuple, action: tuple[int, int],
                 reward: float, next_state_tuple: tuple) -> None:
        actions = _available_actions(np.array(next_state_tuple).reshape(3, 3))
        max_q_next = (
            0.0 if not actions
            else max(self.get_Q(next_state_tuple, a) for a in actions)
        )
        old_q = self.get_Q(state_tuple, action)
        self.Q[(state_tuple, action)] = (
            old_q + self.alpha * (reward + self.gamma * max_q_next - old_q)
        )

    def train(self, episodes: int = 10_000, seed: Optional[int] = None) -> None:
        """Self-play degenerado del notebook: una sola Q para ambos jugadores.

        Replica el bucle de entrenamiento línea a línea con el del notebook:
        recompensa terminal aplicada al último estado-acción (que puede ser de
        X o de O — la patología de asignación de crédito), `current_player *= -1`,
        recompensa `0` en transiciones intermedias.
        """
        if seed is not None:
            set_seed(seed)
        for _ in range(episodes):
            state = _initial_state()
            current_player = X
            while True:
                s = _state_to_tuple(state)
                action = self.choose_action(s, training=True)
                state[action] = current_player
                result = _check_winner(state)
                s_next = _state_to_tuple(state)
                if result is not None:
                    reward = 1 if result == X else (-1 if result == O else 0)
                    self.update_Q(s, action, reward, s_next)
                    break
                else:
                    self.update_Q(s, action, 0, s_next)
                current_player *= -1

    # ------------------------------------------------------------------ #
    # Wrapper para usarlo como oponente del `Game` del compañero (1, 2)
    # ------------------------------------------------------------------ #
    def select_action_eval(self, game) -> tuple[int, int]:
        """Devuelve `(row, col)` para jugar contra el `Game` del compañero.

        Sin exploración (`epsilon = 0`). Traduce la convención del `Game`
        (`{0, 1, 2}`) a la convención del agente (`{0, 1, -1}`) antes de
        consultar la Q-table.
        """
        raw = np.asarray(game.get_game_matrix(), dtype=int)
        state = np.where(raw == 2, -1, raw)
        state_tuple = _state_to_tuple(state)
        return self.choose_action(state_tuple, training=False)


# ---------------------------------------------------------------------------- #
# Adapter torch-compatible para el `Championship` del compañero
# ---------------------------------------------------------------------------- #
class TabularToTorchAdapter(nn.Module):
    """Permite que un agente tabular se enchufe al `Championship` y a
    `Train.opponent` cuando esperan `nn.Module` con `forward(state) -> q_values`.

    El `state_tensor` recibido viene en convención del `Game` (valores en
    `{0, 1, 2}` por casilla). El adapter lo traduce a la convención del agente
    (`{-1, 0, 1}`) antes de buscar `(state_tuple, action)` en la Q-table.

    Política para estados o pares no vistos:
    - `BaseQAgent` → `default_value = 0.0` (fidelidad estricta al profesor:
      el `argmax` del torneo recae en la primera posición legal, igual que en
      el notebook).
    - `ImprovedQAgent` (Fase 4) → `default_value = optimistic_init` (e.g. 0.5).
    """

    def __init__(self, tabular_agent, default_value: float = 0.0):
        super().__init__()
        self.agent = tabular_agent
        self.default = float(default_value)
        # Parámetro dummy para que `.to(device)`, `.eval()`, `.parameters()` no fallen.
        self.dummy = nn.Parameter(torch.zeros(1), requires_grad=False)

    def forward(self, state_tensor: torch.Tensor) -> torch.Tensor:
        """`state_tensor`: `[B, 9]` o `[9]` con valores en `{0, 1, 2}`.

        Devuelve `[B, 9]` con Q-values para las 9 acciones (incluso ocupadas;
        el caller —`Championship` o `Train`— filtra por jugadas legales).
        """
        if state_tensor.dim() == 1:
            state_tensor = state_tensor.unsqueeze(0)
        states = state_tensor.detach().cpu().numpy().astype(int)  # [B, 9] en {0, 1, 2}
        states_translated = np.where(states == 2, -1, states)
        B = states_translated.shape[0]
        q = np.full((B, 9), self.default, dtype=np.float32)
        for b in range(B):
            state_tuple = tuple(states_translated[b])
            for action_int in range(9):
                action = (action_int // 3, action_int % 3)
                q[b, action_int] = self.agent.Q.get((state_tuple, action), self.default)
        return torch.from_numpy(q).to(state_tensor.device)
