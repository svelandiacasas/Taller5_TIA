"""Agente Q-Learning tabular como réplica fiel del agente de referencia.

`BaseQAgent` replica bit a bit el agente Q-Learning de referencia presentado
en la sesión del curso:

- Convención `X = 1`, `O = -1`, `EMPTY = 0`.
- `Q` única compartida entre X y O (defecto intencional).
- `alpha = 0.1`, `gamma = 0.9`, `epsilon = 0.2` (constante, sin decay).
- Recompensas `{+1, 0, -1}` solo terminales.
- 10 000 episodios por defecto.

Las patologías son intencionales — este agente existe como strawman para que
`diagnostics.py` las exhiba empíricamente; **no** debe "mejorarse".

La traducción a la convención del entorno `Game` (IDs `1` y `2`) vive
exclusivamente en wrappers (`select_action_eval` y `TabularToTorchAdapter`),
no en el núcleo del agente, para mantener la fidelidad línea a línea con la
implementación de referencia.
"""
from __future__ import annotations

import random
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from seeds import set_seed


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

    Réplica literal del `check_winner` de referencia: usa `sum`/`abs`/`np.sign`
    sobre filas, columnas y diagonales. Equivalente a la lógica del `Game` del
    entorno, pero opera en la convención `±1` (no en `1/2`).
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

    def choose_action(self, state_tuple: tuple, training: bool = True,
                      tiebreak_random: bool = False) -> tuple[int, int]:
        """Política ε-greedy. En `training=False` usa ε=0 (evaluación).

        El parámetro `tiebreak_random` controla la regla de desempate cuando
        varias acciones empatan en `max(Q)`:

        - `False` (defecto, fiel al profesor): `actions[qs.index(max_q)]`
          devuelve la PRIMERA acción del orden de `_available_actions`. Es el
          sesgo "primer índice válido" del notebook; **se preserva en `train`**.
        - `True` (solo evaluación): `random.choice` entre las empatadas.
          Indispensable para evaluación: con desempate determinista, el agente
          juega la misma partida 10 000 veces contra un oponente determinista
          y la varianza estadística es cero. Aprobado por el usuario en la
          discusión inicial (pregunta 8: "Variabilidad controlada > determinismo
          absoluto").
        """
        actions = _available_actions(np.array(state_tuple).reshape(3, 3))
        eps = self.epsilon if training else 0.0
        if random.random() < eps:
            return random.choice(actions)
        qs = [self.get_Q(state_tuple, a) for a in actions]
        max_q = max(qs)
        if tiebreak_random:
            best = [a for a, q in zip(actions, qs) if q == max_q]
            return random.choice(best)
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
    # Wrapper para usarlo como oponente del entorno `Game` (1, 2)
    # ------------------------------------------------------------------ #
    def select_action_eval(self, game) -> tuple[int, int]:
        """Devuelve `(row, col)` para jugar contra el entorno `Game`.

        Sin exploración (`epsilon = 0`) pero **con desempate aleatorio entre
        argmax**. La fidelidad bit a bit a la implementación de referencia
        solo aplica al bucle de entrenamiento
        (`train` -> `choose_action(training=True)`); esta capa de evaluación
        añade el desempate aleatorio para reducir la varianza de las métricas.

        Traduce la convención del `Game` (`{0, 1, 2}`) a la del agente
        (`{0, 1, -1}`) antes de consultar la Q-table.
        """
        raw = np.asarray(game.get_game_matrix(), dtype=int)
        state = np.where(raw == 2, -1, raw)
        state_tuple = _state_to_tuple(state)
        return self.choose_action(state_tuple, training=False, tiebreak_random=True)


# ---------------------------------------------------------------------------- #
# Adapter torch-compatible para el `Championship` y `Train.opponent`
# ---------------------------------------------------------------------------- #
class TabularToTorchAdapter(nn.Module):
    """Adapter `nn.Module` para enchufar un agente tabular al `Championship`
    y a `Train.opponent`, que esperan `forward(state) -> q_values`.

    El `state_tensor` recibido viene en convención del `Game` (valores en
    `{0, 1, 2}`). El adapter:

    1. Traduce a la convención del agente (`{-1, 0, 1}`, con `X=+1`, `O=-1`).
    2. Si `dual_perspective=True`, voltea el tablero según la paridad de
       fichas para que el jugador-a-mover sea siempre `+1`. Indispensable
       para `ImprovedQAgent` con dual-perspective.
    3. Para cada acción consulta:
       - `agent.get_Q(state_tuple, action)` si `default_value is None`
         (modo recomendado: delega la política de defaults y la
         canonicalización D₄ al agente).
       - `agent.Q.get((state_tuple, action), default_value)` si se
         proporciona `default_value` (modo directo, fidelidad estricta).

    El uso típico es:
    - `BaseQAgent`: `TabularToTorchAdapter(agent)` (defaults; `get_Q` devuelve
      `0.0` para pares no vistos, fiel al profesor).
    - `ImprovedQAgent` (dual_perspective=True): `TabularToTorchAdapter(agent,
      dual_perspective=True)` (canonicalización + `optimistic_init` quedan
      adentro de `agent.get_Q`).
    """

    def __init__(
        self,
        tabular_agent,
        dual_perspective: bool = False,
        default_value: float | None = None,
    ):
        super().__init__()
        self.agent = tabular_agent
        self.dual_perspective = dual_perspective
        self.default_value = default_value
        # Parámetro dummy para que `.to(device)`, `.eval()`, `.parameters()` no fallen.
        self.dummy = nn.Parameter(torch.zeros(1), requires_grad=False)

    def forward(self, state_tensor: torch.Tensor) -> torch.Tensor:
        """`state_tensor`: `[B, 9]` o `[9]` con valores en `{0, 1, 2}`.

        Devuelve `[B, 9]` con Q-values para las 9 acciones (incluso ocupadas;
        el caller filtra por jugadas legales).
        """
        if state_tensor.dim() == 1:
            state_tensor = state_tensor.unsqueeze(0)
        states = state_tensor.detach().cpu().numpy().astype(int)
        states_abs = np.where(states == 2, -1, states)  # X=+1, O=-1, vacío=0

        if self.dual_perspective:
            # Paridad: si #(+1) == #(-1), mueve X (+1); si #(+1) > #(-1), mueve O (-1).
            n_pos = (states_abs == 1).sum(axis=1)
            n_neg = (states_abs == -1).sum(axis=1)
            mover_is_x = n_pos == n_neg

        B = states_abs.shape[0]
        q = np.zeros((B, 9), dtype=np.float32)
        for b in range(B):
            if self.dual_perspective and not mover_is_x[b]:
                state_view = -states_abs[b]
            else:
                state_view = states_abs[b]
            state_tuple = tuple(state_view)
            for action_int in range(9):
                action = (action_int // 3, action_int % 3)
                if self.default_value is not None:
                    q[b, action_int] = self.agent.Q.get(
                        (state_tuple, action), self.default_value,
                    )
                else:
                    q[b, action_int] = self.agent.get_Q(state_tuple, action)
        return torch.from_numpy(q).to(state_tensor.device)
