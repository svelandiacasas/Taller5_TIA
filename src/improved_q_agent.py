"""Q-Learning tabular mejorado para triqui.

`ImprovedQAgent` corrige las patologías diagnosticadas en `BaseQAgent`
y supera al strawman de referencia:

- **Dual perspective** (`dual_perspective=True`): Q se indexa por estados en
  la vista del jugador-a-mover (siempre `+1`). El bucle de entrenamiento usa
  un Bellman tipo *negamax* que niega el valor del próximo estado al cambiar
  de perspectiva. Esto **resuelve la asignación de crédito invertida**: cuando
  O gana, su última jugada queda con valor positivo en Q (perspectiva de O).

- **ε decay exponencial** (`epsilon_start=1.0`, `epsilon_end=0.05`,
  `epsilon_decay=0.9995`): explora intensamente al principio, explota más al
  final. Reemplaza el `ε=0.2` constante del agente de referencia.

- **Reward shaping potential-based** (`reward_shaping=True`,
  `shaping_weight=0.1`): añade `F(s, a, s') = γ·Φ(s') − Φ(s)` con

      Φ(s) = w · (líneas con 2 fichas mías y 0 del rival
                  − líneas con 2 fichas del rival y 0 mías)

  Sigue la forma de Ng, Harada, Russell (1999): dado que Φ es función SOLO
  del estado, la política óptima de la MDP modificada es idéntica a la
  original — el shaping solo acelera el aprendizaje, no lo sesga. Se aplica
  Φ entre la pre-acción y la post-acción ambas en perspectiva del mover (la
  "variante de efecto inmediato" — más débil que la formulación canónica con
  delay de un turno por jugador, pero sin reordenar el ciclo de updates).

  La elección `w = 0.1` es lo suficientemente pequeña para no dominar las
  recompensas terminales `±1`: en el peor caso (4 líneas abiertas a favor,
  0 en contra), Φ alcanza `0.4` y la suma telescópica de F a lo largo de la
  partida es `γ·Φ(s_terminal) − Φ(s_inicial) ≤ γ·Φ_max ≈ 0.4` < 1.

- **Canonicalización D₄** (`use_symmetries=True`): Q se indexa por la forma
  canónica del estado (el lex-mínimo entre las 8 transformaciones del grupo
  diédrico). El espacio de estados pasa de ~5478 a ~700 estados únicos: ~8×
  más eficiente en datos por estado.

- **Inicialización optimista** (`optimistic_init=0.0` por defecto, ajustable):
  para pares `(state, action)` no vistos, `get_Q` devuelve este valor.
  Optimismo (e.g. `0.5`) fomenta exploración temprana incluso con `ε` bajo.

Cada toggle se puede desactivar individualmente para el ablation study de
Fase 6 — la idea es aislar el efecto de cada mejora.
"""
from __future__ import annotations

import contextlib
import random
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from base_q_agent import (
    O,
    X,
    TabularToTorchAdapter,
    _available_actions,
    _check_winner,
    _initial_state,
    _state_to_tuple,
)
from rng_utils import isolated_rng as _isolated_rng
from seeds import set_seed
from symmetry import ACTION_MAPS, canonical_state


# ---------------------------------------------------------------------- #
# Helpers de potencial Φ (líneas con dos fichas y la tercera vacía)
# ---------------------------------------------------------------------- #
def _count_open_twos(state: np.ndarray, player: int) -> int:
    """Número de líneas (3 filas + 3 columnas + 2 diagonales) con exactamente
    2 fichas de `player` y 1 celda vacía."""
    count = 0
    for r in range(3):
        line = state[r, :]
        if int((line == player).sum()) == 2 and int((line == 0).sum()) == 1:
            count += 1
    for c in range(3):
        line = state[:, c]
        if int((line == player).sum()) == 2 and int((line == 0).sum()) == 1:
            count += 1
    diag = np.diag(state)
    if int((diag == player).sum()) == 2 and int((diag == 0).sum()) == 1:
        count += 1
    antidiag = np.diag(np.fliplr(state))
    if int((antidiag == player).sum()) == 2 and int((antidiag == 0).sum()) == 1:
        count += 1
    return count


# ---------------------------------------------------------------------- #
# Agente mejorado
# ---------------------------------------------------------------------- #
class ImprovedQAgent:
    """Q-Learning tabular con todas las mejoras del taller. Cada mejora es
    un toggle independiente para aislarlas en el ablation study (Fase 6)."""

    def __init__(
        self,
        alpha: float = 0.1,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: float = 0.9995,
        use_symmetries: bool = True,
        reward_shaping: bool = True,
        shaping_weight: float = 0.1,
        optimistic_init: float = 0.0,
        dual_perspective: bool = True,
    ):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.epsilon = epsilon_start
        self.use_symmetries = use_symmetries
        self.reward_shaping = reward_shaping
        self.shaping_weight = shaping_weight
        self.optimistic_init = optimistic_init
        self.dual_perspective = dual_perspective
        self.Q: dict[tuple[tuple, tuple[int, int]], float] = {}

    # ------------------------------------------------------------------ #
    # Q-table: lookup y update con canonicalización opcional
    # ------------------------------------------------------------------ #
    def _canonical_key(self, state_tuple: tuple,
                       action: tuple[int, int]) -> tuple[tuple, tuple[int, int]]:
        """Devuelve la clave (state_canon_tuple, action_canon) bajo la cual
        este `(state, action)` se indexa en Q. Sin simetrías, identidad."""
        if not self.use_symmetries:
            return state_tuple, action
        state = np.array(state_tuple).reshape(3, 3)
        canon, t = canonical_state(state)
        action_int = action[0] * 3 + action[1]
        canon_action_int = int(ACTION_MAPS[t, action_int])
        return tuple(canon.flatten().tolist()), (canon_action_int // 3, canon_action_int % 3)

    def get_Q(self, state_tuple: tuple, action: tuple[int, int]) -> float:
        """Q(s, a) con canonicalización D₄ si aplica. Default = optimistic_init."""
        key_state, key_action = self._canonical_key(state_tuple, action)
        return self.Q.get((key_state, key_action), self.optimistic_init)

    def _set_Q(self, state_tuple: tuple, action: tuple[int, int], value: float) -> None:
        key_state, key_action = self._canonical_key(state_tuple, action)
        self.Q[(key_state, key_action)] = value

    # ------------------------------------------------------------------ #
    # Política
    # ------------------------------------------------------------------ #
    def choose_action(self, state_tuple: tuple, training: bool = True,
                      tiebreak_random: bool = False) -> tuple[int, int]:
        """ε-greedy. Igual interfaz que `BaseQAgent.choose_action`."""
        state = np.array(state_tuple).reshape(3, 3)
        actions = _available_actions(state)
        eps = self.epsilon if training else 0.0
        if random.random() < eps:
            return random.choice(actions)
        qs = [self.get_Q(state_tuple, a) for a in actions]
        max_q = max(qs)
        if tiebreak_random:
            best = [a for a, q in zip(actions, qs) if q == max_q]
            return random.choice(best)
        return actions[qs.index(max_q)]

    # ------------------------------------------------------------------ #
    # Reward shaping potential
    # ------------------------------------------------------------------ #
    def _phi(self, state_view: np.ndarray) -> float:
        """Potencial Φ desde la perspectiva del jugador `+1`.

        `state_view` debe estar en la convención `+1 = mover, -1 = opp` (si
        `dual_perspective=True`) o `+1 = X, -1 = O` (si `False`). En ambos
        casos el cómputo es el mismo: ventaja del jugador `+1` en líneas
        con 2 fichas y la tercera vacía.
        """
        my_twos = _count_open_twos(state_view, +1)
        opp_twos = _count_open_twos(state_view, -1)
        return self.shaping_weight * (my_twos - opp_twos)

    # ------------------------------------------------------------------ #
    # Entrenamiento
    # ------------------------------------------------------------------ #
    def train(
        self,
        episodes: int = 50_000,
        seed: Optional[int] = None,
        eval_every: Optional[int] = None,
        eval_episodes: int = 200,
    ) -> dict:
        """Entrena por `episodes` episodios. Devuelve `dict` con el historial.

        Si `eval_every` es entero, evalúa vs Random cada ese número de
        episodios y registra `win_rate_vs_random` en el historial. La
        evaluación está aislada del RNG global (no contamina el entrenamiento).
        """
        if seed is not None:
            set_seed(seed)
        self.epsilon = self.epsilon_start

        history = {
            "episode": [],
            "epsilon": [],
            "win_rate_vs_random": [],
        }

        train_step = (
            self._train_step_dual_perspective if self.dual_perspective
            else self._train_step_absolute_frame
        )

        for ep in range(episodes):
            train_step()
            self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

            if eval_every is not None and (ep + 1) % eval_every == 0:
                wr = self._eval_vs_random(n_episodes=eval_episodes,
                                          seed=12345 + ep + 1)
                history["episode"].append(ep + 1)
                history["epsilon"].append(self.epsilon)
                history["win_rate_vs_random"].append(wr)
        return history

    def _train_step_dual_perspective(self) -> None:
        """Un episodio de self-play en perspectiva del mover, con Bellman
        tipo negamax. Resuelve la asignación de crédito por construcción."""
        state_abs = _initial_state()
        current_player = X  # +1 en frame absoluto

        while True:
            # Vista del mover: mover es siempre +1.
            state_view = state_abs * current_player
            state_tuple = _state_to_tuple(state_view)
            action = self.choose_action(state_tuple, training=True)

            # Aplicar acción en frame absoluto.
            state_abs[action] = current_player

            # Vista post-acción: en el frame del mover, el +1 marca su jugada nueva.
            state_after_view = state_view.copy()
            state_after_view[action] = +1

            # Terminal? `_check_winner` opera en frame con +1/-1 (cualquiera): aquí
            # el mover es +1, así que +1 = victoria del mover, -1 imposible, 0 empate.
            result = _check_winner(state_after_view)

            if self.reward_shaping:
                shaping = self.gamma * self._phi(state_after_view) - self._phi(state_view)
            else:
                shaping = 0.0

            if result is not None:
                reward = float(result) if result != 0 else 0.0  # +1 win, 0 draw
                target = reward + shaping  # sin V_next: terminal
                old_q = self.get_Q(state_tuple, action)
                self._set_Q(state_tuple, action, old_q + self.alpha * (target - old_q))
                break

            # No terminal: vista del oponente del próximo estado (negación).
            opp_view = -state_after_view
            opp_tuple = _state_to_tuple(opp_view)
            opp_legal = _available_actions(opp_view)
            max_q_opp = max(self.get_Q(opp_tuple, a) for a in opp_legal)

            # Negamax: futuro del mover = -futuro del oponente desde su vista.
            target = shaping - self.gamma * max_q_opp
            old_q = self.get_Q(state_tuple, action)
            self._set_Q(state_tuple, action, old_q + self.alpha * (target - old_q))

            current_player = -current_player

    def _train_step_absolute_frame(self) -> None:
        """Un episodio en frame absoluto (X=+1, O=-1) con recompensa siempre
        desde la perspectiva de X — réplica deliberada del bug de
        asignación de crédito de `BaseQAgent`. Solo se usa cuando
        `dual_perspective=False` (ablation: aislar el efecto de la perspectiva)."""
        state_abs = _initial_state()
        current_player = X
        while True:
            state_tuple = _state_to_tuple(state_abs)
            action = self.choose_action(state_tuple, training=True)
            state_abs[action] = current_player
            state_after_tuple = _state_to_tuple(state_abs)
            result = _check_winner(state_abs)

            if self.reward_shaping:
                # En frame absoluto: Φ desde perspectiva de X (X-twos − O-twos).
                shaping = self.gamma * self._phi(state_abs) - self._phi(
                    np.array(state_tuple).reshape(3, 3)
                )
            else:
                shaping = 0.0

            if result is not None:
                # Recompensa desde X siempre: +1 si gana X, -1 si gana O, 0 empate.
                reward = float(result)
                target = reward + shaping
                old_q = self.get_Q(state_tuple, action)
                self._set_Q(state_tuple, action, old_q + self.alpha * (target - old_q))
                break

            next_legal = _available_actions(state_abs)
            max_q_next = max(self.get_Q(state_after_tuple, a) for a in next_legal)
            target = shaping + self.gamma * max_q_next  # Bellman normal, sin negamax
            old_q = self.get_Q(state_tuple, action)
            self._set_Q(state_tuple, action, old_q + self.alpha * (target - old_q))
            current_player = -current_player

    # ------------------------------------------------------------------ #
    # Wrapper para el entorno `Game`
    # ------------------------------------------------------------------ #
    def select_action_eval(self, game) -> tuple[int, int]:
        """Sin exploración (ε=0) y con tiebreak aleatorio entre argmax.

        Traduce `Game (1, 2)` → frame absoluto (`±1`); si `dual_perspective`,
        además voltea según quién mueve para obtener la vista del mover."""
        raw = np.asarray(game.get_game_matrix(), dtype=int)
        state_abs = np.where(raw == 2, -1, raw).astype(int)
        if self.dual_perspective:
            cp = int(game.current_player)
            state_view = state_abs if cp == 1 else -state_abs
        else:
            state_view = state_abs
        state_tuple = _state_to_tuple(state_view)
        return self.choose_action(state_tuple, training=False, tiebreak_random=True)

    # ------------------------------------------------------------------ #
    # Adapter torch-compatible
    # ------------------------------------------------------------------ #
    def to_torch_adapter(self) -> TabularToTorchAdapter:
        """Devuelve un `TabularToTorchAdapter` configurado para este agente
        (propaga `dual_perspective` automáticamente)."""
        return TabularToTorchAdapter(self, dual_perspective=self.dual_perspective)

    # ------------------------------------------------------------------ #
    # Evaluación interna (usada por `train` cuando `eval_every` está activo)
    # ------------------------------------------------------------------ #
    def _eval_vs_random(self, n_episodes: int = 200, seed: int = 42,
                         agent_role: int = 1) -> float:
        """Win rate vs un oponente uniforme aleatorio (X o O según `agent_role`).

        Aislado del RNG global para no contaminar el entrenamiento.
        Importa `Game` localmente para evitar ciclos de dependencia.
        """
        from triqui import Game  # import local para no tocar el import global

        with _isolated_rng():
            set_seed(seed)
            wins = 0
            for _ in range(n_episodes):
                game = Game()
                while not game.game_over:
                    cp = game.current_player
                    if cp == agent_role:
                        r, c = self.select_action_eval(game)
                        game._execute_move(r, c, cp)
                    else:
                        pos = game.available_positions()
                        idx = np.random.randint(len(pos))
                        game._execute_move(int(pos[idx][0]), int(pos[idx][1]), cp)
                if game.get_winner() == agent_role:
                    wins += 1
            return wins / n_episodes
