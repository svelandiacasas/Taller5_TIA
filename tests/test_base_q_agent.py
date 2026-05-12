"""Tests para `BaseQAgent` y `TabularToTorchAdapter`.

Garantías verificadas:

1. **Determinismo**: misma seed → misma Q-table (claves y valores idénticos).
2. **Fidelidad bit a bit**: una reimplementación literal del agente de referencia
   y `BaseQAgent` con la misma seed producen Q-tables idénticas.
3. **Helpers** (`_check_winner`): casos representativos.
4. **Adapter**: shapes correctos, traducción Game(1, 2) → agente(±1) correcta,
   defaults para estados no vistos.
5. **Wrapper de oponente** (`select_action_eval`): jugadas legales contra `Game`.
6. **Smoke**: jugar contra random sin romper.
7. **Patología empírica**: tras N episodios existen valores negativos en `Q`
   (la asignación de crédito errónea documentada en la sección 5 del notebook).
"""
import hashlib
import random as _random

import numpy as np
import pytest
import torch

from triqui import Game
from base_q_agent import (
    BaseQAgent,
    TabularToTorchAdapter,
    _check_winner,
)
from seeds import set_seed


# --------------------------------------------------------------------- #
# 1. Determinismo
# --------------------------------------------------------------------- #
def _hash_q_table(Q: dict) -> str:
    """Hash determinista de la Q-table (independiente del orden de inserción)."""
    items = sorted(Q.items(), key=lambda kv: repr(kv[0]))
    h = hashlib.sha256()
    for k, v in items:
        h.update(repr(k).encode())
        h.update(f"{v:.10f}".encode())
    return h.hexdigest()


def test_base_q_agent_is_deterministic_with_same_seed():
    a1 = BaseQAgent()
    a1.train(episodes=200, seed=42)
    a2 = BaseQAgent()
    a2.train(episodes=200, seed=42)
    assert len(a1.Q) == len(a2.Q)
    assert _hash_q_table(a1.Q) == _hash_q_table(a2.Q)


def test_base_q_agent_differs_with_different_seed():
    a1 = BaseQAgent()
    a1.train(episodes=200, seed=42)
    a2 = BaseQAgent()
    a2.train(episodes=200, seed=7)
    assert _hash_q_table(a1.Q) != _hash_q_table(a2.Q)


# --------------------------------------------------------------------- #
# 2. Fidelidad bit a bit al código de referencia
# --------------------------------------------------------------------- #
def test_base_q_agent_matches_professor_code_exactly():
    """Reimplementación literal del agente Q-Learning de referencia versus
    `BaseQAgent` con la misma seed: Q-tables IDÉNTICAS."""
    EMPTY_, X_, O_ = 0, 1, -1

    def initial_state():
        return np.zeros((3, 3), dtype=int)

    def available_actions(state):
        return [(i, j) for i in range(3) for j in range(3) if state[i, j] == EMPTY_]

    def check_winner(state):
        for i in range(3):
            if abs(sum(state[i, :])) == 3:
                return np.sign(sum(state[i, :]))
            if abs(sum(state[:, i])) == 3:
                return np.sign(sum(state[:, i]))
        d1 = state[0, 0] + state[1, 1] + state[2, 2]
        d2 = state[0, 2] + state[1, 1] + state[2, 0]
        if abs(d1) == 3:
            return np.sign(d1)
        if abs(d2) == 3:
            return np.sign(d2)
        if not available_actions(state):
            return 0
        return None

    def state_to_tuple(state):
        return tuple(state.flatten())

    Q_prof: dict = {}
    alpha, gamma, epsilon = 0.1, 0.9, 0.2

    def get_Q(s, a):
        return Q_prof.get((s, a), 0.0)

    def choose_action(s):
        actions = available_actions(np.array(s).reshape(3, 3))
        if _random.random() < epsilon:
            return _random.choice(actions)
        qs = [get_Q(s, a) for a in actions]
        max_q = max(qs)
        return actions[qs.index(max_q)]

    def update_Q(s, a, r, s_next):
        actions = available_actions(np.array(s_next).reshape(3, 3))
        max_q_next = 0 if not actions else max([get_Q(s_next, a_) for a_ in actions])
        old_q = get_Q(s, a)
        Q_prof[(s, a)] = old_q + alpha * (r + gamma * max_q_next - old_q)

    EPISODES = 200
    SEED = 42

    set_seed(SEED)
    for _ in range(EPISODES):
        state = initial_state()
        current_player = X_
        while True:
            s = state_to_tuple(state)
            action = choose_action(s)
            state[action] = current_player
            result = check_winner(state)
            s_next = state_to_tuple(state)
            if result is not None:
                reward = 1 if result == X_ else -1 if result == O_ else 0
                update_Q(s, action, reward, s_next)
                break
            else:
                update_Q(s, action, 0, s_next)
            current_player *= -1

    agent = BaseQAgent()
    agent.train(episodes=EPISODES, seed=SEED)

    assert set(Q_prof.keys()) == set(agent.Q.keys()), (
        f"Claves distintas: prof={len(Q_prof)} vs agente={len(agent.Q)}"
    )
    for k in Q_prof:
        assert Q_prof[k] == pytest.approx(agent.Q[k], abs=1e-12), (
            f"Mismatch en clave {k}: prof={Q_prof[k]} agente={agent.Q[k]}"
        )


# --------------------------------------------------------------------- #
# 3. Helpers: _check_winner
# --------------------------------------------------------------------- #
def test_check_winner_x_wins_row():
    state = np.array([[1, 1, 1], [-1, -1, 0], [0, 0, 0]], dtype=int)
    assert _check_winner(state) == 1


def test_check_winner_o_wins_diag():
    state = np.array([[-1, 1, 0], [1, -1, 0], [0, 0, -1]], dtype=int)
    assert _check_winner(state) == -1


def test_check_winner_draw():
    state = np.array([[1, -1, 1], [1, -1, -1], [-1, 1, 1]], dtype=int)
    assert _check_winner(state) == 0


def test_check_winner_in_progress():
    state = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 0]], dtype=int)
    assert _check_winner(state) is None


# --------------------------------------------------------------------- #
# 4. Adapter: shapes y traducción 1/2 ↔ ±1
# --------------------------------------------------------------------- #
def test_adapter_returns_correct_shape():
    agent = BaseQAgent()
    agent.train(episodes=50, seed=0)
    adapter = TabularToTorchAdapter(agent)
    state = torch.zeros((1, 9), dtype=torch.float32)
    out = adapter(state)
    assert out.shape == (1, 9)
    assert out.dtype == torch.float32


def test_adapter_handles_batch():
    agent = BaseQAgent()
    agent.train(episodes=50, seed=0)
    adapter = TabularToTorchAdapter(agent)
    batch = torch.zeros((4, 9), dtype=torch.float32)
    out = adapter(batch)
    assert out.shape == (4, 9)


def test_adapter_handles_1d_input():
    """El adapter acepta tanto `[9]` como `[B, 9]`."""
    agent = BaseQAgent()
    agent.train(episodes=20, seed=0)
    adapter = TabularToTorchAdapter(agent)
    state = torch.zeros(9, dtype=torch.float32)
    out = adapter(state)
    assert out.shape == (1, 9)


def test_adapter_returns_zeros_for_unseen_state():
    """`BaseQAgent` con `default=0.0`: estado nunca visto → vector cero."""
    agent = BaseQAgent()  # Q vacía
    adapter = TabularToTorchAdapter(agent, default_value=0.0)
    state = torch.tensor([[1.0, 0, 0, 0, 2, 0, 0, 0, 0]], dtype=torch.float32)
    out = adapter(state)
    assert torch.all(out == 0.0)


def test_adapter_translates_game_convention_to_agent_convention():
    """El adapter recibe estados con `2` (Game) pero indexa Q con `-1` (agente).
    Se siembra una entrada conocida en Q y se confirma el lookup correcto."""
    agent = BaseQAgent()
    # Estado en convención del agente: X esquina (0,0), O centro (1,1)
    agent_state = tuple(np.array([1, 0, 0, 0, -1, 0, 0, 0, 0]))
    agent.Q[(agent_state, (2, 2))] = 0.7  # Q sembrada para acción (2, 2)

    adapter = TabularToTorchAdapter(agent, default_value=0.0)
    # Mismo estado en convención Game (X=1, O=2)
    game_state = torch.tensor([[1.0, 0, 0, 0, 2, 0, 0, 0, 0]], dtype=torch.float32)
    out = adapter(game_state)

    # Acción (2, 2) → índice 8 → debe valer 0.7
    assert out[0, 8].item() == pytest.approx(0.7)
    # Las otras 8 acciones siguen en default 0.0
    assert torch.all(out[0, :8] == 0.0)


def test_adapter_default_value_for_improved_agent_use_case():
    """`default_value` configurable: cuando se use con `ImprovedQAgent` con
    `optimistic_init=0.5`, el adapter debe devolver 0.5 para entradas no vistas."""
    agent = BaseQAgent()  # Q vacía
    adapter = TabularToTorchAdapter(agent, default_value=0.5)
    state = torch.zeros((1, 9), dtype=torch.float32)
    out = adapter(state)
    assert torch.all(out == 0.5)


# --------------------------------------------------------------------- #
# 5. select_action_eval contra el entorno `Game`
# --------------------------------------------------------------------- #
def test_select_action_eval_returns_legal_move_on_empty_board():
    agent = BaseQAgent()
    agent.train(episodes=50, seed=0)
    game = Game()
    move = agent.select_action_eval(game)
    assert isinstance(move, tuple) and len(move) == 2
    r, c = move
    assert 0 <= r <= 2 and 0 <= c <= 2
    assert game.game_matrix[r, c] == 0


def test_select_action_eval_translates_game_convention():
    """El agente debe interpretar el `2` de Game como `-1` (O) en su Q.
    Sembramos preferencia clara y confirmamos que la usa en el lookup."""
    agent = BaseQAgent()
    state_agent = tuple(np.array([1, 0, 0, 0, -1, 0, 0, 0, 0]))
    agent.Q[(state_agent, (2, 2))] = 5.0  # Q alta para acción (2, 2)

    # Posición equivalente en Game: X (=1) en (0,0), O (=2) en (1,1)
    game = Game()
    game._execute_move(0, 0, 1)
    game._execute_move(1, 1, 2)
    assert game.current_player == 1

    move = agent.select_action_eval(game)
    assert move == (2, 2), f"Debió escoger (2, 2) por la Q sembrada, escogió {move}"


# --------------------------------------------------------------------- #
# 6. Smoke: jugar contra random sin romper
# --------------------------------------------------------------------- #
def test_base_q_agent_plays_against_random_without_crashing():
    agent = BaseQAgent()
    agent.train(episodes=100, seed=0)

    np.random.seed(123)
    n_games = 20
    for _ in range(n_games):
        game = Game()
        while not game.game_over:
            cp = game.current_player
            if cp == 1:
                r, c = agent.select_action_eval(game)
            else:
                pos = game.available_positions()
                idx = np.random.randint(len(pos))
                r, c = int(pos[idx][0]), int(pos[idx][1])
            ok = game._execute_move(r, c, cp)
            assert ok, f"Movimiento ilegal del agente: ({r}, {c})"


# --------------------------------------------------------------------- #
# 7. Patología empírica: existen valores negativos en Q
# --------------------------------------------------------------------- #
def test_q_table_contains_negative_values_after_training():
    """Sanity check de la patología de asignación de crédito.

    Cuando O gana, la recompensa es `-1` (perspectiva de X). Esa recompensa se
    asigna al ÚLTIMO movimiento (que fue de O — una jugada GANADORA para O).
    Por tanto Q termina con entradas negativas asociadas a estados-acción que
    fueron óptimos para O. La caracterización detallada vive en `diagnostics.py`.
    """
    agent = BaseQAgent()
    agent.train(episodes=2_000, seed=0)
    negatives = [v for v in agent.Q.values() if v < 0]
    positives = [v for v in agent.Q.values() if v > 0]
    assert len(negatives) > 0, "Esperaba valores negativos en Q (asignación errónea)"
    assert len(positives) > 0, "Esperaba también valores positivos (X gana algunas)"
