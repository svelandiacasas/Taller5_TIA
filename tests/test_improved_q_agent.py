"""Tests para `ImprovedQAgent`.

Cubrimos:

1. `get_Q` con `optimistic_init` para pares no vistos.
2. Canonicalización: simetrías D₄ agrupan estados equivalentes en la misma
   clave de `Q`.
3. `_phi`: open-twos correcto en estados conocidos.
4. Determinismo con misma seed.
5. Sanity de política: tras entrenamiento corto el agente toma jugadas
   ganadoras y bloqueantes triviales.
6. Sanity del teorema de Ng: con/sin shaping coinciden en estados
   terminales triviales.
7. Reducción del tamaño de Q por simetrías (≥ 4×).
8. Adapter: shapes correctos y propagación de dual_perspective.
9. `select_action_eval` contra el entorno `Game` (movimientos legales).
10. Aprende mejor que el agente base contra Random.
"""
import numpy as np
import pytest
import torch

from triqui import Game
from improved_q_agent import (
    ImprovedQAgent,
    _count_open_twos,
)
from base_q_agent import (
    BaseQAgent,
    TabularToTorchAdapter,
    _state_to_tuple,
)
from symmetry import canonical_state, ACTION_MAPS


# --------------------------------------------------------------------- #
# 1. get_Q con optimistic_init
# --------------------------------------------------------------------- #
def test_get_Q_returns_optimistic_init_for_unseen():
    agent = ImprovedQAgent(optimistic_init=0.5, use_symmetries=False,
                           dual_perspective=False)
    state_tuple = (0,) * 9
    assert agent.get_Q(state_tuple, (0, 0)) == 0.5


def test_get_Q_returns_stored_value_when_seeded():
    agent = ImprovedQAgent(optimistic_init=0.5, use_symmetries=False,
                           dual_perspective=False)
    agent._set_Q((0,) * 9, (1, 1), 0.9)
    assert agent.get_Q((0,) * 9, (1, 1)) == pytest.approx(0.9)


# --------------------------------------------------------------------- #
# 2. Canonicalización: estados equivalentes comparten clave Q
# --------------------------------------------------------------------- #
def test_get_Q_groups_d4_equivalent_states_under_symmetries():
    """Dos estados que difieren solo por una transformación D₄ comparten Q."""
    agent = ImprovedQAgent(use_symmetries=True, dual_perspective=False,
                           optimistic_init=0.0)

    # Estado A: X en esquina (0, 0)
    state_A = (1, 0, 0, 0, 0, 0, 0, 0, 0)
    # Estado B: X en esquina (0, 2) — equivalente a A por reflexión horizontal
    state_B = (0, 0, 1, 0, 0, 0, 0, 0, 0)

    # Sembramos Q en A para acción (1, 1) (centro)
    agent._set_Q(state_A, (1, 1), 0.7)

    # Lookup en B para acción (1, 1) — ¿devuelve 0.7?
    val_B = agent.get_Q(state_B, (1, 1))
    assert val_B == pytest.approx(0.7), \
        f"Estados D₄-equivalentes deben compartir Q: A→0.7 pero B→{val_B}"


def test_no_grouping_without_symmetries():
    """Sin simetrías, dos estados D₄-equivalentes son keys distintas."""
    agent = ImprovedQAgent(use_symmetries=False, dual_perspective=False,
                           optimistic_init=0.0)
    state_A = (1, 0, 0, 0, 0, 0, 0, 0, 0)
    state_B = (0, 0, 1, 0, 0, 0, 0, 0, 0)
    agent._set_Q(state_A, (1, 1), 0.7)
    assert agent.get_Q(state_B, (1, 1)) == 0.0  # no encontrado, devuelve default


# --------------------------------------------------------------------- #
# 3. _count_open_twos
# --------------------------------------------------------------------- #
def test_count_open_twos_basic_row():
    state = np.array([[1, 1, 0], [0, 0, 0], [0, 0, 0]], dtype=int)
    assert _count_open_twos(state, +1) == 1
    assert _count_open_twos(state, -1) == 0


def test_count_open_twos_two_open_twos_intersecting():
    """Estado con dos open-twos a favor de +1: fila 0 y diagonal."""
    state = np.array([[1, 1, 0], [0, 1, 0], [0, 0, 0]], dtype=int)
    # +1 en (0,0), (0,1), (1,1). Open twos:
    # Row 0: [1, 1, 0] ✓
    # Col 1: [1, 1, 0] ✓
    # Diag: [1, 1, 0] ✓
    assert _count_open_twos(state, +1) == 3


def test_count_open_twos_blocked_by_opponent():
    """Una línea con 2 mías Y una del rival NO cuenta como open-two."""
    state = np.array([[1, 1, -1], [0, 0, 0], [0, 0, 0]], dtype=int)
    assert _count_open_twos(state, +1) == 0


# --------------------------------------------------------------------- #
# 4. Determinismo
# --------------------------------------------------------------------- #
def test_train_is_deterministic_with_same_seed():
    a1 = ImprovedQAgent()
    a1.train(episodes=300, seed=42)
    a2 = ImprovedQAgent()
    a2.train(episodes=300, seed=42)
    assert len(a1.Q) == len(a2.Q)
    for k, v in a1.Q.items():
        assert k in a2.Q
        assert v == pytest.approx(a2.Q[k], abs=1e-12)


# --------------------------------------------------------------------- #
# 5. Sanity de política: jugadas triviales
# --------------------------------------------------------------------- #
def _state_in_mover_view(absolute, mover):
    """Helper: estado absoluto → vista del mover (mover = +1)."""
    arr = np.asarray(absolute, dtype=int)
    return arr if mover == 1 else -arr


def test_trained_agent_takes_winning_move():
    """Agente con dual_perspective tras 5k episodios: en estado donde X
    tiene dos en línea y casilla libre, debe elegir la que cierra."""
    agent = ImprovedQAgent(use_symmetries=True, dual_perspective=True,
                           reward_shaping=True, shaping_weight=0.1)
    agent.train(episodes=5_000, seed=0)

    # X X .
    # O O .
    # . . .
    # X (mover=+1) debe ganar en (0, 2)
    state_view = np.array([[1, 1, 0], [-1, -1, 0], [0, 0, 0]], dtype=int)
    state_tuple = _state_to_tuple(state_view)
    move = agent.choose_action(state_tuple, training=False)
    assert move == (0, 2), f"Debió ganar en (0,2), eligió {move}"


def test_trained_agent_blocks_immediate_threat():
    """Agente debe bloquear cuando el rival amenaza ganar el próximo turno."""
    agent = ImprovedQAgent(use_symmetries=True, dual_perspective=True,
                           reward_shaping=True, shaping_weight=0.1)
    agent.train(episodes=5_000, seed=0)

    # . X .
    # O O .
    # X . .
    # X (mover=+1) debe BLOQUEAR en (1, 2): O amenaza fila 1.
    state_view = np.array([[0, 1, 0], [-1, -1, 0], [1, 0, 0]], dtype=int)
    state_tuple = _state_to_tuple(state_view)
    move = agent.choose_action(state_tuple, training=False)
    assert move == (1, 2), f"Debió bloquear en (1,2), eligió {move}"


# --------------------------------------------------------------------- #
# 6. Sanity de Ng: shaping no rompe la política óptima en estados triviales
# --------------------------------------------------------------------- #
def test_shaping_does_not_break_optimal_policy_at_trivial_states():
    """Teorema de Ng (1999): potential-based shaping preserva la política
    óptima. Sanity check: con y sin shaping, el agente entrena y coincide
    en la jugada óptima en estados de victoria/bloqueo trivial."""
    common_kwargs = dict(use_symmetries=True, dual_perspective=True)

    a_no_shape = ImprovedQAgent(reward_shaping=False, **common_kwargs)
    a_no_shape.train(episodes=5_000, seed=0)

    a_shape = ImprovedQAgent(reward_shaping=True, shaping_weight=0.1,
                              **common_kwargs)
    a_shape.train(episodes=5_000, seed=0)

    # Estado de victoria trivial
    win_state = _state_to_tuple(np.array([[1, 1, 0], [-1, -1, 0], [0, 0, 0]]))
    move_no = a_no_shape.choose_action(win_state, training=False)
    move_yes = a_shape.choose_action(win_state, training=False)
    assert move_no == (0, 2), f"sin shaping: {move_no}"
    assert move_yes == (0, 2), f"con shaping: {move_yes}"

    # Estado de bloqueo trivial
    block_state = _state_to_tuple(np.array([[0, 1, 0], [-1, -1, 0], [1, 0, 0]]))
    move_no_b = a_no_shape.choose_action(block_state, training=False)
    move_yes_b = a_shape.choose_action(block_state, training=False)
    assert move_no_b == (1, 2), f"sin shaping bloqueo: {move_no_b}"
    assert move_yes_b == (1, 2), f"con shaping bloqueo: {move_yes_b}"


# --------------------------------------------------------------------- #
# 7. Reducción de |Q| por simetrías
# --------------------------------------------------------------------- #
def test_symmetries_reduce_q_size():
    """Con use_symmetries=True, |Q| es significativamente menor que sin
    simetrías. La reducción asintótica teórica es ~8× (orden del grupo D₄
    descontando estados con simetría interna). Con cobertura empírica
    limitada el ratio es menor; aquí pedimos ≥ 4× con 5 000 episodios y
    `epsilon_decay=1.0` (exploración constante para cobertura uniforme)."""
    a_sym = ImprovedQAgent(use_symmetries=True, dual_perspective=True,
                           reward_shaping=False, optimistic_init=0.0,
                           epsilon_decay=1.0)
    a_sym.train(episodes=5_000, seed=0)

    a_no = ImprovedQAgent(use_symmetries=False, dual_perspective=True,
                           reward_shaping=False, optimistic_init=0.0,
                           epsilon_decay=1.0)
    a_no.train(episodes=5_000, seed=0)

    ratio = len(a_no.Q) / len(a_sym.Q)
    assert ratio >= 4.0, (
        f"Reducción esperada ≥ 4×, obtenida {ratio:.2f}× "
        f"(|Q_sym|={len(a_sym.Q)}, |Q_no_sym|={len(a_no.Q)})"
    )


# --------------------------------------------------------------------- #
# 8. Adapter
# --------------------------------------------------------------------- #
def test_adapter_returns_correct_shape_for_improved_agent():
    agent = ImprovedQAgent()
    agent.train(episodes=200, seed=0)
    adapter = agent.to_torch_adapter()
    out = adapter(torch.zeros((4, 9), dtype=torch.float32))
    assert out.shape == (4, 9)
    assert out.dtype == torch.float32


def test_adapter_propagates_dual_perspective():
    """`to_torch_adapter` configura dual_perspective según el flag del agente."""
    a = ImprovedQAgent(dual_perspective=True)
    adapter_a = a.to_torch_adapter()
    assert adapter_a.dual_perspective is True

    b = ImprovedQAgent(dual_perspective=False)
    adapter_b = b.to_torch_adapter()
    assert adapter_b.dual_perspective is False


def test_adapter_handles_perspective_flip():
    """Estado en frame absoluto donde O es mover: el adapter debe consultar
    Q en O-vista, no en X-vista. Se siembra Q en O-vista y se verifica."""
    agent = ImprovedQAgent(use_symmetries=False, dual_perspective=True,
                           optimistic_init=0.0)
    # Estado absoluto: X en (0,0), nadie más. Mover es O.
    # O's view: O = +1, X = -1. State_o_view = -state_abs = [(1,0)→0, ..., (0,0)→0_o_view]
    # Wait: state_abs has -1 at (0,0) meaning... no wait, state_abs has +1 at (0,0) (X = +1)
    # So state_o_view = -state_abs = [-1, 0, 0, 0, 0, 0, 0, 0, 0] in O's view (X is opp = -1)
    state_abs = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0])  # X at (0,0), O to move
    state_o_view_tuple = tuple((-state_abs).tolist())
    # Sembramos en O-vista para acción (1, 1)
    agent._set_Q(state_o_view_tuple, (1, 1), 0.42)

    adapter = agent.to_torch_adapter()
    # Game convention: X=1, O=2, empty=0. Pasamos el mismo estado.
    game_state = torch.tensor([[1.0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=torch.float32)
    out = adapter(game_state)
    # acción (1, 1) → índice 4 → debe valer 0.42
    assert out[0, 4].item() == pytest.approx(0.42), \
        f"adapter con dual_perspective debió encontrar 0.42 en O-vista; obtuvo {out[0, 4].item()}"


# --------------------------------------------------------------------- #
# 9. select_action_eval contra Game
# --------------------------------------------------------------------- #
def test_select_action_eval_returns_legal_move():
    agent = ImprovedQAgent()
    agent.train(episodes=200, seed=0)
    game = Game()
    move = agent.select_action_eval(game)
    assert isinstance(move, tuple) and len(move) == 2
    assert game.game_matrix[move[0], move[1]] == 0


def test_select_action_eval_handles_o_role():
    """Cuando current_player==2 en Game, el agente debe jugar como O y devolver
    una jugada legal."""
    agent = ImprovedQAgent()
    agent.train(episodes=200, seed=0)
    game = Game()
    game._execute_move(0, 0, 1)  # X juega
    assert game.current_player == 2
    move = agent.select_action_eval(game)
    assert game.game_matrix[move[0], move[1]] == 0


# --------------------------------------------------------------------- #
# 10. Aprende: vs Random después de entrenamiento debe ser > Base
# --------------------------------------------------------------------- #
def test_improved_agent_beats_base_at_winning_random():
    """ImprovedQAgent (10k eps, seed=0) debe ganar a Random más que el
    Base agent (10k eps, mismo seed). Smoke check del beneficio agregado."""
    base = BaseQAgent()
    base.train(episodes=10_000, seed=0)

    improved = ImprovedQAgent(use_symmetries=True, reward_shaping=True,
                               dual_perspective=True)
    improved.train(episodes=10_000, seed=0)

    # Eval breve vs Random
    base_wr = base._wr_vs_random_smoke = None  # placeholder
    # Inline eval para evitar dependencia con diagnostics
    def _wr_vs_random(agent, n=300, seed=42):
        from seeds import set_seed
        set_seed(seed)
        wins = 0
        for _ in range(n):
            game = Game()
            while not game.game_over:
                cp = game.current_player
                if cp == 1:
                    r, c = agent.select_action_eval(game)
                    game._execute_move(r, c, cp)
                else:
                    pos = game.available_positions()
                    idx = np.random.randint(len(pos))
                    game._execute_move(int(pos[idx][0]), int(pos[idx][1]), cp)
            if game.get_winner() == 1:
                wins += 1
        return wins / n

    wr_base = _wr_vs_random(base)
    wr_improved = _wr_vs_random(improved)
    assert wr_improved > wr_base, (
        f"Esperaba ImprovedQAgent > BaseQAgent vs Random; "
        f"obtuvo improved={wr_improved:.3f} vs base={wr_base:.3f}"
    )
