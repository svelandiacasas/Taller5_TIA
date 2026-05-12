"""Tests de optimalidad para `MinimaxAgent`.

Garantías verificadas:
- Minimax-vs-Minimax siempre empata.
- Minimax (cualquier rol) nunca pierde contra Random en >= 1000 partidas.
- `optimal_actions` no es vacía mientras haya jugadas legales.
- `value` devuelve los valores correctos en estados terminales y en el inicio.
"""
import numpy as np
import pytest

from triqui import Game
from minimax import MinimaxAgent


# --------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------- #
def _random_action(game):
    legal = game.available_positions()
    idx = np.random.randint(len(legal))
    pos = legal[idx]
    return int(pos[0]), int(pos[1])


def _make_minimax_action(player_id):
    agent = MinimaxAgent(player_id=player_id)
    return lambda game: agent.select_action(game)


def _play_one_game(p1_action, p2_action):
    """Juega una partida y devuelve el ganador (0 empate, 1 ó 2 jugador)."""
    game = Game()
    while not game.game_over:
        cp = game.current_player
        r, c = p1_action(game) if cp == 1 else p2_action(game)
        ok = game._execute_move(r, c, cp)
        if not ok:
            raise RuntimeError(f"Movimiento ilegal ({r}, {c}) por jugador {cp}")
    return game.get_winner()


# --------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------- #
@pytest.fixture(autouse=True)
def _seed_each_test():
    """Semilla determinista por test (los tests no comparten estado RNG)."""
    np.random.seed(0)


# --------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------- #
def test_minimax_vs_minimax_always_draws():
    p1 = _make_minimax_action(player_id=1)
    p2 = _make_minimax_action(player_id=2)
    n = 100
    draws = sum(_play_one_game(p1, p2) == 0 for _ in range(n))
    assert draws == n, f"Esperaba {n} empates, obtuvo {draws}"


def test_minimax_p1_vs_random_never_loses():
    p1 = _make_minimax_action(player_id=1)
    n = 1000
    losses = sum(_play_one_game(p1, _random_action) == 2 for _ in range(n))
    assert losses == 0, f"Minimax(1) perdió {losses}/{n} partidas vs Random"


def test_minimax_p2_vs_random_never_loses():
    p2 = _make_minimax_action(player_id=2)
    n = 1000
    losses = sum(_play_one_game(_random_action, p2) == 1 for _ in range(n))
    assert losses == 0, f"Minimax(2) perdió {losses}/{n} partidas vs Random"


def test_optimal_actions_nonempty_at_start():
    agent = MinimaxAgent(player_id=1)
    game = Game()
    opt = agent.optimal_actions(game)
    assert len(opt) > 0
    # Y todas son legales
    legal = {(int(p[0]), int(p[1])) for p in game.available_positions()}
    for a in opt:
        assert a in legal


def test_value_terminal_states():
    agent = MinimaxAgent(player_id=1)
    # X gana en la fila superior
    win_x = np.array([
        [1, 1, 1],
        [2, 2, 0],
        [0, 0, 0],
    ], dtype=int)
    assert agent.value(win_x, player_to_move=2) == 1.0

    # X pierde
    win_o = np.array([
        [2, 2, 2],
        [1, 1, 0],
        [0, 0, 0],
    ], dtype=int)
    assert agent.value(win_o, player_to_move=1) == -1.0

    # Empate (tablero lleno sin ganador)
    draw = np.array([
        [1, 2, 1],
        [1, 2, 2],
        [2, 1, 1],
    ], dtype=int)
    assert agent.value(draw, player_to_move=1) == 0.0


def test_value_initial_state_is_draw_for_both_players():
    """Triqui con juego perfecto desde el inicio termina en empate."""
    state = np.zeros((3, 3), dtype=int)
    assert MinimaxAgent(player_id=1).value(state, player_to_move=1) == 0.0
    assert MinimaxAgent(player_id=2).value(state, player_to_move=1) == 0.0


def test_select_action_returns_legal_move():
    agent = MinimaxAgent(player_id=1)
    game = Game()
    r, c = agent.select_action(game)
    assert 0 <= r <= 2 and 0 <= c <= 2
    assert game.game_matrix[r, c] == 0


def test_minimax_takes_winning_move_when_available():
    """Si X tiene dos en línea propias y la celda que cierra está libre,
    minimax debe escoger exactamente esa (propiedad ofensiva, dual al test
    de bloqueo defensivo)."""
    # X X .
    # O O .
    # . . .
    # X juega y gana en (0, 2). Aunque O también amenaza fila 1, X mueve primero
    # y cerrar la fila superior es la única jugada de valor +1.
    game = Game()
    game._execute_move(0, 0, 1)  # X
    game._execute_move(1, 0, 2)  # O
    game._execute_move(0, 1, 1)  # X
    game._execute_move(1, 1, 2)  # O
    assert game.current_player == 1
    agent = MinimaxAgent(player_id=1)
    move = agent.select_action(game)
    assert move == (0, 2), f"X debió ganar en (0,2), jugó {move}"


def test_minimax_blocks_immediate_threat():
    """Si el rival tiene línea de 2 y casilla libre, minimax debe bloquear."""
    # Tablero: O ha jugado dos en la fila superior, X debe bloquear (0,2)
    game = Game()
    game._execute_move(0, 0, 1)  # X
    game._execute_move(1, 0, 2)  # O
    game._execute_move(2, 2, 1)  # X
    game._execute_move(1, 1, 2)  # O — ahora amenaza diagonal (0,0)... pero (0,0) ya es X
    # El rival O tiene fila central: posiciones (1,0) y (1,1). Necesita (1,2) para ganar.
    # X debe bloquear en (1,2).
    assert game.current_player == 1
    agent = MinimaxAgent(player_id=1)
    move = agent.select_action(game)
    assert move == (1, 2), f"X debió bloquear en (1,2), jugó {move}"


def test_cache_avoids_recomputation_on_repeated_call():
    """Llamada repetida a `value` con el mismo estado no debe ejecutar nuevo
    cómputo de minimax: el contador `_node_visits` queda igual y el tamaño
    del cache no crece. Comprobado por contadores deterministas, no por tiempo
    de pared (que sería flaky en CI)."""
    MinimaxAgent.clear_cache()
    MinimaxAgent.reset_counters()
    agent = MinimaxAgent(player_id=1)
    state = np.zeros((3, 3), dtype=int)

    v1 = agent.value(state, player_to_move=1)
    visits_after_first = MinimaxAgent._node_visits
    cache_after_first = len(MinimaxAgent._cache)

    v2 = agent.value(state, player_to_move=1)
    visits_after_second = MinimaxAgent._node_visits
    cache_after_second = len(MinimaxAgent._cache)

    assert v1 == v2
    assert visits_after_first > 0, "primera llamada debió ejecutar cómputo real"
    assert visits_after_second == visits_after_first, \
        "segunda llamada debió servirse íntegramente del cache (sin nuevos node visits)"
    assert cache_after_first == cache_after_second
    assert cache_after_first > 0
