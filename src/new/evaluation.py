"""Funciones de evaluación reutilizables.

Centraliza las evaluaciones que estaban dispersas entre `diagnostics.py` y
`ablations.py`. Convención uniforme:

- `agent`: cualquier objeto con `select_action_eval(game) -> (r, c)` (los
  agentes tabulares `BaseQAgent` y `ImprovedQAgent`) **o** `nn.Module` con
  `forward([B, 9]) -> [B, 9] q-values` (los agentes del compañero
  `MasterDQN` / `MasterSARSA`).
- `opponent`: callable `f(game) -> (r, c)` (Algorithm wrappers, Minimax,
  random) o `None` para "uniforme aleatorio".
- `seed`: semilla del RNG global (aislado del entrenamiento via
  `isolated_rng`).
- Salida: `dict` con `wins`, `draws`, `losses`, `win_rate`, `draw_rate`,
  `loss_rate`. `evaluate_robustness` devuelve `pd.DataFrame` long-format.

Implementa `evaluate_robustness` (Fase 7) sobre 4 agentes × 3 oponentes ×
3 valores de K × 2 roles (X / O), con setup parcial del tablero.
"""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from new.minimax import MinimaxAgent
from new.rng_utils import isolated_rng
from new.seeds import set_seed
from triqui import Game
from triqui_algorithm import Algorithm


# ---------------------------------------------------------------------- #
# Interfaz uniforme: extraer una jugada del agente
# ---------------------------------------------------------------------- #
def agent_action(agent, game) -> tuple[int, int]:
    """Devuelve `(r, c)` legal del agente para el estado actual del `game`.

    Soporta agentes tabulares (con `select_action_eval`) y `nn.Module`
    (DQN / SARSA del compañero, que reciben tensor `[1, 9]` y devuelven
    Q-values; filtramos por jugadas legales y tomamos argmax)."""
    if hasattr(agent, "select_action_eval"):
        return agent.select_action_eval(game)
    if isinstance(agent, nn.Module):
        agent.eval()
        state = game.game_matrix.flatten().astype(np.float32)
        state_t = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = agent(state_t).squeeze()
        legal = [int(p[0]) * 3 + int(p[1]) for p in game.available_positions()]
        action = legal[int(q_values[legal].argmax().item())]
        return action // 3, action % 3
    raise TypeError(
        f"agent_action: no sé cómo usar agente de tipo {type(agent).__name__}"
    )


# ---------------------------------------------------------------------- #
# Wrappers de oponente
# ---------------------------------------------------------------------- #
def make_random_opponent() -> Callable:
    def opp(game):
        pos = game.available_positions()
        idx = np.random.randint(len(pos))
        return int(pos[idx][0]), int(pos[idx][1])
    return opp


def make_algorithm_opponent() -> Callable:
    """Algorithm con estado interno (`self.strat`) preservado por partida.

    NOTA: algunas ramas de `Algorithm.play2` (caso `s_t5`) ejecutan el
    movimiento internamente vía `self.juego.play2(...)`. Aquí *solo*
    devolvemos coordenadas; el caller debe detectar el caso y evitar
    doble ejecución (ver `_safe_execute` abajo)."""
    state = {"algo": None, "game_id": None}

    def opp(game):
        if id(game) != state["game_id"]:
            state["algo"] = Algorithm(game)
            state["game_id"] = id(game)
        if game.current_player == 1:
            return state["algo"].play1()
        return state["algo"].play2()
    return opp


def make_minimax_opponent(player_id: int) -> Callable:
    minimax = MinimaxAgent(player_id=player_id)

    def opp(game):
        return minimax.select_action(game)
    return opp


def _safe_execute(game, r: int, c: int, player_id: int, *, may_have_executed: bool) -> None:
    """Ejecuta el movimiento si el `Algorithm` no lo hizo internamente.

    Algunas ramas de `Algorithm.play2` (`s_t5`) ya ejecutan, así que
    detectamos el caso por `turns_played` y solo ejecutamos si hace falta."""
    if may_have_executed:
        # Truco: si `turns_played` ya creció, no llamar de nuevo.
        before = game.turns_played
        # No hay forma de saberlo sin contar antes — lo hace el caller via
        # `play_one_game`, que pasa `may_have_executed=True` solo cuando viene
        # del Algorithm wrapper.
        # Aquí no podemos verificar correctamente; el caller usa una versión
        # más defensiva (ver `play_one_game`).
        pass
    game._execute_move(r, c, player_id)


# ---------------------------------------------------------------------- #
# Bucle de partida unificado
# ---------------------------------------------------------------------- #
def _legal_or_random(r: int, c: int, game: Game) -> tuple[int, int]:
    """Devuelve `(r, c)` si es legal en `game`; si no, escoge una jugada
    legal aleatoria. Necesario porque `Algorithm` puede proponer jugadas
    ilegales en estados con setup parcial (p.ej. `s_t2` asume "primer
    movimiento de X" pero con K=2 hay piezas en el tablero que rompen el
    supuesto). Sin este fallback, `_execute_move` retorna `False` sin
    avanzar `current_player` y el bucle de `play_one_game` se queda
    atascado infinitamente."""
    if game.game_matrix[r, c] == 0:
        return r, c
    pos = game.available_positions()
    idx = np.random.randint(len(pos))
    return int(pos[idx][0]), int(pos[idx][1])


def play_one_game(
    agent,
    opponent: Callable,
    agent_role: int = 1,
    initial_game: Optional[Game] = None,
) -> int:
    """Juega una partida y devuelve el ganador (0 empate, 1 ó 2 jugador).

    Si `initial_game` se provee (con setup parcial), se juega desde ese
    estado; si no, parte de tablero vacío.

    Robustez: si el `agent` o el `opponent` proponen una jugada ilegal,
    se sustituye por una jugada legal aleatoria (ver `_legal_or_random`).
    """
    game = initial_game if initial_game is not None else Game()
    while not game.game_over:
        cp = game.current_player
        if cp == agent_role:
            r, c = agent_action(agent, game)
            r, c = _legal_or_random(r, c, game)
            game._execute_move(r, c, cp)
        else:
            turns_before = game.turns_played
            r, c = opponent(game)
            # Algorithm wrapper puede haber ejecutado internamente (s_t5).
            if game.turns_played == turns_before:
                r, c = _legal_or_random(r, c, game)
                game._execute_move(r, c, cp)
    return int(game.get_winner())


def _wdl_from_winner(winner: int, agent_role: int) -> str:
    if winner == agent_role:
        return "W"
    if winner == 0:
        return "D"
    return "L"


# ---------------------------------------------------------------------- #
# Evaluaciones contra oponentes específicos (compatibles con la API previa)
# ---------------------------------------------------------------------- #
def _evaluate_vs(
    agent,
    opponent: Callable,
    n_episodes: int,
    agent_role: int,
    seed: int,
) -> dict:
    with isolated_rng():
        set_seed(seed)
        wins = draws = losses = 0
        for _ in range(n_episodes):
            winner = play_one_game(agent, opponent, agent_role=agent_role)
            r = _wdl_from_winner(winner, agent_role)
            wins += (r == "W")
            draws += (r == "D")
            losses += (r == "L")
        return {
            "wins": wins, "draws": draws, "losses": losses,
            "win_rate": wins / n_episodes,
            "draw_rate": draws / n_episodes,
            "loss_rate": losses / n_episodes,
        }


def evaluate_vs_random(
    agent, n_episodes: int = 200, agent_role: int = 1, seed: int = 0,
) -> dict:
    return _evaluate_vs(agent, make_random_opponent(), n_episodes, agent_role, seed)


def evaluate_vs_algorithm(
    agent, n_episodes: int = 200, agent_role: int = 1, seed: int = 0,
) -> dict:
    return _evaluate_vs(agent, make_algorithm_opponent(), n_episodes, agent_role, seed)


def evaluate_vs_minimax(
    agent, n_episodes: int = 500, agent_role: int = 1, seed: int = 0,
) -> dict:
    minimax_role = 2 if agent_role == 1 else 1
    return _evaluate_vs(
        agent, make_minimax_opponent(player_id=minimax_role),
        n_episodes, agent_role, seed,
    )


def compute_distance_to_minimax(
    agent, n_episodes: int = 500, agent_role: int = 1, seed: int = 0,
) -> float:
    """`1 − (jugadas_en_optimal / jugadas_totales)` sobre las jugadas del
    agente durante partidas vs Random. `0` = perfecto."""
    minimax = MinimaxAgent(player_id=agent_role)
    total = 0
    optimal = 0
    with isolated_rng():
        set_seed(seed)
        for _ in range(n_episodes):
            game = Game()
            while not game.game_over:
                cp = game.current_player
                if cp == agent_role:
                    state = np.asarray(game.get_game_matrix(), dtype=int)
                    opt_set = minimax.optimal_actions_from_state(state, cp)
                    r, c = agent_action(agent, game)
                    if (r, c) in opt_set:
                        optimal += 1
                    total += 1
                    game._execute_move(r, c, cp)
                else:
                    pos = game.available_positions()
                    idx = np.random.randint(len(pos))
                    game._execute_move(int(pos[idx][0]), int(pos[idx][1]), cp)
        return 0.0 if total == 0 else 1.0 - (optimal / total)


# ---------------------------------------------------------------------- #
# Setup parcial del tablero (sección 10 — robustez)
# ---------------------------------------------------------------------- #
def setup_partial_board(K: int, max_retries: int = 20) -> Optional[Game]:
    """Crea un `Game` con `K` jugadas aleatorias alternadas (X, O, X, O, ...).

    Si tras `K` jugadas el juego se vuelve terminal (línea de 3 fortuita),
    descarta y reintenta. Devuelve `None` tras `max_retries` fallidos
    consecutivos (extremadamente improbable para K ≤ 4)."""
    for _ in range(max_retries):
        game = Game()
        ok = True
        for _ in range(K):
            pos = game.available_positions()
            if len(pos) == 0:
                ok = False
                break
            idx = np.random.randint(len(pos))
            r, c = int(pos[idx][0]), int(pos[idx][1])
            cp = game.current_player
            if not game._execute_move(r, c, cp):
                ok = False
                break
            if game.game_over:
                ok = False
                break
        if ok:
            return game
    return None


def evaluate_robustness(
    agent,
    opponent: Callable,
    seeds: list[int],
    *,
    partidas_per_combo: int = 500,
    K_values: tuple[int, ...] = (0, 2, 4),
    agent_roles: tuple[int, ...] = (1, 2),
    agent_name: str = "agent",
    opponent_name: str = "opponent",
) -> pd.DataFrame:
    """Evalúa el agente bajo distintas condiciones iniciales.

    Para cada combinación `(K, agent_role, seed)` corre `partidas_per_combo`
    partidas y reporta totales W/D/L. La columna `who_starts` indica
    quién hace la PRIMERA jugada después del setup parcial:
    - `who_starts == 'agent'` si `current_player == agent_role` post-setup.
    - `who_starts == 'opponent'` en caso contrario.

    Si `setup_partial_board` falla en una iteración (estado terminal en el
    setup), esa partida se descarta y se reintenta automáticamente.
    """
    rows = []
    for K in K_values:
        # Tras K jugadas alternadas (X primero), turno post-setup = (K % 2) + 1.
        next_player_after_setup = (K % 2) + 1
        for agent_role in agent_roles:
            who_starts = "agent" if next_player_after_setup == agent_role else "opponent"
            for s in seeds:
                with isolated_rng():
                    set_seed(s)
                    wins = draws = losses = 0
                    games_played = 0
                    while games_played < partidas_per_combo:
                        initial = setup_partial_board(K)
                        if initial is None:
                            continue
                        winner = play_one_game(
                            agent, opponent,
                            agent_role=agent_role,
                            initial_game=initial,
                        )
                        r = _wdl_from_winner(winner, agent_role)
                        wins += (r == "W")
                        draws += (r == "D")
                        losses += (r == "L")
                        games_played += 1
                rows.append({
                    "agent": agent_name,
                    "opponent": opponent_name,
                    "K": K,
                    "agent_role": agent_role,
                    "who_starts": who_starts,
                    "seed": s,
                    "W": wins,
                    "D": draws,
                    "L": losses,
                    "n_games": partidas_per_combo,
                })
    return pd.DataFrame(rows)
