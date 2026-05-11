"""Diagnóstico empírico de las tres patologías del `BaseQAgent`.

Genera tres figuras con sus correspondientes CSVs (datos crudos para re-plot
sin re-entrenar):

1. `diagnostics_self_play_degeneracy.{png,csv}` — el agente como O es peor
   que un baseline aleatorio en los mismos estados visitados.

2. `diagnostics_credit_assignment_bug.{png,csv}` — fracción de jugadas
   ganadoras (en sentido semántico) cuyo `Q` quedó con signo invertido.
   Este es el *killer chart* del notebook.

3. `diagnostics_no_convergence.{png,csv}` — multi-semilla del win rate vs
   el `Algorithm` del compañero a lo largo del entrenamiento; alta varianza
   y ausencia de tendencia clara.

Todas las funciones admiten `output_dir` para escribir en un directorio
arbitrario (el smoke test usa `tmp_path`); por defecto escriben en `results/`.

Salida:
- `output_dir/figures/diagnostics_<nombre>.png` a 150 dpi.
- `output_dir/logs/diagnostics_<nombre>.csv` con los datos numéricos.
"""
from __future__ import annotations

import contextlib
import random
import time
from pathlib import Path
from typing import Optional

import matplotlib

# Backend no interactivo: estos diagnósticos solo guardan figuras a disco,
# no las muestran. Evita fallos de Tk en entornos sin display.
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from new.base_q_agent import (
    BaseQAgent,
    EMPTY,
    O,
    X,
    _available_actions,
    _check_winner,
    _initial_state,
    _state_to_tuple,
)
from new.evaluation import (
    evaluate_vs_algorithm,  # noqa: F401  (re-export para back-compat)
    evaluate_vs_random,  # noqa: F401  (re-export para back-compat)
)
from new.minimax import MinimaxAgent
from new.rng_utils import isolated_rng as _isolated_rng
from new.seeds import set_seed
from triqui import Game
from triqui_algorithm import Algorithm


# --------------------------------------------------------------------- #
# Utilidades
# --------------------------------------------------------------------- #
def _agent_state_to_game_state(state: np.ndarray) -> np.ndarray:
    """Convierte tablero del agente (`±1`) a convención `Game` (`1`, `2`)."""
    return np.where(state == O, 2, state).astype(int)


def _agent_player_to_game_player(player: int) -> int:
    """`+1` (X) → `1`; `-1` (O) → `2`."""
    return 1 if player == X else 2


def _ensure_dirs(output_dir: Path) -> tuple[Path, Path]:
    figures = Path(output_dir) / "figures"
    logs = Path(output_dir) / "logs"
    figures.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    return figures, logs


# --------------------------------------------------------------------- #
# Self-play instrumentado: réplica fiel del train del profesor + métricas
# --------------------------------------------------------------------- #
def instrumented_self_play(
    agent: BaseQAgent,
    minimax: MinimaxAgent,
    episodes: int,
    seed: int,
) -> pd.DataFrame:
    """Mirror exacto del bucle `BaseQAgent.train`, instrumentado para registrar
    por cada movimiento si era óptimo según minimax.

    Las actualizaciones de `agent.Q` son IDÉNTICAS a las que haría `agent.train`:
    no se modifica el algoritmo del profesor, solo se observa.

    Parameters
    ----------
    agent : BaseQAgent
        Agente a entrenar (mutado in-place).
    minimax : MinimaxAgent
        Oráculo para evaluar optimalidad por movimiento.
    episodes : int
    seed : int

    Returns
    -------
    pd.DataFrame
        Una fila por movimiento con columnas: `episode`, `move_idx`, `player`
        (`'X'` o `'O'`), `is_optimal`, `n_legal`, `n_optimal`.
    """
    set_seed(seed)
    records: list[dict] = []
    for ep in range(episodes):
        state = _initial_state()
        current_player = X
        move_idx = 0
        while True:
            s = _state_to_tuple(state)
            action = agent.choose_action(s, training=True)

            # Instrumentación: optimalidad de la jugada en el estado actual.
            game_state = _agent_state_to_game_state(state)
            game_player = _agent_player_to_game_player(current_player)
            optimal = minimax.optimal_actions_from_state(game_state, game_player)
            n_legal = len(_available_actions(state))
            records.append({
                "episode": ep,
                "move_idx": move_idx,
                "player": "X" if current_player == X else "O",
                "is_optimal": action in optimal,
                "n_legal": n_legal,
                "n_optimal": len(optimal),
            })

            # Aplicación + Q-update (idéntico al profesor).
            state[action] = current_player
            result = _check_winner(state)
            s_next = _state_to_tuple(state)
            if result is not None:
                reward = 1 if result == X else (-1 if result == O else 0)
                agent.update_Q(s, action, reward, s_next)
                break
            else:
                agent.update_Q(s, action, 0, s_next)
            current_player *= -1
            move_idx += 1
    return pd.DataFrame(records)


# --------------------------------------------------------------------- #
# Patología 1: degeneración del self-play
# --------------------------------------------------------------------- #
def show_self_play_degeneracy(
    seed: int = 0,
    episodes: int = 10_000,
    bin_size: int = 500,
    output_dir: Path = Path("results"),
) -> Path:
    """Cuantifica la tasa de jugadas óptimas (según minimax) que el agente hace
    como O durante el self-play, comparada con el baseline esperado de un
    agente uniforme aleatorio en los mismos estados.

    Si el agente es ≤ random en su rol de O, no es solo "ineficiente", está
    siendo entrenado en sentido contrario por la patología de crédito.
    """
    figures, logs = _ensure_dirs(output_dir)
    minimax = MinimaxAgent(player_id=2)  # zero-sum: optimal set no depende de player_id
    agent = BaseQAgent()

    df_moves = instrumented_self_play(agent, minimax, episodes=episodes, seed=seed)

    # Baseline esperado de random EN LOS MISMOS estados visitados:
    # para cada estado, P(óptimo bajo random) = n_optimal / n_legal.
    df_moves["random_baseline"] = df_moves["n_optimal"] / df_moves["n_legal"]

    # Bin por episodio para curva suave.
    df_moves["episode_bin"] = (df_moves["episode"] // bin_size) * bin_size

    o = df_moves[df_moves["player"] == "O"]
    x = df_moves[df_moves["player"] == "X"]

    grouped_o = o.groupby("episode_bin").agg(
        agent_optimality=("is_optimal", "mean"),
        random_baseline=("random_baseline", "mean"),
        n_o_moves=("is_optimal", "count"),
    ).reset_index()
    grouped_x = x.groupby("episode_bin").agg(
        agent_optimality_x=("is_optimal", "mean"),
        random_baseline_x=("random_baseline", "mean"),
        n_x_moves=("is_optimal", "count"),
    ).reset_index()
    grouped = grouped_o.merge(grouped_x, on="episode_bin")

    csv_path = logs / "diagnostics_self_play_degeneracy.csv"
    grouped.to_csv(csv_path, index=False)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(grouped["episode_bin"], grouped["agent_optimality"],
            label="BaseQAgent como O", color="crimson", linewidth=2)
    ax.plot(grouped["episode_bin"], grouped["random_baseline"],
            label="Random como O (baseline esperado en mismos estados)",
            color="gray", linewidth=2, linestyle="--")
    ax.plot(grouped["episode_bin"], grouped["agent_optimality_x"],
            label="BaseQAgent como X (referencia)",
            color="steelblue", linewidth=1.5, alpha=0.7)
    ax.set_xlabel("Episodios de entrenamiento")
    ax.set_ylabel("Tasa de jugadas óptimas")
    ax.set_title("Degeneración del self-play: el agente como O es peor que random")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="best")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig_path = figures / "diagnostics_self_play_degeneracy.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig_path


# --------------------------------------------------------------------- #
# Patología 2: asignación de crédito invertida (KILLER CHART)
# --------------------------------------------------------------------- #
def analyze_credit_assignment(agent: BaseQAgent) -> pd.DataFrame:
    """Para cada `(state, action)` en `agent.Q`, determina:

    - quién mueve en `state` (paridad de fichas);
    - si la jugada conduce a victoria del que mueve (jugada semánticamente óptima);
    - el signo del valor `Q` aprendido.

    Returns
    -------
    pd.DataFrame
        Columnas: `mover` (`X`/`O`), `wins_immediately` (bool), `q_value` (float).
        Filas: una por entrada de `agent.Q` que corresponda a un estado válido.
    """
    rows = []
    for (state_tuple, action), q_value in agent.Q.items():
        state = np.array(state_tuple).reshape(3, 3)
        n_x = int(np.sum(state == X))
        n_o = int(np.sum(state == O))
        # Validez por paridad: X mueve si #X == #O; O mueve si #X == #O+1.
        if n_x == n_o:
            current = X
        elif n_x == n_o + 1:
            current = O
        else:
            continue
        if state[action] != EMPTY:
            continue  # Acción no era legal en este estado.
        new_state = state.copy()
        new_state[action] = current
        winner = _check_winner(new_state)
        wins_immediately = (winner == current)
        rows.append({
            "mover": "X" if current == X else "O",
            "wins_immediately": bool(wins_immediately),
            "q_value": float(q_value),
        })
    return pd.DataFrame(rows)


def show_credit_assignment_bug(
    seed: int = 0,
    episodes: int = 10_000,
    output_dir: Path = Path("results"),
) -> Path:
    """Killer chart: histograma de `Q-values` segregado por (quien mueve, jugada
    semánticamente óptima). Muestra que las jugadas ganadoras de O acumulan
    valores negativos casi sistemáticamente."""
    figures, logs = _ensure_dirs(output_dir)
    set_seed(seed)
    agent = BaseQAgent()
    agent.train(episodes=episodes, seed=seed)
    df = analyze_credit_assignment(agent)

    # Subconjuntos de interés: jugadas que GANAN inmediatamente.
    x_wins = df[(df["mover"] == "X") & df["wins_immediately"]]["q_value"].values
    o_wins = df[(df["mover"] == "O") & df["wins_immediately"]]["q_value"].values

    def _pct(values: np.ndarray, predicate) -> float:
        return float(np.mean(predicate(values)) * 100) if len(values) else 0.0

    x_pos_pct = _pct(x_wins, lambda v: v > 0)
    x_neg_pct = _pct(x_wins, lambda v: v < 0)
    x_zero_pct = _pct(x_wins, lambda v: v == 0)
    o_pos_pct = _pct(o_wins, lambda v: v > 0)
    o_neg_pct = _pct(o_wins, lambda v: v < 0)
    o_zero_pct = _pct(o_wins, lambda v: v == 0)

    summary = pd.DataFrame({
        "categoria": ["X gana (crédito correcto)", "O gana (crédito invertido)"],
        "n_total": [len(x_wins), len(o_wins)],
        "pct_q_positive": [x_pos_pct, o_pos_pct],
        "pct_q_zero": [x_zero_pct, o_zero_pct],
        "pct_q_negative": [x_neg_pct, o_neg_pct],
        "q_mean": [
            float(np.mean(x_wins)) if len(x_wins) else float("nan"),
            float(np.mean(o_wins)) if len(o_wins) else float("nan"),
        ],
    })
    csv_path = logs / "diagnostics_credit_assignment_bug.csv"
    summary.to_csv(csv_path, index=False)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # (a) Stacked bar: composición de signo por categoría
    ax = axes[0]
    categories = ["X gana\n(crédito correcto)", "O gana\n(crédito invertido)"]
    pos = np.array([x_pos_pct, o_pos_pct]) / 100
    zero = np.array([x_zero_pct, o_zero_pct]) / 100
    neg = np.array([x_neg_pct, o_neg_pct]) / 100
    xpos = np.arange(len(categories))
    ax.bar(xpos, pos, color="forestgreen", label="Q > 0 (signo correcto)")
    ax.bar(xpos, zero, bottom=pos, color="lightgray", label="Q = 0")
    ax.bar(xpos, neg, bottom=pos + zero, color="crimson", label="Q < 0 (signo invertido)")
    ax.set_xticks(xpos)
    ax.set_xticklabels(categories)
    ax.set_ylabel("Fracción de jugadas ganadoras")
    ax.set_ylim(0, 1.05)
    ax.set_title(
        f"Asignación de crédito por mover: "
        f"{o_neg_pct:.1f}% de las jugadas\n"
        f"ganadoras de O quedan con Q < 0"
    )
    ax.legend(loc="upper right")
    for i, t in enumerate([len(x_wins), len(o_wins)]):
        ax.text(i, 1.01, f"n = {t}", ha="center", fontsize=9, color="black")

    # (b) Histograma: distribución de Q-values
    ax = axes[1]
    bins = np.linspace(-1.05, 1.05, 30)
    if len(x_wins):
        ax.hist(x_wins, bins=bins, alpha=0.55, color="forestgreen",
                label=f"X gana (n={len(x_wins)})", edgecolor="black", linewidth=0.5)
    if len(o_wins):
        ax.hist(o_wins, bins=bins, alpha=0.55, color="crimson",
                label=f"O gana (n={len(o_wins)})", edgecolor="black", linewidth=0.5)
    ax.axvline(x=0, color="black", linestyle="--", alpha=0.6)
    ax.set_xlabel("Q-value")
    ax.set_ylabel("Frecuencia")
    ax.set_title("Distribución de Q-values en jugadas que ganan inmediatamente")
    ax.legend(loc="best")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig_path = figures / "diagnostics_credit_assignment_bug.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig_path


# --------------------------------------------------------------------- #
# Patología 3: no convergencia (multi-semilla)
# --------------------------------------------------------------------- #
def _random_agent_win_rate_vs_algorithm(n_episodes: int = 1000, seed: int = 42) -> float:
    """Línea de referencia: % victorias de un agente totalmente aleatorio.

    Reusa `make_algorithm_opponent` de `evaluation.py` (mantiene `Algorithm`
    state por partida) y la lógica defensiva de doble-ejecución."""
    from new.evaluation import make_algorithm_opponent
    with _isolated_rng():
        set_seed(seed)
        algo_opp = make_algorithm_opponent()
        wins = 0
        for _ in range(n_episodes):
            game = Game()
            while not game.game_over:
                cp = game.current_player
                if cp == 1:
                    pos = game.available_positions()
                    idx = np.random.randint(len(pos))
                    game._execute_move(int(pos[idx][0]), int(pos[idx][1]), cp)
                else:
                    turns_before = game.turns_played
                    r, c = algo_opp(game)
                    if game.turns_played == turns_before:
                        game._execute_move(r, c, cp)
            if game.get_winner() == 1:
                wins += 1
        return wins / n_episodes


def _random_agent_win_rate_vs_random(n_episodes: int = 1000, seed: int = 42) -> float:
    """Línea de referencia: % victorias de random-vs-random (X gana)."""
    with _isolated_rng():
        set_seed(seed)
        wins = 0
        for _ in range(n_episodes):
            game = Game()
            while not game.game_over:
                pos = game.available_positions()
                idx = np.random.randint(len(pos))
                cp = game.current_player
                game._execute_move(int(pos[idx][0]), int(pos[idx][1]), cp)
            if game.get_winner() == 1:
                wins += 1
        return wins / n_episodes


def show_no_convergence(
    seeds: list[int],
    episodes: int = 10_000,
    eval_every: int = 500,
    eval_episodes: int = 200,
    output_dir: Path = Path("results"),
    agent_role: int = 1,
) -> Path:
    """Para cada semilla, entrena `BaseQAgent` y al final de cada bloque
    de `eval_every` episodios mide win rate vs DOS oponentes:

    - **Algorithm** (heurístico fuerte del compañero): banda esperablemente
      pegada al 0 — el agente base es incapaz de vencerlo. Es un *upper bound
      negativo* sobre la calidad de la política aprendida.
    - **Random** (baseline débil): donde se ve la varianza entre semillas.
      Si el agente convergiera, las semillas se apelotonarían; no convergen.

    Mostrar ambos paneles simultáneamente cuenta la historia completa: el
    agente *aprende algo* (vence a random) pero *no converge a una política
    estable* (varianza alta entre semillas, sin tendencia clara).
    """
    figures, logs = _ensure_dirs(output_dir)
    n_chunks = episodes // eval_every
    eval_episodes_axis = [(c + 1) * eval_every for c in range(n_chunks)]

    # Líneas de referencia (random como agente).
    rwr_algo = _random_agent_win_rate_vs_algorithm(n_episodes=1000, seed=999)
    rwr_rand = _random_agent_win_rate_vs_random(n_episodes=1000, seed=999)

    mat_algo = np.zeros((len(seeds), n_chunks))
    mat_rand = np.zeros((len(seeds), n_chunks))
    long_records = []
    for si, s in enumerate(seeds):
        set_seed(s)
        agent = BaseQAgent()
        for c in range(n_chunks):
            agent.train(episodes=eval_every, seed=None)  # sigue con su RNG
            res_a = evaluate_vs_algorithm(
                agent, n_episodes=eval_episodes, agent_role=agent_role,
                seed=10_000 * s + c,
            )
            res_r = evaluate_vs_random(
                agent, n_episodes=eval_episodes, agent_role=agent_role,
                seed=20_000 * s + c,
            )
            mat_algo[si, c] = res_a["win_rate"]
            mat_rand[si, c] = res_r["win_rate"]
            long_records.append({
                "seed": s,
                "episode": (c + 1) * eval_every,
                "win_rate_vs_algorithm": res_a["win_rate"],
                "draw_rate_vs_algorithm": res_a["draw_rate"],
                "loss_rate_vs_algorithm": res_a["loss_rate"],
                "win_rate_vs_random": res_r["win_rate"],
                "draw_rate_vs_random": res_r["draw_rate"],
                "loss_rate_vs_random": res_r["loss_rate"],
            })

    df_long = pd.DataFrame(long_records)
    csv_path = logs / "diagnostics_no_convergence.csv"
    df_long.to_csv(csv_path, index=False)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    role_str = "X" if agent_role == 1 else "O"

    def _plot_panel(ax, matrix, ref_wr, ref_label, title, color):
        median = np.median(matrix, axis=0)
        p25 = np.percentile(matrix, 25, axis=0)
        p75 = np.percentile(matrix, 75, axis=0)
        ax.fill_between(eval_episodes_axis, p25, p75, alpha=0.25, color=color,
                        label=f"IQR sobre {len(seeds)} semillas")
        ax.plot(eval_episodes_axis, median, color=color, linewidth=2.5, label="Mediana")
        for si in range(len(seeds)):
            ax.plot(eval_episodes_axis, matrix[si], alpha=0.25, color=color, linewidth=0.7)
        ax.axhline(y=ref_wr, color="gray", linestyle=":", linewidth=1.5, label=ref_label)
        ax.set_xlabel("Episodios de entrenamiento")
        ax.set_ylabel("Win rate (eval)")
        ax.set_title(title)
        ax.set_ylim(-0.02, 1.02)
        ax.legend(loc="best")
        ax.grid(alpha=0.3)

    _plot_panel(
        axes[0], mat_algo, rwr_algo,
        f"Random vs Algorithm: {rwr_algo:.1%}",
        f"vs Algorithm — el agente base nunca vence (jugando como {role_str})",
        "crimson",
    )
    _plot_panel(
        axes[1], mat_rand, rwr_rand,
        f"Random vs Random: {rwr_rand:.1%}",
        f"vs Random — alta varianza entre semillas, sin tendencia clara",
        "darkorange",
    )

    fig.suptitle("No convergencia: el agente base aprende algo (vs Random) "
                 "pero no se estabiliza ni vence al heurístico", fontsize=12)
    fig.tight_layout()
    fig_path = figures / "diagnostics_no_convergence.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig_path


# --------------------------------------------------------------------- #
# Orquestador
# --------------------------------------------------------------------- #
def run_all_diagnostics(
    seeds: list[int] = list(range(10)),
    episodes: int = 10_000,
    eval_every: int = 500,
    eval_episodes: int = 200,
    output_dir: Path = Path("results"),
    bin_size: int = 500,
) -> dict[str, Path]:
    """Corre los tres diagnósticos. Devuelve dict de paths a las figuras.

    Notas:
    - `show_self_play_degeneracy` y `show_credit_assignment_bug` usan `seeds[0]`
      (una sola semilla — el efecto cualitativo es robusto a la semilla).
    - `show_no_convergence` usa todas las semillas (las bandas multi-semilla
      son justamente el punto del diagnóstico).
    """
    print(f"[diagnostics] seeds={seeds}, episodes={episodes}, "
          f"eval_every={eval_every}, eval_episodes={eval_episodes}")
    paths: dict[str, Path] = {}

    t0 = time.time()
    paths["self_play_degeneracy"] = show_self_play_degeneracy(
        seed=seeds[0], episodes=episodes, bin_size=bin_size, output_dir=output_dir,
    )
    print(f"[diagnostics] self_play_degeneracy: {time.time() - t0:.1f}s")

    t0 = time.time()
    paths["credit_assignment_bug"] = show_credit_assignment_bug(
        seed=seeds[0], episodes=episodes, output_dir=output_dir,
    )
    print(f"[diagnostics] credit_assignment_bug: {time.time() - t0:.1f}s")

    t0 = time.time()
    paths["no_convergence"] = show_no_convergence(
        seeds=seeds, episodes=episodes, eval_every=eval_every,
        eval_episodes=eval_episodes, output_dir=output_dir,
    )
    print(f"[diagnostics] no_convergence: {time.time() - t0:.1f}s")

    return paths
