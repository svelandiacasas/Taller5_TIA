"""Análisis de robustez (sección 10 del notebook).

Carga los 4 agentes pre-entrenados (BaseQAgent, ImprovedQAgent, MasterDQN
curriculum, MasterSARSA curriculum) y los evalúa contra los 3 oponentes
principales (Random, Algorithm, Minimax) bajo 3 valores de K (jugadas
iniciales aleatorias) × 2 roles del agente (X / O).

Salida:
- `results/robustness_raw.csv`: long format, una fila por
  `(agent, opponent, K, agent_role, who_starts, seed, W, D, L, n_games)`.
- `results/figures/robustness_<opponent>.png`: 3 heatmaps (uno por oponente),
  agentes en filas × condiciones en columnas, color = win rate (o non-loss
  rate cuando corresponde).
"""
import os
import pickle
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = str(ROOT / "src")
sys.path.insert(0, SRC)
existing = os.environ.get("PYTHONPATH", "")
if SRC not in existing.split(os.pathsep):
    os.environ["PYTHONPATH"] = f"{SRC}{os.pathsep}{existing}" if existing else SRC

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
import torch  # noqa: E402

from master_RL import MasterDQN, MasterSARSA  # noqa: E402
from base_q_agent import BaseQAgent  # noqa: E402
from evaluation import (  # noqa: E402
    evaluate_robustness,
    make_algorithm_opponent,
    make_minimax_opponent,
    make_random_opponent,
)
from improved_q_agent import ImprovedQAgent  # noqa: E402


MODELS_DIR = ROOT / "results" / "models"


# ---------------------------------------------------------------------- #
# Carga de agentes
# ---------------------------------------------------------------------- #
def load_base_agent() -> BaseQAgent:
    with open(MODELS_DIR / "base_seed0.pkl", "rb") as f:
        data = pickle.load(f)
    agent = BaseQAgent(alpha=data["alpha"], gamma=data["gamma"],
                        epsilon=data["epsilon"])
    agent.Q = data["Q"]
    return agent


def load_improved_agent() -> ImprovedQAgent:
    with open(MODELS_DIR / "improved_seed0.pkl", "rb") as f:
        data = pickle.load(f)
    agent = ImprovedQAgent(
        alpha=data["alpha"], gamma=data["gamma"],
        epsilon_start=data["epsilon_start"],
        epsilon_end=data["epsilon_end"],
        epsilon_decay=data["epsilon_decay"],
        use_symmetries=data["use_symmetries"],
        reward_shaping=data["reward_shaping"],
        shaping_weight=data["shaping_weight"],
        optimistic_init=data["optimistic_init"],
        dual_perspective=data["dual_perspective"],
    )
    agent.Q = data["Q"]
    agent.epsilon = data["epsilon"]
    return agent


def load_dqn_agent() -> MasterDQN:
    agent = MasterDQN(input_size=9, action_dim=9, hidden_layers=[128, 128])
    agent.load_state_dict(torch.load(MODELS_DIR / "dqn_curriculum.pt",
                                       map_location="cpu"))
    agent.eval()
    return agent


def load_sarsa_agent() -> MasterSARSA:
    agent = MasterSARSA(input_size=9, action_dim=9, hidden_layers=[128, 128])
    agent.load_state_dict(torch.load(MODELS_DIR / "sarsa_curriculum.pt",
                                       map_location="cpu"))
    agent.eval()
    return agent


# ---------------------------------------------------------------------- #
# Heatmap por oponente
# ---------------------------------------------------------------------- #
def plot_robustness_heatmap(df: pd.DataFrame, opponent_name: str,
                              output_path: Path) -> None:
    """Heatmap: agentes (filas) × (K, who_starts) (columnas), color = WR mediana."""
    sub = df[df["opponent"] == opponent_name].copy()
    sub["WR"] = sub["W"] / sub["n_games"]
    sub["non_loss_rate"] = (sub["W"] + sub["D"]) / sub["n_games"]

    metric_col = "non_loss_rate" if opponent_name in {"Algorithm", "Minimax"} else "WR"
    metric_label = "Non-loss rate" if metric_col == "non_loss_rate" else "Win rate"

    sub["condicion"] = sub.apply(
        lambda r: f"K={r['K']}\n{'X' if r['agent_role'] == 1 else 'O'}\n({r['who_starts']})",
        axis=1,
    )
    pivot = sub.groupby(["agent", "condicion"])[metric_col].median().unstack("condicion")
    # Orden estable de columnas: por (K, agent_role)
    cond_order = sorted(
        sub["condicion"].unique(),
        key=lambda c: (int(c.split("=")[1].split("\n")[0]), c.split("\n")[1]),
    )
    pivot = pivot[cond_order]

    fig, ax = plt.subplots(figsize=(11, 4.5))
    sns.heatmap(
        pivot, annot=True, fmt=".2f", cmap="RdYlGn", vmin=0, vmax=1,
        cbar_kws={"label": metric_label}, ax=ax,
        linewidths=0.5, linecolor="white",
    )
    ax.set_title(
        f"Robustez vs {opponent_name} ({metric_label}, mediana sobre semillas)\n"
        f"Filas: agentes  ·  Columnas: K iniciales aleatorias / rol del agente / "
        f"quién mueve primero post-setup"
    )
    ax.set_xlabel("")
    ax.set_ylabel("")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------- #
# Entry point
# ---------------------------------------------------------------------- #
def main() -> None:
    seeds = list(range(5))   # 5 semillas, suficiente para varianza por celda
    K_values = (0, 2, 4)
    partidas = 200           # más rápido que 500; suficiente para WR estable

    print(f"=== run_robustness.py — {len(seeds)} semillas × K∈{K_values} × "
          f"{partidas} partidas/celda ===")
    t0 = time.time()

    # Cargar agentes.
    print("\nCargando agentes desde results/models/...")
    agents = {
        "Base": load_base_agent(),
        "Improved": load_improved_agent(),
        "DQN": load_dqn_agent(),
        "SARSA": load_sarsa_agent(),
    }
    for name in agents:
        print(f"  ✓ {name}")

    opponents = {
        "Random": make_random_opponent(),
        "Algorithm": make_algorithm_opponent(),
        # Minimax(player_id=2) bloquea cuando el agente es X (role=1); el
        # MinimaxAgent maneja dinámicamente desde su `select_action` con el
        # player_to_move del game, así que un MinimaxAgent(player_id=2) sirve
        # tanto cuando el agente es X como cuando es O (zero-sum).
        "Minimax": make_minimax_opponent(player_id=2),
    }

    # Evaluar todas las combinaciones.
    all_dfs = []
    for ag_name, ag in agents.items():
        for op_name, op in opponents.items():
            print(f"\n[{ag_name}] vs [{op_name}]...", flush=True)
            df = evaluate_robustness(
                agent=ag, opponent=op,
                seeds=seeds, partidas_per_combo=partidas, K_values=K_values,
                agent_name=ag_name, opponent_name=op_name,
            )
            all_dfs.append(df)
            # Reportar agregados
            agg = df.groupby(["K", "agent_role"]).apply(
                lambda d: pd.Series({
                    "WR": (d["W"] / d["n_games"]).mean(),
                    "DR": (d["D"] / d["n_games"]).mean(),
                    "LR": (d["L"] / d["n_games"]).mean(),
                })
            )
            print(agg.round(3).to_string())

    raw_df = pd.concat(all_dfs, ignore_index=True)
    raw_csv = ROOT / "results" / "robustness_raw.csv"
    raw_df.to_csv(raw_csv, index=False)
    print(f"\nRaw CSV: {raw_csv.relative_to(ROOT)}  ({len(raw_df)} filas)")

    # Heatmaps.
    figures_dir = ROOT / "results" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    for op_name in opponents.keys():
        out = figures_dir / f"robustness_{op_name.lower()}.png"
        plot_robustness_heatmap(raw_df, op_name, out)
        print(f"  ✓ {out.relative_to(ROOT)}")

    elapsed = time.time() - t0
    print(f"\n=== Tiempo total: {elapsed:.1f}s ({elapsed / 60:.1f} min) ===")


if __name__ == "__main__":
    # Reconfigurar stdout a UTF-8 (mismo problema que train_neural_agents.py).
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    main()
