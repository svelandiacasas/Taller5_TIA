"""Pre-cómputo de los agentes neuronales (DQN + SARSA + Maestro).

Pipeline simplificado: 1 agente por familia, curriculum vs Maestro →
vs Algorithm → vs Random. Modelos guardados en `results/models/`.

Uso típico (correr una sola vez):

    python scripts/train_neural_agents.py

Tiempo estimado en CPU: ~15 min. Usa los módulos `master_RL.py`,
`triqui_train.py` y `triqui_algorithm.py` definidos en `src/`.
"""
import os
import sys
import time
from pathlib import Path

# `master_RL.MotorRL.train` imprime caracteres Unicode (▶, ✔) que cp1252 no
# soporta. Reconfiguramos stdout/stderr a UTF-8 ANTES de cualquier import
# que dispare prints.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
SRC = str(ROOT / "src")
sys.path.insert(0, SRC)
existing = os.environ.get("PYTHONPATH", "")
if SRC not in existing.split(os.pathsep):
    os.environ["PYTHONPATH"] = f"{SRC}{os.pathsep}{existing}" if existing else SRC

import torch  # noqa: E402
import torch.optim as optim  # noqa: E402

from seeds import set_seed  # noqa: E402
from master_RL import EpsilonGreedy, MasterDQN, MasterSARSA, MotorRL  # noqa: E402
from triqui_algorithm import Algorithm  # noqa: E402
from triqui_train import Train  # noqa: E402


# ---------------------------------------------------------------------- #
# Wrappers para usar Algorithm como `opponent` en `Train`
# ---------------------------------------------------------------------- #
def make_algorithm_opponent():
    """Callable opponent que mantiene una instancia de `Algorithm` por partida.

    Algorithm tiene estado interno (`self.strat`) que se debe preservar a lo
    largo de los turnos del MISMO juego pero reiniciar entre juegos.
    Detectamos cambio de partida por `id(game)`."""
    state = {"algo": None, "game_id": None}

    def opponent(game):
        if id(game) != state["game_id"]:
            state["algo"] = Algorithm(game)
            state["game_id"] = id(game)
        if game.current_player == 1:
            return state["algo"].play1()
        return state["algo"].play2()

    return opponent


# ---------------------------------------------------------------------- #
# Helpers de entrenamiento
# ---------------------------------------------------------------------- #
def build_agent(agent_class, device: str = "cpu"):
    """Instancia un MasterDQN o MasterSARSA con la arquitectura para triqui."""
    return agent_class(
        input_size=9, action_dim=9, hidden_layers=[128, 128]
    ).to(device)


def train_phase(
    agent,
    opponent,
    episodes: int,
    epsilon_start: float,
    epsilon_end: float,
    epsilon_decay: float,
    *,
    gamma: float = 0.99,
    lr: float = 1e-3,
    batch_size: int = 64,
    device: str = "cpu",
    log_freq: int = 5_000,
):
    """Entrena `agent` por `episodes` episodios contra `opponent` con la
    estrategia de exploración indicada."""
    env = Train(device=device, rewards_config=[3.0, 1.0, -3.0, -10.0])
    env.opponent = opponent  # asignación post-instanciación (patrón del entorno)

    optimizer = optim.Adam(agent.parameters(), lr=lr)
    exploration = EpsilonGreedy(
        start=epsilon_start, end=epsilon_end, decay=epsilon_decay
    )
    motor = MotorRL(
        agent=agent, env=env, optimizer=optimizer, device=device,
        exploration_strategy=exploration, gamma=gamma, batch_size=batch_size,
    )
    motor.train(episodes=episodes, log_freq=log_freq)


# ---------------------------------------------------------------------- #
# Maestro: 1 DQN entrenado vs Random
# ---------------------------------------------------------------------- #
def train_maestro(seed: int, episodes: int = 20_000, device: str = "cpu"):
    print(f"\n=== Maestro (DQN vs Random, {episodes} eps, seed={seed}) ===", flush=True)
    set_seed(seed)
    agent = build_agent(MasterDQN, device=device)
    train_phase(
        agent, opponent=None, episodes=episodes,
        epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.99995,
        device=device,
    )
    return agent


# ---------------------------------------------------------------------- #
# Curriculum: vs Maestro -> vs Algorithm -> vs Random
# ---------------------------------------------------------------------- #
def train_curriculum(
    agent_class,
    maestro,
    seed: int,
    episodes_per_phase: int = 20_000,
    device: str = "cpu",
):
    name = agent_class.__name__
    print(f"\n=== {name} curriculum (3 fases x {episodes_per_phase} eps, seed={seed}) ===",
          flush=True)
    set_seed(seed)
    agent = build_agent(agent_class, device=device)

    # El Maestro pasa al modo eval para no actualizar sus pesos durante el
    # entrenamiento del estudiante.
    maestro.eval()

    print(f"  Fase 1/3: vs Maestro", flush=True)
    train_phase(
        agent, opponent=maestro, episodes=episodes_per_phase,
        epsilon_start=1.0, epsilon_end=0.1, epsilon_decay=0.99995,
        device=device,
    )

    print(f"  Fase 2/3: vs Algorithm", flush=True)
    train_phase(
        agent, opponent=make_algorithm_opponent(), episodes=episodes_per_phase,
        epsilon_start=0.3, epsilon_end=0.05, epsilon_decay=0.99995,
        device=device,
    )

    print(f"  Fase 3/3: vs Random", flush=True)
    train_phase(
        agent, opponent=None, episodes=episodes_per_phase,
        epsilon_start=0.1, epsilon_end=0.05, epsilon_decay=0.99995,
        device=device,
    )

    return agent


# ---------------------------------------------------------------------- #
# Entry point
# ---------------------------------------------------------------------- #
def main() -> None:
    device = "cpu"  # CLAUDE.md: device explícito cpu para triqui
    models_dir = ROOT / "results" / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== train_neural_agents.py — device={device}, salida={models_dir} ===")
    t0 = time.time()

    maestro = train_maestro(seed=0, episodes=20_000, device=device)
    torch.save(maestro.state_dict(), models_dir / "maestro.pt")
    print(f"  guardado: maestro.pt", flush=True)

    dqn = train_curriculum(MasterDQN, maestro, seed=1,
                            episodes_per_phase=20_000, device=device)
    torch.save(dqn.state_dict(), models_dir / "dqn_curriculum.pt")
    print(f"  guardado: dqn_curriculum.pt", flush=True)

    sarsa = train_curriculum(MasterSARSA, maestro, seed=2,
                              episodes_per_phase=20_000, device=device)
    torch.save(sarsa.state_dict(), models_dir / "sarsa_curriculum.pt")
    print(f"  guardado: sarsa_curriculum.pt", flush=True)

    elapsed = time.time() - t0
    print(f"\n=== Tiempo total: {elapsed:.1f}s ({elapsed / 60:.1f} min) ===")


if __name__ == "__main__":
    main()
