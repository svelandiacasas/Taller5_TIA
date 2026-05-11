"""Sanity run de `ImprovedQAgent` (Fase 4).

Entrena con seed 0, 50 000 episodios y todos los toggles activos. Reporta:

- Tiempo de entrenamiento.
- |Q| con vs sin simetrías (ratio esperado ~ 7-8×).
- Win rate vs Random (Base techo ≈ 87 ± 5 %).
- Win rate vs Algorithm (Base = 0 %).

Guarda un resumen JSON en `results/logs/improved_sanity_seed0.json` y la
curva de aprendizaje en `results/figures/improved_sanity_curve.png`.
"""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from new.improved_q_agent import ImprovedQAgent  # noqa: E402
from new.diagnostics import evaluate_vs_algorithm, evaluate_vs_random  # noqa: E402


def main() -> None:
    SEED = 0
    EPISODES = 50_000

    print(f"=== Sanity run ImprovedQAgent (seed={SEED}, episodes={EPISODES}) ===\n")

    # Run principal: todos los toggles activos.
    agent = ImprovedQAgent(
        alpha=0.1,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=0.9995,
        use_symmetries=True,
        reward_shaping=True,
        shaping_weight=0.1,
        optimistic_init=0.0,
        dual_perspective=True,
    )
    t0 = time.time()
    history = agent.train(
        episodes=EPISODES, seed=SEED,
        eval_every=2_000, eval_episodes=200,
    )
    elapsed = time.time() - t0
    print(f"Entrenamiento full toggles: {elapsed:.1f}s para {EPISODES} eps")
    print(f"  |Q| final = {len(agent.Q)}")
    print(f"  epsilon final = {agent.epsilon:.4f}")

    # |Q| sin simetrías para comparar reducción.
    agent_no_sym = ImprovedQAgent(
        alpha=0.1, gamma=0.99,
        epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.9995,
        use_symmetries=False, reward_shaping=True, shaping_weight=0.1,
        optimistic_init=0.0, dual_perspective=True,
    )
    t0 = time.time()
    agent_no_sym.train(episodes=EPISODES, seed=SEED)
    elapsed_no = time.time() - t0
    print(f"\nEntrenamiento sin simetrías: {elapsed_no:.1f}s")
    print(f"  |Q| sin simetrías = {len(agent_no_sym.Q)}")
    print(f"  Reducción: {len(agent_no_sym.Q) / len(agent.Q):.2f}×")

    # Eval final vs Random y vs Algorithm.
    res_random = evaluate_vs_random(agent, n_episodes=2_000, agent_role=1, seed=42)
    res_algo = evaluate_vs_algorithm(agent, n_episodes=200, agent_role=1, seed=42)
    print(f"\nEvaluación con tiebreak random:")
    print(f"  vs Random (n=2000):    W={res_random['wins']:4d}  D={res_random['draws']:4d}  "
          f"L={res_random['losses']:4d}  -> WR={res_random['win_rate']:.3f}")
    print(f"  vs Algorithm (n=200):  W={res_algo['wins']:4d}  D={res_algo['draws']:4d}  "
          f"L={res_algo['losses']:4d}  -> WR={res_algo['win_rate']:.3f}  DR={res_algo['draw_rate']:.3f}")

    # Curva de aprendizaje.
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(history["episode"], history["win_rate_vs_random"],
            color="darkorange", linewidth=2, label="ImprovedQAgent vs Random (eval cada 2k eps)")
    ax.axhline(y=0.87, color="gray", linestyle="--",
               label="BaseQAgent ceiling (≈ 0.87 a 10k eps)")
    ax.set_xlabel("Episodios de entrenamiento")
    ax.set_ylabel("Win rate vs Random")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"Curva de aprendizaje ImprovedQAgent (seed={SEED}, full toggles)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig_path = ROOT / "results" / "figures" / "improved_sanity_curve.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nFigura: {fig_path.relative_to(ROOT)}")

    # Guardar el agente entrenado (pickle del Q-dict + hiperparametros).
    import pickle
    model_path = ROOT / "results" / "models" / "improved_seed0.pkl"
    with open(model_path, "wb") as f:
        pickle.dump({
            "Q": agent.Q,
            "alpha": agent.alpha,
            "gamma": agent.gamma,
            "epsilon": agent.epsilon,
            "epsilon_start": agent.epsilon_start,
            "epsilon_end": agent.epsilon_end,
            "epsilon_decay": agent.epsilon_decay,
            "use_symmetries": agent.use_symmetries,
            "reward_shaping": agent.reward_shaping,
            "shaping_weight": agent.shaping_weight,
            "optimistic_init": agent.optimistic_init,
            "dual_perspective": agent.dual_perspective,
        }, f)
    print(f"Modelo: {model_path.relative_to(ROOT)}")

    # Resumen JSON.
    summary = {
        "seed": SEED,
        "episodes": EPISODES,
        "training_time_s_with_symmetries": round(elapsed, 2),
        "training_time_s_without_symmetries": round(elapsed_no, 2),
        "q_size_with_symmetries": len(agent.Q),
        "q_size_without_symmetries": len(agent_no_sym.Q),
        "q_reduction_factor": round(len(agent_no_sym.Q) / len(agent.Q), 2),
        "epsilon_final": round(agent.epsilon, 4),
        "win_rate_vs_random": round(res_random["win_rate"], 4),
        "draw_rate_vs_random": round(res_random["draw_rate"], 4),
        "loss_rate_vs_random": round(res_random["loss_rate"], 4),
        "win_rate_vs_algorithm": round(res_algo["win_rate"], 4),
        "draw_rate_vs_algorithm": round(res_algo["draw_rate"], 4),
        "loss_rate_vs_algorithm": round(res_algo["loss_rate"], 4),
        "history_episodes": history["episode"],
        "history_win_rate_vs_random": [round(x, 4) for x in history["win_rate_vs_random"]],
    }
    out_path = ROOT / "results" / "logs" / "improved_sanity_seed0.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"Resumen: {out_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
