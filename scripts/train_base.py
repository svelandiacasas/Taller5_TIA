"""Entrena `BaseQAgent` con seed=0 y guarda el modelo a disco.

Modelo: `results/models/base_seed0.pkl` (pickle del dict de Q-table + meta).
Tiempo: ~1-2 s.
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

from base_q_agent import BaseQAgent  # noqa: E402


def main() -> None:
    SEED = 0
    EPISODES = 10_000

    print(f"=== train_base.py — seed={SEED}, episodes={EPISODES} ===")
    t0 = time.time()
    agent = BaseQAgent()
    agent.train(episodes=EPISODES, seed=SEED)
    elapsed = time.time() - t0
    print(f"  entrenamiento: {elapsed:.2f}s, |Q|={len(agent.Q)}")

    model_path = ROOT / "results" / "models" / "base_seed0.pkl"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump({
            "Q": agent.Q,
            "alpha": agent.alpha,
            "gamma": agent.gamma,
            "epsilon": agent.epsilon,
        }, f)
    print(f"  guardado: {model_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
