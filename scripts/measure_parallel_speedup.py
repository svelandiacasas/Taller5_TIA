"""Mide el speedup de `run_parallel_seeds` vs ejecución secuencial.

Sujeto: `BaseQAgent` con 10 semillas × 10 000 episodios. Es el régimen ligero
que usaremos en Fase 6 para iterar; con `ImprovedQAgent` × 5 variantes × 10
semillas serán ~3× más caro.

Salida en consola + JSON en `results/logs/parallel_speedup.json`.
"""
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = str(ROOT / "src")
sys.path.insert(0, SRC)
# Propagar a procesos hijos en Windows (`spawn` no hereda sys.path).
existing = os.environ.get("PYTHONPATH", "")
if SRC not in existing.split(os.pathsep):
    os.environ["PYTHONPATH"] = f"{SRC}{os.pathsep}{existing}" if existing else SRC

from new.base_q_agent import BaseQAgent  # noqa: E402
from new.multi_seed import run_parallel_seeds  # noqa: E402
from new.seeds import set_seed  # noqa: E402


def measure_sequential(seeds: list[int], episodes: int) -> tuple[float, list[BaseQAgent]]:
    t0 = time.time()
    agents = []
    for s in seeds:
        set_seed(s)
        a = BaseQAgent()
        a.train(episodes=episodes, seed=s)
        agents.append(a)
    return time.time() - t0, agents


def measure_parallel(seeds: list[int], episodes: int, n_workers: int) -> tuple[float, list[BaseQAgent]]:
    t0 = time.time()
    results = run_parallel_seeds(
        agent_class=BaseQAgent,
        agent_kwargs={},
        train_kwargs={"episodes": episodes},
        seeds=seeds,
        n_workers=n_workers,
    )
    elapsed = time.time() - t0
    return elapsed, [r["agent"] for r in results]


def verify_q_tables_match(seq_agents: list[BaseQAgent],
                          par_agents: list[BaseQAgent]) -> bool:
    for seq, par in zip(seq_agents, par_agents):
        if set(seq.Q.keys()) != set(par.Q.keys()):
            return False
        for k, v in seq.Q.items():
            if abs(v - par.Q[k]) > 1e-12:
                return False
    return True


def main() -> None:
    SEEDS = list(range(10))
    EPISODES = 10_000

    print(f"=== Speedup measurement: BaseQAgent x {len(SEEDS)} seeds x "
          f"{EPISODES} eps ===\n")

    print("Secuencial...", flush=True)
    seq_time, seq_agents = measure_sequential(SEEDS, EPISODES)
    print(f"  Tiempo: {seq_time:.2f}s")

    n_workers_to_try = [2, 4, 8, 10]
    parallel_runs: dict[int, dict] = {}
    for nw in n_workers_to_try:
        print(f"\nParalelo (n_workers={nw})...", flush=True)
        par_time, par_agents = measure_parallel(SEEDS, EPISODES, nw)
        match = verify_q_tables_match(seq_agents, par_agents)
        speedup = seq_time / par_time if par_time > 0 else float("inf")
        parallel_runs[nw] = {
            "time_s": round(par_time, 2),
            "speedup_vs_seq": round(speedup, 2),
            "q_tables_match_sequential": match,
        }
        print(f"  Tiempo: {par_time:.2f}s  speedup: {speedup:.2f}x  "
              f"Q match: {match}")

    summary = {
        "seeds": SEEDS,
        "episodes_per_seed": EPISODES,
        "agent_class": "BaseQAgent",
        "sequential_time_s": round(seq_time, 2),
        "parallel": parallel_runs,
    }

    out_path = ROOT / "results" / "logs" / "parallel_speedup.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\nResumen JSON: {out_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
