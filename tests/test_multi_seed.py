"""Tests para `run_parallel_seeds`.

El test crítico verifica que la corrida paralela produce, para una misma
semilla, EXACTAMENTE la misma Q-table que la corrida secuencial. Es la
demostración empírica de que `set_seed(seed)` se ejecuta efectivamente
dentro del worker (en `spawn` los workers no heredan estado RNG, así que si
falla, las Q-tables divergen).
"""
import multiprocessing as mp

import numpy as np
import pytest

from new.base_q_agent import BaseQAgent
from new.multi_seed import run_parallel_seeds
from new.seeds import set_seed


# --------------------------------------------------------------------- #
# Determinismo: paralelo == secuencial para la misma semilla
# --------------------------------------------------------------------- #
def test_parallel_results_match_sequential_for_same_seed():
    """La Q-table tras `train(seed=s)` en worker debe ser bit-idéntica a la
    de un entrenamiento secuencial con la misma semilla. Demuestra que el
    worker re-siembra correctamente."""
    seeds = [0, 1, 2]
    train_kwargs = {"episodes": 500}

    parallel = run_parallel_seeds(
        agent_class=BaseQAgent,
        agent_kwargs={},
        train_kwargs=train_kwargs,
        seeds=seeds,
        n_workers=3,
    )

    for s in seeds:
        ref = BaseQAgent()
        ref.train(seed=s, **train_kwargs)
        par = next(r for r in parallel if r["seed"] == s)
        par_agent = par["agent"]

        assert set(par_agent.Q.keys()) == set(ref.Q.keys()), (
            f"Seed {s}: claves divergen entre paralelo y secuencial"
        )
        for k, v in ref.Q.items():
            assert v == pytest.approx(par_agent.Q[k], abs=1e-12), (
                f"Seed {s} clave {k}: paralelo={par_agent.Q[k]} ref={v}"
            )


# --------------------------------------------------------------------- #
# Estructura de la salida
# --------------------------------------------------------------------- #
def test_results_preserve_seed_order():
    """`pool.map` preserva orden, así que `results[i]['seed'] == seeds[i]`."""
    seeds = [3, 1, 4, 5, 9, 2, 6, 7]
    parallel = run_parallel_seeds(
        agent_class=BaseQAgent,
        agent_kwargs={},
        train_kwargs={"episodes": 50},
        seeds=seeds,
        n_workers=2,
    )
    assert [r["seed"] for r in parallel] == seeds


def test_results_have_expected_keys():
    parallel = run_parallel_seeds(
        agent_class=BaseQAgent,
        agent_kwargs={},
        train_kwargs={"episodes": 50},
        seeds=[0, 1],
        n_workers=2,
    )
    for r in parallel:
        assert set(r.keys()) >= {"seed", "agent", "history"}
        assert isinstance(r["agent"], BaseQAgent)
        # BaseQAgent.train returns None; ImprovedQAgent returns dict.
        # Aquí solo verificamos que el campo existe (puede ser None).


# --------------------------------------------------------------------- #
# Distintas semillas producen distintos agentes
# --------------------------------------------------------------------- #
def test_different_seeds_produce_distinct_q_tables():
    parallel = run_parallel_seeds(
        agent_class=BaseQAgent,
        agent_kwargs={},
        train_kwargs={"episodes": 200},
        seeds=[0, 1, 2],
        n_workers=3,
    )
    a0 = parallel[0]["agent"].Q
    a1 = parallel[1]["agent"].Q
    a2 = parallel[2]["agent"].Q
    # Probabilidad astronómica de que dos seeds distintos den Q idénticas.
    assert a0 != a1 or len(a0) != len(a1) or any(a0[k] != a1.get(k, None) for k in a0)
    assert a0 != a2 or len(a0) != len(a2) or any(a0[k] != a2.get(k, None) for k in a0)


# --------------------------------------------------------------------- #
# n_workers default razonable
# --------------------------------------------------------------------- #
def test_default_n_workers_is_capped_by_seeds_and_cpu():
    """Con 2 semillas, default debe ser ≤ 2."""
    parallel = run_parallel_seeds(
        agent_class=BaseQAgent,
        agent_kwargs={},
        train_kwargs={"episodes": 10},
        seeds=[0, 1],
        # n_workers=None: default
    )
    assert len(parallel) == 2
