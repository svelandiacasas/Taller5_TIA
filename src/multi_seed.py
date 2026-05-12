"""Paralelización por semillas con `multiprocessing.Pool`.

API mínima:

    results = run_parallel_seeds(
        agent_class=BaseQAgent,
        agent_kwargs={"alpha": 0.1},
        train_kwargs={"episodes": 10_000},
        seeds=list(range(10)),
        n_workers=10,
    )

`results` es una lista de dicts `{"seed": int, "agent": agent, "history": dict | None}`,
ordenada por la posición de la semilla en `seeds`.

**Determinismo**: cada worker llama `set_seed(seed)` al inicio, ANTES de
instanciar el agente o entrenar. En Windows (`spawn`) los workers no heredan
estado RNG del padre; en Linux (`fork`) lo heredan, lo cual es PEOR (RNG
correlacionado entre workers). Re-sembrar siempre garantiza que el resultado
paralelo sea idéntico al secuencial para la misma semilla.

**Pickling**: los agentes tabulares (`BaseQAgent`, `ImprovedQAgent`) son
`pickleable` directamente porque su estado es un `dict`. El `nn.Module`
`TabularToTorchAdapter` también pickle-ea, pero no se usa aquí (se construye
on-demand en el padre tras recibir el agente del worker).

**`PYTHONPATH`**: `conftest.py` propaga `src/` al entorno para que los workers
puedan importar `new.*`. Los scripts CLI deben hacer lo mismo manualmente
(ver `scripts/measure_parallel_speedup.py`).
"""
from __future__ import annotations

import multiprocessing as mp
from typing import Any, Optional


def _train_one_seed(args: tuple) -> dict:
    """Worker: entrena un agente para una semilla, devuelve dict con resultados.

    Definido a nivel de módulo (no anidado) para que `multiprocessing` lo
    pueda pickle-ar y enviar a los procesos hijos.
    """
    agent_class, agent_kwargs, train_kwargs, seed = args
    # Importar set_seed dentro del worker para no asumir que el padre ya lo
    # importó (en `spawn`, el worker arranca limpio).
    from seeds import set_seed

    set_seed(seed)
    agent = agent_class(**agent_kwargs)
    # `train` puede aceptar `seed` (lo hace internamente también: redundancia
    # defensiva sin costo).
    history = agent.train(seed=seed, **train_kwargs)
    return {"seed": seed, "agent": agent, "history": history}


def run_parallel_seeds(
    agent_class: type,
    agent_kwargs: Optional[dict[str, Any]] = None,
    train_kwargs: Optional[dict[str, Any]] = None,
    seeds: Optional[list[int]] = None,
    n_workers: Optional[int] = None,
) -> list[dict]:
    """Lanza un proceso por semilla con `multiprocessing.Pool`.

    Parameters
    ----------
    agent_class : type
        Clase del agente, e.g. `BaseQAgent` o `ImprovedQAgent`.
    agent_kwargs : dict, optional
        Argumentos para `agent_class(**agent_kwargs)`.
    train_kwargs : dict, optional
        Argumentos para `agent.train(seed=seed, **train_kwargs)`.
        El parámetro `seed` lo aporta el orquestador, no incluirlo aquí.
    seeds : list[int], optional
        Semillas a entrenar (default: `list(range(10))`).
    n_workers : int, optional
        Número de procesos en el pool. Default:
        `min(len(seeds), max(1, cpu_count() - 2))`.

    Returns
    -------
    list[dict]
        Una entrada por semilla, ordenada por posición en `seeds`. Cada
        dict tiene claves `"seed"`, `"agent"`, `"history"`.
    """
    agent_kwargs = agent_kwargs or {}
    train_kwargs = train_kwargs or {}
    if seeds is None:
        seeds = list(range(10))
    if n_workers is None:
        n_workers = min(len(seeds), max(1, mp.cpu_count() - 2))

    args_list = [(agent_class, agent_kwargs, train_kwargs, s) for s in seeds]

    with mp.Pool(processes=n_workers) as pool:
        results = pool.map(_train_one_seed, args_list)

    # `pool.map` garantiza que `results[i]` corresponde a `args_list[i]`
    # (orden preservado), así que `results` queda ordenado por `seeds`.
    return results
