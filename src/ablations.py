"""Ablation study del `ImprovedQAgent`.

Cinco variantes (1 Full + 4 con un toggle desactivado), entrenadas con
`run_parallel_seeds` (Fase 5) y evaluadas con 4 métricas:

1. `win_rate_vs_random`
2. `win_rate_vs_algorithm` (el techo realista es ~0; el draw rate alto es
   el verdadero objetivo y queda implícito como `1 - loss_rate_vs_algorithm`).
3. `loss_rate_vs_minimax` (Minimax nunca pierde, así que el agente solo
   puede empatar o perder; menor = mejor).
4. `distance_to_minimax`: `1 − (jugadas_en_optimal_set / jugadas_totales)`,
   medido SOBRE LA TRAYECTORIA del agente jugando contra Random
   (queremos su política de explotación, no de exploración).

Cada celda `(variante, seed)` reporta los 4 valores. El agregado por variante
es `mean ± IC95 bootstrap` con `paired_bootstrap_test` contra Full para
detectar diferencias significativas.

Salidas (en `output_dir`):
- `ablation_raw.csv`: long format, una fila por `(variante, seed, métrica, valor)`.
- `ablation_summary.csv`: una fila por `(variante, métrica)` con
  `mean`, `ci_low`, `ci_high`, `n_seeds`, `p_value_vs_full`.
- `figures/ablation_<metric>.png`: bar chart por variante con error bars
  IC95, una figura por métrica (4 en total).

**Sobre el volumen de evaluación**: usamos 500 partidas por matchup, no las
≥ 10 000 que pide el enunciado del taller. Razón: la unidad de variabilidad
relevante para el ablation es la *semilla* (10 puntos por celda → IC95
bootstrap sobre 10), no la partida. Con 500 partidas la varianza intra-seed
es σ ≈ √(0.25/500) ≈ 2 %, suficientemente baja para que la varianza
inter-seed domine. Las 10 000 partidas se reservan para la evaluación final
del notebook integrador, no para cada celda del ablation.
"""
from __future__ import annotations

import multiprocessing as mp
import time
from pathlib import Path
from typing import Any, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from bootstrap import bootstrap_ci, paired_bootstrap_test  # noqa: E402
from evaluation import (  # noqa: E402
    compute_distance_to_minimax,
    evaluate_vs_algorithm,
    evaluate_vs_minimax,
    evaluate_vs_random,
)
from minimax import MinimaxAgent  # noqa: E402
from rng_utils import isolated_rng  # noqa: E402
from seeds import set_seed  # noqa: E402
from triqui import Game  # noqa: E402


# ---------------------------------------------------------------------- #
# Variantes del ablation
# ---------------------------------------------------------------------- #
# Detalle: la variante "− decay" usa `epsilon_start = epsilon_end = 0.2` para
# que ε quede CONSTANTE en 0.2 (matches BaseQAgent del notebook). Con
# `epsilon_decay=1.0` y `epsilon_start=1.0` el agente quedaría 100% aleatorio
# y la ablation no sería informativa.
ABLATION_VARIANTS: dict[str, dict[str, Any]] = {
    "Full": dict(
        alpha=0.1, gamma=0.99,
        epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.9995,
        reward_shaping=True, shaping_weight=0.1,
        use_symmetries=True, dual_perspective=True,
        optimistic_init=0.0,
    ),
    "− decay": dict(
        alpha=0.1, gamma=0.99,
        epsilon_start=0.2, epsilon_end=0.2, epsilon_decay=1.0,
        reward_shaping=True, shaping_weight=0.1,
        use_symmetries=True, dual_perspective=True,
        optimistic_init=0.0,
    ),
    "− shaping": dict(
        alpha=0.1, gamma=0.99,
        epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.9995,
        reward_shaping=False, shaping_weight=0.1,
        use_symmetries=True, dual_perspective=True,
        optimistic_init=0.0,
    ),
    "− symmetries": dict(
        alpha=0.1, gamma=0.99,
        epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.9995,
        reward_shaping=True, shaping_weight=0.1,
        use_symmetries=False, dual_perspective=True,
        optimistic_init=0.0,
    ),
    "− dual persp.": dict(
        alpha=0.1, gamma=0.99,
        epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.9995,
        reward_shaping=True, shaping_weight=0.1,
        use_symmetries=True, dual_perspective=False,
        optimistic_init=0.0,
    ),
}

METRICS = [
    "win_rate_vs_random",
    "win_rate_vs_algorithm",
    "loss_rate_vs_minimax",
    "distance_to_minimax",
]


# ---------------------------------------------------------------------- #
# Evaluaciones nuevas (no estaban en diagnostics.py)
# ---------------------------------------------------------------------- #
def evaluate_agent_metrics(
    agent,
    n_eval_games: int = 500,
    agent_role: int = 1,
    eval_seed_base: int = 0,
) -> dict:
    """Calcula los 4 valores de métrica para un agente entrenado.

    Cada evaluación usa un seed derivado de `eval_seed_base` para
    reproducibilidad y para que las trayectorias entre métricas no se
    correlacionen artificialmente.
    """
    res_random = evaluate_vs_random(
        agent, n_episodes=n_eval_games, agent_role=agent_role,
        seed=eval_seed_base + 1,
    )
    res_algo = evaluate_vs_algorithm(
        agent, n_episodes=n_eval_games, agent_role=agent_role,
        seed=eval_seed_base + 2,
    )
    res_minimax = evaluate_vs_minimax(
        agent, n_episodes=n_eval_games, agent_role=agent_role,
        seed=eval_seed_base + 3,
    )
    distance = compute_distance_to_minimax(
        agent, n_episodes=n_eval_games, agent_role=agent_role,
        seed=eval_seed_base + 4,
    )
    return {
        "win_rate_vs_random": res_random["win_rate"],
        "win_rate_vs_algorithm": res_algo["win_rate"],
        "loss_rate_vs_minimax": res_minimax["loss_rate"],
        "distance_to_minimax": distance,
    }


# ---------------------------------------------------------------------- #
# Worker para multiprocessing.Pool
# ---------------------------------------------------------------------- #
def _train_and_evaluate_variant(args: tuple) -> dict:
    """Worker: entrena una variante para una semilla y evalúa las 4 métricas."""
    from improved_q_agent import ImprovedQAgent  # import local en worker
    from seeds import set_seed as _set_seed

    variant_name, variant_kwargs, seed, episodes, eval_games, agent_role = args
    _set_seed(seed)
    agent = ImprovedQAgent(**variant_kwargs)
    agent.train(episodes=episodes, seed=seed)
    metrics = evaluate_agent_metrics(
        agent,
        n_eval_games=eval_games,
        agent_role=agent_role,
        eval_seed_base=seed * 1_000,
    )
    return {"variant": variant_name, "seed": seed, "metrics": metrics}


# ---------------------------------------------------------------------- #
# Orquestador
# ---------------------------------------------------------------------- #
def _ensure_dirs(output_dir: Path) -> tuple[Path, Path]:
    figures = Path(output_dir) / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    return figures, Path(output_dir)


def run_ablation_study(
    seeds: Optional[list[int]] = None,
    episodes: int = 50_000,
    eval_games: int = 500,
    agent_role: int = 1,
    n_workers: Optional[int] = None,
    output_dir: Path = Path("results"),
    variants: Optional[dict[str, dict[str, Any]]] = None,
    bootstrap_seed: int = 42,
) -> pd.DataFrame:
    """Corre todas las variantes × todas las semillas en paralelo. Devuelve
    el `summary_df` y deja `ablation_raw.csv`, `ablation_summary.csv` y
    las 4 figuras en `output_dir`."""
    if variants is None:
        variants = ABLATION_VARIANTS
    if seeds is None:
        seeds = list(range(10))

    figures_dir, out_dir = _ensure_dirs(output_dir)

    n_jobs = len(variants) * len(seeds)
    if n_workers is None:
        n_workers = min(n_jobs, max(1, mp.cpu_count() - 2))

    jobs = [
        (vname, vkwargs, s, episodes, eval_games, agent_role)
        for vname, vkwargs in variants.items()
        for s in seeds
    ]

    print(f"[ablation] {n_jobs} jobs × {episodes} eps × {eval_games} eval games "
          f"(workers={n_workers})")
    t0 = time.time()
    with mp.Pool(processes=n_workers) as pool:
        results = pool.map(_train_and_evaluate_variant, jobs)
    elapsed = time.time() - t0
    print(f"[ablation] tiempo total: {elapsed:.1f}s ({elapsed / 60:.1f} min)")

    raw_df = _build_raw_df(results)
    raw_df.to_csv(out_dir / "ablation_raw.csv", index=False)

    summary_df = _build_summary_df(raw_df, variants, bootstrap_seed=bootstrap_seed)
    summary_df.to_csv(out_dir / "ablation_summary.csv", index=False)

    variant_order = list(variants.keys())
    for metric in METRICS:
        if metric in summary_df["metric"].unique():
            _plot_ablation_metric(summary_df, metric, figures_dir, variant_order)

    return summary_df


def _build_raw_df(results: list[dict]) -> pd.DataFrame:
    rows = []
    for r in results:
        for metric, value in r["metrics"].items():
            rows.append({
                "variant": r["variant"],
                "seed": r["seed"],
                "metric": metric,
                "value": value,
            })
    return pd.DataFrame(rows)


def _build_summary_df(
    raw_df: pd.DataFrame,
    variants: dict[str, dict[str, Any]],
    bootstrap_seed: int = 42,
) -> pd.DataFrame:
    summary_rows = []
    metrics_present = list(raw_df["metric"].unique())
    full_data = {
        m: raw_df[(raw_df["variant"] == "Full") & (raw_df["metric"] == m)]["value"].values
        for m in metrics_present
    }
    for vname in variants.keys():
        for m in metrics_present:
            values = raw_df[(raw_df["variant"] == vname) & (raw_df["metric"] == m)]["value"].values
            if len(values) == 0:
                continue
            est, low, high = bootstrap_ci(values, n_resamples=10_000,
                                            seed=bootstrap_seed)
            if vname == "Full":
                p_value = 1.0
            elif len(full_data.get(m, [])) != len(values):
                # Tamaños distintos (no debería pasar) → no se puede pareado.
                p_value = float("nan")
            else:
                p_value = paired_bootstrap_test(values, full_data[m],
                                                  n_resamples=10_000,
                                                  seed=bootstrap_seed)
            summary_rows.append({
                "variant": vname,
                "metric": m,
                "mean": est,
                "ci_low": low,
                "ci_high": high,
                "n_seeds": len(values),
                "p_value_vs_full": p_value,
            })
    return pd.DataFrame(summary_rows)


def _p_value_label(p: float, is_full: bool = False) -> str:
    """Anotación visual para p-values en las figuras.

    `is_full=True` produce '(reference)'. Para variantes ablation, traduce el
    p-value a estrellas (`*** **  *`) o un valor numérico. El caso degenerado
    `p == 1.0` (cero varianza en ambos lados) se etiqueta `p≈1` para
    distinguirlo del verdadero referente."""
    if is_full:
        return "(reference)"
    if np.isnan(p):
        return "n/a"
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    if p >= 0.99:
        return "p≈1"
    return f"p={p:.2f}"


def _plot_ablation_metric(
    summary_df: pd.DataFrame,
    metric: str,
    figures_dir: Path,
    variant_order: list[str],
) -> Path:
    df = summary_df[summary_df["metric"] == metric].set_index("variant")
    df = df.loc[[v for v in variant_order if v in df.index]]

    means = df["mean"].values
    err_low = means - df["ci_low"].values
    err_high = df["ci_high"].values - means

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(df))
    colors = ["steelblue" if v == "Full" else "lightcoral" for v in df.index]
    ax.bar(x, means, yerr=[err_low, err_high], capsize=8, color=colors,
           edgecolor="black", linewidth=0.5)

    # Etiqueta de p-value sobre cada barra.
    y_max = float(np.max(df["ci_high"].values))
    y_floor = 1.0 if y_max < 0.05 else y_max
    y_offset = 0.04 * y_floor
    for i, (variant, row) in enumerate(df.iterrows()):
        ax.text(
            i, row["ci_high"] + y_offset,
            _p_value_label(row["p_value_vs_full"], is_full=(variant == "Full")),
            ha="center", va="bottom", fontsize=10,
        )

    # Extender el eje Y para que las anotaciones queden DENTRO del axes
    # (sin esto, anotaciones cerca del borde superior se renderizan fuera).
    ax.set_ylim(0, max(y_max + 6 * y_offset, 0.05))

    ax.set_xticks(x)
    ax.set_xticklabels(df.index, rotation=15, ha="right")
    ax.set_ylabel(metric)
    n_seeds = int(df["n_seeds"].iloc[0])
    ax.set_title(
        f"Ablation: {metric}  "
        f"(media ± IC95 bootstrap, n={n_seeds} semillas; "
        f"p-value vs Full)"
    )
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    out_path = figures_dir / f"ablation_{metric}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path
