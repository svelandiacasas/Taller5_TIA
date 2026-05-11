"""Smoke test del ablation study.

Configuración mínima: 2 variantes × 2 seeds × 1 000 episodios × 100 eval games.
Verifica estructura del DataFrame, presencia de archivos y propiedades
estadísticas básicas (Full vs Full → p_value=1.0, IC contiene la media).

Tiempo objetivo: < 30 s.
"""
import pandas as pd
import pytest

from new.ablations import (
    ABLATION_VARIANTS,
    METRICS,
    compute_distance_to_minimax,
    evaluate_agent_metrics,
    evaluate_vs_minimax,
    run_ablation_study,
)
from new.improved_q_agent import ImprovedQAgent


# --------------------------------------------------------------------- #
# Smoke test del orquestador
# --------------------------------------------------------------------- #
def test_ablation_smoke(tmp_path):
    test_variants = {
        "Full": ABLATION_VARIANTS["Full"],
        "− shaping": ABLATION_VARIANTS["− shaping"],
    }
    df = run_ablation_study(
        seeds=[0, 1],
        episodes=1_000,
        eval_games=100,
        n_workers=2,
        output_dir=tmp_path,
        variants=test_variants,
    )

    # Estructura del DataFrame
    expected_cols = {"variant", "metric", "mean", "ci_low", "ci_high",
                      "n_seeds", "p_value_vs_full"}
    assert expected_cols.issubset(df.columns)

    # Variantes presentes
    assert set(df["variant"].unique()) == {"Full", "− shaping"}

    # Métricas presentes
    for m in METRICS:
        assert m in df["metric"].unique(), f"métrica faltante: {m}"

    # Full vs Full → p_value = 1.0
    full_rows = df[df["variant"] == "Full"]
    assert all(p == 1.0 for p in full_rows["p_value_vs_full"])

    # IC contiene la media
    for _, row in df.iterrows():
        assert row["ci_low"] <= row["mean"] <= row["ci_high"], (
            f"{row['variant']} / {row['metric']}: media fuera del IC"
        )

    # n_seeds correcto
    assert all(df["n_seeds"] == 2)

    # Archivos creados
    assert (tmp_path / "ablation_raw.csv").exists()
    assert (tmp_path / "ablation_summary.csv").exists()
    raw = pd.read_csv(tmp_path / "ablation_raw.csv")
    assert len(raw) == 2 * 2 * len(METRICS)  # 2 variants × 2 seeds × N metrics
    assert {"variant", "seed", "metric", "value"}.issubset(raw.columns)

    # 4 figuras
    for m in METRICS:
        assert (tmp_path / "figures" / f"ablation_{m}.png").exists(), (
            f"figura faltante para {m}"
        )


# --------------------------------------------------------------------- #
# Eval functions: smoke
# --------------------------------------------------------------------- #
def test_evaluate_vs_minimax_basic():
    """Cualquier agente vs Minimax: nunca debe ganar (Minimax es perfecto)."""
    agent = ImprovedQAgent()
    agent.train(episodes=200, seed=0)
    res = evaluate_vs_minimax(agent, n_episodes=20, agent_role=1, seed=0)
    assert res["wins"] == 0
    assert res["wins"] + res["draws"] + res["losses"] == 20


def test_compute_distance_to_minimax_in_range():
    agent = ImprovedQAgent()
    agent.train(episodes=200, seed=0)
    dist = compute_distance_to_minimax(agent, n_episodes=20,
                                        agent_role=1, seed=0)
    assert 0.0 <= dist <= 1.0


def test_evaluate_agent_metrics_returns_all_four():
    agent = ImprovedQAgent()
    agent.train(episodes=200, seed=0)
    metrics = evaluate_agent_metrics(agent, n_eval_games=20,
                                       agent_role=1, eval_seed_base=0)
    assert set(metrics.keys()) == set(METRICS)
    for k, v in metrics.items():
        assert isinstance(v, float)
        assert 0.0 <= v <= 1.0


# --------------------------------------------------------------------- #
# ABLATION_VARIANTS sanity
# --------------------------------------------------------------------- #
def test_ablation_variants_have_all_required_keys():
    """Cada variante debe especificar TODOS los kwargs (no defaults implícitos
    para evitar deriva accidental)."""
    required = {
        "alpha", "gamma",
        "epsilon_start", "epsilon_end", "epsilon_decay",
        "reward_shaping", "shaping_weight",
        "use_symmetries", "dual_perspective", "optimistic_init",
    }
    for vname, vkwargs in ABLATION_VARIANTS.items():
        assert required.issubset(vkwargs.keys()), (
            f"{vname}: faltan keys {required - set(vkwargs.keys())}"
        )


def test_ablation_variants_differ_only_in_one_toggle():
    """Cada variante (excepto Full) difiere de Full en exactamente UN toggle."""
    full = ABLATION_VARIANTS["Full"]
    # Para "− decay" la diferencia abarca tres parámetros relacionados (start,
    # end, decay) pero se considera un solo toggle conceptual.
    expected_diffs = {
        "− decay": {"epsilon_start", "epsilon_end", "epsilon_decay"},
        "− shaping": {"reward_shaping"},
        "− symmetries": {"use_symmetries"},
        "− dual persp.": {"dual_perspective"},
    }
    for vname, expected in expected_diffs.items():
        diffs = {k for k in full.keys() if full[k] != ABLATION_VARIANTS[vname][k]}
        assert diffs == expected, (
            f"{vname} difiere en {diffs}, esperaba {expected}"
        )
