"""Smoke tests para `diagnostics.py`.

Confirman que las tres funciones diagnóstico:
- corren sin lanzar excepciones con configuración mínima,
- producen las figuras y CSVs esperados en el output_dir indicado,
- escriben datos no vacíos.

Los tests usan `tmp_path` para no contaminar `results/`.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from diagnostics import (
    analyze_credit_assignment,
    evaluate_vs_algorithm,
    instrumented_self_play,
    run_all_diagnostics,
    show_credit_assignment_bug,
    show_no_convergence,
    show_self_play_degeneracy,
)
from base_q_agent import BaseQAgent
from minimax import MinimaxAgent


def _csv_for(fig_path: Path) -> Path:
    return fig_path.parent.parent / "logs" / (fig_path.stem + ".csv")


def test_instrumented_self_play_basic(tmp_path):
    agent = BaseQAgent()
    minimax = MinimaxAgent(player_id=2)
    df = instrumented_self_play(agent, minimax, episodes=20, seed=0)
    assert len(df) > 0
    assert set(df.columns) == {"episode", "move_idx", "player", "is_optimal", "n_legal", "n_optimal"}
    assert set(df["player"].unique()).issubset({"X", "O"})
    assert df["n_optimal"].min() >= 1
    assert df["n_legal"].max() <= 9
    # La Q del agente debe haberse poblado (entrenó realmente).
    assert len(agent.Q) > 0


def test_analyze_credit_assignment_finds_o_winning_with_negative_q():
    """Tras N episodios, debe haber jugadas semánticamente óptimas para O
    con Q negativo (el bug). Confirma que el análisis las encuentra."""
    agent = BaseQAgent()
    agent.train(episodes=2_000, seed=0)
    df = analyze_credit_assignment(agent)
    assert len(df) > 0
    o_wins = df[(df["mover"] == "O") & df["wins_immediately"]]
    assert len(o_wins) > 0, "Esperaba al menos una jugada ganadora de O analizada"
    # La mayoría tienen Q < 0
    frac_neg = (o_wins["q_value"] < 0).mean()
    assert frac_neg > 0.5, f"Esperaba > 50% de O-winning con Q<0, obtuvo {frac_neg:.1%}"


def test_evaluate_vs_algorithm_returns_valid_rates():
    agent = BaseQAgent()
    agent.train(episodes=300, seed=0)
    res = evaluate_vs_algorithm(agent, n_episodes=20, agent_role=1, seed=0)
    assert set(res.keys()) >= {"wins", "draws", "losses", "win_rate", "draw_rate", "loss_rate"}
    assert res["wins"] + res["draws"] + res["losses"] == 20
    assert 0.0 <= res["win_rate"] <= 1.0


def test_show_self_play_degeneracy_creates_outputs(tmp_path):
    fig_path = show_self_play_degeneracy(
        seed=0, episodes=300, bin_size=100, output_dir=tmp_path,
    )
    assert fig_path.exists()
    csv = _csv_for(fig_path)
    assert csv.exists()
    df = pd.read_csv(csv)
    assert len(df) > 0
    assert {"episode_bin", "agent_optimality", "random_baseline"}.issubset(df.columns)


def test_show_credit_assignment_bug_creates_outputs(tmp_path):
    fig_path = show_credit_assignment_bug(
        seed=0, episodes=500, output_dir=tmp_path,
    )
    assert fig_path.exists()
    csv = _csv_for(fig_path)
    assert csv.exists()
    df = pd.read_csv(csv)
    assert len(df) == 2  # X-gana, O-gana
    assert {"categoria", "n_total", "pct_q_positive", "pct_q_negative"}.issubset(df.columns)


def test_show_no_convergence_creates_outputs(tmp_path):
    fig_path = show_no_convergence(
        seeds=[0, 1, 2],
        episodes=300,
        eval_every=100,
        eval_episodes=20,
        output_dir=tmp_path,
    )
    assert fig_path.exists()
    csv = _csv_for(fig_path)
    assert csv.exists()
    df = pd.read_csv(csv)
    assert set(df["seed"].unique()) == {0, 1, 2}
    assert len(df) == 3 * 3  # 3 seeds × 3 chunks
    # Nuevas columnas multi-oponente
    expected_cols = {
        "seed", "episode",
        "win_rate_vs_algorithm", "draw_rate_vs_algorithm", "loss_rate_vs_algorithm",
        "win_rate_vs_random", "draw_rate_vs_random", "loss_rate_vs_random",
    }
    assert expected_cols.issubset(df.columns)


def test_run_all_diagnostics_smoke(tmp_path):
    """Smoke: el orquestador corre y deja las tres figuras + CSVs."""
    paths = run_all_diagnostics(
        seeds=[0, 1, 2],
        episodes=300,
        eval_every=100,
        eval_episodes=20,
        output_dir=tmp_path,
        bin_size=100,
    )
    assert set(paths.keys()) == {"self_play_degeneracy", "credit_assignment_bug", "no_convergence"}
    for name, fig_path in paths.items():
        assert fig_path.exists(), f"{name}: figura no creada"
        assert _csv_for(fig_path).exists(), f"{name}: CSV no creado"
