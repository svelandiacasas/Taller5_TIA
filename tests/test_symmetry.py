"""Tests para `symmetry.py` (canonicalización D₄)."""
import numpy as np
import pytest

from new.symmetry import (
    ACTION_MAPS,
    INVERSE_MAP,
    _apply_transform,
    canonical_action,
    canonical_state,
    restore_action,
)


@pytest.fixture(autouse=True)
def _seed():
    np.random.seed(0)


# --------------------------------------------------------------------- #
# Invariancia: las 8 transformaciones de s producen el mismo canónico
# --------------------------------------------------------------------- #
def test_canonical_state_invariant_under_d4():
    """Los 8 conjugados de un estado bajo D₄ comparten la misma forma canónica."""
    state = np.array([[1, 0, -1], [0, 1, 0], [0, 0, -1]], dtype=int)
    canon0, _ = canonical_state(state)
    for t in range(8):
        transformed = _apply_transform(state, t)
        canon_t, _ = canonical_state(transformed)
        np.testing.assert_array_equal(
            canon0, canon_t,
            err_msg=f"transformación t={t} altera el canónico",
        )


def test_canonical_state_returns_lex_smallest():
    """El canónico es el lexicográficamente mínimo entre las 8 transformaciones."""
    state = np.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]], dtype=int)
    canon, _ = canonical_state(state)
    all_transforms = [tuple(_apply_transform(state, t).flatten().tolist()) for t in range(8)]
    assert tuple(canon.flatten().tolist()) == min(all_transforms)


def test_symmetric_state_has_identity_canonical():
    """Un estado D₄-simétrico (e.g. solo centro) coincide con su canónico, t=0."""
    state = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=int)
    canon, t = canonical_state(state)
    np.testing.assert_array_equal(canon, state)
    assert t == 0


# --------------------------------------------------------------------- #
# Acciones: canonical_action y restore_action son inversas
# --------------------------------------------------------------------- #
def test_restore_action_inverts_canonical_action_for_random_states():
    """Para 100 estados aleatorios y todas sus acciones legales:
    restore_action(s, canonical_action(s, a)) == a."""
    for trial in range(100):
        state = np.zeros((3, 3), dtype=int)
        n_pieces = int(np.random.randint(0, 9))
        positions = np.random.choice(9, size=n_pieces, replace=False)
        for i, p in enumerate(positions):
            state[p // 3, p % 3] = 1 if i % 2 == 0 else -1
        for r in range(3):
            for c in range(3):
                if state[r, c] != 0:
                    continue
                a = r * 3 + c
                canon_a = canonical_action(state, a)
                restored = restore_action(state, canon_a)
                assert restored == a, (
                    f"trial {trial} state={state.flatten().tolist()} "
                    f"a={a} canon={canon_a} restored={restored}"
                )


# --------------------------------------------------------------------- #
# Tablas precomputadas: consistencia interna
# --------------------------------------------------------------------- #
def test_action_maps_have_correct_shape():
    assert ACTION_MAPS.shape == (8, 9)
    assert INVERSE_MAP.shape == (8,)


def test_action_maps_inverse_consistency():
    """ACTION_MAPS[INVERSE_MAP[t]][ACTION_MAPS[t][a]] == a para todo (t, a)."""
    for t in range(8):
        t_inv = int(INVERSE_MAP[t])
        for a in range(9):
            forward = int(ACTION_MAPS[t, a])
            back = int(ACTION_MAPS[t_inv, forward])
            assert back == a, f"t={t} a={a}: forward={forward} back={back}"


def test_action_maps_are_permutations():
    """Cada fila de ACTION_MAPS es una permutación de {0..8}."""
    for t in range(8):
        row = ACTION_MAPS[t].tolist()
        assert sorted(row) == list(range(9)), f"t={t}: no es permutación"


def test_identity_maps_actions_to_themselves():
    """t=0 (identidad) deja las acciones intactas."""
    np.testing.assert_array_equal(ACTION_MAPS[0], np.arange(9))


def test_rot180_action_map_matches_geometric_rotation():
    """t=2 (rot180): celda (i, j) → (2-i, 2-j)."""
    for r in range(3):
        for c in range(3):
            a = r * 3 + c
            new_r, new_c = 2 - r, 2 - c
            expected = new_r * 3 + new_c
            assert int(ACTION_MAPS[2, a]) == expected, f"({r}, {c}) → ({new_r}, {new_c})"
