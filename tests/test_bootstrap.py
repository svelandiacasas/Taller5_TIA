"""Tests para `bootstrap_ci` y `paired_bootstrap_test`.

Cubrimos:
- `bootstrap_ci` con datos normales: el IC contiene la media verdadera con
  alta probabilidad y la estimación está dentro del IC.
- Reproducibilidad: misma seed → mismo resultado.
- `paired_bootstrap_test` con dos muestras de la misma distribución: el
  p-value NO debe ser sistemáticamente bajo (calibración del null).
- `paired_bootstrap_test` con efecto grande: el p-value debe ser bajo.
- `paired_bootstrap_test` no es sesgado: ~5 % de rechazos a nivel 0.05
  cuando H₀ es verdadera.
"""
import numpy as np
import pytest

from new.bootstrap import bootstrap_ci, paired_bootstrap_test


# --------------------------------------------------------------------- #
# bootstrap_ci
# --------------------------------------------------------------------- #
def test_bootstrap_ci_estimate_is_inside_ci():
    rng = np.random.default_rng(0)
    values = rng.normal(loc=5.0, scale=2.0, size=50)
    estimate, low, high = bootstrap_ci(values, n_resamples=2000, seed=0)
    assert low <= estimate <= high


def test_bootstrap_ci_contains_true_mean_for_normal_data():
    """Para una muestra n=200 ~ N(5, 2), el IC bootstrap al 95 % debe contener 5.0."""
    rng = np.random.default_rng(42)
    values = rng.normal(loc=5.0, scale=2.0, size=200)
    estimate, low, high = bootstrap_ci(values, n_resamples=2000, seed=0)
    assert low <= 5.0 <= high


def test_bootstrap_ci_is_reproducible_with_same_seed():
    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    e1, l1, h1 = bootstrap_ci(values, n_resamples=500, seed=123)
    e2, l2, h2 = bootstrap_ci(values, n_resamples=500, seed=123)
    assert (e1, l1, h1) == (e2, l2, h2)


def test_bootstrap_ci_different_seeds_give_close_but_distinct_intervals():
    rng = np.random.default_rng(0)
    values = rng.normal(0, 1, size=100)
    _, l1, h1 = bootstrap_ci(values, n_resamples=2000, seed=1)
    _, l2, h2 = bootstrap_ci(values, n_resamples=2000, seed=2)
    # Distintas seeds → distintos intervalos (con probabilidad casi 1)
    assert (l1, h1) != (l2, h2)
    # ... pero similares (mismo dato base)
    assert abs(l1 - l2) < 0.5
    assert abs(h1 - h2) < 0.5


def test_bootstrap_ci_works_with_median():
    """`statistic=np.median` no soporta `axis=...` en scipy/numpy old;
    nuestro fallback al loop debería funcionar igual."""
    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 100.0])  # mediana robusta a outliers
    estimate, low, high = bootstrap_ci(values, statistic=np.median,
                                        n_resamples=500, seed=0)
    assert estimate == 3.5  # median of [1, 2, 3, 4, 5, 100]
    assert low <= estimate <= high


def test_bootstrap_ci_raises_on_empty_input():
    with pytest.raises(ValueError):
        bootstrap_ci([])


def test_bootstrap_ci_raises_on_invalid_confidence():
    with pytest.raises(ValueError):
        bootstrap_ci([1.0, 2.0], confidence=1.5)


# --------------------------------------------------------------------- #
# paired_bootstrap_test
# --------------------------------------------------------------------- #
def test_paired_test_high_p_value_for_identical_distributions():
    """Si `a` y `b` provienen de la misma distribución, p-value típicamente alto."""
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, size=50)
    b = rng.normal(0, 1, size=50)
    p = paired_bootstrap_test(a, b, n_resamples=2000, seed=0)
    assert p > 0.10, f"H₀ verdadera pero p-value bajo: {p}"


def test_paired_test_low_p_value_for_clear_effect():
    """Si `a` tiene media 0 y `b` tiene media 2, p-value debe ser muy bajo."""
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, size=50)
    b = rng.normal(2, 1, size=50)
    p = paired_bootstrap_test(a, b, n_resamples=2000, seed=0)
    assert p < 0.01, f"Efecto claro pero p-value alto: {p}"


def test_paired_test_is_calibrated_under_null():
    """Test de calibración: con 50 trials de muestras de la misma distribución,
    la fracción de p-values < 0.05 debe ser cercana a 0.05 (entre 0 y 0.20).

    No es estricto (50 trials × intrínseca varianza → tolerancia amplia)
    pero detecta sesgos sistemáticos."""
    n_trials = 50
    n_samples_per = 30
    rng = np.random.default_rng(0)
    p_values = []
    for trial in range(n_trials):
        a = rng.normal(0, 1, size=n_samples_per)
        b = rng.normal(0, 1, size=n_samples_per)
        p = paired_bootstrap_test(a, b, n_resamples=1000, seed=trial)
        p_values.append(p)
    rejection_rate = float(np.mean(np.array(p_values) < 0.05))
    # Bajo H₀ esperamos ~5 % de rechazos. Permitimos hasta 20 % por varianza
    # de muestreo a 50 trials (binomial(50, 0.05) tiene cola al 99 %≈10).
    assert rejection_rate <= 0.20, (
        f"Test pareado no calibrado: rejection rate {rejection_rate:.2%} "
        f"esperaba ≤ 20 % (true ~5 %)"
    )
    # Y también que la media de p-values esté en una banda razonable
    # (uniforme(0, 1) tiene media 0.5).
    assert 0.30 <= np.mean(p_values) <= 0.70, (
        f"Distribución de p-values posiblemente sesgada: "
        f"mean={np.mean(p_values):.3f}"
    )


def test_paired_test_is_reproducible_with_same_seed():
    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    b = np.array([1.5, 1.8, 3.2, 4.1, 4.9])
    p1 = paired_bootstrap_test(a, b, n_resamples=500, seed=42)
    p2 = paired_bootstrap_test(a, b, n_resamples=500, seed=42)
    assert p1 == p2


def test_paired_test_raises_on_shape_mismatch():
    with pytest.raises(ValueError):
        paired_bootstrap_test([1.0, 2.0], [1.0, 2.0, 3.0])


def test_paired_test_raises_on_empty():
    with pytest.raises(ValueError):
        paired_bootstrap_test([], [])
