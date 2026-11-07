import itertools

import numpy as np
import pytest

from magic_squares.cp_sat import solve_magic_square
from magic_squares.qubo import (
    build_qubo,
    decode,
    direct_penalty,
    encode,
    solve_simulated_annealing,
    solve_swap_annealing,
    swap_energy,
)
from magic_squares.verify import check_magic_square, is_magic_square, magic_constant

LO_SHU = np.array([[2, 7, 6], [9, 5, 1], [4, 3, 8]])
DURER = np.array([[16, 3, 2, 13], [5, 10, 11, 8], [9, 6, 7, 12], [4, 15, 14, 1]])


# ---------------------------------------------------------------- verify
def test_magic_constant():
    assert [magic_constant(n) for n in range(1, 6)] == [1, 5, 15, 34, 65]


def test_known_squares_are_magic():
    assert is_magic_square(LO_SHU)
    assert is_magic_square(DURER)


def test_rejects_repeated_values_even_if_sums_match():
    # Todas las líneas suman 15 pero no es una permutación de 1..9
    assert not is_magic_square(np.full((3, 3), 5))
    assert not check_magic_square(np.full((3, 3), 5))["is_permutation"]


def test_rejects_wrong_diagonal():
    bad = LO_SHU[:, [1, 0, 2]]  # intercambiar columnas mantiene filas/columnas, rompe diagonales
    assert not is_magic_square(bad)


# ---------------------------------------------------------------- CP-SAT
@pytest.mark.parametrize("n", [1, 3, 4, 5, 6])
def test_cp_sat_finds_magic_square(n):
    grid, info = solve_magic_square(n, time_limit=60)
    assert grid is not None, info
    assert is_magic_square(grid)


def test_cp_sat_n2_infeasible():
    grid, info = solve_magic_square(2, time_limit=10)
    assert grid is None
    assert info["status"] == "INFEASIBLE"


# ---------------------------------------------------------------- QUBO
@pytest.mark.parametrize("n", [2, 3, 4])
def test_qubo_matrix_matches_definition(n):
    """x^T Q x + offset debe coincidir con la energía evaluada desde la fórmula."""
    q = build_qubo(n, A=7.0, B=1.0)
    rng = np.random.default_rng(42)
    for density in (0.05, 0.3, 0.7):
        for _ in range(20):
            x = (rng.random(q.num_vars) < density).astype(int)
            assert q.energy(x) == pytest.approx(direct_penalty(x, n, 7.0, 1.0))


def test_qubo_is_upper_triangular():
    q = build_qubo(3)
    assert np.allclose(q.Q, np.triu(q.Q))


@pytest.mark.parametrize("grid", [LO_SHU, DURER])
def test_qubo_energy_zero_for_magic_squares(grid):
    q = build_qubo(grid.shape[0])
    assert q.energy(encode(grid)) == pytest.approx(0.0)


def test_qubo_ground_states_are_exactly_magic_squares_n3():
    """Recorre las 9! permutaciones: E = 0 exactamente en los 8 cuadrados mágicos de orden 3."""
    q = build_qubo(3)
    zero, magic = 0, 0
    for perm in itertools.permutations(range(1, 10)):
        g = np.array(perm).reshape(3, 3)
        e = q.energy(encode(g))
        is_m = is_magic_square(g)
        assert (abs(e) < 1e-9) == is_m
        assert e >= -1e-9
        zero += abs(e) < 1e-9
        magic += is_m
    assert zero == magic == 8


def test_qubo_penalizes_non_permutations():
    """El fallo de la versión original: un mismo valor en celdas de distinta fila y columna."""
    q = build_qubo(3)
    g = LO_SHU.copy()
    g[0, 0] = 5  # ahora 5 aparece dos veces (en (0,0) y (1,1)) y falta el 2
    assert q.energy(encode(g)) > 0


def test_decode_rejects_invalid_cell():
    x = encode(LO_SHU)
    x[:9] = 0  # celda (0,0) sin valor
    assert decode(x, 3) is None
    assert np.array_equal(decode(encode(LO_SHU), 3), LO_SHU)


@pytest.mark.parametrize("n", [3, 4, 5])
def test_swap_energy_equals_qubo_energy_on_permutations(n):
    """En permutaciones, la energía reducida del swap annealing es la energía QUBO completa."""
    q = build_qubo(n)
    rng = np.random.default_rng(0)
    for _ in range(50):
        g = (rng.permutation(n * n) + 1).reshape(n, n)
        assert swap_energy(g) == pytest.approx(q.energy(encode(g)))


def test_swap_annealing_incremental_energy_is_consistent():
    """Con pocos pasos no resuelve: la energía llevada de forma incremental debe
    coincidir con la recalculada desde cero sobre la mejor permutación."""
    for seed in range(5):
        grid, info = solve_swap_annealing(5, num_reads=3, num_steps=300, seed=seed)
        best = info["best_grid"]
        assert sorted(best.flatten().tolist()) == list(range(1, 26))
        assert info["best_energy"] == pytest.approx(swap_energy(best))


@pytest.mark.parametrize("n", [3, 4, 5, 6])
def test_swap_annealing_solves(n):
    grid, info = solve_swap_annealing(n, num_reads=20, seed=0, stop_on_success=True)
    assert grid is not None, info
    assert is_magic_square(grid)


def test_qubo_simulated_annealing_n3():
    grid, info = solve_simulated_annealing(3, num_reads=200, num_sweeps=10000, seed=0)
    assert grid is not None, info
    assert is_magic_square(grid)
