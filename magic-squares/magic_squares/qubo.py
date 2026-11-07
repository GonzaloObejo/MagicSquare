"""Cuadrado mágico como QUBO (Quadratic Unconstrained Binary Optimization).

Codificación one-hot: x[i, j, k] = 1  <=>  la celda (i, j) contiene el valor k + 1,
con i, j en 0..n-1 y k en 0..n^2-1  (n^4 variables binarias).

Energía (todas las penalizaciones son cuadrados, luego E >= 0):

    E(x) = A * sum_{celdas (i,j)} ( sum_k x[i,j,k] - 1 )^2          (a) un valor por celda
         + A * sum_{valores k}    ( sum_{i,j} x[i,j,k] - 1 )^2      (b) cada valor una sola vez
         + B * sum_{líneas L}     ( sum_{(i,j) en L} sum_k (k+1) x[i,j,k] - M )^2
                                                                     (c) filas, columnas y diagonales

Con A, B > 0:  E(x) = 0  <=>  x codifica un cuadrado mágico. (a) y (b) juntas
obligan a que la asignación sea una permutación de 1..n^2, por lo que no hacen
falta restricciones extra de "no repetir en fila/columna".

La matriz se construye en NumPy, independiente de cualquier proveedor, y se
resuelve en local con simulated annealing (dwave-samplers, gratuito y sin token).

Uso:
    python -m magic_squares.qubo 3
    python -m magic_squares.qubo 4 --reads 200 --sweeps 20000
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

import numpy as np

from magic_squares.verify import check_magic_square, magic_constant


@dataclass
class MagicSquareQUBO:
    n: int
    Q: np.ndarray        # matriz triangular superior (N_vars × N_vars)
    offset: float        # constante: E(x) = x^T Q x + offset
    A: float
    B: float

    @property
    def num_vars(self) -> int:
        return self.Q.shape[0]

    def index(self, i: int, j: int, k: int) -> int:
        n2 = self.n * self.n
        return (i * self.n + j) * n2 + k

    def energy(self, x) -> float:
        x = np.asarray(x, dtype=float)
        return float(x @ self.Q @ x + self.offset)

    def to_dict(self) -> dict:
        """Formato {(u, v): coef} que aceptan dimod / dwave-samplers."""
        rows, cols = np.nonzero(self.Q)
        return {(int(u), int(v)): float(self.Q[u, v]) for u, v in zip(rows, cols)}


def _add_squared(Q: np.ndarray, idx: np.ndarray, coef: np.ndarray, const: float, w: float) -> float:
    """Suma w * (sum_t coef_t x_{idx_t} + const)^2 a Q (triangular superior).

    Usa x^2 = x para binarias. Devuelve la contribución constante w * const^2.
    Se asume que `idx` no tiene índices repetidos.
    """
    outer = w * np.outer(coef, coef)
    # Términos cruzados: 2 c_s c_t x_s x_t  (solo s < t en el triángulo superior)
    ii, jj = np.meshgrid(idx, idx, indexing="ij")
    lo, hi = np.minimum(ii, jj), np.maximum(ii, jj)
    off = ii != jj
    np.add.at(Q, (lo[off], hi[off]), outer[off])  # cada par aparece dos veces -> 2 c_s c_t
    # Términos diagonales: c^2 x + 2 const c x
    Q[idx, idx] += w * (coef ** 2 + 2.0 * const * coef)
    return w * const ** 2


def build_qubo(n: int, A: float | None = None, B: float = 1.0) -> MagicSquareQUBO:
    """Construye la QUBO del cuadrado mágico de orden n.

    A: peso de las restricciones de permutación. Cualquier A > 0 da los mismos
       mínimos (E = 0), pero A afecta mucho al annealing: si es grande el paisaje
       se vuelve muy abrupto. Por defecto 2 n^2, elegido empíricamente (ver README).
    """
    if n < 1:
        raise ValueError("n debe ser >= 1")
    n2 = n * n
    if A is None:
        A = 2.0 * n2
    M = magic_constant(n)
    nv = n2 * n2
    Q = np.zeros((nv, nv))
    offset = 0.0
    values = np.arange(1, n2 + 1, dtype=float)
    ones = np.ones(n2)

    def cell_block(i, j):
        base = (i * n + j) * n2
        return np.arange(base, base + n2)

    # (a) un único valor por celda
    for i in range(n):
        for j in range(n):
            offset += _add_squared(Q, cell_block(i, j), ones, -1.0, A)

    # (b) cada valor aparece exactamente una vez en todo el cuadrado
    for k in range(n2):
        idx = np.arange(k, nv, n2)
        offset += _add_squared(Q, idx, ones, -1.0, A)

    # (c) restricciones de suma
    lines = [[(i, j) for j in range(n)] for i in range(n)]
    lines += [[(i, j) for i in range(n)] for j in range(n)]
    lines.append([(i, i) for i in range(n)])
    lines.append([(i, n - 1 - i) for i in range(n)])
    for line in lines:
        idx = np.concatenate([cell_block(i, j) for i, j in line])
        coef = np.tile(values, len(line))
        offset += _add_squared(Q, idx, coef, -float(M), B)

    return MagicSquareQUBO(n=n, Q=Q, offset=offset, A=A, B=B)


def encode(grid) -> np.ndarray:
    """Cuadrado n×n con valores 1..n^2  ->  vector binario one-hot."""
    g = np.asarray(grid)
    n = g.shape[0]
    n2 = n * n
    x = np.zeros(n2 * n2, dtype=np.int8)
    for i in range(n):
        for j in range(n):
            x[(i * n + j) * n2 + int(g[i, j]) - 1] = 1
    return x


def decode(x, n: int):
    """Vector binario -> cuadrado n×n. Devuelve None si alguna celda no es one-hot."""
    n2 = n * n
    cells = np.asarray(x).reshape(n, n, n2)
    if not np.all(cells.sum(axis=2) == 1):
        return None
    return cells.argmax(axis=2) + 1


def direct_penalty(x, n: int, A: float, B: float) -> float:
    """Evalúa E(x) directamente desde la definición (sin la matriz Q). Sirve para tests."""
    n2 = n * n
    M = magic_constant(n)
    c = np.asarray(x, dtype=float).reshape(n, n, n2)
    v = np.arange(1, n2 + 1)
    e = A * np.sum((c.sum(axis=2) - 1) ** 2)
    e += A * np.sum((c.sum(axis=(0, 1)) - 1) ** 2)
    s = (c * v).sum(axis=2)  # valor "efectivo" de cada celda
    lines = list(s.sum(axis=1)) + list(s.sum(axis=0)) + [np.trace(s), np.trace(np.fliplr(s))]
    e += B * sum((L - M) ** 2 for L in lines)
    return float(e)


def solve_simulated_annealing(
    n: int,
    num_reads: int = 100,
    num_sweeps: int = 10000,
    A: float | None = None,
    B: float = 1.0,
    seed: int | None = 0,
):
    """Resuelve con el simulated annealing de dwave-samplers.

    Devuelve (grid, info). grid es None si ninguna lectura alcanza energía 0.
    """
    from dwave.samplers import SimulatedAnnealingSampler

    qubo = build_qubo(n, A, B)
    sampler = SimulatedAnnealingSampler()
    t0 = time.perf_counter()
    result = sampler.sample_qubo(
        qubo.to_dict(), num_reads=num_reads, num_sweeps=num_sweeps, seed=seed
    )
    elapsed = time.perf_counter() - t0

    energies = result.record.energy + qubo.offset
    samples = result.record.sample
    occ = result.record.num_occurrences
    best = int(np.argmin(energies))
    x = np.zeros(qubo.num_vars, dtype=np.int8)
    x[np.asarray(result.variables, dtype=int)] = samples[best]

    n_ok = int(occ[np.isclose(energies, 0)].sum())
    info = {
        "wall_time": elapsed,
        "best_energy": float(energies[best]),
        "success_rate": n_ok / int(occ.sum()),
        "num_vars": qubo.num_vars,
    }
    grid = decode(x, n)
    if grid is None or not check_magic_square(grid)["is_magic"]:
        return None, info
    return grid, info


def main(argv=None):
    p = argparse.ArgumentParser(description="Cuadrado mágico como QUBO (simulated annealing local)")
    p.add_argument("n", type=int)
    p.add_argument("--reads", type=int, default=200, help="número de lecturas (reinicios)")
    p.add_argument("--sweeps", type=int, default=10000, help="barridos por lectura")
    p.add_argument("--A", type=float, default=None, help="peso de permutación (defecto 2 n^2)")
    p.add_argument("--B", type=float, default=1.0, help="peso de las sumas")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    grid, info = solve_simulated_annealing(args.n, args.reads, args.sweeps, args.A, args.B, args.seed)
    print(f"n = {args.n}, M = {magic_constant(args.n)}, variables binarias = {info['num_vars']}")
    print(
        f"Tiempo: {info['wall_time']:.2f} s | mejor energía: {info['best_energy']:.0f} | "
        f"lecturas con E=0: {info['success_rate']:.1%}"
    )
    if grid is None:
        print("No se alcanzó un cuadrado mágico (prueba más --reads / --sweeps o ajusta --A).")
        return 1
    print(grid)
    print("Verificación: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
