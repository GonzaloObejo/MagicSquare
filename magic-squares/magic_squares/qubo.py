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

La matriz se construye en NumPy, independiente de cualquier proveedor. Hay dos
solvers locales (sin licencia ni token):
  - solve_swap_annealing: annealing por intercambios que conserva la permutación (defecto).
  - solve_simulated_annealing: SA genérico por bits de dwave-samplers sobre la QUBO completa.

Uso:
    python -m magic_squares.qubo 6
    python -m magic_squares.qubo 3 --method bitflip
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


def _cell_lines(n: int) -> list[list[int]]:
    """Para cada celda (aplanada), índices de las líneas a las que pertenece.

    Líneas: 0..n-1 filas, n..2n-1 columnas, 2n diagonal principal, 2n+1 antidiagonal.
    """
    out = []
    for i in range(n):
        for j in range(n):
            ls = [i, n + j]
            if i == j:
                ls.append(2 * n)
            if i + j == n - 1:
                ls.append(2 * n + 1)
            out.append(ls)
    return out


def swap_energy(grid, B: float = 1.0) -> float:
    """Energía QUBO de una permutación: solo queda el término (c) de sumas.

    Coincide con build_qubo(n).energy(encode(grid)) porque (a) y (b) valen 0.
    """
    g = np.asarray(grid)
    M = magic_constant(g.shape[0])
    sums = list(g.sum(axis=1)) + list(g.sum(axis=0)) + [np.trace(g), np.trace(np.fliplr(g))]
    return float(B * sum((s - M) ** 2 for s in sums))


def solve_swap_annealing(
    n: int,
    num_reads: int = 20,
    num_steps: int | None = None,
    T0: float | None = None,
    T1: float = 0.05,
    seed: int | None = 0,
    stop_on_success: bool = False,
):
    """Simulated annealing sobre la misma energía QUBO, restringido a permutaciones.

    El annealing por bits (`solve_simulated_annealing`) tiene que atravesar
    estados que violan las restricciones one-hot: intercambiar dos valores son
    4 cambios de bit. Aquí el estado es siempre una permutación de 1..n^2 y el
    movimiento es intercambiar los valores de dos celdas, de modo que las
    penalizaciones (a) y (b) son 0 por construcción y E = término (c).
    Equivale a usar un "mezclador" que conserva las restricciones, como en
    QAOA con mezcladores XY o en annealers con restricciones one-hot nativas.

    La variación de energía de un intercambio se calcula en O(1) actualizando
    solo las sumas de las líneas afectadas.

    Devuelve (grid, info). grid es None si ninguna lectura alcanza E = 0;
    info["best_grid"] contiene siempre la mejor permutación encontrada.
    """
    if n < 1:
        raise ValueError("n debe ser >= 1")
    n2 = n * n
    M = magic_constant(n)
    if num_steps is None:
        num_steps = 2000 * n2 * n
    if T0 is None:
        T0 = float(n2)  # del orden de un |ΔE| típico al principio
    rng = np.random.default_rng(seed)
    lines = _cell_lines(n)
    n_lines = 2 * n + 2
    cool = (T1 / T0) ** (1.0 / max(num_steps - 1, 1))

    best_grid, best_e = None, float("inf")
    successes, steps_to_solution, reads_done = 0, [], 0
    t_start = time.perf_counter()

    for _ in range(num_reads):
        reads_done += 1
        g = rng.permutation(n2) + 1  # g[c] = valor de la celda c (aplanada)
        dev = [0] * n_lines          # dev[l] = suma(l) - M
        for c, ls in enumerate(lines):
            for l in ls:
                dev[l] += int(g[c])
        dev = [d - M for d in dev]
        e = sum(d * d for d in dev)

        # Números aleatorios por bloques (mucho más rápido que llamar a rng en cada paso)
        a_all = rng.integers(0, n2, num_steps)
        b_all = rng.integers(0, n2 - 1, num_steps) if n2 > 1 else np.zeros(num_steps, int)
        u_all = rng.random(num_steps)
        g = g.tolist()
        T = T0
        step = 0
        while step < num_steps and e > 0:
            a = int(a_all[step])
            b = int(b_all[step])
            if b >= a:
                b += 1  # b != a
            d = g[b] - g[a]  # la celda a gana d, la b pierde d
            change = {}
            for l in lines[a]:
                change[l] = change.get(l, 0) + d
            for l in lines[b]:
                change[l] = change.get(l, 0) - d
            de = 0
            for l, c in change.items():
                if c:
                    de += 2 * dev[l] * c + c * c  # (dev+c)^2 - dev^2
            if de <= 0 or u_all[step] < np.exp(-de / T):
                g[a], g[b] = g[b], g[a]
                for l, c in change.items():
                    dev[l] += c
                e += de
            T *= cool
            step += 1

        if e < best_e:
            best_e, best_grid = e, np.array(g).reshape(n, n)
        if e == 0:
            successes += 1
            steps_to_solution.append(step)
            if stop_on_success:
                break

    elapsed = time.perf_counter() - t_start
    info = {
        "wall_time": elapsed,
        "best_energy": float(best_e),
        "success_rate": successes / reads_done,
        "reads_done": reads_done,
        "mean_steps_to_solution": float(np.mean(steps_to_solution)) if steps_to_solution else None,
        "num_vars": n2 * n2,
        "best_grid": best_grid,
    }
    if best_e != 0 or not check_magic_square(best_grid)["is_magic"]:
        return None, info
    return best_grid, info


def main(argv=None):
    p = argparse.ArgumentParser(description="Cuadrado mágico como QUBO (simulated annealing local)")
    p.add_argument("n", type=int)
    p.add_argument(
        "--method",
        choices=["swap", "bitflip"],
        default="swap",
        help="swap: annealing que conserva la permutación (defecto); "
        "bitflip: SA genérico de dwave-samplers sobre la QUBO completa",
    )
    p.add_argument("--reads", type=int, default=None, help="lecturas/reinicios (defecto 20 swap, 200 bitflip)")
    p.add_argument("--sweeps", type=int, default=10000, help="[bitflip] barridos por lectura")
    p.add_argument("--steps", type=int, default=None, help="[swap] pasos por lectura (defecto 2000·n^3)")
    p.add_argument("--A", type=float, default=None, help="[bitflip] peso de permutación (defecto 2 n^2)")
    p.add_argument("--B", type=float, default=1.0, help="[bitflip] peso de las sumas")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    if args.method == "swap":
        grid, info = solve_swap_annealing(
            args.n, args.reads or 20, args.steps, seed=args.seed, stop_on_success=True
        )
    else:
        grid, info = solve_simulated_annealing(
            args.n, args.reads or 200, args.sweeps, args.A, args.B, args.seed
        )
    print(f"n = {args.n}, M = {magic_constant(args.n)}, variables binarias = {info['num_vars']}")
    print(f"Método: {args.method}")
    print(
        f"Tiempo: {info['wall_time']:.2f} s | mejor energía: {info['best_energy']:.0f} | "
        f"lecturas con E=0: {info['success_rate']:.1%}"
        + (f" ({info['reads_done']} lecturas)" if "reads_done" in info else "")
    )
    if grid is None:
        print("No se alcanzó un cuadrado mágico (prueba más --reads / --steps / --sweeps).")
        return 1
    print(grid)
    print("Verificación: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
