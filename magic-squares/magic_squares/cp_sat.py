"""Cuadrado mágico como problema de programación por restricciones (OR-Tools CP-SAT).

Uso:
    python -m magic_squares.cp_sat 5
    python -m magic_squares.cp_sat 8 --time-limit 30 --workers 8
"""

from __future__ import annotations

import argparse
import time

import numpy as np
from ortools.sat.python import cp_model

from magic_squares.verify import check_magic_square, magic_constant


def build_model(n: int, symmetry_breaking: bool = True):
    """Construye el modelo CP-SAT. Devuelve (model, cells)."""
    if n < 1:
        raise ValueError("n debe ser >= 1")
    m = magic_constant(n)
    model = cp_model.CpModel()
    cells = [[model.NewIntVar(1, n * n, f"x_{i}_{j}") for j in range(n)] for i in range(n)]

    model.AddAllDifferent([c for row in cells for c in row])
    for i in range(n):
        model.Add(sum(cells[i]) == m)
    for j in range(n):
        model.Add(sum(cells[i][j] for i in range(n)) == m)
    model.Add(sum(cells[i][i] for i in range(n)) == m)
    model.Add(sum(cells[i][n - 1 - i] for i in range(n)) == m)

    # Las 8 simetrías del cuadrado (rotaciones y reflexiones) transforman una
    # solución en otra. Fijamos la esquina superior izquierda como la menor de
    # las cuatro esquinas y la superior derecha menor que la inferior izquierda:
    # queda exactamente un representante de cada clase de equivalencia.
    if symmetry_breaking and n >= 2:
        tl, tr = cells[0][0], cells[0][n - 1]
        bl, br = cells[n - 1][0], cells[n - 1][n - 1]
        model.Add(tl < tr)
        model.Add(tl < bl)
        model.Add(tl < br)
        model.Add(tr < bl)
    return model, cells


def solve_magic_square(
    n: int,
    time_limit: float | None = 60.0,
    workers: int = 8,
    seed: int = 0,
    symmetry_breaking: bool = True,
):
    """Resuelve el cuadrado mágico de orden n.

    Devuelve (grid, info): grid es un np.ndarray n×n o None si no hay solución
    (n = 2) o se agota el tiempo; info contiene estado y tiempos.
    """
    model, cells = build_model(n, symmetry_breaking)
    solver = cp_model.CpSolver()
    if time_limit is not None:
        solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.num_workers = workers
    solver.parameters.random_seed = seed

    t0 = time.perf_counter()
    status = solver.Solve(model)
    elapsed = time.perf_counter() - t0

    info = {
        "status": solver.StatusName(status),
        "wall_time": elapsed,
        "solver_time": solver.WallTime(),
    }
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        grid = np.array([[solver.Value(c) for c in row] for row in cells])
        return grid, info
    return None, info


def main(argv=None):
    p = argparse.ArgumentParser(description="Cuadrado mágico con OR-Tools CP-SAT")
    p.add_argument("n", type=int, help="orden del cuadrado")
    p.add_argument("--time-limit", type=float, default=60.0, help="segundos (por defecto 60)")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-symmetry-breaking", action="store_true")
    args = p.parse_args(argv)

    grid, info = solve_magic_square(
        args.n, args.time_limit, args.workers, args.seed, not args.no_symmetry_breaking
    )
    print(f"n = {args.n}, constante mágica M = {magic_constant(args.n)}")
    print(f"Estado: {info['status']}  ({info['wall_time']:.3f} s)")
    if grid is None:
        print("No se encontró solución." + (" (No existe para n = 2.)" if args.n == 2 else ""))
        return 1
    print(grid)
    report = check_magic_square(grid)
    print("Verificación:", "OK" if report["is_magic"] else f"FALLO {report}")
    return 0 if report["is_magic"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
