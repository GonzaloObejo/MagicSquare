"""Utilidades comunes: constante mágica y verificación de soluciones."""

from __future__ import annotations

import numpy as np


def magic_constant(n: int) -> int:
    """Constante mágica M = n(n^2 + 1) / 2."""
    return n * (n * n + 1) // 2


def check_magic_square(matrix) -> dict:
    """Devuelve un informe detallado de si `matrix` es un cuadrado mágico normal.

    Comprueba que contiene exactamente los números 1..n^2 y que todas las filas,
    columnas y ambas diagonales suman la constante mágica.
    """
    a = np.asarray(matrix)
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError(f"Se esperaba una matriz cuadrada, recibido shape={a.shape}")
    n = a.shape[0]
    m = magic_constant(n)

    row_sums = a.sum(axis=1)
    col_sums = a.sum(axis=0)
    diag_sums = [int(np.trace(a)), int(np.trace(np.fliplr(a)))]
    is_permutation = sorted(a.flatten().tolist()) == list(range(1, n * n + 1))

    row_mismatch = np.flatnonzero(row_sums != m).tolist()
    col_mismatch = np.flatnonzero(col_sums != m).tolist()
    diag_mismatch = [d for d, s in zip(("main", "anti"), diag_sums) if s != m]

    return {
        "is_magic": is_permutation and not row_mismatch and not col_mismatch and not diag_mismatch,
        "is_permutation": is_permutation,
        "magic_constant": m,
        "row_sums": row_sums.tolist(),
        "col_sums": col_sums.tolist(),
        "diag_sums": diag_sums,
        "row_mismatch": row_mismatch,
        "col_mismatch": col_mismatch,
        "diag_mismatch": diag_mismatch,
    }


def is_magic_square(matrix) -> bool:
    return check_magic_square(matrix)["is_magic"]
