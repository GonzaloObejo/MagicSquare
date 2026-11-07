"""Ejemplo de referencia (documentación de Fixstars Amplify): Sudoku 16x16 como QUBO.

Sirvió como punto de partida para la formulación del cuadrado mágico. Requiere
`pip install amplify` y un token de Amplify en la variable AMPLIFY_TOKEN;
no forma parte de los tests.
"""
import os
from amplify import VariableGenerator, solve, one_hot, FixstarsClient
import numpy as np
    
n = 4  # block size
N = n * n  # total number of blocks
    
# Initial configuration for n = 4 (N = 16)
# Reference: https://www.free-sudoku-puzzle.com/puzzle_fours/solve/3/238
initial = np.array(
    [
        [0, 0, 0, 7, 0, 0, 0, 1, 5, 0, 3, 16, 4, 0, 15, 0],
        [0, 11, 0, 0, 0, 0, 5, 0, 0, 2, 12, 6, 0, 0, 7, 14],
        [4, 0, 0, 0, 7, 8, 9, 0, 11, 0, 1, 15, 0, 0, 10, 0],
        [10, 0, 0, 0, 0, 0, 0, 15, 13, 0, 9, 7, 8, 0, 0, 1],
        [13, 0, 0, 16, 15, 0, 4, 9, 0, 0, 14, 0, 11, 0, 1, 0],
        [8, 0, 5, 0, 0, 0, 10, 0, 0, 0, 0, 0, 15, 0, 0, 0],
        [0, 0, 0, 11, 0, 0, 0, 8, 16, 7, 0, 9, 0, 0, 0, 0],
        [0, 0, 0, 14, 0, 0, 3, 0, 4, 0, 0, 5, 13, 0, 0, 0],
        [0, 0, 0, 0, 3, 0, 0, 14, 0, 0, 4, 0, 9, 12, 8, 15],
        [0, 0, 0, 0, 0, 1, 7, 10, 0, 15, 8, 11, 0, 0, 0, 0],
        [0, 0, 0, 0, 11, 12, 0, 0, 0, 0, 16, 0, 3, 5, 0, 0],
        [0, 0, 0, 0, 0, 16, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 6, 16, 0, 15, 1, 5, 0, 14, 2, 0, 0],
        [0, 0, 0, 3, 0, 0, 0, 0, 9, 0, 0, 14, 0, 1, 0, 4],
        [2, 0, 12, 0, 0, 0, 0, 0, 0, 16, 13, 0, 6, 0, 3, 5],
        [1, 0, 0, 0, 0, 15, 0, 0, 2, 11, 6, 12, 7, 9, 0, 10],
    ]
)
    
    
# Display a sudoku grid with numbers
def print_sudoku(sudoku):
    width = len(str(N))
    for i in range(len(sudoku)):
        line = ""
        if i % n == 0 and i != 0:
            print("-" * ((width + 1) * n * n + 2 * (n - 1)))
        for j in range(len(sudoku[i])):
            if j % n == 0 and j != 0:
                line += "| "
            line += str(sudoku[i][j]).rjust(width) + " "
        print(line)
    
    
q = VariableGenerator().array("Binary", N, N, N)
    
for i, j in zip(*np.where(initial != 0)):
    k = initial[i, j] - 1
    
    q[i, :, k] = 0  # Constraint (a)
    q[:, j, k] = 0  # Constraint (b)
    q[i, j, :] = 0  # Constraint (d)
    for m in range(N):
        q[(n * (i // n) + m // n), (n * (j // n) + m % n), k] = 0  # Constraint (c)
    
    q[i, j, k] = 1  # Substitute 1
    
# (a): Constraints that each row cannot contain the same number
row_constraints = one_hot(q, axis=1)
    
# (b): Constraints that each column cannot contain the same number
col_constraints = one_hot(q, axis=0)
    
# (d): Constraints that only one number can be in a cell
num_constraints = one_hot(q, axis=2)
    
# (c): Constraints that a nxn subgrid cannot contain the same number
block_constraints = [
    one_hot(sum([q[i + m // n, j + m % n, k] for m in range(N)]))
    for i in range(0, N, n)
    for j in range(0, N, n)
    for k in range(N)
]
    
constraints = (
    row_constraints + col_constraints + num_constraints + sum(block_constraints)
)
    
client = FixstarsClient()
client.parameters.timeout = 10000  # timeout is 10 seconds
client.token = os.environ["AMPLIFY_TOKEN"]  # export AMPLIFY_TOKEN=<tu_token>
    
# Solve the problem
result = solve(constraints, client)
if len(result) == 0:
    raise RuntimeError("Some constraints are unsatisfied.")
    
values = result.best.values
    
q_values = q.evaluate(values)
answer = np.array([np.where(np.array(q_values[i]) != 0)[1] + 1 for i in range(N)])
print_sudoku(answer)