import os
import numpy as np
from amplify import VariableGenerator, solve, one_hot, FixstarsClient
from amplify import sum as sum_poly
from datetime import timedelta

# Step 1: Define Problem Parameters
n = 5  # Size of the magic square (e.g., 3x3)
N = n**2  # Total number of cells
M = n * (n**2 + 1) // 2  # Magic constant

# Step 2: Create a 3D Binary Matrix for One-Hot Encoding
gen = VariableGenerator()
m = gen.array("Binary", n, n, N)  # 3D binary matrix (rows, columns, values)
num_constraint=0
for k in range(N):
    
    num_constraint += sum_poly(
        -1 * m[i, j, k]
        for j in range(n)
        for i in range(n)
)
num_constraint += sum_poly(
        2 * m[i, j, k1] * m[i, j, k2]
        for j in range(n)
        for k1 in range(N)
        for i in range(n)
        for k2 in range(N)
        if (k2 > k1) )
global_constraints=0
global_constraints += sum_poly(
        m[i, j, k1] * m[i, j, k2]
        for j in range(n)
        for k1 in range(N)
        for i in range(n)
        for k2 in range(N)
        if (k2 > k1) )
global_constraints += sum_poly(
        m[i, j1, k] * m[i, j2, k]
        for j1 in range(n)
        for k in range(N)
        for i in range(n)
        for j2 in range(n)
        if (j2 > j1) )
global_constraints += sum_poly(
        m[i1, j, k] * m[i2, j, k]
        for i1 in range(n)
        for k in range(N)
        for j in range(n)
        for i2 in range(n)
        if (i2 > i1) )
global_constraints += sum_poly(
        -1 * m[i, j, k]
        for i in range(n)
        for k in range(N)
        for j in range(n))


ms_penalty = 0

#-------------------------------
# 1) Row sums == M
#-------------------------------
for i in range(n):
    #
    # Single-variable (diagonal) terms:  sum_k (k+1)^2 * m[i,j,k]
    #
    ms_penalty += sum_poly(
        (k + 1) * (k + 1) * m[i, j, k]
        for j in range(n)
        for k in range(N)
    )
    #
    # Pairwise (cross) terms:  2 * sum_{(j,k) < (j',k')} (k+1)(k'+1) * m[i,j,k]*m[i,j',k']
    #
    ms_penalty += sum_poly(
        2 * (k1 + 1) * (k2 + 1) * m[i, j1, k1] * m[i, j2, k2]
        for j1 in range(n)
        for k1 in range(N)
        for j2 in range(n)
        for k2 in range(N)
        if (j2 > j1) or (j2 == j1 and k2 > k1)
    )
    #
    # Linear term in m[i,j,k]:  -2*M * sum_{j,k} (k+1) * m[i,j,k]
    #
    ms_penalty += sum_poly(
        -2 * M * (k + 1) * m[i, j, k]
        for j in range(n)
        for k in range(N)
    )

#-------------------------------
# 2) Column sums == M
#-------------------------------
for j in range(n):
    # Single-variable terms
    ms_penalty += sum_poly(
        (k + 1) * (k + 1) * m[i, j, k]
        for i in range(n)
        for k in range(N)
    )
    # Pairwise cross terms
    ms_penalty += sum_poly(
        2 * (k1 + 1) * (k2 + 1) * m[i1, j, k1] * m[i2, j, k2]
        for i1 in range(n)
        for k1 in range(N)
        for i2 in range(n)
        for k2 in range(N)
        if (i2 > i1) or (i2 == i1 and k2 > k1)
    )
    # Linear terms
    ms_penalty += sum_poly(
        -2 * M * (k + 1) * m[i, j, k]
        for i in range(n)
        for k in range(N)
    )

#-------------------------------
# 3) Main diagonal 
#-------------------------------
# Single-variable terms
ms_penalty += sum_poly(
    (k + 1) * (k + 1) * m[i, i, k]
    for i in range(n)
    for k in range(N)
)
# Pairwise cross terms
ms_penalty += sum_poly(
    2 * (k1 + 1) * (k2 + 1) * m[i1, i1, k1] * m[i2, i2, k2]
    for i1 in range(n)
    for k1 in range(N)
    for i2 in range(n)
    for k2 in range(N)
    if (i2 > i1) or (i2 == i1 and k2 > k1)
)
# Linear terms
ms_penalty += sum_poly(
    -2 * M * (k + 1) * m[i, i, k]
    for i in range(n)
    for k in range(N)
)

#-------------------------------
# 4) Second diagonal 
#-------------------------------
ms_penalty += sum_poly(
    (k + 1) * (k + 1) * m[i, n - 1 - i, k]
    for i in range(n)
    for k in range(N)
)
ms_penalty += sum_poly(
    2 * (k1 + 1) * (k2 + 1) * m[i1, n - 1 - i1, k1] * m[i2, n - 1 - i2, k2]
    for i1 in range(n)
    for k1 in range(N)
    for i2 in range(n)
    for k2 in range(N)
    if (i2 > i1) or (i2 == i1 and k2 > k1)
)
ms_penalty += sum_poly(
    -2 * M * (k + 1) * m[i, n - 1 - i, k]
    for i in range(n)
    for k in range(N)
)
# Combine all constraints into one objective
constraints = (
    ms_penalty
    + n*n*n*global_constraints
    + n*n*n*num_constraint
)

# Configure the client
client = FixstarsClient()
client.parameters.timeout = timedelta(milliseconds=5000)
client.token = os.environ["AMPLIFY_TOKEN"]  # export AMPLIFY_TOKEN=<tu_token>

# Solve the problem
result = solve(constraints, client)
if len(result) == 0:
    raise RuntimeError("Some constraints are unsatisfied.")


# Retrieve solution
values = result.best.values
m_values = m.evaluate(values)

# Construct the magic square
answer = np.zeros((n, n), dtype=int)
for i in range(n):
    for j in range(n):
        (k_idx,) = np.where(m_values[i, j, :] == 1)
        answer[i, j] = k_idx[0] + 1

# Print the result
print("Magic Constant =", M)
print("Magic Square:")
print(answer)

# Use the execution_time provided by the Result object
print(f"\nTime taken to solve (solver execution time): {result.execution_time} seconds")
def check_magic_square(matrix):
    matrix = np.array(matrix)
    size = matrix.shape[0]
    expected_sum = size * (size**2 + 1) // 2  # Formula for magic constant
    
    # Calculate row, column, and diagonal sums
    row_sums = np.sum(matrix, axis=1)
    col_sums = np.sum(matrix, axis=0)
    diag_sum1 = np.sum(np.diag(matrix))
    diag_sum2 = np.sum(np.diag(np.fliplr(matrix)))
    
    # Check if all sums match the expected sum
    row_mismatch = np.where(row_sums != expected_sum)[0].tolist()
    col_mismatch = np.where(col_sums != expected_sum)[0].tolist()
    diag_mismatch = diag_sum1 != expected_sum or diag_sum2 != expected_sum
    
    # Result
    result = {
        "Magic Square": len(row_mismatch) == 0 and len(col_mismatch) == 0 and not diag_mismatch,
        "Expected Sum": expected_sum,
        "Row Sums": row_sums.tolist(),
        "Column Sums": col_sums.tolist(),
        "Diagonal Sums": [diag_sum1, diag_sum2],
        "Row Mismatches": row_mismatch,
        "Column Mismatches": col_mismatch,
        "Diagonal Mismatch": diag_mismatch
    }
    
    return result
print(check_magic_square(answer))
