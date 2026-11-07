import time
from ortools.sat.python import cp_model

def solve_magic_square(n):
    # Start timing
    start_time = time.time()
    
    # Calculate the magic constant (sum for each row, column, and diagonal)
    magic_constant = n * (n * n + 1) // 2
    
    # Create the model
    model = cp_model.CpModel()

    # Create variables: a 2D array of integers in the range [1, n^2]
    cells = [[model.NewIntVar(1, n * n, f'cell_{i}_{j}') for j in range(n)] for i in range(n)]

    # Flatten the cells for convenience
    all_cells = [cells[i][j] for i in range(n) for j in range(n)]

    # All numbers must be unique
    model.AddAllDifferent(all_cells)

    # Constraints for rows
    for i in range(n):
        model.Add(sum(cells[i][j] for j in range(n)) == magic_constant)

    # Constraints for columns
    for j in range(n):
        model.Add(sum(cells[i][j] for i in range(n)) == magic_constant)

    # Main diagonal
    model.Add(sum(cells[i][i] for i in range(n)) == magic_constant)

    # Secondary diagonal
    model.Add(sum(cells[i][n - 1 - i] for i in range(n)) == magic_constant)

    # Create solver and solve the model
    solver = cp_model.CpSolver()
    status = solver.Solve(model)

    # End timing
    end_time = time.time()

    # Print the solution if found
    if status == cp_model.FEASIBLE or status == cp_model.OPTIMAL:
        print(f"Magic constant: {magic_constant}")
        for i in range(n):
            row_values = [solver.Value(cells[i][j]) for j in range(n)]
            print(row_values)
    else:
        print("No solution found.")

    # Print the time taken
    print(f"Solved in {end_time - start_time:.4f} seconds.")

if __name__ == "__main__":
    n = int(input("Enter the size of the magic square (n): "))
    solve_magic_square(n)
