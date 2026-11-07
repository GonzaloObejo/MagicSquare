# Cuadrados mágicos: CP-SAT vs. QUBO

Un cuadrado mágico de orden *n* es una matriz *n×n* con los números 1…n² sin repetir, donde todas las filas, columnas y las dos diagonales suman la constante mágica **M = n(n²+1)/2**.

Este proyecto compara dos paradigmas para resolverlo: **programación por restricciones** (Google OR-Tools CP-SAT) y **optimización binaria cuadrática sin restricciones** (QUBO) resuelta con un *annealer* en la nube (Fixstars Amplify AE).

## Estructura

```
magic-squares/
├── cp_sat/
│   └── magic_square_cp_sat.py      # Modelo de programación por restricciones (OR-Tools)
├── qubo/
│   ├── magic_square_qubo.py        # Formulación QUBO del cuadrado mágico (Amplify)
│   └── sudoku_qubo_example.py      # Ejemplo de referencia: Sudoku 16x16 como QUBO
├── requirements.txt
└── README.md
```

## a) Programación por restricciones — `cp_sat/magic_square_cp_sat.py`

- Variables enteras `x[i][j] ∈ [1, n²]`.
- Restricción `AllDifferent` sobre todas las celdas.
- Restricciones lineales de suma para filas, columnas y las dos diagonales.
- Mide el tiempo de resolución para el *n* indicado por el usuario.

## b) QUBO — `qubo/magic_square_qubo.py`

- Codificación *one-hot*: variable binaria `m[i,j,k] = 1` si la celda (i,j) contiene el valor k+1 (n·n·n² variables).
- Cada restricción de suma se convierte en una penalización cuadrática `(Σ valor − M)²`, desarrollada en términos lineales y cruzados.
- Penalizaciones adicionales (con peso n³) para: un único valor por celda y no repetir valores en filas/columnas.
- Resuelto con **Fixstars Amplify AE**; el resultado se verifica con `check_magic_square`.

`qubo/sudoku_qubo_example.py` es el ejemplo de referencia (Sudoku 16×16 como QUBO, de la documentación de Amplify) usado como punto de partida para la formulación.

## Instalación y ejecución

Requiere Python 3.10+.

```bash
pip install -r requirements.txt

# CP-SAT (pide n por consola)
python cp_sat/magic_square_cp_sat.py

# QUBO (necesita un token de https://amplify.fixstars.com)
export AMPLIFY_TOKEN="<tu_token>"          # Windows PowerShell: $env:AMPLIFY_TOKEN="<tu_token>"
python qubo/magic_square_qubo.py
```

## Autor

**Gonzalo Obejo** — [GitHub](https://github.com/GonzaloObejo)
