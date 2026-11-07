# Cuadrados mágicos: CP-SAT vs. QUBO

Un cuadrado mágico de orden *n* es una matriz *n×n* con los números 1…n² sin repetir, donde todas las filas, columnas y las dos diagonales suman la constante mágica **M = n(n²+1)/2**.

Este proyecto resuelve el problema con dos paradigmas y los compara:

- **Programación por restricciones** con Google OR-Tools CP-SAT.
- **QUBO** (*Quadratic Unconstrained Binary Optimization*), la formulación que usan los *annealers* cuánticos y digitales, resuelta en local con *simulated annealing*.

## Estructura

```
magic-squares/
├── magic_squares/
│   ├── verify.py        # Constante mágica y verificación de soluciones
│   ├── cp_sat.py        # Modelo CP-SAT (OR-Tools)
│   └── qubo.py          # Construcción de la QUBO + solver de simulated annealing
├── tests/
│   └── test_magic_squares.py
├── examples/
│   └── sudoku_qubo_amplify.py   # Referencia: Sudoku 16×16 como QUBO (requiere Fixstars Amplify)
├── pyproject.toml
└── requirements.txt
```

## Instalación

Requiere Python 3.10+.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Uso

```bash
python -m magic_squares.cp_sat 8                 # CP-SAT, n = 8
python -m magic_squares.cp_sat 12 --time-limit 120

python -m magic_squares.qubo 3                   # QUBO + simulated annealing
python -m magic_squares.qubo 4 --reads 500 --sweeps 50000 --A 32

pytest                                           # tests
```

## a) Programación por restricciones — `magic_squares/cp_sat.py`

- Variables enteras `x[i][j] ∈ [1, n²]`.
- `AllDifferent` sobre todas las celdas y restricciones lineales de suma para filas, columnas y diagonales.
- **Ruptura de simetrías**: cada cuadrado mágico tiene 8 variantes equivalentes (rotaciones y reflexiones). Se imponen restricciones sobre las esquinas (`tl < tr < bl` y `tl < br`) para quedarse con un único representante de cada clase.
- Detecta que n = 2 no tiene solución (`INFEASIBLE`).

Tiempos medidos (8 *workers*, tiempo real):

| n | 3 | 4 | 5 | 6 | 8 | 10 | 12 |
|---|---|---|---|---|---|---|---|
| tiempo (s) | 0,01 | 0,05 | 0,09 | 0,17 | 0,91 | 2,65 | 32,1 |

## b) QUBO — `magic_squares/qubo.py`

Codificación *one-hot*: `x[i,j,k] = 1` si la celda (i,j) contiene el valor k+1, lo que da n⁴ variables binarias. La energía es una suma de penalizaciones al cuadrado:

```
E(x) = A · Σ_celdas (Σ_k x[i,j,k] − 1)²                (a) un valor por celda
     + A · Σ_k      (Σ_{i,j} x[i,j,k] − 1)²            (b) cada valor aparece una sola vez
     + B · Σ_líneas (Σ_{(i,j)∈L} Σ_k (k+1)·x[i,j,k] − M)²   (c) filas, columnas y diagonales
```

Como todas las penalizaciones son cuadrados, **E(x) ≥ 0, y E(x) = 0 si y solo si x es un cuadrado mágico**, para cualquier A, B > 0. La matriz Q se construye con NumPy expandiendo cada cuadrado (usando x² = x) y se resuelve con el `SimulatedAnnealingSampler` de `dwave-samplers`, que es local y no necesita licencia ni token.

### Fallos corregidos respecto a la primera versión

1. **Faltaba la restricción (b).** Solo se prohibía repetir un valor dentro de una misma fila o columna, de modo que el mismo número podía aparecer en celdas de filas y columnas distintas. Con esta restricción, (a) y (b) obligan a que el resultado sea una permutación de 1…n², y sobran las restricciones de fila/columna.
2. Un término `−Σ x` premiaba activar variables, y la restricción de celda estaba sumada dos veces.
3. El decodificador fallaba si una celda acababa sin valor. Ahora devuelve `None` y la solución se valida siempre con `check_magic_square`.
4. El token de Amplify estaba escrito en el código. Ahora no hay dependencia de Amplify.

### Cómo se comprueba la formulación (tests)

- La energía calculada con la matriz, `xᵀQx + offset`, coincide con la evaluada directamente desde la fórmula para cientos de vectores binarios aleatorios (n = 2, 3, 4).
- Se recorren las **9! = 362 880 permutaciones** de orden 3: la energía es 0 exactamente en los 8 cuadrados mágicos y positiva en el resto.
- La energía es 0 para cuadrados mágicos conocidos (Lo Shu, Durero) y positiva si hay valores repetidos.
- CP-SAT encuentra soluciones válidas para n = 1, 3, 4, 5, 6, y devuelve `INFEASIBLE` para n = 2.

### Resultados del *annealing* y conclusión

| n | variables | resultado (200 lecturas × 10⁴ barridos) |
|---|---|---|
| 3 | 81 | resuelto: ≈1–3 % de las lecturas llegan a E = 0 (≈4 s) |
| 4 | 256 | no resuelto: mejor energía ≈ 10, incluso con 5·10⁴ barridos |

El peso A no cambia cuáles son las soluciones, pero sí la dificultad del *annealing*. Con A grande (≈ n⁴) el paisaje de energía se vuelve muy abrupto y no se llega a E = 0 ni siquiera para n = 3. Por defecto se usa A = 2n², que funcionó bien en las pruebas.

La diferencia con CP-SAT se explica por la codificación. Con *one-hot*, intercambiar dos valores entre celdas exige cambiar 4 bits a la vez, y todos los estados intermedios incumplen (a) o (b). Un *annealer* que solo cambia un bit cada vez apenas puede moverse entre soluciones válidas. CP-SAT, en cambio, propaga las restricciones de forma global y resuelve n = 10 en menos de 3 s.

## Autor

**Gonzalo Obejo** — [GitHub](https://github.com/GonzaloObejo)
