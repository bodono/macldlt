# macldlt

Python wrapper for Apple Accelerate's sparse LDL^T factorization.

`macldlt` exposes the symmetric indefinite sparse solver from Apple's
[Accelerate framework](https://developer.apple.com/documentation/accelerate/sparse_solvers)
to Python via [pybind11](https://github.com/pybind/pybind11). It accepts
SciPy sparse matrices and NumPy arrays directly, with no manual conversion
needed.

**macOS only** — requires macOS 13.0+ for full functionality (including
`SparseGetInertia`).

## Installation

```bash
pip install macldlt
```

Or install from source:

```bash
git clone https://github.com/bodono/macldlt.git
cd macldlt
pip install -e ".[test]"
```

Building from source requires:
- macOS 13.0+
- A C++17 compiler (Xcode command-line tools)
- Python >= 3.10
- pybind11 >= 2.12

## Quick start

```python
import numpy as np
import scipy.sparse as sp
from macldlt import LDLTSolver

# Build a symmetric positive-definite matrix
A = sp.csc_matrix(np.array([
    [ 4.0, 1.0],
    [ 1.0, 3.0],
]))

solver = LDLTSolver(A)
b = np.array([1.0, 2.0])
x = solver.solve(b)
print(x)  # [0.09090909 0.63636364]
```

## API reference

### `LDLTSolver(A, triangle="upper", ordering="amd", factorization="ldlt")`

Create a solver by performing symbolic analysis and numeric factorization of
`A`.

**Parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `A` | scipy sparse matrix | *(required)* | Square symmetric sparse matrix (CSC, CSR, or COO). CSR and COO are converted to CSC internally. |
| `triangle` | `str` | `"upper"` | Which triangle of `A` is stored: `"upper"` or `"lower"`. |
| `ordering` | `str` | `"amd"` | Fill-reducing ordering for symbolic analysis: `"default"`, `"amd"`, `"metis"`, or `"colamd"`. |
| `factorization` | `str` | `"ldlt"` | Factorization variant: `"ldlt"`, `"ldlt_tpp"`, `"ldlt_sbk"`, or `"ldlt_unpivoted"`. |

**Notes:**

- Symmetry is assumed but **not checked**. Only the specified triangle is read;
  the other triangle is ignored. If you pass a full symmetric matrix, set
  `triangle` to whichever triangle contains the data you want used.
- The solver is **not thread-safe**. Do not call methods concurrently on the
  same instance from multiple threads.

**Example:**

```python
# Upper triangle of a 3x3 symmetric matrix
A_upper = sp.csc_matrix(np.array([
    [4.0, 1.0, 0.0],
    [0.0, 3.0, 2.0],
    [0.0, 0.0, 5.0],
]))
solver = LDLTSolver(A_upper, triangle="upper")
```

---

### `solver.analyze(A)`

Redo symbolic analysis for a **new sparsity pattern**. This discards both the
existing symbolic and numeric factorizations and rebuilds them. You must call
`factor()` after this before solving.

Call this when the nonzero structure of your matrix changes (e.g., entries
appear or disappear). If only the numerical values change, use `refactor()` or
`refactor_values()` instead.

```python
A_new_pattern = sp.csc_matrix(...)  # different sparsity structure
solver.analyze(A_new_pattern)
solver.factor(A_new_pattern)
x = solver.solve(b)
```

---

### `solver.factor(A)`

Compute a fresh numeric factorization for the current sparsity pattern. The
matrix `A` must have the same sparsity pattern as the matrix used in the most
recent `analyze()` call (or the constructor).

Use this when you want a clean numeric factorization, discarding any previous
one.

```python
solver.factor(A_updated)
x = solver.solve(b)
```

---

### `solver.refactor(A)`

Reuse the existing symbolic analysis **and** numeric factorization workspace
for a new matrix with **identical sparsity pattern**. This calls Accelerate's
`SparseRefactor`, which can be faster than a full `factor()`.

The matrix `A` is a SciPy sparse matrix. Its sparsity pattern must exactly
match the pattern from the most recent `analyze()` or constructor call.

```python
# Solve for many matrices with the same pattern
for values in value_sequence:
    A_new = sp.csc_matrix((values, indices, indptr), shape=(n, n))
    solver.refactor(A_new)
    x = solver.solve(b)
```

---

### `solver.refactor_values(values)`

Fast-path refactor using only the flat nonzero-values array. This skips all
SciPy matrix parsing, format conversion, and sparsity pattern validation, making
it the fastest way to update the numeric factorization.

**Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `values` | `numpy.ndarray` | 1D float64 array of nonzero values in CSC storage order, matching the original sparsity pattern. Length must equal the number of stored nonzeros. Non-float64 arrays are cast automatically. |

**Warning:** No pattern validation is performed. Passing values from a matrix
with a different sparsity pattern is undefined behavior.

```python
# Extract the pattern once
A = sp.csc_matrix(...)
solver = LDLTSolver(A)
indices, indptr = A.indices.copy(), A.indptr.copy()

# In a tight loop, just pass new values
for new_values in value_generator:
    solver.refactor_values(new_values)
    x = solver.solve(b)
```

---

### `solver.solve(rhs)`

Solve `Ax = rhs` and return a **new** NumPy array containing `x`.

**Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `rhs` | `numpy.ndarray` | 1D array of length `n` for a single right-hand side, or 2D array of shape `(n, k)` for `k` right-hand sides. |

**Returns:** `numpy.ndarray` — same shape as `rhs`.

```python
# Single right-hand side
x = solver.solve(b)

# Multiple right-hand sides (n x k)
B = np.column_stack([b1, b2, b3])
X = solver.solve(B)
```

---

### `solver.solve_inplace(rhs_and_solution)`

Solve `Ax = b` in place, overwriting `rhs_and_solution` with the result. This
avoids allocating a new array.

**Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `rhs_and_solution` | `numpy.ndarray` | Writeable array to overwrite. Must be C-contiguous float64 for 1D, or F-contiguous float64 for 2D. |

```python
b = np.array([1.0, 2.0, 3.0])
solver.solve_inplace(b)
# b now contains the solution
```

---

### `solver.inertia()`

Return the inertia of the factored matrix as a tuple
`(num_negative, num_zero, num_positive)`, where:

- `num_negative` — number of negative pivots
- `num_zero` — number of zero pivots
- `num_positive` — number of positive pivots

The sum `num_negative + num_zero + num_positive` equals `n`.

```python
neg, zero, pos = solver.inertia()
if zero > 0:
    print("Matrix is singular")
if neg == 0 and zero == 0:
    print("Matrix is positive definite")
```

---

### `solver.info()`

Return a dictionary with solver state and workspace information.

**Returns:** `dict` with keys:

| Key | Description |
|---|---|
| `n` | Matrix dimension |
| `symbolic_status` | Status of symbolic factorization (e.g., `"SparseStatusOK"`) |
| `numeric_status` | Status of numeric factorization (e.g., `"SparseStatusOK"`) |
| `factor_workspace_allocated_bytes` | Bytes allocated for factorization workspace |
| `solve_workspace_allocated_bytes` | Bytes allocated for solve workspace |
| `factor_workspace_required_bytes` | Bytes required for factorization (if symbolic analysis done) |
| `symbolic_workspace_double` | Symbolic workspace size reported by Accelerate |
| `factor_size_double` | Factor size reported by Accelerate |
| `solve_workspace_required_bytes_1rhs` | Solve workspace bytes for a single RHS (if numeric factorization done) |
| `solve_workspace_static` | Static solve workspace component |
| `solve_workspace_per_rhs` | Per-RHS solve workspace component |

---

### Properties

| Property | Type | Description |
|---|---|---|
| `solver.n` | `int` | Matrix dimension. |
| `solver.symbolic_status` | `str` | Symbolic factorization status string. |
| `solver.numeric_status` | `str` | Numeric factorization status string. |

## Typical workflow

```
Constructor ──► solve()          # one-shot usage
     │
     ├── refactor() ──► solve()  # same pattern, new values (from scipy matrix)
     │
     ├── refactor_values() ──► solve()  # same pattern, fastest path
     │
     └── analyze() ──► factor() ──► solve()  # new sparsity pattern
```

1. **One-shot solve:** Pass `A` to the constructor, then call `solve()`.
2. **Repeated solves, same pattern:** Call `refactor(A_new)` or
   `refactor_values(new_vals)` then `solve()`. The symbolic analysis from the
   constructor is reused.
3. **New sparsity pattern:** Call `analyze(A_new)` then `factor(A_new)` then
   `solve()`.

## Triangle conventions

Accelerate's symmetric solver reads only one triangle of the matrix. You
specify which triangle is stored via the `triangle` parameter:

```python
import numpy as np
import scipy.sparse as sp

A_full = np.array([
    [4.0, 1.0],
    [1.0, 3.0],
])

# If you store the upper triangle:
A_upper = sp.csc_matrix(np.triu(A_full))
solver = LDLTSolver(A_upper, triangle="upper")

# If you store the lower triangle:
A_lower = sp.csc_matrix(np.tril(A_full))
solver = LDLTSolver(A_lower, triangle="lower")
```

If you pass a full symmetric matrix with `triangle="upper"`, only the upper
triangle entries are used — the lower triangle is ignored (and vice versa).

## License

MIT
