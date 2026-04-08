# macldlt v0.0.0

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

Perform symbolic analysis and numeric factorization of `A`. The solver is
immediately ready to call `solve()`.

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
- For a new sparsity pattern, create a new solver.

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

### `solver.refactor(values)`

Reuse the existing symbolic analysis and numeric factorization workspace for
new values with the **same sparsity pattern**. This calls Accelerate's
`SparseRefactor`, which can be faster than a full `factor()`.

**Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `values` | `numpy.ndarray` | 1D float64 array of nonzero values in CSC storage order, matching the original sparsity pattern. Length must equal the number of stored nonzeros (i.e., `A.data` from the original scipy sparse matrix). Non-float64 arrays are cast automatically. |

```python
# In a tight loop, just pass new values
for new_values in value_generator:
    solver.refactor(new_values)
    x = solver.solve(b)
```

---

### `solver.solve(rhs, inplace=False)`

Solve `Ax = rhs`.

By default, allocates and returns a new NumPy array. With `inplace=True`,
overwrites `rhs` in place and returns it (avoids allocation).

**Parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `rhs` | `numpy.ndarray` | *(required)* | 1D array of length `n`, or 2D array of shape `(n, k)`. |
| `inplace` | `bool` | `False` | If `True`, solve in place. Requires writeable C-contiguous float64 (1D) or F-contiguous float64 (2D). |

**Returns:** `numpy.ndarray` — the solution, same shape as `rhs`.

```python
# Allocating solve
x = solver.solve(b)

# In-place solve (no allocation)
solver.solve(b, inplace=True)
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
solver = LDLTSolver(A) ──► solve()              # one-shot usage
              │
              └── refactor(values) ──► solve()   # same pattern, new values

solver = LDLTSolver(A_new) ──► solve()           # new sparsity pattern
```

1. **One-shot solve:** Pass `A` to the constructor, then call `solve()`.
2. **Repeated solves, same pattern:** Call `refactor(new_vals)` then `solve()`.
   The symbolic analysis from the constructor is reused.
3. **New sparsity pattern:** Create a new `LDLTSolver` with the new matrix.

## Benchmarking

The main performance win comes from reusing symbolic analysis when only the
numeric values change. A small benchmark script is included:

```bash
python benchmarks/refactor_benchmark.py --n 2000 --density 0.002 --iterations 25
```

This compares repeated `refactor()` calls against rebuilding a fresh
`LDLTSolver` each iteration for the same sparsity pattern.

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
