import argparse
import time

import numpy as np
import scipy.sparse as sp

from macldlt import LDLTSolver


def make_problem(n, density, seed):
    rng = np.random.default_rng(seed)
    r = sp.random(n, n, density=density, random_state=rng, format="csc")
    m = r.T @ r + 0.1 * sp.eye(n, format="csc")
    return sp.triu(m, format="csc"), rng


def main():
    parser = argparse.ArgumentParser(
        description="Compare repeated refactor() calls against repeated reconstruction."
    )
    parser.add_argument("--n", type=int, default=2000)
    parser.add_argument("--density", type=float, default=0.002)
    parser.add_argument("--iterations", type=int, default=25)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    a_upper, rng = make_problem(args.n, args.density, args.seed)
    solver = LDLTSolver(a_upper)
    b = rng.standard_normal(args.n)

    refactor_values = []
    for _ in range(args.iterations):
        scale = 1.0 + 0.01 * rng.standard_normal(a_upper.data.shape[0])
        refactor_values.append(a_upper.data * scale)

    t0 = time.perf_counter()
    for values in refactor_values:
        solver.refactor(values)
        solver.solve(b)
    refactor_seconds = time.perf_counter() - t0

    t0 = time.perf_counter()
    for values in refactor_values:
        a_new = sp.csc_matrix((values, a_upper.indices, a_upper.indptr), shape=a_upper.shape)
        LDLTSolver(a_new).solve(b)
    rebuild_seconds = time.perf_counter() - t0

    print(f"n={args.n} density={args.density} iterations={args.iterations}")
    print(f"refactor+solve: {refactor_seconds:.3f}s")
    print(f"rebuild+solve:  {rebuild_seconds:.3f}s")
    if refactor_seconds > 0:
        print(f"speedup:        {rebuild_seconds / refactor_seconds:.2f}x")


if __name__ == "__main__":
    main()
