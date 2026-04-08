import numpy as np

import scipy.sparse as sp

import traceback



import macldlt





def full_from_triangle(U, triangle="upper"):

    if triangle == "upper":

        return U + np.triu(U, 1).T

    elif triangle == "lower":

        return U + np.tril(U, -1).T

    else:

        raise ValueError("triangle must be 'upper' or 'lower'")





def check_close(name, got, expected, tol=1e-8):

    err = np.max(np.abs(got - expected))

    ok = err < tol

    print(f"{name}: {'PASS' if ok else 'FAIL'}  max_abs_err={err:.3e}")

    return ok





def check_residual(name, A, x, b, tol=1e-8):

    r = np.linalg.norm(A @ x - b)

    ok = r < tol

    print(f"{name}: {'PASS' if ok else 'FAIL'}  residual={r:.3e}")

    return ok





def test_edge_cases():

    print("--- Running Edge Cases & Validation Tests ---")

    ok_all = True

    

    A_full = np.array([

        [4.0, 1.0],

        [1.0, 3.0]

    ], dtype=np.float64)

    

    # 1. Test CSR conversion and Lower Triangle

    print("Testing CSR matrix and lower triangle...")

    A_csr = sp.csr_matrix(np.tril(A_full))

    solver_lower = macldlt.LDLTSolver(A_csr, triangle="lower")

    b = np.array([1.0, 2.0], dtype=np.float64)

    x = solver_lower.solve(b)

    x_ref = np.linalg.solve(A_full, b)

    ok_all &= check_close("CSR / Lower Triangle", x, x_ref)

    

    # 2. Test Re-analyze (changing sparsity pattern intentionally)

    print("\nTesting analyze() and factor() for new patterns...")

    A_new_full = np.array([

        [5.0, 0.0, 2.0],

        [0.0, 4.0, 1.0],

        [2.0, 1.0, 6.0]

    ], dtype=np.float64)

    

    # FIX: Use tril since solver_lower expects lower triangle!

    A_new = sp.csc_matrix(np.tril(A_new_full)) 

    

    solver_lower.analyze(A_new) # Re-analyze for 3x3

    solver_lower.factor(A_new)  # Factor the new matrix

    b_new = np.array([1.0, 2.0, 3.0])

    x_new = solver_lower.solve(b_new)

    x_new_ref = np.linalg.solve(A_new_full, b_new)

    ok_all &= check_close("Re-analyze 3x3", x_new, x_new_ref)



    # 3. Test memory layout strictness for inplace

    print("\nTesting memory layout strictness...")

    B_2d = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]) # C-contiguous by default

    try:

        solver_lower.solve_inplace(B_2d)

        print("2D C-contiguous inplace rejection: FAIL (unexpected success)")

        ok_all = False

    except ValueError as e:

        print(f"2D C-contiguous inplace rejection: PASS ({e})")



    # 4. Test Shape Mismatches

    print("\nTesting shape mismatches...")

    try:

        solver_lower.solve(np.array([1.0, 2.0])) # solver expects size 3

        print("RHS dimension mismatch rejection: FAIL (unexpected success)")

        ok_all = False

    except ValueError as e:

        print(f"RHS dimension mismatch rejection: PASS ({e})")

        

    try:

        A_rect = sp.csc_matrix(np.ones((3, 4)))

        macldlt.LDLTSolver(A_rect)

        print("Non-square matrix rejection: FAIL (unexpected success)")

        ok_all = False

    except ValueError as e:

        print(f"Non-square matrix rejection: PASS ({e})")



    # 5. Test Singular Matrix

    print("\nTesting singular matrix handling...")

    A_singular = sp.csc_matrix(np.array([[1.0, 1.0], [1.0, 1.0]]))

    

    # FIX: LDLT successfully factors singular matrices. Check inertia instead.

    solver_singular = macldlt.LDLTSolver(A_singular)

    _, zero_pivots, _ = solver_singular.inertia()

    

    if zero_pivots > 0:

        print(f"Singular matrix detection: PASS ({zero_pivots} zero pivots detected)")

    else:

        print("Singular matrix detection: FAIL (expected >0 zero pivots)")

        ok_all = False



    print("\nEDGE CASES OVERALL:", "PASS" if ok_all else "FAIL")

    print("-" * 50, "\n")

    return ok_all





def main():

    print("--- Running Standard Tests ---")

    np.set_printoptions(precision=4, suppress=True)



    # Small symmetric indefinite matrix, stored as upper triangle

    A_full = np.array(

        [

            [4.0, 1.0, 0.0, 0.0],

            [1.0, -3.0, 2.0, 0.0],

            [0.0, 2.0, 5.0, 1.0],

            [0.0, 0.0, 1.0, -2.0],

        ],

        dtype=np.float64,

    )

    A_tri = np.triu(A_full)

    A = sp.csc_matrix(A_tri)



    print("Creating solver...")

    solver = macldlt.LDLTSolver(

        A,

        triangle="upper",

        ordering="amd",

        factorization="ldlt_tpp",   # switch to "ldlt_unpivoted" later if you want

    )

    print("Solver created")

    print("info:", solver.info())

    print("inertia:", solver.inertia())

    print()



    ok_all = True



    # Test 1: 1D solve

    b = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)

    x = solver.solve(b)

    x_ref = np.linalg.solve(A_full, b)



    ok_all &= check_close("1D solution", x, x_ref)

    ok_all &= check_residual("1D residual", A_full, x, b)

    print()



    # Test 2: 2D solve

    B = np.asfortranarray(

        np.array(

            [

                [1.0, 2.0],

                [0.0, 1.0],

                [3.0, 4.0],

                [1.0, 0.0],

            ],

            dtype=np.float64,

        )

    )

    X = solver.solve(B)

    X_ref = np.linalg.solve(A_full, B)



    ok_all &= check_close("2D solution", X, X_ref)

    ok_all &= check_residual("2D residual", A_full, X, B)

    print(f"2D output F-contiguous: {X.flags['F_CONTIGUOUS']}")

    print()



    # Test 3: inplace solve

    b_inplace = b.copy()

    solver.solve_inplace(b_inplace)

    ok_all &= check_close("1D inplace", b_inplace, x_ref)

    print()



    B_inplace = B.copy(order="F")

    solver.solve_inplace(B_inplace)

    ok_all &= check_close("2D inplace", B_inplace, X_ref)

    print()



    # Test 4: refactor with same sparsity pattern

    A2_full = A_full.copy()

    A2_full[0, 0] += 0.5

    A2_full[1, 1] -= 0.25

    A2_full[2, 2] += 0.75

    A2_full[3, 3] -= 0.1

    A2_full[0, 1] += 0.2

    A2_full[1, 0] += 0.2

    A2_full[2, 3] -= 0.15

    A2_full[3, 2] -= 0.15



    A2 = sp.csc_matrix(np.triu(A2_full))

    solver.refactor(A2)



    x2 = solver.solve(b)

    x2_ref = np.linalg.solve(A2_full, b)



    ok_all &= check_close("Refactor solution", x2, x2_ref)

    ok_all &= check_residual("Refactor residual", A2_full, x2, b)

    print("inertia after refactor:", solver.inertia())

    print()



    # Test 5: changed sparsity pattern should fail

    A_bad_full = A2_full.copy()

    A_bad_full[0, 2] = 1.0

    A_bad_full[2, 0] = 1.0

    A_bad = sp.csc_matrix(np.triu(A_bad_full))



    try:

        solver.refactor(A_bad)

        print("Pattern change rejection: FAIL (unexpected success)")

        ok_all = False

    except Exception as e:

        print("Pattern change rejection: PASS")

        print("  caught:", type(e).__name__, str(e))

    print()



    print("STANDARD TESTS OVERALL:", "PASS" if ok_all else "FAIL")

    print("-" * 50, "\n")

    return ok_all





if __name__ == "__main__":

    try:

        ok_edge = test_edge_cases()

        ok_main = main()

        

        if ok_edge and ok_main:

            print(">>> ALL TEST SUITES PASSED SUCCESSFULLY <<<")

        else:

            print(">>> SOME TESTS FAILED <<<")

            

    except Exception:

        traceback.print_exc()

        raise


