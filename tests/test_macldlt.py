import numpy as np
import numpy.testing as npt
import pytest
import scipy.sparse as sp

import macldlt


def _full_from_upper(U):
    """Reconstruct full symmetric matrix from upper triangle."""
    return U + np.triu(U, 1).T


def _full_from_lower(L):
    """Reconstruct full symmetric matrix from lower triangle."""
    return L + np.tril(L, -1).T


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def A4_upper():
    """4x4 symmetric indefinite matrix, upper triangle stored."""
    A_full = np.array([
        [4.0, 1.0, 0.0, 0.0],
        [1.0, -3.0, 2.0, 0.0],
        [0.0, 2.0, 5.0, 1.0],
        [0.0, 0.0, 1.0, -2.0],
    ])
    A_tri = np.triu(A_full)
    return sp.csc_matrix(A_tri), A_full


@pytest.fixture
def solver4(A4_upper):
    """Solver for the 4x4 fixture."""
    A_sp, _ = A4_upper
    return macldlt.LDLTSolver(A_sp, triangle="upper", factorization="ldlt_tpp")


# ---------------------------------------------------------------------------
# Basic solve
# ---------------------------------------------------------------------------

class TestSolve1D:
    def test_solve_1d(self, solver4, A4_upper):
        _, A_full = A4_upper
        b = np.array([1.0, 2.0, 3.0, 4.0])
        x = solver4.solve(b)
        x_ref = np.linalg.solve(A_full, b)
        npt.assert_allclose(x, x_ref, atol=1e-12)

    def test_residual_1d(self, solver4, A4_upper):
        _, A_full = A4_upper
        b = np.array([1.0, 2.0, 3.0, 4.0])
        x = solver4.solve(b)
        npt.assert_allclose(A_full @ x, b, atol=1e-12)

    def test_solve_1d_integer_rhs(self, solver4, A4_upper):
        """Integer RHS should be cast to float64 automatically."""
        _, A_full = A4_upper
        b = np.array([1, 2, 3, 4])
        x = solver4.solve(b)
        x_ref = np.linalg.solve(A_full, b.astype(float))
        npt.assert_allclose(x, x_ref, atol=1e-12)


class TestSolve2D:
    def test_solve_2d(self, solver4, A4_upper):
        _, A_full = A4_upper
        B = np.array([[1.0, 2.0], [0.0, 1.0], [3.0, 4.0], [1.0, 0.0]])
        X = solver4.solve(B)
        X_ref = np.linalg.solve(A_full, B)
        npt.assert_allclose(X, X_ref, atol=1e-12)

    def test_solve_2d_f_contiguous_output(self, solver4):
        B = np.asfortranarray(np.ones((4, 3)))
        X = solver4.solve(B)
        assert X.flags["F_CONTIGUOUS"]

    def test_residual_2d(self, solver4, A4_upper):
        _, A_full = A4_upper
        B = np.array([[1.0, 2.0], [0.0, 1.0], [3.0, 4.0], [1.0, 0.0]])
        X = solver4.solve(B)
        npt.assert_allclose(A_full @ X, B, atol=1e-12)


# ---------------------------------------------------------------------------
# In-place solve
# ---------------------------------------------------------------------------

class TestSolveInplace:
    def test_inplace_1d(self, solver4, A4_upper):
        _, A_full = A4_upper
        b = np.array([1.0, 2.0, 3.0, 4.0])
        x = b.copy()
        solver4.solve(x, inplace=True)
        npt.assert_allclose(x, np.linalg.solve(A_full, b), atol=1e-12)

    def test_inplace_2d(self, solver4, A4_upper):
        _, A_full = A4_upper
        B = np.asfortranarray(
            np.array([[1.0, 2.0], [0.0, 1.0], [3.0, 4.0], [1.0, 0.0]])
        )
        X = B.copy(order="F")
        solver4.solve(X, inplace=True)
        npt.assert_allclose(X, np.linalg.solve(A_full, B), atol=1e-12)

    def test_inplace_1d_matches_solve(self, solver4):
        b = np.array([1.0, 2.0, 3.0, 4.0])
        x_copy = solver4.solve(b)
        x_inplace = b.copy()
        solver4.solve(x_inplace, inplace=True)
        npt.assert_allclose(x_inplace, x_copy, atol=1e-14)

    def test_inplace_rejects_2d_c_contiguous(self, solver4):
        B = np.ones((4, 2))  # C-contiguous by default
        with pytest.raises(ValueError, match="F-contiguous"):
            solver4.solve(B, inplace=True)

    def test_inplace_rejects_readonly(self, solver4):
        b = np.array([1.0, 2.0, 3.0, 4.0])
        b.flags.writeable = False
        with pytest.raises(ValueError, match="writeable"):
            solver4.solve(b, inplace=True)


# ---------------------------------------------------------------------------
# Refactor (same pattern, new values)
# ---------------------------------------------------------------------------

class TestRefactor:
    def test_refactor_correctness(self, solver4, A4_upper):
        _, A_full = A4_upper
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
        solver4.refactor(A2.data)

        b = np.array([1.0, 2.0, 3.0, 4.0])
        x = solver4.solve(b)
        x_ref = np.linalg.solve(A2_full, b)
        npt.assert_allclose(x, x_ref, atol=1e-12)


# ---------------------------------------------------------------------------
# Triangle / format variants
# ---------------------------------------------------------------------------

class TestTriangleAndFormat:
    def test_lower_triangle(self):
        A_full = np.array([[4.0, 1.0], [1.0, 3.0]])
        A_lower = sp.csc_matrix(np.tril(A_full))
        solver = macldlt.LDLTSolver(A_lower, triangle="lower")
        b = np.array([1.0, 2.0])
        x = solver.solve(b)
        npt.assert_allclose(x, np.linalg.solve(A_full, b), atol=1e-12)

    def test_csr_input(self):
        A_full = np.array([[4.0, 1.0], [1.0, 3.0]])
        A_csr = sp.csr_matrix(np.tril(A_full))
        solver = macldlt.LDLTSolver(A_csr, triangle="lower")
        b = np.array([1.0, 2.0])
        x = solver.solve(b)
        npt.assert_allclose(x, np.linalg.solve(A_full, b), atol=1e-12)

    def test_coo_input(self):
        A_full = np.array([[4.0, 1.0], [1.0, 3.0]])
        A_coo = sp.coo_matrix(np.triu(A_full))
        solver = macldlt.LDLTSolver(A_coo)
        b = np.array([1.0, 2.0])
        x = solver.solve(b)
        npt.assert_allclose(x, np.linalg.solve(A_full, b), atol=1e-12)


# ---------------------------------------------------------------------------
# Inertia
# ---------------------------------------------------------------------------

class TestInertia:
    def test_positive_definite(self):
        A = sp.csc_matrix(np.array([[4.0, 1.0], [1.0, 3.0]]))
        solver = macldlt.LDLTSolver(A)
        neg, zero, pos = solver.inertia()
        assert neg == 0
        assert zero == 0
        assert pos == 2

    def test_indefinite(self, solver4):
        neg, zero, pos = solver4.inertia()
        assert neg + zero + pos == 4
        assert neg > 0
        assert pos > 0

    def test_singular_has_zero_pivots(self):
        A = sp.csc_matrix(np.array([[1.0, 1.0], [1.0, 1.0]]))
        solver = macldlt.LDLTSolver(A)
        _, zero, _ = solver.inertia()
        assert zero > 0


# ---------------------------------------------------------------------------
# Factorization variants
# ---------------------------------------------------------------------------

class TestFactorizationTypes:
    @pytest.mark.parametrize("ftype", ["ldlt", "ldlt_tpp", "ldlt_sbk"])
    def test_factorization_variant(self, ftype, A4_upper):
        A_sp, A_full = A4_upper
        solver = macldlt.LDLTSolver(A_sp, factorization=ftype)
        b = np.array([1.0, 2.0, 3.0, 4.0])
        x = solver.solve(b)
        npt.assert_allclose(x, np.linalg.solve(A_full, b), atol=1e-12)

    def test_ldlt_unpivoted_on_spd(self):
        """ldlt_unpivoted only works on positive definite matrices."""
        A_full = np.array([[4.0, 1.0], [1.0, 3.0]])
        A = sp.csc_matrix(np.triu(A_full))
        solver = macldlt.LDLTSolver(A, factorization="ldlt_unpivoted")
        b = np.array([1.0, 2.0])
        x = solver.solve(b)
        npt.assert_allclose(x, np.linalg.solve(A_full, b), atol=1e-12)


# ---------------------------------------------------------------------------
# Ordering variants
# ---------------------------------------------------------------------------

class TestOrderingTypes:
    @pytest.mark.parametrize("order", ["default", "amd"])
    def test_ordering_variant(self, order, A4_upper):
        A_sp, A_full = A4_upper
        solver = macldlt.LDLTSolver(A_sp, ordering=order)
        b = np.array([1.0, 2.0, 3.0, 4.0])
        x = solver.solve(b)
        npt.assert_allclose(x, np.linalg.solve(A_full, b), atol=1e-12)


# ---------------------------------------------------------------------------
# Info and status
# ---------------------------------------------------------------------------

class TestInfo:
    def test_info_keys(self, solver4):
        info = solver4.info()
        assert "n" in info
        assert "symbolic_status" in info
        assert "numeric_status" in info
        assert info["n"] == 4

    def test_status_properties(self, solver4):
        assert solver4.symbolic_status == "SparseStatusOK"
        assert solver4.numeric_status == "SparseStatusOK"

    def test_n_property(self, solver4):
        assert solver4.n == 4


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

class TestValidation:
    def test_rejects_nonsquare(self):
        A = sp.csc_matrix(np.ones((3, 4)))
        with pytest.raises(ValueError, match="square"):
            macldlt.LDLTSolver(A)

    def test_rejects_wrong_rhs_length(self, solver4):
        with pytest.raises(ValueError, match="wrong"):
            solver4.solve(np.array([1.0, 2.0]))

    def test_rejects_3d_rhs(self, solver4):
        with pytest.raises(ValueError):
            solver4.solve(np.ones((4, 2, 2)))

    def test_rejects_dense_matrix(self):
        with pytest.raises(ValueError, match="sparse"):
            macldlt.LDLTSolver(np.eye(3))

    def test_rejects_invalid_triangle(self):
        A = sp.csc_matrix(np.eye(2))
        with pytest.raises(ValueError, match="triangle"):
            macldlt.LDLTSolver(A, triangle="middle")

    def test_rejects_invalid_factorization(self):
        A = sp.csc_matrix(np.eye(2))
        with pytest.raises(ValueError, match="factorization"):
            macldlt.LDLTSolver(A, factorization="cholesky")

    def test_rejects_invalid_ordering(self):
        A = sp.csc_matrix(np.eye(2))
        with pytest.raises(ValueError, match="ordering"):
            macldlt.LDLTSolver(A, ordering="random")


# ---------------------------------------------------------------------------
# Larger system
# ---------------------------------------------------------------------------

class TestLarger:
    def test_random_spd_100(self):
        rng = np.random.default_rng(42)
        n = 100
        density = 0.05
        R = sp.random(n, n, density=density, random_state=rng, format="csc")
        M = R.T @ R + sp.eye(n)
        A_upper = sp.triu(M, format="csc")

        solver = macldlt.LDLTSolver(A_upper)
        b = rng.standard_normal(n)
        x = solver.solve(b)
        npt.assert_allclose(M.toarray() @ x, b, atol=1e-8)

    def test_random_spd_multi_rhs(self):
        rng = np.random.default_rng(99)
        n = 50
        R = sp.random(n, n, density=0.1, random_state=rng, format="csc")
        M = R.T @ R + sp.eye(n)
        A_upper = sp.triu(M, format="csc")

        solver = macldlt.LDLTSolver(A_upper)
        B = rng.standard_normal((n, 5))
        X = solver.solve(B)
        npt.assert_allclose(M.toarray() @ X, B, atol=1e-8)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_1x1_matrix(self):
        A = sp.csc_matrix(np.array([[3.0]]))
        solver = macldlt.LDLTSolver(A)
        x = solver.solve(np.array([6.0]))
        npt.assert_allclose(x, [2.0], atol=1e-14)

    def test_diagonal_matrix(self):
        d = np.array([2.0, 5.0, 0.5, 10.0])
        A = sp.diags(d, format="csc")
        solver = macldlt.LDLTSolver(A)
        b = np.array([1.0, 1.0, 1.0, 1.0])
        x = solver.solve(b)
        npt.assert_allclose(x, 1.0 / d, atol=1e-14)

    def test_float32_data_coercion(self):
        """float32 sparse data should be cast to float64 internally."""
        A_full = np.array([[4.0, 1.0], [1.0, 3.0]], dtype=np.float32)
        A = sp.csc_matrix(np.triu(A_full))
        solver = macldlt.LDLTSolver(A)
        b = np.array([1.0, 2.0])
        x = solver.solve(b)
        x_ref = np.linalg.solve(A_full.astype(np.float64), b)
        npt.assert_allclose(x, x_ref, atol=1e-12)

    def test_solve_inplace_2d_matches_solve(self, solver4):
        B = np.asfortranarray(
            np.array([[1.0, 2.0], [0.0, 1.0], [3.0, 4.0], [1.0, 0.0]])
        )
        X_copy = solver4.solve(B)
        X_inplace = B.copy(order="F")
        solver4.solve(X_inplace, inplace=True)
        npt.assert_allclose(X_inplace, X_copy, atol=1e-14)


# ---------------------------------------------------------------------------
# Factor / refactor lifecycle
# ---------------------------------------------------------------------------

class TestFactorLifecycle:
    def test_refactor_then_solve_multiple(self, solver4, A4_upper):
        """Refactor then solve with multiple different RHS vectors."""
        _, A_full = A4_upper
        A2_full = A_full.copy()
        A2_full[0, 0] += 1.0
        A2_full[2, 2] += 1.0
        A2 = sp.csc_matrix(np.triu(A2_full))
        solver4.refactor(A2.data)

        for i in range(5):
            b = np.random.default_rng(i).standard_normal(4)
            x = solver4.solve(b)
            npt.assert_allclose(A2_full @ x, b, atol=1e-12)


# ---------------------------------------------------------------------------
# Refactor with values array
# ---------------------------------------------------------------------------

class TestRefactorValues:
    def test_correctness(self, A4_upper):
        A_sp, A_full = A4_upper
        solver = macldlt.LDLTSolver(A_sp, triangle="upper")

        A2_full = A_full.copy()
        A2_full[0, 0] += 1.0
        A2_full[2, 2] += 2.0
        A2 = sp.csc_matrix(np.triu(A2_full))

        solver.refactor(A2.data)
        b = np.array([1.0, 2.0, 3.0, 4.0])
        x = solver.solve(b)
        npt.assert_allclose(A2_full @ x, b, atol=1e-12)

    def test_repeated_refactor(self, A4_upper):
        """Multiple refactor calls in a loop."""
        A_sp, A_full = A4_upper
        solver = macldlt.LDLTSolver(A_sp, triangle="upper")
        b = np.array([1.0, 2.0, 3.0, 4.0])

        for i in range(10):
            A_mod = A_full.copy()
            A_mod[0, 0] += 0.1 * i
            A_mod[2, 2] += 0.1 * i
            A_csc = sp.csc_matrix(np.triu(A_mod))

            solver.refactor(A_csc.data)
            x = solver.solve(b)
            npt.assert_allclose(A_mod @ x, b, atol=1e-12)

    def test_solve_after_refactor_is_correct(self):
        A_full_pd = np.array([[4.0, 1.0], [1.0, 3.0]])
        A_pd = sp.csc_matrix(np.triu(A_full_pd))
        solver = macldlt.LDLTSolver(A_pd, factorization="ldlt_tpp")

        A_full_2 = np.array([[10.0, 2.0], [2.0, 8.0]])
        A_2 = sp.csc_matrix(np.triu(A_full_2))
        solver.refactor(A_2.data)

        b = np.array([1.0, 2.0])
        x = solver.solve(b)
        npt.assert_allclose(A_full_2 @ x, b, atol=1e-12)

    def test_rejects_wrong_length(self, solver4):
        with pytest.raises(ValueError, match="elements"):
            solver4.refactor(np.array([1.0, 2.0]))

    def test_rejects_2d(self, solver4):
        with pytest.raises((ValueError, TypeError)):
            solver4.refactor(np.ones((3, 3)))

    def test_float32_coercion(self, A4_upper):
        """float32 values should be cast automatically."""
        A_sp, A_full = A4_upper
        solver = macldlt.LDLTSolver(A_sp, triangle="upper")

        values_f32 = A_sp.data.astype(np.float32)
        solver.refactor(values_f32)
        b = np.array([1.0, 2.0, 3.0, 4.0])
        x = solver.solve(b)
        npt.assert_allclose(A_full @ x, b, atol=1e-12)
