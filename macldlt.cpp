#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <Accelerate/Accelerate.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <new>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace py = pybind11;

namespace
{

[[noreturn]] void throw_value_error(const std::string& msg)
{
    throw py::value_error(msg);
}

[[noreturn]] void throw_runtime_error(const std::string& msg)
{
    throw std::runtime_error(msg);
}

std::string sparse_status_to_string(SparseStatus_t s)
{
    switch (s)
    {
        case SparseStatusOK:
            return "SparseStatusOK";
        case SparseFactorizationFailed:
            return "SparseFactorizationFailed";
        case SparseMatrixIsSingular:
            return "SparseMatrixIsSingular";
        case SparseInternalError:
            return "SparseInternalError";
        case SparseParameterError:
            return "SparseParameterError";
        case SparseStatusReleased:
            return "SparseStatusReleased";
        default:
            return "SparseStatus(unknown)";
    }
}

SparseTriangle_t parse_triangle(std::string_view triangle)
{
    if (triangle == "upper") return SparseUpperTriangle;
    if (triangle == "lower") return SparseLowerTriangle;
    throw_value_error("triangle must be 'upper' or 'lower'");
}

SparseFactorization_t parse_factorization(std::string_view factorization)
{
    if (factorization == "ldlt") return SparseFactorizationLDLT;
    if (factorization == "ldlt_tpp") return SparseFactorizationLDLTTPP;
    if (factorization == "ldlt_sbk") return SparseFactorizationLDLTSBK;
    if (factorization == "ldlt_unpivoted") return SparseFactorizationLDLTUnpivoted;

    throw_value_error(
        "factorization must be one of: 'ldlt', 'ldlt_tpp', 'ldlt_sbk', "
        "'ldlt_unpivoted'");
}

SparseOrder_t parse_ordering(std::string_view ordering)
{
    if (ordering == "default") return SparseOrderDefault;
    if (ordering == "amd") return SparseOrderAMD;
    if (ordering == "metis") return SparseOrderMetis;
    if (ordering == "colamd") return SparseOrderCOLAMD;

    throw_value_error("ordering must be one of: 'default', 'amd', 'metis', 'colamd'");
}

void check_square_shape_attr(py::handle obj)
{
    const py::tuple shape = py::getattr(obj, "shape");
    if (shape.size() != 2)
    {
        throw_value_error("expected a 2D sparse matrix");
    }

    const ssize_t m = shape[0].cast<ssize_t>();
    const ssize_t n = shape[1].cast<ssize_t>();
    if (m != n)
    {
        throw_value_error("matrix must be square");
    }
}

void check_int_fits(int64_t v, const char* name)
{
    if (v < 0 || v > static_cast<int64_t>(std::numeric_limits<int>::max()))
    {
        throw_value_error(std::string(name) + " does not fit in C int");
    }
}

void check_long_fits(int64_t v, const char* name)
{
    if (v < 0 || v > static_cast<int64_t>(std::numeric_limits<long>::max()))
    {
        throw_value_error(std::string(name) + " does not fit in C long");
    }
}

py::object scipy_issparse()
{
    return py::module_::import("scipy.sparse").attr("issparse");
}

struct ScipyCSCView
{
    py::object matrix;  // keeps canonicalized CSC matrix alive

    py::array data;
    py::array indices;
    py::array indptr;

    py::array_t<double, py::array::c_style | py::array::forcecast> data_d;
    py::array_t<int, py::array::c_style | py::array::forcecast> indices_i;
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> indptr_i64;

    int n = 0;

    explicit ScipyCSCView(py::handle A, bool copy_if_needed = false)
    {
        py::object obj = py::reinterpret_borrow<py::object>(A);

        if (!scipy_issparse()(obj).cast<bool>())
        {
            throw_value_error("expected a SciPy sparse matrix");
        }

        check_square_shape_attr(obj);

        const std::string fmt = py::str(py::getattr(obj, "format"));
        if (fmt == "csc")
        {
            matrix = obj;
        }
        else if (fmt == "csr")
        {
            matrix = py::getattr(obj, "tocsc")(py::arg("copy") = copy_if_needed);
        }
        else
        {
            matrix = py::getattr(obj, "tocsc")();
        }

        const bool is_canonical =
            py::getattr(matrix, "has_canonical_format").cast<bool>();
        const bool is_sorted =
            py::getattr(matrix, "has_sorted_indices").cast<bool>();

        if (!is_canonical || !is_sorted)
        {
            // Avoid mutating the user's original matrix in-place.
            if (matrix.is(obj) && !copy_if_needed)
            {
                matrix = py::getattr(matrix, "copy")();
            }
            py::getattr(matrix, "sum_duplicates")();
            py::getattr(matrix, "sort_indices")();
        }

        data = py::getattr(matrix, "data");
        indices = py::getattr(matrix, "indices");
        indptr = py::getattr(matrix, "indptr");

        data_d = py::array_t<double, py::array::c_style | py::array::forcecast>(data);
        indices_i = py::array_t<int, py::array::c_style | py::array::forcecast>(indices);
        indptr_i64 = py::array_t<int64_t, py::array::c_style | py::array::forcecast>(indptr);

        const py::tuple shape = py::getattr(matrix, "shape");
        const int64_t n64 = shape[0].cast<int64_t>();
        check_int_fits(n64, "matrix dimension");
        n = static_cast<int>(n64);

        if (data_d.ndim() != 1 || indices_i.ndim() != 1 || indptr_i64.ndim() != 1)
        {
            throw_value_error("CSC arrays must be 1D");
        }

        if (indptr_i64.shape(0) != static_cast<ssize_t>(n + 1))
        {
            throw_value_error("invalid CSC indptr length");
        }

        if (indices_i.shape(0) != data_d.shape(0))
        {
            throw_value_error("CSC indices and data must have the same length");
        }

        auto indptr_r = indptr_i64.unchecked<1>();
        for (ssize_t i = 0; i < indptr_i64.shape(0); ++i)
        {
            if (indptr_r(i) < 0)
            {
                throw_value_error("indptr must be nonnegative");
            }
        }
        for (ssize_t i = 1; i < indptr_i64.shape(0); ++i)
        {
            if (indptr_r(i) < indptr_r(i - 1))
            {
                throw_value_error("indptr must be nondecreasing");
            }
        }

        auto idx_r = indices_i.unchecked<1>();
        for (ssize_t i = 0; i < indices_i.shape(0); ++i)
        {
            if (idx_r(i) < 0 || idx_r(i) >= n)
            {
                throw_value_error("row index out of bounds");
            }
        }
    }
};

std::vector<long> make_column_starts_long(const py::array_t<int64_t>& indptr)
{
    if (indptr.ndim() != 1)
    {
        throw_value_error("indptr must be 1D");
    }

    auto r = indptr.unchecked<1>();
    std::vector<long> out(static_cast<size_t>(indptr.shape(0)));

    for (ssize_t i = 0; i < indptr.shape(0); ++i)
    {
        const int64_t v = r(i);
        if (v < 0)
        {
            throw_value_error("indptr must be nonnegative");
        }
        check_long_fits(v, "indptr entry");
        out[static_cast<size_t>(i)] = static_cast<long>(v);
    }

    return out;
}

DenseVector_Double make_dense_vector(py::array_t<double, py::array::c_style>& x)
{
    auto buf = x.request();
    if (buf.ndim != 1)
    {
        throw_value_error("dense RHS must be 1D");
    }
    check_int_fits(static_cast<int64_t>(buf.shape[0]), "rhs length");

    DenseVector_Double v{};
    v.count = static_cast<int>(buf.shape[0]);
    v.data = static_cast<double*>(buf.ptr);
    return v;
}

DenseMatrix_Double make_dense_matrix(py::array_t<double, py::array::f_style>& x)
{
    auto buf = x.request();
    if (buf.ndim != 2)
    {
        throw_value_error("dense RHS must be 2D");
    }
    check_int_fits(static_cast<int64_t>(buf.shape[0]), "rhs row count");
    check_int_fits(static_cast<int64_t>(buf.shape[1]), "rhs column count");

    const ssize_t stride_bytes = buf.strides[1];
    if (stride_bytes < 0 || stride_bytes % static_cast<ssize_t>(sizeof(double)) != 0)
    {
        throw_value_error("invalid F-contiguous double matrix stride");
    }

    const ssize_t stride_elems = stride_bytes / static_cast<ssize_t>(sizeof(double));
    check_int_fits(static_cast<int64_t>(stride_elems), "rhs column stride");

    DenseMatrix_Double M{};
    M.rowCount = static_cast<int>(buf.shape[0]);
    M.columnCount = static_cast<int>(buf.shape[1]);
    M.columnStride = static_cast<int>(stride_elems);
    M.attributes.transpose = false;
    M.attributes.triangle = SparseLowerTriangle;
    M.attributes.kind = SparseOrdinary;
    M.attributes._reserved = 0;
    M.attributes._allocatedBySparse = 0;
    M.data = static_cast<double*>(buf.ptr);
    return M;
}

bool is_c_contiguous_1d_double(py::handle arr)
{
    if (!py::isinstance<py::array_t<double>>(arr)) return false;
    py::array a = py::reinterpret_borrow<py::array>(arr);
    if (a.ndim() != 1) return false;
    return (a.flags() & py::array::c_style) != 0;
}

bool is_f_contiguous_2d_double(py::handle arr)
{
    if (!py::isinstance<py::array_t<double>>(arr)) return false;
    py::array a = py::reinterpret_borrow<py::array>(arr);
    if (a.ndim() != 2) return false;
    return (a.flags() & py::array::f_style) != 0;
}

class LDLTSolver
{
public:
    LDLTSolver(py::object A,
               std::string_view triangle = "upper",
               std::string_view ordering = "amd",
               std::string_view factorization = "ldlt")
        : triangle_(parse_triangle(triangle)),
          ordering_(parse_ordering(ordering)),
          factorization_(parse_factorization(factorization))
    {
        analyze_and_factor(A);
    }

    ~LDLTSolver()
    {
        cleanup_numeric();
        cleanup_symbolic();
    }

    LDLTSolver(const LDLTSolver&) = delete;
    LDLTSolver& operator=(const LDLTSolver&) = delete;
    LDLTSolver(LDLTSolver&&) = delete;
    LDLTSolver& operator=(LDLTSolver&&) = delete;

    void refactor(py::array_t<double, py::array::c_style | py::array::forcecast> values)
    {
        ensure_numeric();

        const ssize_t expected_nnz = pattern_indices_.shape(0);
        if (values.ndim() != 1 || values.shape(0) != expected_nnz)
        {
            throw_value_error(
                "values must be a 1D array with " + std::to_string(expected_nnz) +
                " elements (matching the sparsity pattern)");
        }

        numeric_owner_ = py::none();
        numeric_values_ = std::move(values);
        refactor_current_numeric();
    }

    py::array solve(py::array rhs, bool inplace = false)
    {
        ensure_numeric();

        if (inplace)
        {
            if (!rhs.writeable())
            {
                throw_value_error("inplace solve requires a writeable array");
            }

            if (rhs.ndim() == 1)
            {
                if (!is_c_contiguous_1d_double(rhs))
                {
                    throw_value_error(
                        "1D inplace solve requires a C-contiguous float64 array");
                }

                auto x = py::reinterpret_borrow<
                    py::array_t<double, py::array::c_style>>(rhs);
                if (x.shape(0) != n_)
                {
                    throw_value_error("rhs has wrong length");
                }

                solve_inplace_impl(x);
                return rhs;
            }

            if (rhs.ndim() == 2)
            {
                if (!is_f_contiguous_2d_double(rhs))
                {
                    throw_value_error(
                        "2D inplace solve requires an F-contiguous float64 array");
                }

                auto X = py::reinterpret_borrow<
                    py::array_t<double, py::array::f_style>>(rhs);
                if (X.shape(0) != n_)
                {
                    throw_value_error("rhs has wrong leading dimension");
                }

                solve_inplace_impl(X);
                return rhs;
            }

            throw_value_error("rhs must be 1D or 2D");
        }

        if (rhs.ndim() == 1)
        {
            auto b = py::array_t<double, py::array::c_style | py::array::forcecast>(rhs);
            if (b.shape(0) != n_)
            {
                throw_value_error("rhs has wrong length");
            }

            py::array_t<double, py::array::c_style> x(n_);
            std::memcpy(x.mutable_data(), b.data(), static_cast<size_t>(n_) * sizeof(double));
            solve_inplace_impl(x);
            return x;
        }

        if (rhs.ndim() == 2)
        {
            auto B = py::array_t<double, py::array::f_style | py::array::forcecast>(rhs);
            if (B.shape(0) != n_)
            {
                throw_value_error("rhs has wrong leading dimension");
            }

            auto X = py::array_t<double, py::array::f_style>({B.shape(0), B.shape(1)});
            std::memcpy(X.mutable_data(), B.data(),
                        static_cast<size_t>(B.size()) * sizeof(double));
            solve_inplace_impl(X);
            return X;
        }

        throw_value_error("rhs must be 1D or 2D");
    }

    py::tuple inertia() const
    {
        ensure_numeric();

        int num_positive = 0;
        int num_zero = 0;
        int num_negative = 0;
        SparseGetInertia(numeric_, &num_positive, &num_zero, &num_negative);
        return py::make_tuple(num_negative, num_zero, num_positive);
    }

    int n() const { return n_; }

    std::string symbolic_status() const
    {
        return symbolic_valid_ ? sparse_status_to_string(symbolic_.status) : "not_analyzed";
    }

    std::string numeric_status() const
    {
        return numeric_valid_ ? sparse_status_to_string(numeric_.status) : "not_factored";
    }

    py::dict info() const
    {
        py::dict d;
        d["n"] = n_;
        d["symbolic_status"] = symbolic_status();
        d["numeric_status"] = numeric_status();

        d["factor_workspace_allocated_bytes"] =
            static_cast<py::int_>(factor_workspace_size_);
        d["solve_workspace_allocated_bytes"] =
            static_cast<py::int_>(solve_workspace_size_);

        if (symbolic_valid_)
        {
            d["factor_workspace_required_bytes"] =
                static_cast<py::int_>(required_factor_workspace_bytes());
            d["symbolic_workspace_double"] =
                static_cast<py::int_>(symbolic_.workspaceSize_Double);
            d["factor_size_double"] =
                static_cast<py::int_>(symbolic_.factorSize_Double);
        }

        if (numeric_valid_)
        {
            d["solve_workspace_required_bytes_1rhs"] =
                static_cast<py::int_>(required_solve_workspace_bytes(1));
            d["solve_workspace_static"] =
                static_cast<py::int_>(numeric_.solveWorkspaceRequiredStatic);
            d["solve_workspace_per_rhs"] =
                static_cast<py::int_>(numeric_.solveWorkspaceRequiredPerRHS);
        }

        return d;
    }

private:
    static constexpr size_t kRequiredAlignment = 16;

    void ensure_symbolic() const
    {
        if (!symbolic_valid_)
        {
            throw_runtime_error("symbolic factorization not available; call analyze() first");
        }
    }

    void ensure_numeric() const
    {
        if (!numeric_valid_)
        {
            throw_runtime_error("numeric factorization not available");
        }
    }

    void analyze_and_factor(const py::object& A)
    {
        cleanup_numeric();
        cleanup_symbolic();

        ScipyCSCView csc(A, false);

        keep_pattern_owner_ = csc.matrix;
        pattern_indices_ = csc.indices_i;
        pattern_indptr_ = csc.indptr_i64;
        column_starts_ = make_column_starts_long(csc.indptr_i64);
        n_ = csc.n;

        structure_ = {};
        structure_.rowCount = n_;
        structure_.columnCount = n_;
        structure_.columnStarts = column_starts_.data();

        // Accelerate API takes mutable pointers here, but the index structure
        // is treated as read-only input by this wrapper.
        structure_.rowIndices = const_cast<int*>(pattern_indices_.data());

        structure_.attributes.transpose = false;
        structure_.attributes.triangle = triangle_;
        structure_.attributes.kind = SparseSymmetric;
        structure_.attributes._reserved = 0;
        structure_.attributes._allocatedBySparse = 0;
        structure_.blockSize = 1;

        SparseSymbolicFactorOptions sfopts{};
        sfopts.control = SparseDefaultControl;
        sfopts.orderMethod = ordering_;
        sfopts.order = nullptr;
        sfopts.ignoreRowsAndColumns = nullptr;
        sfopts.malloc = malloc;
        sfopts.free = free;
        sfopts.reportError = nullptr;

        SparseOpaqueSymbolicFactorization sym{};
        {
            py::gil_scoped_release release;
            sym = SparseFactor(factorization_, structure_, sfopts);
        }

        if (sym.status != SparseStatusOK)
        {
            throw_runtime_error(
                "symbolic factorization failed: " + sparse_status_to_string(sym.status));
        }

        symbolic_ = sym;
        symbolic_valid_ = true;

        ensure_factor_workspace(required_factor_workspace_bytes());

        // Numeric factorization
        numeric_owner_ = csc.matrix;
        numeric_values_ = csc.data_d;
        numeric_matrix_ = make_sparse_matrix_from_current_numeric();

        SparseOpaqueFactorization_Double num{};
        {
            py::gil_scoped_release release;
            num = SparseFactor(symbolic_, numeric_matrix_);
        }

        if (num.status != SparseStatusOK)
        {
            clear_numeric_state_only();
            throw_runtime_error(
                "numeric factorization failed: " + sparse_status_to_string(num.status));
        }

        numeric_ = num;
        numeric_valid_ = true;

        ensure_factor_workspace(required_factor_workspace_bytes());
        ensure_solve_workspace(required_solve_workspace_bytes(1));
    }

    void clear_symbolic_state_only()
    {
        symbolic_ = {};
        symbolic_valid_ = false;

        keep_pattern_owner_ = py::none();
        pattern_indices_ = {};
        pattern_indptr_ = {};
        column_starts_.clear();
        structure_ = {};
        n_ = 0;
    }

    void clear_numeric_state_only()
    {
        numeric_ = {};
        numeric_valid_ = false;

        numeric_owner_ = py::none();
        numeric_values_ = {};
        numeric_matrix_ = {};
    }

    void cleanup_symbolic()
    {
        if (symbolic_valid_)
        {
            SparseCleanup(symbolic_);
        }
        clear_symbolic_state_only();
    }

    void cleanup_numeric()
    {
        if (numeric_valid_)
        {
            SparseCleanup(numeric_);
        }
        clear_numeric_state_only();
    }

    SparseMatrix_Double make_sparse_matrix_from_current_numeric()
    {
        SparseMatrix_Double A{};
        A.structure = structure_;

        // Accelerate API takes a mutable pointer here, but values are treated as
        // input matrix coefficients by this wrapper.
        A.data = const_cast<double*>(numeric_values_.data());
        return A;
    }

    void refactor_current_numeric()
    {
        numeric_matrix_ = make_sparse_matrix_from_current_numeric();

        const size_t required = required_factor_workspace_bytes();
        ensure_factor_workspace(required);

        {
            py::gil_scoped_release release;
            SparseRefactor(
                numeric_matrix_,
                &numeric_,
                aligned_ptr(factor_workspace_.get(), factor_workspace_size_, required));
        }

        if (numeric_.status != SparseStatusOK)
        {
            throw_runtime_error("refactor failed: " +
                                sparse_status_to_string(numeric_.status));
        }
    }

    size_t required_factor_workspace_bytes() const
    {
        ensure_symbolic();
        return static_cast<size_t>(symbolic_.workspaceSize_Double);
    }

    size_t required_solve_workspace_bytes(int nrhs) const
    {
        ensure_numeric();
        return static_cast<size_t>(numeric_.solveWorkspaceRequiredStatic) +
               static_cast<size_t>(numeric_.solveWorkspaceRequiredPerRHS) *
                   static_cast<size_t>(std::max(1, nrhs));
    }

    static void* aligned_ptr(std::byte* workspace, size_t size, size_t required_size)
    {
        void* ptr = workspace;
        void* aligned = std::align(kRequiredAlignment, required_size, ptr, size);
        if (aligned == nullptr)
        {
            throw_runtime_error("failed to align workspace buffer");
        }
        return aligned;
    }

    void ensure_factor_workspace(size_t required)
    {
        const size_t need = required + kRequiredAlignment;
        if (factor_workspace_size_ < need)
        {
            factor_workspace_.reset(new std::byte[need]);
            factor_workspace_size_ = need;
        }
    }

    void ensure_solve_workspace(size_t required)
    {
        const size_t need = required + kRequiredAlignment;
        if (solve_workspace_size_ < need)
        {
            solve_workspace_.reset(new std::byte[need]);
            solve_workspace_size_ = need;
        }
    }

    void solve_inplace_impl(py::array_t<double, py::array::c_style>& x)
    {
        const size_t required = required_solve_workspace_bytes(1);
        ensure_solve_workspace(required);

        DenseVector_Double v = make_dense_vector(x);

        {
            py::gil_scoped_release release;
            SparseSolve(
                numeric_,
                v,
                aligned_ptr(solve_workspace_.get(), solve_workspace_size_, required));
        }

        if (numeric_.status != SparseStatusOK)
        {
            throw_runtime_error("solve failed: " + sparse_status_to_string(numeric_.status));
        }
    }

    void solve_inplace_impl(py::array_t<double, py::array::f_style>& X)
    {
        const size_t required =
            required_solve_workspace_bytes(static_cast<int>(X.shape(1)));
        ensure_solve_workspace(required);

        DenseMatrix_Double M = make_dense_matrix(X);

        {
            py::gil_scoped_release release;
            SparseSolve(
                numeric_,
                M,
                aligned_ptr(solve_workspace_.get(), solve_workspace_size_, required));
        }

        if (numeric_.status != SparseStatusOK)
        {
            throw_runtime_error("solve failed: " + sparse_status_to_string(numeric_.status));
        }
    }

    SparseTriangle_t triangle_;
    SparseOrder_t ordering_;
    SparseFactorization_t factorization_;

    int n_ = 0;

    py::object keep_pattern_owner_ = py::none();
    py::array_t<int, py::array::c_style | py::array::forcecast> pattern_indices_;
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> pattern_indptr_;
    std::vector<long> column_starts_;
    SparseMatrixStructure structure_{};

    py::object numeric_owner_ = py::none();
    py::array_t<double, py::array::c_style | py::array::forcecast> numeric_values_;
    SparseMatrix_Double numeric_matrix_{};

    SparseOpaqueSymbolicFactorization symbolic_{};
    bool symbolic_valid_ = false;

    SparseOpaqueFactorization_Double numeric_{};
    bool numeric_valid_ = false;

    std::unique_ptr<std::byte[]> factor_workspace_;
    size_t factor_workspace_size_ = 0;

    std::unique_ptr<std::byte[]> solve_workspace_;
    size_t solve_workspace_size_ = 0;
};

}  // namespace

PYBIND11_MODULE(macldlt, m)
{
    m.doc() = "pybind11 wrapper for Apple Accelerate sparse LDLT with symbolic reuse";

    py::class_<LDLTSolver>(m, "LDLTSolver")
        .def(py::init<py::object, std::string_view, std::string_view, std::string_view>(),
             py::arg("A"),
             py::arg("triangle") = "upper",
             py::arg("ordering") = "amd",
             py::arg("factorization") = "ldlt",
             R"pbdoc(
Construct an LDLT solver for a symmetric sparse matrix.

Notes
-----
This class is not thread-safe. Do not call refactor() or solve() concurrently
on the same solver instance from multiple Python threads.

Symmetry is assumed but not checked. The selected stored triangle is assumed to
match the matrix data actually provided. Passing a nonsymmetric matrix, or
choosing the wrong stored triangle, may produce incorrect results or solver
failure.

The constructor performs both symbolic analysis and numeric factorization. To
update the numeric values without re-analyzing, use refactor(). For a new
sparsity pattern, create a new solver.

Parameters
----------
A : scipy.sparse.csc_matrix or csr_matrix
    Square sparse matrix. Accelerate uses CSC storage internally. CSR is
    accepted and converted to CSC.

triangle : {'upper', 'lower'}
    Which stored triangle represents the symmetric matrix.

ordering : {'default', 'amd', 'metis', 'colamd'}
    Fill-reducing ordering used during symbolic analysis.

factorization : {'ldlt', 'ldlt_tpp', 'ldlt_sbk', 'ldlt_unpivoted'}
    LDLT variant. 'ldlt' is Accelerate's default LDLT.
)pbdoc")
        .def("refactor",
             &LDLTSolver::refactor,
             py::arg("values"),
             R"pbdoc(
Reuse the existing numeric factorization for new values with the same sparsity
pattern.

Parameters
----------
values : numpy.ndarray
    1D float64 array of nonzero values, in the same CSC storage order as
    the matrix passed to the constructor. The length must match the number
    of stored entries in the sparsity pattern (i.e., ``A.data`` from the
    original scipy sparse matrix).

    Non-float64 arrays are cast automatically, but passing float64 avoids
    the copy.
)pbdoc")
        .def("solve",
             &LDLTSolver::solve,
             py::arg("rhs"),
             py::arg("inplace") = false,
             R"pbdoc(
Solve Ax=b or AX=B.

By default allocates and returns a new array. With inplace=True, overwrites
rhs in place and returns it. Inplace requires:
- 1D: writeable, C-contiguous float64
- 2D: writeable, F-contiguous float64
)pbdoc")
        .def("inertia",
             &LDLTSolver::inertia,
             "Return (num_negative, num_zero, num_positive) pivots.")
        .def("info", &LDLTSolver::info)
        .def_property_readonly("n", &LDLTSolver::n)
        .def_property_readonly("symbolic_status", &LDLTSolver::symbolic_status)
        .def_property_readonly("numeric_status", &LDLTSolver::numeric_status);
}
