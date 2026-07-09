#ifndef FILTERMATH_H
#define FILTERMATH_H

/**
 * FilterMath.h — Unified dispatch layer for accelerated linear algebra.
 *
 * Dispatch priority (highest to lowest):
 *   CUDA (if available and matrix size >= threshold)
 *   SVE2 (ARMv9 with FCMA/I8MM)
 *   NEON (ARM64)
 *   Eigen (fallback, all platforms)
 *
 * GPU acceleration is enabled for matrices larger than FILTERMATH_CUDA_THRESHOLD
 * to amortize PCIe transfer overhead. Smaller matrices remain on CPU.
 *
 * On non-ARM platforms (x86, etc.) paths fall through to CUDA or Eigen.
 */

#include <Eigen/Dense>
#include <Eigen/Cholesky>
#include <algorithm>   // std::min, std::max
#include <type_traits> // std::enable_if_t (fixed-size fast-path SFINAE)

// ---------- platform detection ----------
#if defined(__aarch64__) || defined(_M_ARM64)
  #define FILTERMATH_ARM64 1
#else
  #define FILTERMATH_ARM64 0
#endif

// ---------- CUDA detection ----------
#if defined(OPTMATH_USE_CUDA) || defined(__CUDACC__)
  #define FILTERMATH_HAS_CUDA 1
  #include <optmath/cuda_backend.hpp>
#else
  #define FILTERMATH_HAS_CUDA 0
#endif

#if FILTERMATH_ARM64
  #include <optmath/neon_kernels.hpp>
  #include <optmath/sve2_kernels.hpp>
#endif

// Size threshold for GPU dispatch (matrices smaller than this stay on CPU)
// PCIe latency typically 10-20us, so GPU is only faster for N >= 32
#ifndef FILTERMATH_CUDA_MIN_DIM
  #define FILTERMATH_CUDA_MIN_DIM 32
#endif

// Enable/disable CUDA dispatch at runtime (can be toggled)
namespace filtermath {
namespace config {
    inline bool& cuda_enabled() {
        static bool enabled = true;
        return enabled;
    }
    inline void set_cuda_enabled(bool en) { cuda_enabled() = en; }
}
}

namespace filtermath {

// ========================================================================
//  GEMM  —  CUDA > SVE2 > NEON > Eigen
// ========================================================================
/**
 * General matrix-matrix multiply C = A * B with hardware dispatch.
 * Selects the fastest available backend based on matrix size and platform:
 * CUDA for large matrices (>= FILTERMATH_CUDA_MIN_DIM), then SVE2, NEON, or Eigen.
 */
inline Eigen::MatrixXf gemm(const Eigen::MatrixXf& A, const Eigen::MatrixXf& B) {
    const Eigen::Index min_dim = std::min({A.rows(), A.cols(), B.cols()});

#if FILTERMATH_HAS_CUDA
    // GPU dispatch for large matrices (amortizes PCIe transfer)
    if (config::cuda_enabled() &&
        min_dim >= FILTERMATH_CUDA_MIN_DIM &&
        optmath::cuda::is_available()) {
        return optmath::cuda::cuda_gemm(A, B);
    }
#endif

#if FILTERMATH_ARM64
    if (optmath::sve2::is_available())
        return optmath::sve2::sve2_gemm_blocked(A, B);
    if (optmath::neon::is_available())
        return optmath::neon::neon_gemm(A, B);
#endif
    return A * B;
}

// ------------------------------------------------------------------------
//  GEMM fixed-size fast path (compile-time dimensions)
// ------------------------------------------------------------------------
// The filters (UKF/SRUKF sigma-point covariance, RBPF conditional KF) operate
// on small Eigen matrices whose dimensions are known at compile time (2..10 per
// side, never the >= 32 that would justify a GPU round-trip). Routing those
// through the dynamic `MatrixXf` overload above is pure overhead: each call
// materialises the operands into heap-backed MatrixXf temporaries, runs the
// runtime size/backend-availability branches, and heap-allocates the result —
// wrapped around what the compiler could otherwise emit as a fully-unrolled,
// vectorised, stack-allocated product.
//
// This overload is selected by ordinary overload resolution *only* when both
// operands have compile-time-known extents (identity binding beats the
// user-defined fixed->dynamic conversion the MatrixXf overload would need).
// Genuinely dynamic operands (EKF, the RBPF model dynamics/observation matrices)
// still bind to the MatrixXf overload, so the large-matrix CUDA/SVE2/NEON
// dispatch is fully preserved. Accepting MatrixBase<Derived> also lets Eigen
// expressions (e.g. `B.transpose()`) bind without an intermediate temporary.
template<typename DerivedA, typename DerivedB,
         typename = std::enable_if_t<
             DerivedA::RowsAtCompileTime != Eigen::Dynamic &&
             DerivedA::ColsAtCompileTime != Eigen::Dynamic &&
             DerivedB::RowsAtCompileTime != Eigen::Dynamic &&
             DerivedB::ColsAtCompileTime != Eigen::Dynamic>>
inline Eigen::Matrix<float, DerivedA::RowsAtCompileTime, DerivedB::ColsAtCompileTime>
gemm(const Eigen::MatrixBase<DerivedA>& A, const Eigen::MatrixBase<DerivedB>& B) {
    Eigen::Matrix<float, DerivedA::RowsAtCompileTime, DerivedB::ColsAtCompileTime> C;
    C.noalias() = A * B;  // C is a fresh local, distinct from A/B — noalias is safe
    return C;
}

// ========================================================================
//  Matrix-vector multiply  —  CUDA > NEON > Eigen
// ========================================================================
/**
 * Matrix-vector product y = A * v with hardware dispatch.
 * Uses CUDA gemv for large matrices, NEON on ARM, or Eigen fallback.
 */
inline Eigen::VectorXf mat_vec_mul(const Eigen::MatrixXf& A, const Eigen::VectorXf& v) {
    const int n = A.rows();

#if FILTERMATH_HAS_CUDA
    // GPU gemv for large matrices
    if (config::cuda_enabled() &&
        n >= FILTERMATH_CUDA_MIN_DIM &&
        optmath::cuda::is_available()) {
        return optmath::cuda::cuda_mat_vec_mul(A, v);
    }
#endif

#if FILTERMATH_ARM64
    if (optmath::neon::is_available())
        return optmath::neon::neon_mat_vec_mul(A, v);
#endif
    return A * v;
}

// ------------------------------------------------------------------------
//  Matrix-vector fixed-size fast path (compile-time dimensions)
// ------------------------------------------------------------------------
// Same rationale as the fixed-size gemm overload above: for small compile-time
// operands this avoids the MatrixXf materialisation, the runtime dispatch
// branches, and the heap-allocated result vector. Dynamic operands keep the
// MatrixXf overload (and its CUDA gemv path). Enabled only when the matrix has
// compile-time extents and the RHS has a compile-time row count.
template<typename DerivedA, typename DerivedV,
         typename = std::enable_if_t<
             DerivedA::RowsAtCompileTime != Eigen::Dynamic &&
             DerivedA::ColsAtCompileTime != Eigen::Dynamic &&
             DerivedV::RowsAtCompileTime != Eigen::Dynamic>>
inline Eigen::Matrix<float, DerivedA::RowsAtCompileTime, 1>
mat_vec_mul(const Eigen::MatrixBase<DerivedA>& A, const Eigen::MatrixBase<DerivedV>& v) {
    Eigen::Matrix<float, DerivedA::RowsAtCompileTime, 1> y;
    y.noalias() = A * v;
    return y;
}

// ========================================================================
//  Cholesky (lower-triangular L where A = L L^T)
//  Returns empty matrix on failure.
//  Dispatch: NEON > Eigen.
//  OptMathKernels v0.5.10+ exposes a cuSOLVER cuda_cholesky (potrf), but the
//  covariance factors here are small (typically <= ~10x10), so a host<->device
//  round-trip costs far more than the factorization itself. GPU Cholesky is
//  therefore intentionally NOT on this path; it is reserved for a future
//  large-matrix dispatch. See DEVELOPMENT_NOTES.md.
// ========================================================================
/**
 * Compute lower-triangular Cholesky factor L such that A = L * L^T.
 * Returns an empty matrix if A is not positive definite (signals failure
 * so the caller can apply a jitter-and-retry or LDLT fallback).
 */
inline Eigen::MatrixXf cholesky(const Eigen::MatrixXf& A) {
    // CUDA cuSOLVER path deliberately omitted for these small SPD matrices
    // (PCIe round-trip > compute). cuda_cholesky exists upstream if needed later.

#if FILTERMATH_ARM64
    if (optmath::neon::is_available()) {
        Eigen::MatrixXf L = optmath::neon::neon_cholesky(A);
        if (L.size() > 0) return L;
    }
#endif
    Eigen::LLT<Eigen::MatrixXf> llt(A);
    if (llt.info() == Eigen::Success)
        return llt.matrixL();
    return Eigen::MatrixXf();  // signal failure
}

// ========================================================================
//  Matrix inverse  —  NEON > Eigen
//  Returns empty matrix on failure.
//  No GPU path: OptMathKernels provides cuda_cholesky_inverse (SPD only), not a
//  general dense inverse, and these matrices are small. Prefer solve_spd().
// ========================================================================
/**
 * Compute the full inverse A^{-1} via NEON or Eigen full-pivot LU.
 * Returns an empty matrix if A is singular. Prefer solve_spd() or
 * kalman_gain() when possible, as they avoid explicit inversion.
 */
inline Eigen::MatrixXf inverse(const Eigen::MatrixXf& A) {

#if FILTERMATH_ARM64
    if (optmath::neon::is_available()) {
        Eigen::MatrixXf Ainv = optmath::neon::neon_inverse(A);
        if (Ainv.size() > 0) return Ainv;
    }
#endif
    // Eigen fallback via full-pivot LU
    Eigen::FullPivLU<Eigen::MatrixXf> lu(A);
    if (lu.isInvertible())
        return lu.inverse();
    return Eigen::MatrixXf();
}

// ========================================================================
//  SPD solve  A x = b  (A symmetric positive-definite)
//  Returns empty vector on failure.
//  Dispatch: NEON > Eigen. cuSOLVER cuda_solve / cuda_cholesky_solve exist in
//  OptMathKernels v0.5.10+ but are not used here (small matrices; PCIe-bound).
// ========================================================================
/**
 * Solve A * x = b for x where A is symmetric positive-definite.
 * Uses LDLT decomposition (robust to near-singular A). Returns an empty
 * vector on failure so the caller can fall back to a different strategy.
 */
inline Eigen::VectorXf solve_spd(const Eigen::MatrixXf& A, const Eigen::VectorXf& b) {

#if FILTERMATH_ARM64
    if (optmath::neon::is_available()) {
        Eigen::VectorXf x = optmath::neon::neon_solve_spd(A, b);
        if (x.size() > 0) return x;
    }
#endif
    Eigen::LDLT<Eigen::MatrixXf> ldlt(A);
    if (ldlt.info() == Eigen::Success && ldlt.isPositive())
        return ldlt.solve(b);
    return Eigen::VectorXf();
}

// ========================================================================
//  SPD solve for multiple RHS:  A X = B  →  X = A^{-1} B
//  Returns empty matrix on failure.
// ========================================================================
/**
 * Solve A * X = B for matrix X where A is SPD, using column-wise solve_spd.
 * Falls back to a single LDLT solve of the full system if any column fails.
 */
inline Eigen::MatrixXf solve_spd_mat(const Eigen::MatrixXf& A, const Eigen::MatrixXf& B) {
    // Try LDLT solve for all columns at once (most efficient path)
    Eigen::LDLT<Eigen::MatrixXf> ldlt(A);
    if (ldlt.info() == Eigen::Success && ldlt.isPositive())
        return ldlt.solve(B);

    // Fallback: column-wise solve_spd (uses NEON path if available)
    Eigen::MatrixXf X(B.rows(), B.cols());
    for (int j = 0; j < B.cols(); ++j) {
        Eigen::VectorXf col = solve_spd(A, B.col(j));
        if (col.size() == 0)
            return Eigen::MatrixXf();
        X.col(j) = col;
    }
    return X;
}

// ========================================================================
//  Triangular solve (lower): L x = b
// ========================================================================
/**
 * Solve L * x = b by forward substitution where L is lower-triangular.
 */
inline Eigen::VectorXf trsv_lower(const Eigen::MatrixXf& L, const Eigen::VectorXf& b) {
#if FILTERMATH_ARM64
    if (optmath::neon::is_available())
        return optmath::neon::neon_trsv_lower(L, b);
#endif
    return L.triangularView<Eigen::Lower>().solve(b);
}

// ========================================================================
//  Triangular solve (upper): U x = b
// ========================================================================
/**
 * Solve U * x = b by back substitution where U is upper-triangular.
 */
inline Eigen::VectorXf trsv_upper(const Eigen::MatrixXf& U, const Eigen::VectorXf& b) {
#if FILTERMATH_ARM64
    if (optmath::neon::is_available())
        return optmath::neon::neon_trsv_upper(U, b);
#endif
    return U.triangularView<Eigen::Upper>().solve(b);
}

// ========================================================================
//  Kalman gain via SPD solve:  K = P H^T S^{-1}
//  Avoids explicit inverse by solving S K^T = (P H^T)^T
// ========================================================================
/**
 * Compute the Kalman gain K = PHt * S^{-1} without forming S^{-1} explicitly.
 * Tries SPD column-wise solve first, then explicit inverse, then SVD pseudoinverse
 * as a last resort. PHt = P * H^T and S is the innovation covariance.
 */
inline Eigen::MatrixXf kalman_gain(const Eigen::MatrixXf& PHt,
                                    const Eigen::MatrixXf& S) {
    // Solve S * K^T = PHt^T  →  K^T = S^{-1} * PHt^T  →  K = PHt * S^{-1}
    // More stable: solve column-by-column or use LDLT
    Eigen::MatrixXf K_T = solve_spd_mat(S, PHt.transpose());
    if (K_T.size() > 0)
        return K_T.transpose();

    // Last resort: explicit inverse
    Eigen::MatrixXf S_inv = inverse(S);
    if (S_inv.size() > 0)
        return gemm(PHt, S_inv);

    // SVD pseudoinverse
    Eigen::JacobiSVD<Eigen::MatrixXf> svd(S, Eigen::ComputeFullU | Eigen::ComputeFullV);
    return PHt * svd.solve(Eigen::MatrixXf::Identity(S.rows(), S.cols()));
}

// ========================================================================
//  GPU Availability Query
// ========================================================================
/** Return true if a CUDA GPU is both present and enabled at runtime. */
inline bool gpu_available() {
#if FILTERMATH_HAS_CUDA
    return config::cuda_enabled() && optmath::cuda::is_available();
#else
    return false;
#endif
}

// ========================================================================
//  GPU Synchronization (call after async operations)
// ========================================================================
/** Block until all pending CUDA operations complete. No-op when CUDA is absent. */
inline void gpu_sync() {
#if FILTERMATH_HAS_CUDA
    if (optmath::cuda::is_available()) {
        optmath::cuda::CudaContext::get().synchronize();
    }
#endif
}

// ========================================================================
//  Vector reduction operations (GPU-accelerated for large vectors)
// ========================================================================
/** Sum all elements of v. Uses CUDA parallel reduction for large vectors. */
inline float reduce_sum(const Eigen::VectorXf& v) {
#if FILTERMATH_HAS_CUDA
    if (config::cuda_enabled() &&
        v.size() >= FILTERMATH_CUDA_MIN_DIM * FILTERMATH_CUDA_MIN_DIM &&
        optmath::cuda::is_available()) {
        return optmath::cuda::cuda_reduce_sum(v);
    }
#endif
    return v.sum();
}

/** Return the maximum element of v. Uses CUDA parallel reduction for large vectors. */
inline float reduce_max(const Eigen::VectorXf& v) {
#if FILTERMATH_HAS_CUDA
    if (config::cuda_enabled() &&
        v.size() >= FILTERMATH_CUDA_MIN_DIM * FILTERMATH_CUDA_MIN_DIM &&
        optmath::cuda::is_available()) {
        return optmath::cuda::cuda_reduce_max(v);
    }
#endif
    return v.maxCoeff();
}

// ========================================================================
//  Vectorized exp (for log-weight to weight conversion)
// ========================================================================
/** Element-wise exp(v). Uses CUDA for large vectors (e.g. particle weight conversion). */
inline Eigen::VectorXf vec_exp(const Eigen::VectorXf& v) {
#if FILTERMATH_HAS_CUDA
    if (config::cuda_enabled() &&
        v.size() >= FILTERMATH_CUDA_MIN_DIM * FILTERMATH_CUDA_MIN_DIM &&
        optmath::cuda::is_available()) {
        return optmath::cuda::cuda_exp(v);
    }
#endif
    return v.array().exp().matrix();
}

// ========================================================================
//  Vectorized log
// ========================================================================
/** Element-wise log(v). Uses CUDA for large vectors. */
inline Eigen::VectorXf vec_log(const Eigen::VectorXf& v) {
#if FILTERMATH_HAS_CUDA
    if (config::cuda_enabled() &&
        v.size() >= FILTERMATH_CUDA_MIN_DIM * FILTERMATH_CUDA_MIN_DIM &&
        optmath::cuda::is_available()) {
        return optmath::cuda::cuda_log(v);
    }
#endif
    return v.array().log().matrix();
}

// ========================================================================
//  GPU-accelerated covariance computation from weighted residuals
//  P = sum_i W[i] * (X[:,i] - mean) * (X[:,i] - mean)^T
//  More efficient than N rank-1 updates when N is large.
// ========================================================================
/**
 * Compute a weighted sum of outer products: P = (W.*R) * R^T where R is the
 * residual matrix (columns are deviations from the mean). Used for covariance
 * computation in UKF sigma-point and particle filter operations.
 */
inline Eigen::MatrixXf weighted_outer_sum(
    const Eigen::MatrixXf& residuals,  // NX × N_sigma: each column is (x_i - mean)
    const Eigen::VectorXf& weights)    // N_sigma weights
{
    const int n = residuals.rows();
    const int m = residuals.cols();

#if FILTERMATH_HAS_CUDA
    // GPU implementation: scale columns, then GEMM: (W.*R) * R^T
    if (config::cuda_enabled() &&
        n >= FILTERMATH_CUDA_MIN_DIM &&
        optmath::cuda::is_available()) {
        // Create weighted residual matrix
        Eigen::MatrixXf weighted_res(n, m);
        for (int j = 0; j < m; ++j) {
            weighted_res.col(j) = weights(j) * residuals.col(j);
        }
        return optmath::cuda::cuda_gemm(weighted_res, residuals.transpose());
    }
#endif

#if FILTERMATH_ARM64
    if (m >= 4) {  // Use GEMM for many sigma points
        Eigen::MatrixXf weighted_res(n, m);
        for (int j = 0; j < m; ++j) {
            weighted_res.col(j) = weights(j) * residuals.col(j);
        }
        if (optmath::sve2::is_available())
            return optmath::sve2::sve2_gemm_blocked(weighted_res, residuals.transpose());
        if (optmath::neon::is_available())
            return optmath::neon::neon_gemm(weighted_res, residuals.transpose());
    }
#endif

    // Eigen fallback
    Eigen::MatrixXf weighted_res(n, m);
    for (int j = 0; j < m; ++j) {
        weighted_res.col(j) = weights(j) * residuals.col(j);
    }
    return weighted_res * residuals.transpose();
}

} // namespace filtermath

#endif // FILTERMATH_H
