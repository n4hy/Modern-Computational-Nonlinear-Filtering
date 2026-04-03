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
#include <algorithm>  // std::min, std::max

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

// ========================================================================
//  Matrix-vector multiply  —  CUDA > NEON > Eigen
// ========================================================================
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

// ========================================================================
//  Cholesky (lower-triangular L where A = L L^T)
//  Returns empty matrix on failure.
//  Dispatch: NEON > Eigen (CUDA cuSOLVER not yet implemented in OptimizedKernels)
// ========================================================================
inline Eigen::MatrixXf cholesky(const Eigen::MatrixXf& A) {
    // Note: CUDA dispatch disabled - cuda_cholesky not implemented in OptimizedKernels
    // TODO: Enable when cuSOLVER wrappers are added to OptimizedKernels

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
//  Note: CUDA dispatch disabled - cuda_inverse not implemented in OptimizedKernels
// ========================================================================
inline Eigen::MatrixXf inverse(const Eigen::MatrixXf& A) {
    // TODO: Enable CUDA when cuSOLVER wrappers are added to OptimizedKernels

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
//  Dispatch: NEON > Eigen (CUDA cuSOLVER not yet implemented in OptimizedKernels)
// ========================================================================
inline Eigen::VectorXf solve_spd(const Eigen::MatrixXf& A, const Eigen::VectorXf& b) {
    // TODO: Enable CUDA when cuSOLVER wrappers are added to OptimizedKernels

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
inline Eigen::MatrixXf solve_spd_mat(const Eigen::MatrixXf& A, const Eigen::MatrixXf& B) {
    // Use column-wise solve_spd for each column of B
    Eigen::MatrixXf X(B.rows(), B.cols());
    for (int j = 0; j < B.cols(); ++j) {
        Eigen::VectorXf col = solve_spd(A, B.col(j));
        if (col.size() == 0) {
            // Fallback: LDLT solve all at once
            Eigen::LDLT<Eigen::MatrixXf> ldlt(A);
            if (ldlt.info() == Eigen::Success)
                return ldlt.solve(B);
            return Eigen::MatrixXf();
        }
        X.col(j) = col;
    }
    return X;
}

// ========================================================================
//  Triangular solve (lower): L x = b
// ========================================================================
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
