#ifndef FILTERMATH_H
#define FILTERMATH_H

/**
 * FilterMath.h — Unified dispatch layer for accelerated linear algebra.
 *
 * Dispatch priority:
 *   GEMM:                    SVE2 → NEON → Eigen
 *   Cholesky/Inverse/Solve:  NEON → Eigen
 *   mat-vec multiply:        NEON → Eigen
 *
 * On non-ARM platforms (x86, etc.) all paths fall through to Eigen,
 * so the filter code compiles and runs correctly everywhere.
 */

#include <Eigen/Dense>
#include <Eigen/Cholesky>

// ---------- platform detection ----------
#if defined(__aarch64__) || defined(_M_ARM64)
  #define FILTERMATH_ARM64 1
#else
  #define FILTERMATH_ARM64 0
#endif

#if FILTERMATH_ARM64
  #include <optmath/neon_kernels.hpp>
  #include <optmath/sve2_kernels.hpp>
#endif

namespace filtermath {

// ========================================================================
//  GEMM  —  SVE2 > NEON > Eigen
// ========================================================================
inline Eigen::MatrixXf gemm(const Eigen::MatrixXf& A, const Eigen::MatrixXf& B) {
#if FILTERMATH_ARM64
    if (optmath::sve2::is_available())
        return optmath::sve2::sve2_gemm_blocked(A, B);
    if (optmath::neon::is_available())
        return optmath::neon::neon_gemm(A, B);
#endif
    return A * B;
}

// ========================================================================
//  Matrix-vector multiply  —  NEON > Eigen
// ========================================================================
inline Eigen::VectorXf mat_vec_mul(const Eigen::MatrixXf& A, const Eigen::VectorXf& v) {
#if FILTERMATH_ARM64
    if (optmath::neon::is_available())
        return optmath::neon::neon_mat_vec_mul(A, v);
#endif
    return A * v;
}

// ========================================================================
//  Cholesky (lower-triangular L where A = L L^T)
//  Returns empty matrix on failure.
// ========================================================================
inline Eigen::MatrixXf cholesky(const Eigen::MatrixXf& A) {
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
// ========================================================================
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
// ========================================================================
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

} // namespace filtermath

#endif // FILTERMATH_H
