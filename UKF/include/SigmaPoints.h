#ifndef SIGMA_POINTS_H
#define SIGMA_POINTS_H

#include <Eigen/Dense>
#include <Eigen/Cholesky>
#include <cmath>
#include <iostream>  // std::cerr in the decomposition-failure warning below
#include "FilterMath.h"

namespace UKFCore {

template<int NX>
struct SigmaPoints {
    static constexpr int NSIG = 2 * NX + 1;
    using State    = Eigen::Matrix<float, NX, 1>;
    using StateMat = Eigen::Matrix<float, NX, NX>;
    using SigmaMat = Eigen::Matrix<float, NX, NSIG>;
    using Weights  = Eigen::Matrix<float, NSIG, 1>;

    SigmaMat X;     // Columns are sigma points
    Weights  Wm;    // Mean weights
    Weights  Wc;    // Covariance weights

    float lambda; // Stored lambda parameter

    SigmaPoints() {
        X.setZero();
        Wm.setZero();
        Wc.setZero();
        lambda = 0.0f;
    }
};

/**
 * Scaled-UT weights (van der Merwe): lambda = alpha^2 (n + kappa) - n,
 * Wm0 = lambda/(n+lambda), Wc0 = Wm0 + (1 - alpha^2 + beta), Wi = 1/(2(n+lambda)).
 *
 * Spread floor: when n + lambda < 0.5 (e.g. the classic alpha = 1e-3, kappa = 0)
 * the sigma points would collapse onto the mean, so the spread is reset to
 * n + lambda = n (lambda = 0). This is done by raising alpha to
 * alpha_eff^2 = n / (n + kappa) with kappa KEPT, and Wc0 uses alpha_eff. The
 * previous floor instead rewrote kappa = n/alpha^2 - n but kept the caller's
 * alpha in Wc0, giving Wc0 = 1 - alpha^2 + beta (~3) where the spread it
 * actually produced (alpha = 1) calls for beta (= 2). Derivation, n = 1,
 * x ~ N(0,1), y = x^2 (true var 2): the UT returns var = alpha^2 kappa + beta,
 * exact for beta = 2 iff alpha^2 kappa = 0; the old floor had alpha^2 kappa =
 * 1 - alpha^2 (var ~3), the new one alpha_eff^2 kappa = 0 for kappa = 0 (var 2).
 * Also removes the alpha = 0 division (kappa = inf, lambda = NaN).
 */
template<int NX>
void compute_sigma_weights(float alpha, float beta, float kappa, SigmaPoints<NX>& out) {
    const float n = static_cast<float>(NX);
    float alpha2 = alpha * alpha;
    float lambda = alpha2 * (n + kappa) - n;

    if (!(n + lambda >= 0.5f)) {
        if (n + kappa > 0.0f) {
            alpha2 = n / (n + kappa);
        } else {
            // n + kappa <= 0 cannot be repaired by scaling alpha; use the
            // unscaled symmetric set (alpha = 1, kappa = 0).
            alpha2 = 1.0f;
            kappa = 0.0f;
        }
        lambda = alpha2 * (n + kappa) - n;
    }

    out.lambda = lambda;
    out.Wm(0) = lambda / (n + lambda);
    out.Wc(0) = lambda / (n + lambda) + (1.0f - alpha2 + beta);

    const float w_i = 1.0f / (2.0f * (n + lambda));
    for (int i = 1; i < SigmaPoints<NX>::NSIG; ++i) {
        out.Wm(i) = w_i;
        out.Wc(i) = w_i;
    }
}

/**
 * Square-root factor S (S S^T = A) from an Eigen LDLT of A.
 *
 * Eigen's LDLT pivots: it factors P A P^T = L D L^T with P = transpositionsP(),
 * so A = (P^T L sqrt(D)) (P^T L sqrt(D))^T. Using L sqrt(D) alone (as both
 * fallbacks here previously did) factors the PERMUTED matrix -- e.g. it swaps
 * the two variances of [[1e4, 2e4], [2e4, 4e4]]. D is floored at 1e-10.
 */
template<int NX>
Eigen::Matrix<float, NX, NX> ldlt_sqrt_factor(const Eigen::LDLT<Eigen::Matrix<float, NX, NX>>& ldlt) {
    const Eigen::Matrix<float, NX, 1> d = ldlt.vectorD().cwiseMax(1e-10f).cwiseSqrt();
    const Eigen::Matrix<float, NX, NX> L = ldlt.matrixL();  // dense copy of the unit-lower factor
    const Eigen::Matrix<float, NX, NX> LD = L * d.asDiagonal();
    return ldlt.transpositionsP().transpose() * LD;
}

/**
 * Generates Sigma Points using Merwe Scaled Sigma Point algorithm.
 */
template<int NX>
void generate_sigma_points(const Eigen::Matrix<float, NX, 1>& x,
                           const Eigen::Matrix<float, NX, NX>& P,
                           float alpha,
                           float beta,
                           float kappa,
                           SigmaPoints<NX>& out) {

    compute_sigma_weights<NX>(alpha, beta, kappa, out);

    // Sigma Points Generation — use accelerated Cholesky with fallback chain
    Eigen::MatrixXf L_dyn = filtermath::cholesky(P);
    Eigen::Matrix<float, NX, NX> L;

    if (L_dyn.size() > 0) {
        L = L_dyn;
    } else {
        // Add jitter and retry
        Eigen::Matrix<float, NX, NX> P_jitter = P + 1e-6f * Eigen::Matrix<float, NX, NX>::Identity();
        L_dyn = filtermath::cholesky(P_jitter);
        if (L_dyn.size() > 0) {
            L = L_dyn;
        } else {
            // Try LDLT decomposition
            Eigen::LDLT<Eigen::Matrix<float, NX, NX>> ldlt(P_jitter);
            if (ldlt.info() == Eigen::Success && ldlt.isPositive()) {
                L = ldlt_sqrt_factor<NX>(ldlt);
            } else {
                // Ultimate fallback to scaled identity based on P diagonal
                std::cerr << "[SigmaPoints] WARNING: Covariance decomposition failed, using diagonal fallback" << std::endl;
                L = Eigen::Matrix<float, NX, NX>::Zero();
                for (int j = 0; j < NX; ++j) {
                    L(j, j) = std::sqrt(std::max(P(j, j), 1e-8f));
                }
            }
        }
    }
    const float scale = std::sqrt(static_cast<float>(NX) + out.lambda);

    out.X.col(0) = x;
    for (int i = 0; i < NX; ++i) {
        out.X.col(i + 1)      = x + scale * L.col(i);
        out.X.col(i + 1 + NX) = x - scale * L.col(i);
    }
}

/**
 * Reconstruct Mean from Sigma Points.
 * Computes x_mean = sum_i Wm[i] * X[:,i].
 */
template<int NX>
Eigen::Matrix<float, NX, 1> compute_mean(const SigmaPoints<NX>& sigmas) {
    Eigen::Matrix<float, NX, 1> x_mean = Eigen::Matrix<float, NX, 1>::Zero();
    for (int i = 0; i < SigmaPoints<NX>::NSIG; ++i) {
        x_mean += sigmas.Wm(i) * sigmas.X.col(i);
    }
    return x_mean;
}

/**
 * Reconstruct Covariance from Sigma Points and Mean
 * Adds noise covariance Q_or_R to the result.
 */
template<int NX>
Eigen::Matrix<float, NX, NX> compute_covariance(const SigmaPoints<NX>& sigmas,
                                                 const Eigen::Matrix<float, NX, 1>& mean,
                                                 const Eigen::Matrix<float, NX, NX>& noise_cov) {
    Eigen::Matrix<float, NX, NX> P = Eigen::Matrix<float, NX, NX>::Zero();
    for (int i = 0; i < SigmaPoints<NX>::NSIG; ++i) {
        Eigen::Matrix<float, NX, 1> diff = sigmas.X.col(i) - mean;
        P += sigmas.Wc(i) * diff * diff.transpose();
    }
    P += noise_cov;
    // Symmetrize
    P = 0.5f * (P + P.transpose());
    return P;
}

/**
 * Generates Sigma Points directly from the square root (Cholesky factor) S
 * where P = S * S^T. This avoids recomputing the Cholesky decomposition.
 */
template<int NX>
void generate_sigma_points_from_sqrt(const Eigen::Matrix<float, NX, 1>& x,
                                      const Eigen::Matrix<float, NX, NX>& S,
                                      float alpha,
                                      float beta,
                                      float kappa,
                                      SigmaPoints<NX>& out) {

    compute_sigma_weights<NX>(alpha, beta, kappa, out);

    // Sigma Points: use S directly (no Cholesky needed)
    const float scale = std::sqrt(static_cast<float>(NX) + out.lambda);

    out.X.col(0) = x;
    for (int i = 0; i < NX; ++i) {
        out.X.col(i + 1)      = x + scale * S.col(i);
        out.X.col(i + 1 + NX) = x - scale * S.col(i);
    }
}

} // namespace UKFCore

#endif // SIGMA_POINTS_H
