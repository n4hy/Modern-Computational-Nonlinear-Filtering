#ifndef SIGMA_POINTS_H
#define SIGMA_POINTS_H

#include <Eigen/Dense>
#include <Eigen/Cholesky>
#include <cmath>

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
 * Generates Sigma Points using Merwe Scaled Sigma Point algorithm.
 *
 * @param x Mean state
 * @param P Covariance
 * @param alpha Spread parameter
 * @param beta Prior knowledge of distribution (Gaussian = 2)
 * @param kappa Secondary scaling parameter
 * @param out_sigmas Output struct
 */
template<int NX>
void generate_sigma_points(const Eigen::Matrix<float, NX, 1>& x,
                           const Eigen::Matrix<float, NX, NX>& P,
                           float alpha,
                           float beta,
                           float kappa,
                           SigmaPoints<NX>& out) {

    float n = static_cast<float>(NX);
    float lambda = alpha * alpha * (n + kappa) - n;

    // Prevent near-zero or negative (n + lambda) which collapses sigma point spread
    // For alpha=1, kappa=0: lambda=0, n_lambda=n (safe)
    // For alpha=1e-3, kappa=0: lambda ≈ -n, n_lambda ≈ 0 (dangerous)
    float n_lambda = n + lambda;
    if (n_lambda < 0.5f) {
        // Adjust kappa so that n_lambda = n (i.e., lambda = 0)
        // lambda = alpha^2 * (n + kappa) - n = 0  =>  kappa = n/alpha^2 - n
        kappa = n / (alpha * alpha) - n;
        lambda = alpha * alpha * (n + kappa) - n;
        n_lambda = n + lambda;
    }

    out.lambda = lambda;

    // Weights
    // W_0
    out.Wm(0) = lambda / (n + lambda);
    out.Wc(0) = lambda / (n + lambda) + (1.0f - alpha * alpha + beta);

    // W_i for i = 1 ... 2n
    float w_i = 1.0f / (2.0f * (n + lambda));
    for (int i = 1; i < SigmaPoints<NX>::NSIG; ++i) {
        out.Wm(i) = w_i;
        out.Wc(i) = w_i;
    }

    // Sigma Points Generation
    // Factorize P using Eigen LLT
    Eigen::LLT<Eigen::Matrix<float, NX, NX>> llt(P);
    Eigen::Matrix<float, NX, NX> L;
    if (llt.info() == Eigen::Success) {
        L = llt.matrixL();
    } else {
        // Not SPD, add jitter and retry
        Eigen::Matrix<float, NX, NX> P_jitter = P + 1e-6f * Eigen::Matrix<float, NX, NX>::Identity();
        Eigen::LLT<Eigen::Matrix<float, NX, NX>> llt_jitter(P_jitter);
        if (llt_jitter.info() == Eigen::Success) {
            L = llt_jitter.matrixL();
        } else {
            // Ultimate fallback to identity
            L = Eigen::Matrix<float, NX, NX>::Identity();
        }
    }
    float scale = std::sqrt(n + lambda);

    out.X.col(0) = x;
    for (int i = 0; i < NX; ++i) {
        out.X.col(i + 1)      = x + scale * L.col(i);
        out.X.col(i + 1 + NX) = x - scale * L.col(i);
    }
}

/**
 * Reconstruct Mean from Sigma Points
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
 *
 * @param x Mean state
 * @param S Lower triangular Cholesky factor of covariance
 * @param alpha Spread parameter
 * @param beta Prior knowledge of distribution (Gaussian = 2)
 * @param kappa Secondary scaling parameter
 * @param out_sigmas Output struct
 */
template<int NX>
void generate_sigma_points_from_sqrt(const Eigen::Matrix<float, NX, 1>& x,
                                      const Eigen::Matrix<float, NX, NX>& S,
                                      float alpha,
                                      float beta,
                                      float kappa,
                                      SigmaPoints<NX>& out) {

    float n = static_cast<float>(NX);
    float lambda = alpha * alpha * (n + kappa) - n;

    // Prevent near-zero or negative (n + lambda) which collapses sigma point spread
    float n_lambda = n + lambda;
    if (n_lambda < 0.5f) {
        kappa = n / (alpha * alpha) - n;
        lambda = alpha * alpha * (n + kappa) - n;
        n_lambda = n + lambda;
    }

    out.lambda = lambda;

    // Weights
    out.Wm(0) = lambda / (n + lambda);
    out.Wc(0) = lambda / (n + lambda) + (1.0f - alpha * alpha + beta);

    float w_i = 1.0f / (2.0f * (n + lambda));
    for (int i = 1; i < SigmaPoints<NX>::NSIG; ++i) {
        out.Wm(i) = w_i;
        out.Wc(i) = w_i;
    }

    // Sigma Points: use S directly (no Cholesky needed)
    float scale = std::sqrt(n + lambda);

    out.X.col(0) = x;
    for (int i = 0; i < NX; ++i) {
        out.X.col(i + 1)      = x + scale * S.col(i);
        out.X.col(i + 1 + NX) = x - scale * S.col(i);
    }
}

} // namespace UKFCore

#endif // SIGMA_POINTS_H
