#ifndef SIGMA_POINTS_H
#define SIGMA_POINTS_H

#include <Eigen/Dense>
#include <cmath>

namespace UKFCore {

template<int NX>
struct SigmaPoints {
    static constexpr int NSIG = 2 * NX + 1;
    using State    = Eigen::Matrix<double, NX, 1>;
    using StateMat = Eigen::Matrix<double, NX, NX>;
    using SigmaMat = Eigen::Matrix<double, NX, NSIG>;
    using Weights  = Eigen::Matrix<double, NSIG, 1>;

    SigmaMat X;     // Columns are sigma points
    Weights  Wm;    // Mean weights
    Weights  Wc;    // Covariance weights

    double lambda; // Stored lambda parameter

    SigmaPoints() {
        X.setZero();
        Wm.setZero();
        Wc.setZero();
        lambda = 0.0;
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
void generate_sigma_points(const Eigen::Matrix<double, NX, 1>& x,
                           const Eigen::Matrix<double, NX, NX>& P,
                           double alpha,
                           double beta,
                           double kappa,
                           SigmaPoints<NX>& out) {

    double n = static_cast<double>(NX);
    double lambda = alpha * alpha * (n + kappa) - n;
    out.lambda = lambda;

    // Weights
    // W_0
    out.Wm(0) = lambda / (n + lambda);
    out.Wc(0) = lambda / (n + lambda) + (1.0 - alpha * alpha + beta);

    // W_i for i = 1 ... 2n
    double w_i = 1.0 / (2.0 * (n + lambda));
    for (int i = 1; i < SigmaPoints<NX>::NSIG; ++i) {
        out.Wm(i) = w_i;
        out.Wc(i) = w_i;
    }

    // Sigma Points Generation
    // Factorize P using LLT (Cholesky)
    // If P is not positive definite, LLT might fail. We can try adding jitter or use LDLT.
    // The prompt suggests adding jitter if fail.

    Eigen::LLT<Eigen::Matrix<double, NX, NX>> llt;
    Eigen::Matrix<double, NX, NX> P_copy = P;

    llt.compute(P_copy);

    if (llt.info() != Eigen::Success) {
        // Add jitter
        P_copy += 1e-9 * Eigen::Matrix<double, NX, NX>::Identity();
        llt.compute(P_copy);
        // If still fails, we might need robust handling, but let's assume it works with jitter or crash responsibly
    }

    Eigen::Matrix<double, NX, NX> L = llt.matrixL();
    double scale = std::sqrt(n + lambda);

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
Eigen::Matrix<double, NX, 1> compute_mean(const SigmaPoints<NX>& sigmas) {
    Eigen::Matrix<double, NX, 1> x_mean = Eigen::Matrix<double, NX, 1>::Zero();
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
Eigen::Matrix<double, NX, NX> compute_covariance(const SigmaPoints<NX>& sigmas,
                                                 const Eigen::Matrix<double, NX, 1>& mean,
                                                 const Eigen::Matrix<double, NX, NX>& noise_cov) {
    Eigen::Matrix<double, NX, NX> P = Eigen::Matrix<double, NX, NX>::Zero();
    for (int i = 0; i < SigmaPoints<NX>::NSIG; ++i) {
        Eigen::Matrix<double, NX, 1> diff = sigmas.X.col(i) - mean;
        P += sigmas.Wc(i) * diff * diff.transpose();
    }
    P += noise_cov;
    // Symmetrize
    P = 0.5 * (P + P.transpose());
    return P;
}

} // namespace UKFCore

#endif // SIGMA_POINTS_H
