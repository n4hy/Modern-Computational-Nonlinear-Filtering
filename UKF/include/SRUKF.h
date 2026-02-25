#ifndef SRUKF_H
#define SRUKF_H

#include <Eigen/Dense>
#include <Eigen/QR>
#include <iostream>
#include "StateSpaceModel.h"
#include "SigmaPoints.h"
#include <optmath/neon_kernels.hpp>

namespace UKFCore {

/**
 * Square Root Unscented Kalman Filter (SRUKF)
 *
 * Propagates the Cholesky factor S of the covariance P = S*S^T
 * instead of P directly. This provides better numerical stability
 * and guarantees positive definiteness.
 *
 * Uses QR decomposition for covariance updates and Cholesky
 * updates for efficient square root propagation.
 */
template<int NX, int NY>
class SRUKF {
public:
    using Model = UKFModel::StateSpaceModel<NX, NY>;
    using State = typename Model::State;
    using Observation = typename Model::Observation;
    using StateMat = typename Model::StateMat;
    using ObsMat = typename Model::ObsMat;
    using CrossMat = Eigen::Matrix<float, NX, NY>;
    using SigmaPts = SigmaPoints<NX>;

    // Parameters - dimension adaptive
    float alpha = 1.0f;    // Will be adjusted based on dimension
    float beta = 2.0f;     // Optimal for Gaussian
    float kappa = -1.0f;   // Will be set in initialize()

    SRUKF(Model& model) : model_(model) {
        x_.setZero();
        S_.setIdentity();  // Square root of covariance
    }

    void initialize(const State& x0, const StateMat& P0) {
        x_ = x0;
        // Dimension-adaptive parameters for SRUKF
        // Low/medium dimensions: alpha=1.0 for better weak observability handling
        // High dimensions: alpha=1e-3 to prevent numerical issues
        if (kappa < 0) {
            beta = 2.0f;  // Optimal for Gaussian
            if (NX <= 5) {
                alpha = 1.0f;
                kappa = 3.0f - static_cast<float>(NX);
            } else {
                alpha = 1.0f;
                kappa = 0.0f;
            }
        }

        // Compute Cholesky factor of P0 using NEON-accelerated decomposition
        Eigen::MatrixXf L0 = optmath::neon::neon_cholesky(P0);
        if (L0.size() == 0) {
            StateMat P_jitter = P0 + 1e-6f * StateMat::Identity();
            L0 = optmath::neon::neon_cholesky(P_jitter);
            if (L0.size() == 0) {
                Eigen::LDLT<StateMat> ldlt(P_jitter);
                if (ldlt.info() != Eigen::Success || !ldlt.isPositive()) {
                    L0 = StateMat::Identity();  // Last resort
                } else {
                    StateMat P_ldlt = ldlt.matrixL();
                    Eigen::VectorXf D_sqrt = ldlt.vectorD().cwiseSqrt();
                    L0 = P_ldlt * D_sqrt.asDiagonal();
                }
            }
        }
        S_ = L0;
    }

    /**
     * Prediction Step using Square Root formulation
     * Returns cross-covariance P_{x_k, x_{k+1}} for smoothing
     */
    StateMat predict(float t_k, const Eigen::Ref<const State>& u_k) {
        // 1. Generate Sigma Points from current estimate using S
        StateMat P = S_ * S_.transpose();
        SigmaPts sigmas;
        generate_sigma_points<NX>(x_, P, alpha, beta, kappa, sigmas);

        // 2. Propagate Sigma Points
        typename SigmaPts::SigmaMat X_pred;
        for (int i = 0; i < SigmaPts::NSIG; ++i) {
            X_pred.col(i) = model_.f(sigmas.X.col(i), t_k, u_k);
        }

        // 3. Compute Predicted Mean
        State x_pred_mean = State::Zero();
        for (int i = 0; i < SigmaPts::NSIG; ++i) {
            x_pred_mean += sigmas.Wm(i) * X_pred.col(i);
        }

        // 4. Compute Square Root of Predicted Covariance using QR decomposition
        StateMat Q = model_.Q(t_k);
        Eigen::MatrixXf S_Q_dyn = optmath::neon::neon_cholesky(Q);
        if (S_Q_dyn.size() == 0) {
            Q += 1e-8f * StateMat::Identity();
            S_Q_dyn = optmath::neon::neon_cholesky(Q);
            if (S_Q_dyn.size() == 0) {
                Eigen::LDLT<StateMat> ldlt_Q(Q);
                if (ldlt_Q.info() != Eigen::Success || !ldlt_Q.isPositive()) {
                    S_Q_dyn = StateMat::Identity() * 1e-4f;  // Last resort
                } else {
                    StateMat Q_ldlt = ldlt_Q.matrixL();
                    Eigen::VectorXf D_sqrt = ldlt_Q.vectorD().cwiseSqrt();
                    S_Q_dyn = Q_ldlt * D_sqrt.asDiagonal();
                }
            }
        }
        StateMat S_Q = S_Q_dyn;

        // Build matrix for QR: [sqrt(Wc[1])*X_diff[1], ..., sqrt(Wc[n])*X_diff[n], S_Q]
        // Note: skip i=0 since it has special weight that can be negative
        Eigen::Matrix<float, NX, 3*NX> chi_diff;
        for (int i = 1; i < SigmaPts::NSIG; ++i) {
            State diff = X_pred.col(i) - x_pred_mean;
            float wc_sign = (sigmas.Wc(i) >= 0) ? 1.0f : -1.0f;
            chi_diff.col(i-1) = std::sqrt(std::abs(sigmas.Wc(i))) * diff * wc_sign;
        }
        chi_diff.block(0, 2*NX, NX, NX) = S_Q;  // Copy full S_Q matrix

        // QR decomposition to get S_pred
        Eigen::HouseholderQR<Eigen::Matrix<float, NX, 3*NX>> qr(chi_diff);
        // Extract R factor (upper triangular), transpose to get lower triangular
        Eigen::Matrix<float, NX, 3*NX> R_matrix = qr.matrixQR().template triangularView<Eigen::Upper>();
        StateMat S_pred = R_matrix.block(0, 0, NX, NX).transpose();

        // Handle the i=0 term with rank-1 update (cholupdate)
        State diff_0 = X_pred.col(0) - x_pred_mean;
        float wc_0 = sigmas.Wc(0);
        if (wc_0 < 0) {
            // Rank-1 downdate
            cholupdate_downdate(S_pred, diff_0, std::sqrt(std::abs(wc_0)));
        } else {
            // Rank-1 update
            cholupdate(S_pred, diff_0, std::sqrt(wc_0));
        }

        // 5. Compute Cross Covariance P_{x_k, x_{k+1}} for smoothing
        // Build weighted diff matrices for NEON GEMM: P_cross = Dx_w * Dp^T
        Eigen::Matrix<float, NX, SigmaPts::NSIG> Dx_w, Dp;
        for (int i = 0; i < SigmaPts::NSIG; ++i) {
            Dp.col(i) = X_pred.col(i) - x_pred_mean;
            Dx_w.col(i) = sigmas.Wc(i) * (sigmas.X.col(i) - x_);
        }
        StateMat P_cross = optmath::neon::neon_gemm(Dx_w, Dp.transpose());

        // Update state
        x_ = x_pred_mean;
        S_ = S_pred;

        return P_cross;
    }

    /**
     * Update Step using Square Root formulation
     */
    void update(float t_k, const Observation& y_k) {
        // 1. Generate Sigma Points from predicted state
        StateMat P = S_ * S_.transpose();
        SigmaPts sigmas;
        generate_sigma_points<NX>(x_, P, alpha, beta, kappa, sigmas);

        // 2. Propagate through measurement function
        Eigen::Matrix<float, NY, SigmaPts::NSIG> Y_pred;
        for (int i = 0; i < SigmaPts::NSIG; ++i) {
            Y_pred.col(i) = model_.h(sigmas.X.col(i), t_k);
        }

        // 3. Compute Predicted Measurement Mean
        Observation y_hat = Observation::Zero();
        for (int i = 0; i < SigmaPts::NSIG; ++i) {
            y_hat += sigmas.Wm(i) * Y_pred.col(i);
        }

        // 4. Compute Square Root of Innovation Covariance using QR
        ObsMat R = model_.R(t_k);

        // Compute innovation covariance square root
        // For numerical robustness, compute P_yy directly from all sigma points
        // rather than using QR decomposition
        constexpr int NSIG = SigmaPts::NSIG;
        ObsMat P_yy = ObsMat::Zero();

        for (int i = 0; i < NSIG; ++i) {
            Observation diff_y = Y_pred.col(i) - y_hat;
            P_yy += sigmas.Wc(i) * (diff_y * diff_y.transpose());
        }
        P_yy += R;  // Add measurement noise

        // Ensure positive definite
        P_yy = 0.5f * (P_yy + P_yy.transpose());

        // Compute square root via NEON-accelerated Cholesky
        Eigen::MatrixXf S_yy_dyn = optmath::neon::neon_cholesky(P_yy);
        if (S_yy_dyn.size() == 0) {
            P_yy += 1e-6f * ObsMat::Identity();
            S_yy_dyn = optmath::neon::neon_cholesky(P_yy);
            if (S_yy_dyn.size() == 0) {
                Eigen::LDLT<ObsMat> ldlt_Pyy(P_yy);
                if (ldlt_Pyy.info() != Eigen::Success || !ldlt_Pyy.isPositive()) {
                    // Last resort: use Cholesky of R (measurement noise defines correct scale)
                    Eigen::LLT<ObsMat> llt_R_fallback(R);
                    if (llt_R_fallback.info() == Eigen::Success) {
                        S_yy_dyn = llt_R_fallback.matrixL();
                    } else {
                        // R itself is diagonal, use sqrt of diagonal
                        S_yy_dyn = ObsMat::Zero();
                        for (int i = 0; i < NY; ++i)
                            S_yy_dyn(i,i) = std::sqrt(R(i,i));
                    }
                } else {
                    ObsMat Pyy_ldlt = ldlt_Pyy.matrixL();
                    Eigen::VectorXf D_sqrt = ldlt_Pyy.vectorD().cwiseSqrt();
                    S_yy_dyn = Pyy_ldlt * D_sqrt.asDiagonal();
                }
            }
        }
        ObsMat S_yy = S_yy_dyn;

        // Check S_yy for numerical issues — use measurement noise scale
        for (int i = 0; i < NY; ++i) {
            if (!std::isfinite(S_yy(i,i)) || S_yy(i,i) < 1e-10f) {
                S_yy(i,i) = std::sqrt(R(i,i));  // Scale-adaptive fallback
            }
        }

        // 5. Compute Cross Covariance Pxy using NEON GEMM
        Eigen::Matrix<float, NX, NSIG> Dx_w;
        Eigen::Matrix<float, NY, NSIG> Dy;
        for (int i = 0; i < NSIG; ++i) {
            Dy.col(i) = Y_pred.col(i) - y_hat;
            Dx_w.col(i) = sigmas.Wc(i) * (sigmas.X.col(i) - x_);
        }
        Eigen::MatrixXf Pxy = optmath::neon::neon_gemm(Dx_w, Dy.transpose());

        // 6. Kalman Gain: K = Pxy * P_yy^{-1}
        // Prefer triangular solve (O(N^2)) over explicit inverse (O(N^3)) since we have S_yy
        // K = Pxy * (S_yy * S_yy^T)^{-1} = Pxy * S_yy^{-T} * S_yy^{-1}
        // Solve: S_yy * temp = Pxy^T  =>  temp = S_yy^{-1} * Pxy^T
        // Then:  S_yy^T * K^T = temp  =>  K^T = S_yy^{-T} * temp
        Eigen::Matrix<float, NX, NY> K;
        Eigen::Matrix<float, NY, NX> temp_T = S_yy.template triangularView<Eigen::Lower>()
                                               .solve(Pxy.transpose());
        Eigen::Matrix<float, NY, NX> K_T = S_yy.transpose().template triangularView<Eigen::Upper>()
                                            .solve(temp_T);
        K = K_T.transpose();

        // 7. State Update with NaN detection and rollback
        State x_prev = x_;
        StateMat S_prev = S_;

        Observation innovation = y_k - y_hat;
        State correction = K * innovation;  // Use Eigen directly
        x_ = x_ + correction;

        // 8. Covariance Update using square root form
        // P = P - K*S_yy*S_yy^T*K^T = S*S^T - K*S_yy*S_yy^T*K^T
        // Use QR to compute: [S^T, (K*S_yy)^T]^T and extract updated S
        Eigen::Matrix<float, NX, NY> U = K * S_yy;

        // Use cholupdate for each column of U (rank-NY downdate)
        StateMat S_updated = S_;
        for (int i = 0; i < NY; ++i) {
            cholupdate_downdate(S_updated, U.col(i), 1.0f);
        }
        S_ = S_updated;

        // NaN detection: rollback state and covariance if update produced NaN
        if (!x_.allFinite() || !S_.allFinite()) {
            x_ = x_prev;
            S_ = S_prev;
        }
    }

    // Getters
    const State& getState() const { return x_; }
    StateMat getCovariance() const { return S_ * S_.transpose(); }
    const StateMat& getSqrtCovariance() const { return S_; }

    // Setters
    void setState(const State& x) { x_ = x; }
    void setSqrtCovariance(const StateMat& S) { S_ = S; }

private:
    Model& model_;
    State x_;
    StateMat S_;  // Square root of covariance (Cholesky factor)

    /**
     * Rank-1 Cholesky update: S_new such that
     * S_new * S_new^T = S * S^T + alpha^2 * v * v^T
     */
    void cholupdate(StateMat& S, const State& v, float alpha) {
        State v_scaled = alpha * v;
        constexpr float eps = 1e-10f;
        for (int k = 0; k < NX; ++k) {
            if (std::abs(S(k,k)) < eps) {
                S(k,k) = eps;  // Regularize to prevent division by zero
            }
            float r = std::sqrt(S(k,k)*S(k,k) + v_scaled(k)*v_scaled(k));
            float c = r / S(k,k);
            float s = v_scaled(k) / S(k,k);
            S(k,k) = r;
            if (k < NX - 1) {
                if (std::abs(c) > eps) {
                    S.block(k+1, k, NX-k-1, 1) = (S.block(k+1, k, NX-k-1, 1) + s * v_scaled.segment(k+1, NX-k-1)) / c;
                }
                v_scaled.segment(k+1, NX-k-1) = c * v_scaled.segment(k+1, NX-k-1) - s * S.block(k+1, k, NX-k-1, 1);
            }
        }
    }

    /**
     * Rank-1 Cholesky downdate: S_new such that
     * S_new * S_new^T = S * S^T - alpha^2 * v * v^T
     */
    void cholupdate_downdate(StateMat& S, const State& v, float alpha) {
        State v_scaled = alpha * v;
        constexpr float eps = 1e-10f;
        for (int k = 0; k < NX; ++k) {
            if (std::abs(S(k,k)) < eps) {
                S(k,k) = eps;  // Regularize to prevent division by zero
            }
            float r_sq = S(k,k)*S(k,k) - v_scaled(k)*v_scaled(k);
            if (r_sq <= 0) {
                // Scale-adaptive jitter relative to diagonal element
                r_sq = 1e-6f * S(k,k) * S(k,k);
            }
            float r = std::sqrt(r_sq);
            float c = r / S(k,k);
            float s = v_scaled(k) / S(k,k);
            S(k,k) = r;
            if (k < NX - 1) {
                if (std::abs(c) > eps) {
                    S.block(k+1, k, NX-k-1, 1) = (S.block(k+1, k, NX-k-1, 1) - s * v_scaled.segment(k+1, NX-k-1)) / c;
                }
                v_scaled.segment(k+1, NX-k-1) = c * v_scaled.segment(k+1, NX-k-1) - s * S.block(k+1, k, NX-k-1, 1);
            }
        }
    }

    /**
     * Cholesky update for observation dimension
     */
    void cholupdate_obs(ObsMat& S, const Observation& v, float alpha) {
        Observation v_scaled = alpha * v;
        constexpr float eps = 1e-10f;
        for (int k = 0; k < NY; ++k) {
            if (std::abs(S(k,k)) < eps) {
                S(k,k) = eps;  // Regularize to prevent division by zero
            }
            float r = std::sqrt(S(k,k)*S(k,k) + v_scaled(k)*v_scaled(k));
            float c = r / S(k,k);
            float s = v_scaled(k) / S(k,k);
            S(k,k) = r;
            if (k < NY - 1) {
                if (std::abs(c) > eps) {
                    S.block(k+1, k, NY-k-1, 1) = (S.block(k+1, k, NY-k-1, 1) + s * v_scaled.segment(k+1, NY-k-1)) / c;
                }
                v_scaled.segment(k+1, NY-k-1) = c * v_scaled.segment(k+1, NY-k-1) - s * S.block(k+1, k, NY-k-1, 1);
            }
        }
    }

    /**
     * Cholesky downdate for observation dimension
     */
    void cholupdate_downdate_obs(ObsMat& S, const Observation& v, float alpha) {
        Observation v_scaled = alpha * v;
        constexpr float eps = 1e-10f;
        for (int k = 0; k < NY; ++k) {
            if (std::abs(S(k,k)) < eps) {
                S(k,k) = eps;  // Regularize to prevent division by zero
            }
            float r_sq = S(k,k)*S(k,k) - v_scaled(k)*v_scaled(k);
            if (r_sq <= 0) {
                // Scale-adaptive jitter relative to diagonal element
                r_sq = 1e-6f * S(k,k) * S(k,k);
            }
            float r = std::sqrt(r_sq);
            float c = r / S(k,k);
            float s = v_scaled(k) / S(k,k);
            S(k,k) = r;
            if (k < NY - 1) {
                if (std::abs(c) > eps) {
                    S.block(k+1, k, NY-k-1, 1) = (S.block(k+1, k, NY-k-1, 1) - s * v_scaled.segment(k+1, NY-k-1)) / c;
                }
                v_scaled.segment(k+1, NY-k-1) = c * v_scaled.segment(k+1, NY-k-1) - s * S.block(k+1, k, NY-k-1, 1);
            }
        }
    }
};

} // namespace UKFCore

#endif // SRUKF_H
