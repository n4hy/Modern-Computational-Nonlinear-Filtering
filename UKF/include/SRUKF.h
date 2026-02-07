#ifndef SRUKF_H
#define SRUKF_H

#include <Eigen/Dense>
#include <Eigen/QR>
#include <iostream>
#include "StateSpaceModel.h"
#include "SigmaPoints.h"

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
                alpha = 1e-3f;
                kappa = 0.0f;
            }
        }

        // Compute Cholesky factor of P0
        Eigen::LLT<StateMat> llt(P0);
        if (llt.info() != Eigen::Success) {
            StateMat P_jitter = P0 + 1e-6f * StateMat::Identity();
            llt.compute(P_jitter);
        }
        S_ = llt.matrixL();
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
        Eigen::LLT<StateMat> llt_Q(Q);
        if (llt_Q.info() != Eigen::Success) {
            // Q is not positive definite, add regularization
            Q += 1e-8f * StateMat::Identity();
            llt_Q.compute(Q);
        }
        StateMat S_Q = llt_Q.matrixL();  // Square root of Q

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
        StateMat P_cross = StateMat::Zero();
        for (int i = 0; i < SigmaPts::NSIG; ++i) {
            State diff_x = sigmas.X.col(i) - x_;
            State diff_x_pred = X_pred.col(i) - x_pred_mean;
            P_cross += sigmas.Wc(i) * (diff_x * diff_x_pred.transpose());  // Use Eigen directly
        }

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
        Eigen::LLT<ObsMat> llt_R(R);
        if (llt_R.info() != Eigen::Success) {
            // R is not positive definite, add regularization
            R += 1e-8f * ObsMat::Identity();
            llt_R.compute(R);
        }
        ObsMat S_R = llt_R.matrixL();

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

        // Compute square root via Cholesky
        Eigen::LLT<ObsMat> llt_Pyy(P_yy);
        if (llt_Pyy.info() != Eigen::Success) {
            P_yy += 1e-6f * ObsMat::Identity();
            llt_Pyy.compute(P_yy);
        }
        ObsMat S_yy = llt_Pyy.matrixL();

        // Check S_yy for numerical issues
        for (int i = 0; i < NY; ++i) {
            if (!std::isfinite(S_yy(i,i)) || S_yy(i,i) < 1e-10f) {
                S_yy(i,i) = 0.01f;  // Restore to reasonable value
            }
        }

        // 5. Compute Cross Covariance Pxy
        CrossMat Pxy = CrossMat::Zero();
        for (int i = 0; i < NSIG; ++i) {
            State diff_x = sigmas.X.col(i) - x_;
            Observation diff_y = Y_pred.col(i) - y_hat;
            Pxy += sigmas.Wc(i) * (diff_x * diff_y.transpose());  // Use Eigen directly
        }

        // 6. Kalman Gain: K = Pxy * inv(S_yy * S_yy^T)
        // Since P_yy = S_yy * S_yy^T, we have K = Pxy * P_yy^{-1}
        // Solve in two steps:
        //   1. S_yy * temp^T = Pxy^T  => temp^T = S_yy^{-1} * Pxy^T
        //   2. S_yy^T * K^T = temp^T  => K^T = (S_yy^T)^{-1} * temp^T
        Eigen::Matrix<float, NY, NX> temp_T = S_yy.template triangularView<Eigen::Lower>()
                                               .solve(Pxy.transpose());
        Eigen::Matrix<float, NY, NX> K_T = S_yy.transpose().template triangularView<Eigen::Upper>()
                                            .solve(temp_T);
        Eigen::Matrix<float, NX, NY> K = K_T.transpose();

        // 7. State Update
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
                // Numerical issue, add small jitter
                r_sq = 1e-8f;
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
                r_sq = 1e-8f;
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
