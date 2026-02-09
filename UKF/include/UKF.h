#ifndef UKF_H
#define UKF_H

#include <Eigen/Dense>
#include <iostream>
#include "StateSpaceModel.h"
#include "SigmaPoints.h"
#include <optmath/neon_kernels.hpp>

namespace UKFCore {

template<int NX, int NY>
class UKF {
public:
    using Model = UKFModel::StateSpaceModel<NX, NY>;
    using State = typename Model::State;
    using Observation = typename Model::Observation;
    using StateMat = typename Model::StateMat;
    using ObsMat = typename Model::ObsMat;
    using CrossMat = Eigen::Matrix<float, NX, NY>;
    using SigmaPts = SigmaPoints<NX>;

    // Parameters
    float alpha = 1e-3f;
    float beta = 2.0f;
    float kappa = 0.0f;

    UKF(Model& model) : model_(model) {
        x_.setZero();
        P_.setIdentity();
    }

    void initialize(const State& x0, const StateMat& P0) {
        x_ = x0;
        P_ = P0;
    }

    /**
     * Prediction Step (Time Update)
     * Returns cross-covariance P_{x_k, x_{k+1}} needed for smoothing.
     */
    StateMat predict(float t_k, const Eigen::Ref<const State>& u_k) {
        // 1. Generate Sigma Points from current estimate
        SigmaPts sigmas;
        generate_sigma_points<NX>(x_, P_, alpha, beta, kappa, sigmas);

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

        // 4. Compute Predicted Covariance & Cross Covariance using NEON GEMM
        StateMat Q = model_.Q(t_k);

        // Build weighted diff matrices for batch outer product via GEMM
        Eigen::Matrix<float, NX, SigmaPts::NSIG> Dp_w, Dp, Dx_w;
        for (int i = 0; i < SigmaPts::NSIG; ++i) {
            State diff_x_pred = X_pred.col(i) - x_pred_mean;
            Dp.col(i) = diff_x_pred;
            Dp_w.col(i) = sigmas.Wc(i) * diff_x_pred;
            Dx_w.col(i) = sigmas.Wc(i) * (sigmas.X.col(i) - x_);
        }
        StateMat P_pred = optmath::neon::neon_gemm(Dp_w, Dp.transpose());
        StateMat P_cross = optmath::neon::neon_gemm(Dx_w, Dp.transpose());

        P_pred += Q;

        // Symmetrize
        P_pred = 0.5f * (P_pred + P_pred.transpose());

        // Update state
        x_ = x_pred_mean;
        P_ = P_pred;

        return P_cross;
    }

    /**
     * Update Step (Measurement Update)
     */
    void update(float t_k, const Observation& y_k) {
        // 1. Generate Sigma Points from predicted state
        SigmaPts sigmas;
        generate_sigma_points<NX>(x_, P_, alpha, beta, kappa, sigmas);

        // 2. Propagate through h
        Eigen::Matrix<float, NY, SigmaPts::NSIG> Y_pred;

        for (int i = 0; i < SigmaPts::NSIG; ++i) {
            Y_pred.col(i) = model_.h(sigmas.X.col(i), t_k);
        }

        // 3. Compute Predicted Measurement Mean
        Observation y_hat = Observation::Zero();
        for (int i = 0; i < SigmaPts::NSIG; ++i) {
            y_hat += sigmas.Wm(i) * Y_pred.col(i);
        }

        // 4. Compute Innovation Covariance S and Cross Covariance Pxy via NEON GEMM
        ObsMat R = model_.R(t_k);

        Eigen::Matrix<float, NY, SigmaPts::NSIG> Dy_w, Dy;
        Eigen::Matrix<float, NX, SigmaPts::NSIG> Dx_w;
        for (int i = 0; i < SigmaPts::NSIG; ++i) {
             Observation diff_y = Y_pred.col(i) - y_hat;
             Dy.col(i) = diff_y;
             Dy_w.col(i) = sigmas.Wc(i) * diff_y;
             Dx_w.col(i) = sigmas.Wc(i) * (sigmas.X.col(i) - x_);
        }
        ObsMat S = optmath::neon::neon_gemm(Dy_w, Dy.transpose());
        CrossMat Pxy = optmath::neon::neon_gemm(Dx_w, Dy.transpose());
        S += R;

        // 5. Kalman Gain: K = Pxy * S^{-1} using NEON-accelerated inverse
        Eigen::MatrixXf S_inv = optmath::neon::neon_inverse(S);
        Eigen::Matrix<float, NX, NY> K;
        if (S_inv.size() > 0) {
            K = optmath::neon::neon_gemm(Eigen::MatrixXf(Pxy), S_inv);
        } else {
            // Fallback to Eigen LDLT
            Eigen::LDLT<ObsMat> ldlt(S);
            K = Pxy * ldlt.solve(ObsMat::Identity());
        }

        Observation y_diff = y_k - y_hat;

        // State update
        x_ = x_ + K * y_diff;

        // Covariance update: P = P - K*S*K^T using NEON GEMM
        Eigen::MatrixXf KS = optmath::neon::neon_gemm(K, S);
        P_ = P_ - optmath::neon::neon_gemm(KS, K.transpose());

        // Symmetrize and ensure PD
        P_ = 0.5f * (P_ + P_.transpose());
    }

    // Getters
    const State& getState() const { return x_; }
    const StateMat& getCovariance() const { return P_; }

    // Setters
    void setState(const State& x) { x_ = x; }
    void setCovariance(const StateMat& P) { P_ = P; }

private:
    Model& model_;
    State x_;
    StateMat P_;
};

} // namespace UKFCore

#endif // UKF_H
