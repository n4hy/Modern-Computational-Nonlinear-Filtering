#ifndef UKF_H
#define UKF_H

#include <Eigen/Dense>
#include <iostream>
#include "StateSpaceModel.h"
#include "SigmaPoints.h"

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

        // 4. Compute Predicted Covariance & Cross Covariance P_{x_k, x_{k+1}}
        StateMat P_pred = StateMat::Zero();
        StateMat P_cross = StateMat::Zero();

        StateMat Q = model_.Q(t_k);

        for (int i = 0; i < SigmaPts::NSIG; ++i) {
            State diff_x = sigmas.X.col(i) - x_;
            State diff_x_pred = X_pred.col(i) - x_pred_mean;

            // Use Eigen operators for outer products
            P_pred += sigmas.Wc(i) * (diff_x_pred * diff_x_pred.transpose());
            P_cross += sigmas.Wc(i) * (diff_x * diff_x_pred.transpose());
        }

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

        // 4. Compute Innovation Covariance S and Cross Covariance Pxy
        ObsMat S = ObsMat::Zero();
        CrossMat Pxy = CrossMat::Zero();
        ObsMat R = model_.R(t_k);

        for (int i = 0; i < SigmaPts::NSIG; ++i) {
             State diff_x = sigmas.X.col(i) - x_;
             Observation diff_y = Y_pred.col(i) - y_hat;

             S += sigmas.Wc(i) * (diff_y * diff_y.transpose());
             Pxy += sigmas.Wc(i) * (diff_x * diff_y.transpose());
        }
        S += R;

        // 5. Kalman Gain and Update
        // K = Pxy * S^{-1}
        Eigen::LDLT<ObsMat> ldlt(S);

        Eigen::Matrix<float, NX, NY> K = Pxy * ldlt.solve(ObsMat::Identity());

        Observation y_diff = y_k - y_hat;

        // State update
        x_ = x_ + K * y_diff;

        // Covariance update
        P_ = P_ - K * S * K.transpose();

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
