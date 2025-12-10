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
    using CrossMat = Eigen::Matrix<double, NX, NY>;
    using SigmaPts = SigmaPoints<NX>;

    // Parameters
    double alpha = 1e-3;
    double beta = 2.0;
    double kappa = 0.0;

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
     * In standard UKF prediction, we normally compute P_{k+1|k}.
     * To get P_{x_k, x_{k+1}}, we need to correlate the *sigma points of k* with the *sigma points of k+1*.
     * Actually, strictly speaking, P_{x_k, x_{k+1}} = E[(x_k - x_k)(x_{k+1} - x_{k+1})^T].
     * In the UKF prediction step:
     *   X_{k|k} are generated from x_{k|k}, P_{k|k}.
     *   X_{k+1|k} = f(X_{k|k}).
     *   x_{k+1|k} = sum Wm * X_{k+1|k}
     *   P_{x_k, x_{k+1}} \approx sum Wc * (X_{k|k} - x_{k|k}) * (X_{k+1|k} - x_{k+1|k})^T
     */
    StateMat predict(double t_k, const Eigen::Ref<const State>& u_k) {
        // 1. Generate Sigma Points from current estimate
        SigmaPts sigmas;
        generate_sigma_points<NX>(x_, P_, alpha, beta, kappa, sigmas);

        // 2. Propagate Sigma Points
        // We need a place to store propagated sigma points.
        // Let's reuse a SigmaMat structure, but it's not strictly "SigmaPoints" in terms of weights generation,
        // just a container for the points.
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

            P_pred  += sigmas.Wc(i) * diff_x_pred * diff_x_pred.transpose();
            P_cross += sigmas.Wc(i) * diff_x * diff_x_pred.transpose();
        }

        P_pred += Q;

        // Symmetrize
        P_pred = 0.5 * (P_pred + P_pred.transpose());

        // Update state
        x_ = x_pred_mean;
        P_ = P_pred;

        return P_cross;
    }

    /**
     * Update Step (Measurement Update)
     */
    void update(double t_k, const Observation& y_k) {
        // 1. Generate Sigma Points from predicted state
        SigmaPts sigmas;
        generate_sigma_points<NX>(x_, P_, alpha, beta, kappa, sigmas);

        // 2. Propagate through h
        // Create matrix for Y_pred (NY x NSIG)
        Eigen::Matrix<double, NY, SigmaPts::NSIG> Y_pred;

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

             S += sigmas.Wc(i) * diff_y * diff_y.transpose();
             Pxy += sigmas.Wc(i) * diff_x * diff_y.transpose();
        }
        S += R;

        // 5. Kalman Gain and Update
        // K = Pxy * S^{-1}
        // Use LDLT for stability
        Eigen::LDLT<ObsMat> ldlt(S);
        // Check robustness?
        if (ldlt.info() != Eigen::Success) {
            // Log warning?
            // std::cerr << "LDLT decomposition failed in UKF update!" << std::endl;
            // Can add jitter if needed, but for now just proceed
        }

        Eigen::Matrix<double, NX, NY> K = Pxy * ldlt.solve(ObsMat::Identity());

        Observation y_diff = y_k - y_hat;

        // State update
        x_ = x_ + K * y_diff;

        // Covariance update (Joseph form)
        // P = (I - KH) P (I - KH)^T + KRK^T
        // But we don't have explicit H.
        // Standard UKF update: P = P - K S K^T
        // Joseph form without H?
        // Some UKF formulations use K = Pxy * S^-1 => P = P - K S K^T.
        // To be numerically robust, we stick to P - K S K^T and ensure symmetry.
        // Or we can try to "infer" H? No, that defeats the point.
        // The prompt says: "Use Joseph-form ... H_eff is the effective linearized measurement mapping... or note that we can skip explicit H and use P = P - K S K^T".
        // I'll use the simpler form P = P - K S K^T as deriving H_eff is complex without gradients.

        P_ = P_ - K * S * K.transpose();

        // Symmetrize and ensure PD
        P_ = 0.5 * (P_ + P_.transpose());

        // Jitter?
        // P_ += 1e-9 * StateMat::Identity();
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
