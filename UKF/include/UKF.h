#ifndef UKF_H
#define UKF_H

#include <Eigen/Dense>
#include <cmath>
#include <iostream>
#include <numbers>
#include "StateSpaceModel.h"
#include "SigmaPoints.h"
#include "FilterMath.h"

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

    /** Construct UKF with a reference to the nonlinear state-space model. */
    UKF(Model& model) : model_(model) {
        x_.setZero();
        P_.setIdentity();
    }

    /** Set initial state estimate and covariance. */
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

        // 3. Compute Predicted Mean (circular for angular states, as in SRUKF)
        const State x_pred_mean = weighted_mean_state(X_pred, sigmas.Wm);

        // 4. Compute Predicted Covariance & Cross Covariance via GEMM
        StateMat Q = model_.Q(t_k);

        Eigen::Matrix<float, NX, SigmaPts::NSIG> Dp_w, Dp, Dx_w;
        for (int i = 0; i < SigmaPts::NSIG; ++i) {
            const State diff_x_pred = wrap_state(X_pred.col(i) - x_pred_mean);
            Dp.col(i) = diff_x_pred;
            Dp_w.col(i) = sigmas.Wc(i) * diff_x_pred;
            Dx_w.col(i) = sigmas.Wc(i) * wrap_state(sigmas.X.col(i) - x_);
        }
        StateMat P_pred = filtermath::gemm(Dp_w, Dp.transpose());
        StateMat P_cross = filtermath::gemm(Dx_w, Dp.transpose());

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
        // A non-finite measurement is treated as missed: skip the update and
        // keep the prediction. Fusing it would turn x_ and P_ into NaN.
        if (!y_k.allFinite()) return;

        // 1. Generate Sigma Points from predicted state
        SigmaPts sigmas;
        generate_sigma_points<NX>(x_, P_, alpha, beta, kappa, sigmas);

        // 2. Propagate through h
        Eigen::Matrix<float, NY, SigmaPts::NSIG> Y_pred;

        for (int i = 0; i < SigmaPts::NSIG; ++i) {
            Y_pred.col(i) = model_.h(sigmas.X.col(i), t_k);
        }

        // 3. Compute Predicted Measurement Mean (circular for angular observations)
        const Observation y_hat = weighted_mean_obs(Y_pred, sigmas.Wm);

        // 4. Compute Innovation Covariance S and Cross Covariance Pxy via GEMM
        ObsMat R = model_.R(t_k);

        Eigen::Matrix<float, NY, SigmaPts::NSIG> Dy_w, Dy;
        Eigen::Matrix<float, NX, SigmaPts::NSIG> Dx_w;
        for (int i = 0; i < SigmaPts::NSIG; ++i) {
             const Observation diff_y = wrap_obs(Y_pred.col(i) - y_hat);
             Dy.col(i) = diff_y;
             Dy_w.col(i) = sigmas.Wc(i) * diff_y;
             Dx_w.col(i) = sigmas.Wc(i) * wrap_state(sigmas.X.col(i) - x_);
        }
        ObsMat S = filtermath::gemm(Dy_w, Dy.transpose());
        CrossMat Pxy = filtermath::gemm(Dx_w, Dy.transpose());
        S += R;

        // 5. Kalman Gain via SPD solve (avoids explicit inverse)
        Eigen::Matrix<float, NX, NY> K = filtermath::kalman_gain(
            Eigen::MatrixXf(Pxy), Eigen::MatrixXf(S));

        const Observation y_diff = wrap_obs(y_k - y_hat);

        // State update
        x_ = x_ + K * y_diff;

        // Covariance update — Joseph-style symmetric four-term form defended by
        // a multi-tier recovery ladder.
        //
        // Form: P = P - K*Pxy^T - Pxy*K^T + K*S*K^T. Algebraically identical to
        // P - K*S*K^T (since Pxy = K*S makes K*Pxy^T = Pxy*K^T = K*S*K^T), but
        // symmetric by construction so rounding partially cancels — the same
        // PSD-preserving intent as the EKF/RBPF Joseph updates, without needing
        // an explicit H. Fixed-size types (NX/NY known at compile time) keep
        // every gemm on FilterMath.h's compile-time fast path.
        Eigen::Matrix<float, NX, NY> KS    = filtermath::gemm(K, S);
        StateMat                     KSKt  = filtermath::gemm(KS, K.transpose());
        StateMat                     KPxyT = filtermath::gemm(K, Pxy.transpose());
        StateMat P_new = P_ - KPxyT - KPxyT.transpose() + KSKt;
        P_new = 0.5f * (P_new + P_new.transpose());

        // Recovery ladder: LLT → LLT+relative-jitter → LDLT reconstruction →
        // diagonal clamp. Mirrors SRUKF.h so both filters have identical
        // numerical defenses.
        Eigen::LLT<StateMat> llt_check(P_new);
        if (llt_check.info() == Eigen::Success) {
            P_ = P_new;
        } else {
            // Relative jitter: 1e-6 floor with 1e-8 * trace(P)/NX scaling so
            // small-covariance states are not artificially inflated.
            float trace_over_n = std::max(P_new.trace() / static_cast<float>(NX), 0.0f);
            float jitter = std::max(1e-6f, 1e-8f * trace_over_n);
            StateMat P_jitter = P_new + jitter * StateMat::Identity();
            Eigen::LLT<StateMat> llt_retry(P_jitter);
            if (llt_retry.info() == Eigen::Success) {
                P_ = P_jitter;
            } else {
                // LDLT reconstruction (rebuild PSD approximant from spectrum).
                Eigen::LDLT<StateMat> ldlt(P_jitter);
                if (ldlt.info() == Eigen::Success && ldlt.isPositive()) {
                    const StateMat sqrtP = ldlt_sqrt_factor<NX>(ldlt);  // pivot-aware
                    P_ = sqrtP * sqrtP.transpose();
                } else {
                    // Last-resort clamp: keep the (pre-update, hence conservative)
                    // predicted variances, discard off-diagonals. The diagonal is
                    // copied out first: `P_ = P_.diagonal()...asDiagonal()` aliases
                    // -- Eigen zeroes P_ before reading its diagonal, yielding
                    // 1e-10 * I instead of diag(P_).
                    const State d = P_.diagonal().cwiseMax(1e-10f);
                    P_ = d.asDiagonal();
                }
            }
        }
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

    // ---- angular handling (model_.isAngularState / isAngularObservation) ----
    // Mirrors SRUKF: circular weighted means and deviations wrapped to [-pi, pi].
    // All four are identities for models that flag nothing as angular.
    static constexpr float kTwoPi = 2.0f * std::numbers::pi_v<float>;

    State wrap_state(State d) const {
        for (int j = 0; j < NX; ++j)
            if (model_.isAngularState(j)) d(j) = std::remainder(d(j), kTwoPi);
        return d;
    }
    Observation wrap_obs(Observation d) const {
        for (int j = 0; j < NY; ++j)
            if (model_.isAngularObservation(j)) d(j) = std::remainder(d(j), kTwoPi);
        return d;
    }
    template<int ROWS, typename IsAngular>
    static Eigen::Matrix<float, ROWS, 1> weighted_mean(
            const Eigen::Matrix<float, ROWS, SigmaPts::NSIG>& X,
            const typename SigmaPts::Weights& W, IsAngular is_angular) {
        Eigen::Matrix<float, ROWS, 1> m = X * W;
        for (int j = 0; j < ROWS; ++j) {
            if (!is_angular(j)) continue;
            float s = 0.0f, c = 0.0f;
            for (int i = 0; i < SigmaPts::NSIG; ++i) {
                s += W(i) * std::sin(X(j, i));
                c += W(i) * std::cos(X(j, i));
            }
            m(j) = std::atan2(s, c);
        }
        return m;
    }
    State weighted_mean_state(const typename SigmaPts::SigmaMat& X,
                              const typename SigmaPts::Weights& W) const {
        return weighted_mean<NX>(X, W, [this](int j) { return model_.isAngularState(j); });
    }
    Observation weighted_mean_obs(const Eigen::Matrix<float, NY, SigmaPts::NSIG>& Y,
                                  const typename SigmaPts::Weights& W) const {
        return weighted_mean<NY>(Y, W, [this](int j) { return model_.isAngularObservation(j); });
    }
};

} // namespace UKFCore

#endif // UKF_H
