#ifndef UKF_SMOOTHER_H
#define UKF_SMOOTHER_H

// =============================================================================
// UKFSmoother -- full-interval Unscented RTS (URTS) smoother, Sarkka form.
// -----------------------------------------------------------------------------
// Implements Sarkka, "Bayesian Filtering and Smoothing" (Cambridge, 2013),
// Algorithm 10.1 literally: the forward pass stores only the filtered
// posteriors (m_k, P_k). On the backward pass, at each epoch we REGENERATE
// sigma points from (m_k, P_k), re-propagate them through f(., t_k, u_{k+1})
// -- t_k is the time the state is propagated FROM, the convention of the
// fixed-lag smoothers and the benchmark (commit 3dbbf4c) -- and recompute the predicted mean m_tilde_{k+1}, predicted
// covariance P_tilde_{k+1} (with Q added), and cross-covariance D_{k+1}
// fresh. The smoother gain is then
//
//     G_k    = D_{k+1} * (P_tilde_{k+1})^{-1}
//     m_k^s  = m_k     + G_k * (m_{k+1}^s - m_tilde_{k+1})
//     P_k^s  = P_k     + G_k * (P_{k+1}^s - P_tilde_{k+1}) * G_k^T
//
// which is the URTS recursion. Angular states (model_.isAngularState) use a
// circular predicted mean and wrapped deviations, as UKF::predict does (until
// 2026-11-16 this pass used linear means: rotating a bearing problem across
// +-pi moved the smoothed bearing by up to 1.4 sigma).
//
// smooth(n>0) then runs n iterations of the iterated posterior linearisation
// smoother (IPLS, PosteriorLinearization.h): sigma-point statistical linear
// regression of f and h w.r.t. the current smoothed posteriors, then an exact
// affine Kalman/RTS solve from the ORIGINAL prior. On a linear model every
// iteration returns the exact RTS answer. The filtered accessors always report
// the original forward UKF pass. (Before 2026-11-16 iteration re-ran the
// forward UKF from the smoothed x_0 with covariance P0, counting the data
// twice.)
// =============================================================================

#include <cmath>
#include <numbers>
#include <optional>
#include <stdexcept>
#include <vector>
#include <Eigen/Dense>
#include "UKF.h"
#include "SigmaPoints.h"
#include "FilterMath.h"
#include "PosteriorLinearization.h"

namespace UKFCore {

template<int NX, int NY>
class UKFSmoother {
public:
    using UKFType = UKF<NX, NY>;
    using Model = typename UKFType::Model;
    using State = typename Model::State;
    using Observation = typename Model::Observation;
    using StateMat = typename Model::StateMat;
    using SigmaPts = SigmaPoints<NX>;

    explicit UKFSmoother(Model& model) : model_(model), ukf_(model) {}

    void initialize(const State& x0, const StateMat& P0) {
        x0_ = x0; P0_ = P0; meas_.clear(); init_obs_.reset();
        smoothed_states_.clear(); smoothed_covs_.clear();
        reset_forward(x0_, P0_);
    }

    /**
     * Fuse a measurement taken at the initialisation time t0, without
     * propagating (same contract as the fixed-lag smoothers). Refines
     * history_[0] in place; call at most once, before the first step().
     */
    void observe_initial(float t0, const Observation& y_0) {
        init_obs_ = InitObs{t0, y_0};
        apply_initial_observation();
    }

    /**
     * Consume one measurement (stored for iteration replay); no smoothing yet.
     * Propagates FROM t_prev (where the estimate is) TO t_k, then fuses y_k,
     * which observes the state at t_k. A non-finite y_k is a missed
     * measurement (UKF::update skips it).
     * Throws std::runtime_error if the model makes the state non-finite; the
     * filter is restored to its pre-step state first.
     */
    void step(float t_prev, float t_k, const Observation& y_k,
              const Eigen::Ref<const State>& u_k) {
        meas_.push_back(Meas{t_prev, t_k, y_k, State(u_k)});
        forward_step(meas_.back());
    }

    /**
     * Backward URTS over the full interval, then n_iterations IPLS iterations.
     * Idempotent: always starts from the stored forward pass.
     */
    void smooth(int n_iterations = 0) {
        backward_pass();
        for (int it = 0; it < n_iterations; ++it) {
            if (smoothed_states_.size() != meas_.size() + 1) break;
            const auto r = ipls::iterate<NX, NY>(
                model_, x0_, P0_, init_obs_, meas_, smoothed_states_,
                [&](size_t k, SigmaPts& sig) {
                    generate_sigma_points<NX>(smoothed_states_[k], smoothed_covs_[k],
                                              ukf_.alpha, ukf_.beta, ukf_.kappa, sig);
                });
            for (size_t k = 0; k < r.m.size(); ++k) {
                smoothed_states_[k] = r.m[k].template cast<float>();
                smoothed_covs_[k] = r.P[k].template cast<float>();
            }
        }
    }

    int size() const { return static_cast<int>(history_.size()); }
    // Bounds-checked: an index past size() throws std::out_of_range.
    State  filtered_state(int k)      const { return history_.at(static_cast<size_t>(k)).x; }
    StateMat filtered_covariance(int k) const { return history_.at(static_cast<size_t>(k)).P; }
    State  smoothed_state(int k)      const { return smoothed_states_.at(static_cast<size_t>(k)); }
    StateMat smoothed_covariance(int k) const { return smoothed_covs_.at(static_cast<size_t>(k)); }
    State  smoothed_initial() const { return smoothed_states_.at(0); }
    State  final_filtered_state() const { return history_.back().x; }

private:
    struct Meas { float t_prev; float t; Observation y; State u; };
    struct InitObs { float t; Observation y; };
    struct Filtered { State x; StateMat P; };  // Only the posterior is retained.

    Model& model_;
    UKFType ukf_;
    State x0_;
    StateMat P0_;
    std::vector<Filtered> history_;         // history_[0] = prior; history_[k>=1] = x_{k|k}.
    std::vector<Meas> meas_;                // meas_[j] drives the step history_[j] -> history_[j+1].
    std::optional<InitObs> init_obs_;       // replayed on every iteration
    std::vector<State> smoothed_states_;
    std::vector<StateMat> smoothed_covs_;

    void reset_forward(const State& x, const StateMat& P) {
        ukf_.initialize(x, P);
        history_.clear();
        history_.push_back(Filtered{x, P});
    }

    void apply_initial_observation() {
        if (!init_obs_ || history_.empty()) return;
        ukf_.update(init_obs_->t, init_obs_->y);
        require_finite(history_[0].x, history_[0].P);
        history_[0] = Filtered{ukf_.getState(), ukf_.getCovariance()};
    }

    void forward_step(const Meas& m) {
        if (history_.empty()) return;
        const Filtered before{ukf_.getState(), ukf_.getCovariance()};
        ukf_.predict(m.t_prev, m.u);  // returned cross-cov is not used by the URTS backward pass
        ukf_.update(m.t, m.y);
        require_finite(before.x, before.P);
        history_.push_back(Filtered{ukf_.getState(), ukf_.getCovariance()});
    }

    // A non-finite state cannot be skipped: it would desynchronise history_
    // from meas_ (the old guard returned early but left the UKF itself NaN,
    // so every later step was lost as well). Restore and report instead.
    void require_finite(const State& x_before, const StateMat& P_before) {
        if (ukf_.getState().allFinite() && ukf_.getCovariance().allFinite()) return;
        ukf_.initialize(x_before, P_before);
        throw std::runtime_error("UKFSmoother: model produced a non-finite state; "
                                 "filter restored to its pre-step estimate");
    }

    // Wrap the angular components of a state (or state difference) to [-pi, pi].
    State wrap_angular(State d) const {
        for (int r = 0; r < NX; ++r)
            if (model_.isAngularState(r)) d(r) = std::remainder(d(r), 2.0f * std::numbers::pi_v<float>);
        return d;
    }

    // Sarkka Alg. 10.1: at each j, regenerate sigma points from the FILTERED
    // posterior (m_j, P_j), propagate through model_.f with the same (t, u)
    // that drove the forward step j->j+1, recompute m_tilde, P_tilde, D_cross
    // fresh, then apply the URTS mean/covariance update.
    void backward_pass() {
        const int N = static_cast<int>(history_.size());
        smoothed_states_.assign(N, State::Zero());
        smoothed_covs_.assign(N, StateMat::Identity());
        if (N == 0) return;
        smoothed_states_[N-1] = history_.back().x;
        smoothed_covs_[N-1]   = history_.back().P;
        if (N == 1) return;
        if (static_cast<int>(meas_.size()) < N - 1) return;  // defensive: no drive data to replay

        for (int j = N - 2; j >= 0; --j) {
            const State&    m_j   = history_[j].x;
            const StateMat& P_j   = history_[j].P;
            const float     t_j   = meas_[j].t_prev;  // propagate FROM t_j, as the forward step did
            const State&    u_jp1 = meas_[j].u;

            // 1) Regenerate sigma points from (m_j, P_j).
            SigmaPts sig;
            generate_sigma_points<NX>(m_j, P_j, ukf_.alpha, ukf_.beta, ukf_.kappa, sig);

            // 2) Propagate through f.
            Eigen::Matrix<float, NX, SigmaPts::NSIG> Xp;
            for (int i = 0; i < SigmaPts::NSIG; ++i)
                Xp.col(i) = model_.f(sig.X.col(i), t_j, u_jp1);

            // 3) Predicted mean (circular for angular states, as UKF::predict).
            State m_tilde = State::Zero();
            for (int i = 0; i < SigmaPts::NSIG; ++i)
                m_tilde += sig.Wm(i) * Xp.col(i);
            for (int r = 0; r < NX; ++r) {
                if (!model_.isAngularState(r)) continue;
                float sn = 0.0f, cs = 0.0f;
                for (int i = 0; i < SigmaPts::NSIG; ++i) {
                    sn += sig.Wm(i) * std::sin(Xp(r, i));
                    cs += sig.Wm(i) * std::cos(Xp(r, i));
                }
                m_tilde(r) = std::atan2(sn, cs);
            }

            // 4) Weighted deviations for GEMM assembly of P_tilde and D_cross.
            //    Angular predicted deviations are wrapped; the sigma offsets
            //    dx are the true offsets and are left alone.
            Eigen::Matrix<float, NX, SigmaPts::NSIG> Dp, Dp_w, Dx_w;
            for (int i = 0; i < SigmaPts::NSIG; ++i) {
                State dp = wrap_angular(State(Xp.col(i) - m_tilde));
                State dx = sig.X.col(i) - m_j;
                Dp.col(i)   = dp;
                Dp_w.col(i) = sig.Wc(i) * dp;
                Dx_w.col(i) = sig.Wc(i) * dx;
            }
            StateMat P_tilde = filtermath::gemm(Dp_w, Dp.transpose()) + model_.Q(t_j);
            P_tilde = 0.5f * (P_tilde + P_tilde.transpose());
            StateMat D_cross = filtermath::gemm(Dx_w, Dp.transpose());

            // 5) Smoother gain G_j = D_cross * inv(P_tilde) via SPD solve.
            StateMat G = filtermath::kalman_gain(
                Eigen::MatrixXf(D_cross), Eigen::MatrixXf(P_tilde));

            // 6) Smoothed mean.
            State diff_m = wrap_angular(State(smoothed_states_[j+1] - m_tilde));
            Eigen::VectorXf update_m = filtermath::mat_vec_mul(
                Eigen::MatrixXf(G), Eigen::VectorXf(diff_m));
            smoothed_states_[j] = wrap_angular(State(m_j + State(update_m)));

            // 7) Smoothed covariance: P_j + G (P_s_{j+1} - P_tilde) G^T.
            StateMat diff_P = smoothed_covs_[j+1] - P_tilde;
            Eigen::MatrixXf term1 = filtermath::gemm(Eigen::MatrixXf(G), Eigen::MatrixXf(diff_P));
            Eigen::MatrixXf term2 = filtermath::gemm(term1, Eigen::MatrixXf(G.transpose()));
            StateMat P_s = P_j + StateMat(term2);
            smoothed_covs_[j] = 0.5f * (P_s + P_s.transpose());
        }
    }
};

} // namespace UKFCore

#endif // UKF_SMOOTHER_H
