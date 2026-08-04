#ifndef UKF_SMOOTHER_H
#define UKF_SMOOTHER_H

// =============================================================================
// UKFSmoother -- full-interval Unscented RTS (URTS) smoother, Sarkka form.
// -----------------------------------------------------------------------------
// Implements Sarkka, "Bayesian Filtering and Smoothing" (Cambridge, 2013),
// Algorithm 10.1 literally: the forward pass stores only the filtered
// posteriors (m_k, P_k). On the backward pass, at each epoch we REGENERATE
// sigma points from (m_k, P_k), re-propagate them through f(., t_{k+1},
// u_{k+1}), and recompute the predicted mean m_tilde_{k+1}, predicted
// covariance P_tilde_{k+1} (with Q added), and cross-covariance D_{k+1}
// fresh. The smoother gain is then
//
//     G_k    = D_{k+1} * (P_tilde_{k+1})^{-1}
//     m_k^s  = m_k     + G_k * (m_{k+1}^s - m_tilde_{k+1})
//     P_k^s  = P_k     + G_k * (P_{k+1}^s - P_tilde_{k+1}) * G_k^T
//
// which is the URTS recursion. Iteration (smooth(n>0)) re-runs the forward
// filter from the smoothed initial condition (keeping the ORIGINAL prior
// covariance so the data still dominate) and re-smooths -- the iterated
// smoother, meant to run "as time is available". No RTS-linear covariance
// bookkeeping is retained.
// =============================================================================

#include <vector>
#include <Eigen/Dense>
#include "UKF.h"
#include "SigmaPoints.h"
#include "FilterMath.h"

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
        x0_ = x0; P0_ = P0; meas_.clear();
        reset_forward(x0_, P0_);
    }

    /** Consume one measurement (stored for iteration replay); no smoothing yet. */
    void step(float t_k, const Observation& y_k, const Eigen::Ref<const State>& u_k) {
        meas_.push_back(Meas{t_k, y_k, State(u_k)});
        forward_step(t_k, y_k, u_k);
    }

    /** Backward URTS over the full interval; n_iterations>0 -> iterated smoother. */
    void smooth(int n_iterations = 0) {
        backward_pass();
        for (int it = 0; it < n_iterations; ++it) {
            if (smoothed_states_.empty()) break;
            const State xs0 = smoothed_states_.front();
            reset_forward(xs0, P0_);                 // re-linearise, keep original prior
            for (const auto& m : meas_) forward_step(m.t, m.y, m.u);
            backward_pass();
        }
    }

    int size() const { return static_cast<int>(history_.size()); }
    State  filtered_state(int k)      const { return history_[k].x; }
    StateMat filtered_covariance(int k) const { return history_[k].P; }
    State  smoothed_state(int k)      const { return smoothed_states_[k]; }
    StateMat smoothed_covariance(int k) const { return smoothed_covs_[k]; }
    State  smoothed_initial() const { return smoothed_states_.front(); }
    State  final_filtered_state() const { return history_.back().x; }

private:
    struct Meas { float t; Observation y; State u; };
    struct Filtered { State x; StateMat P; };  // Only the posterior is retained.

    Model& model_;
    UKFType ukf_;
    State x0_;
    StateMat P0_;
    std::vector<Filtered> history_;         // history_[0] = prior; history_[k>=1] = x_{k|k}.
    std::vector<Meas> meas_;                // meas_[j] drives the step history_[j] -> history_[j+1].
    std::vector<State> smoothed_states_;
    std::vector<StateMat> smoothed_covs_;

    void reset_forward(const State& x, const StateMat& P) {
        ukf_.initialize(x, P);
        history_.clear();
        history_.push_back(Filtered{x, P});
    }

    void forward_step(float t_k, const Observation& y_k,
                      const Eigen::Ref<const State>& u_k) {
        if (history_.empty()) return;
        ukf_.predict(t_k, u_k);       // returned cross-cov is not used by the URTS backward pass
        ukf_.update(t_k, y_k);
        if (!ukf_.getState().allFinite() || !ukf_.getCovariance().allFinite()) return;
        history_.push_back(Filtered{ukf_.getState(), ukf_.getCovariance()});
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
            const float     t_jp1 = meas_[j].t;
            const State&    u_jp1 = meas_[j].u;

            // 1) Regenerate sigma points from (m_j, P_j).
            SigmaPts sig;
            generate_sigma_points<NX>(m_j, P_j, ukf_.alpha, ukf_.beta, ukf_.kappa, sig);

            // 2) Propagate through f.
            Eigen::Matrix<float, NX, SigmaPts::NSIG> Xp;
            for (int i = 0; i < SigmaPts::NSIG; ++i)
                Xp.col(i) = model_.f(sig.X.col(i), t_jp1, u_jp1);

            // 3) Predicted mean.
            State m_tilde = State::Zero();
            for (int i = 0; i < SigmaPts::NSIG; ++i)
                m_tilde += sig.Wm(i) * Xp.col(i);

            // 4) Weighted deviations for GEMM assembly of P_tilde and D_cross.
            Eigen::Matrix<float, NX, SigmaPts::NSIG> Dp, Dp_w, Dx_w;
            for (int i = 0; i < SigmaPts::NSIG; ++i) {
                State dp = Xp.col(i)    - m_tilde;
                State dx = sig.X.col(i) - m_j;
                Dp.col(i)   = dp;
                Dp_w.col(i) = sig.Wc(i) * dp;
                Dx_w.col(i) = sig.Wc(i) * dx;
            }
            StateMat P_tilde = filtermath::gemm(Dp_w, Dp.transpose()) + model_.Q(t_jp1);
            P_tilde = 0.5f * (P_tilde + P_tilde.transpose());
            StateMat D_cross = filtermath::gemm(Dx_w, Dp.transpose());

            // 5) Smoother gain G_j = D_cross * inv(P_tilde) via SPD solve.
            StateMat G = filtermath::kalman_gain(
                Eigen::MatrixXf(D_cross), Eigen::MatrixXf(P_tilde));

            // 6) Smoothed mean.
            State diff_m = smoothed_states_[j+1] - m_tilde;
            Eigen::VectorXf update_m = filtermath::mat_vec_mul(
                Eigen::MatrixXf(G), Eigen::VectorXf(diff_m));
            smoothed_states_[j] = m_j + State(update_m);

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
