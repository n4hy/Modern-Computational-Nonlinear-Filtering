#ifndef SRUKF_SMOOTHER_H
#define SRUKF_SMOOTHER_H

// =============================================================================
// SRUKFSmoother -- FULL-INTERVAL (batch) square-root unscented RTS smoother,
// with optional ITERATION.
// -----------------------------------------------------------------------------
// The companion SRUKFFixedLagSmoother smooths over a bounded lag window in real
// time.  This class instead keeps the ENTIRE forward trajectory and runs a single
// backward Rauch-Tung-Striebel pass once all measurements are in -- the natural
// "revisit the whole capture after the pass is over" refinement, using every
// measurement to inform every epoch.
//
// Iteration (smooth(n_iterations>0)) addresses the nonlinear case: the forward
// filter's early epochs are linearised (via the unscented transform) around a
// still-uncertain prior, so their sigma points -- and hence the smoothed estimate
// -- are only as good as that starting point.  Re-running the forward filter from
// the SMOOTHED initial condition (keeping the original prior covariance so the
// data still dominate) re-centres those early sigma points near the truth and
// re-smooths; repeating drives the batch toward the maximum-likelihood / fully
// smoothed solution.  This is a practical iterated smoother (cf. the iterated
// unscented RTS smoother), meant to run opportunistically, "as time is available".
//
// The RTS recursion mirrors SRUKFFixedLagSmoother::perform_smoothing exactly
// (same NEON-accelerated gemm/kalman_gain/cholesky path); only the bookkeeping
// (full history, no trimming, deferred single pass, measurement replay) differs.
// =============================================================================

#include <vector>
#include <Eigen/Dense>
#include "SRUKF.h"
#include "SRUKFFixedLagSmoother.h"   // reuse SRUKFHistoryEntry<NX>
#include "FilterMath.h"

namespace UKFCore {

template<int NX, int NY>
class SRUKFSmoother {
public:
    using SRUKFType = SRUKF<NX, NY>;
    using Model = typename SRUKFType::Model;
    using State = typename Model::State;
    using Observation = typename Model::Observation;
    using StateMat = typename Model::StateMat;

    explicit SRUKFSmoother(Model& model) : model_(model), srukf_(model) {}

    /** Set the prior; clears any stored trajectory/measurements. */
    void initialize(const State& x0, const StateMat& P0) {
        x0_ = x0;
        P0_ = P0;
        meas_.clear();
        reset_forward(x0_, P0_);
    }

    /**
     * Consume one measurement: advance the forward SRUKF and store the history
     * needed by the backward pass.  The (t, y, u) triple is retained so the
     * iterated smoother can replay it.  No smoothing happens here -- call
     * smooth() once, after the last measurement.
     */
    void step(float t_k, const Observation& y_k, const Eigen::Ref<const State>& u_k) {
        meas_.push_back(Meas{t_k, y_k, State(u_k)});
        forward_step(t_k, y_k, u_k);
    }

    /**
     * Run the backward RTS pass over the full interval.  With n_iterations > 0,
     * re-run the forward filter from the smoothed initial condition and re-smooth
     * that many additional times (the original prior covariance is kept so the
     * measurements still dominate; only the linearisation point improves).
     */
    void smooth(int n_iterations = 0) {
        backward_pass();
        for (int it = 0; it < n_iterations; ++it) {
            if (smoothed_states_.empty()) break;
            const State xs0 = smoothed_states_.front();   // re-linearise here
            reset_forward(xs0, P0_);                       // keep the ORIGINAL prior
            for (const auto& m : meas_) forward_step(m.t, m.y, m.u);
            backward_pass();
        }
    }

    /** Number of stored epochs (initial prior + one per measurement). */
    int size() const { return static_cast<int>(history_.size()); }

    State  filtered_state(int k)      const { return history_[k].x_filt; }
    StateMat filtered_covariance(int k) const {
        return history_[k].S_filt * history_[k].S_filt.transpose();
    }
    State  smoothed_state(int k)      const { return smoothed_states_[k]; }
    StateMat smoothed_covariance(int k) const {
        return smoothed_S_[k] * smoothed_S_[k].transpose();
    }
    const StateMat& smoothed_sqrt_covariance(int k) const { return smoothed_S_[k]; }

    /** Smoothed estimate at the START of the interval (the batch-refined prior). */
    State  smoothed_initial() const { return smoothed_states_.front(); }
    State  final_filtered_state() const { return history_.back().x_filt; }

private:
    struct Meas { float t; Observation y; State u; };

    Model& model_;
    SRUKFType srukf_;
    State x0_;
    StateMat P0_;
    std::vector<SRUKFHistoryEntry<NX>> history_;
    std::vector<Meas> meas_;
    std::vector<State> smoothed_states_;
    std::vector<StateMat> smoothed_S_;

    static StateMat sqrt_spd(const StateMat& P) {
        Eigen::MatrixXf L = filtermath::cholesky(P);
        if (L.size() == 0) L = filtermath::cholesky(P + 1e-6f * StateMat::Identity());
        if (L.size() == 0) {
            Eigen::LLT<StateMat> llt(P + 1e-8f * StateMat::Identity());
            return (llt.info() == Eigen::Success) ? StateMat(llt.matrixL())
                                                  : StateMat(StateMat::Identity());
        }
        return StateMat(L);
    }

    void reset_forward(const State& x, const StateMat& P) {
        srukf_.initialize(x, P);
        history_.clear();
        SRUKFHistoryEntry<NX> e;
        e.x_filt = x;
        e.S_filt = sqrt_spd(P);
        history_.push_back(e);
    }

    void forward_step(float t_k, const Observation& y_k,
                      const Eigen::Ref<const State>& u_k) {
        if (history_.empty()) return;
        SRUKFHistoryEntry<NX>& last = history_.back();
        StateMat P_cross = srukf_.predict(t_k, u_k);
        last.x_pred_next = srukf_.getState();
        last.S_pred_next = srukf_.getSqrtCovariance();
        last.P_cross_next = P_cross;
        srukf_.update(t_k, y_k);
        if (!srukf_.getState().allFinite() || !srukf_.getSqrtCovariance().allFinite())
            return;                                    // NaN guard: drop this step
        SRUKFHistoryEntry<NX> e;
        e.x_filt = srukf_.getState();
        e.S_filt = srukf_.getSqrtCovariance();
        history_.push_back(e);
    }

    // Backward RTS over the full stored history (identical math to the fixed-lag
    // smoother's perform_smoothing, run once over every epoch).
    void backward_pass() {
        const int N = static_cast<int>(history_.size());
        smoothed_states_.assign(N, State::Zero());
        smoothed_S_.assign(N, StateMat::Identity());
        if (N == 0) return;
        smoothed_states_[N-1] = history_.back().x_filt;
        smoothed_S_[N-1]      = history_.back().S_filt;

        for (int j = N - 2; j >= 0; --j) {
            const auto& e = history_[j];
            Eigen::MatrixXf P_pred = filtermath::gemm(
                Eigen::MatrixXf(e.S_pred_next),
                Eigen::MatrixXf(e.S_pred_next.transpose()));
            StateMat P_pred_jp1 = P_pred;

            StateMat G = filtermath::kalman_gain(
                Eigen::MatrixXf(e.P_cross_next), Eigen::MatrixXf(P_pred_jp1));

            State diff_x = smoothed_states_[j+1] - e.x_pred_next;
            Eigen::VectorXf dx = filtermath::mat_vec_mul(
                Eigen::MatrixXf(G), Eigen::VectorXf(diff_x));
            smoothed_states_[j] = e.x_filt + State(dx);

            Eigen::MatrixXf P_f = filtermath::gemm(
                Eigen::MatrixXf(e.S_filt), Eigen::MatrixXf(e.S_filt.transpose()));
            Eigen::MatrixXf P_s_jp1 = filtermath::gemm(
                Eigen::MatrixXf(smoothed_S_[j+1]),
                Eigen::MatrixXf(smoothed_S_[j+1].transpose()));
            StateMat diff_P = StateMat(P_s_jp1) - P_pred_jp1;
            Eigen::MatrixXf t1 = filtermath::gemm(Eigen::MatrixXf(G), Eigen::MatrixXf(diff_P));
            Eigen::MatrixXf t2 = filtermath::gemm(t1, Eigen::MatrixXf(G.transpose()));
            StateMat P_s = StateMat(P_f) + StateMat(t2);
            P_s = 0.5f * (P_s + P_s.transpose());
            smoothed_S_[j] = sqrt_spd(P_s);
        }
    }
};

} // namespace UKFCore

#endif // SRUKF_SMOOTHER_H
