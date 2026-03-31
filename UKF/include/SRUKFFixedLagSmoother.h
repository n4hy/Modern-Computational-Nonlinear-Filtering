#ifndef SRUKF_FIXED_LAG_SMOOTHER_H
#define SRUKF_FIXED_LAG_SMOOTHER_H

#include <deque>
#include <vector>
#include <iostream>
#include <Eigen/Dense>
#include "SRUKF.h"
#include "FilterMath.h"

namespace UKFCore {

/**
 * Structure to hold historical filter state for SRUKF Smoothing.
 */
template<int NX>
struct SRUKFHistoryEntry {
    using State = Eigen::Matrix<float, NX, 1>;
    using StateMat = Eigen::Matrix<float, NX, NX>;

    State x_filt;      // x_{k|k}
    StateMat S_filt;   // S_{k|k} (square root of P_{k|k})

    State x_pred_next; // x_{k+1|k}
    StateMat S_pred_next; // S_{k+1|k}

    StateMat P_cross_next; // P_{x_k, x_{k+1}}
};

template<int NX, int NY>
class SRUKFFixedLagSmoother {
public:
    using SRUKFType = SRUKF<NX, NY>;
    using Model = typename SRUKFType::Model;
    using State = typename Model::State;
    using Observation = typename Model::Observation;
    using StateMat = typename Model::StateMat;

    SRUKFFixedLagSmoother(Model& model, int lag)
        : srukf_(model), lag_(lag) {}

    void initialize(const State& x0, const StateMat& P0) {
        srukf_.initialize(x0, P0);
        history_.clear();

        SRUKFHistoryEntry<NX> entry;
        entry.x_filt = x0;

        // Compute square root of P0 using accelerated Cholesky
        Eigen::MatrixXf L0 = filtermath::cholesky(P0);
        if (L0.size() == 0) {
            StateMat P_jitter = P0 + 1e-6f * StateMat::Identity();
            L0 = filtermath::cholesky(P_jitter);
        }
        if (L0.size() > 0) {
            entry.S_filt = L0;
        } else {
            entry.S_filt = StateMat::Identity();
        }

        history_.push_back(entry);
    }

    void step(float t_k, const Observation& y_k, const Eigen::Ref<const State>& u_k) {
        if (history_.empty()) {
            return;
        }

        SRUKFHistoryEntry<NX>& last_entry = history_.back();

        // Predict
        StateMat P_cross = srukf_.predict(t_k, u_k);

        last_entry.x_pred_next = srukf_.getState();
        last_entry.S_pred_next = srukf_.getSqrtCovariance();
        last_entry.P_cross_next = P_cross;

        // Update
        srukf_.update(t_k, y_k);

        // NaN guard
        if (!srukf_.getState().allFinite() || !srukf_.getSqrtCovariance().allFinite()) {
            return;
        }

        SRUKFHistoryEntry<NX> new_entry;
        new_entry.x_filt = srukf_.getState();
        new_entry.S_filt = srukf_.getSqrtCovariance();
        history_.push_back(new_entry);

        if (history_.size() > static_cast<size_t>(lag_ + 2)) {
            history_.pop_front();
        }

        perform_smoothing();
    }

    // Accessors
    State get_filtered_state() const { return srukf_.getState(); }
    StateMat get_filtered_covariance() const { return srukf_.getCovariance(); }

    State get_smoothed_state(int lag) const {
        if (lag < 0 || lag >= static_cast<int>(history_.size())) return State::Zero();
        int idx = static_cast<int>(history_.size()) - 1 - lag;
        return smoothed_states_[idx];
    }

    StateMat get_smoothed_covariance(int lag) const {
        if (lag < 0 || lag >= static_cast<int>(history_.size())) return StateMat::Identity();
        int idx = static_cast<int>(history_.size()) - 1 - lag;
        return smoothed_S_[idx] * smoothed_S_[idx].transpose();
    }

    StateMat get_smoothed_sqrt_covariance(int lag) const {
        if (lag < 0 || lag >= static_cast<int>(history_.size())) return StateMat::Identity();
        int idx = static_cast<int>(history_.size()) - 1 - lag;
        return smoothed_S_[idx];
    }

private:
    SRUKFType srukf_;
    int lag_;
    std::deque<SRUKFHistoryEntry<NX>> history_;

    std::vector<State> smoothed_states_;
    std::vector<StateMat> smoothed_S_;  // Square root of smoothed covariances

    void perform_smoothing() {
        int N = static_cast<int>(history_.size());
        if (N == 0) return;

        smoothed_states_.resize(N);
        smoothed_S_.resize(N);

        smoothed_states_[N-1] = history_.back().x_filt;
        smoothed_S_[N-1] = history_.back().S_filt;

        for (int j = N - 2; j >= 0; --j) {
            const auto& entry_j = history_[j];

            const State& x_f_j = entry_j.x_filt;
            const StateMat& S_f_j = entry_j.S_filt;

            const State& x_pred_jp1 = entry_j.x_pred_next;
            const StateMat& S_pred_jp1 = entry_j.S_pred_next;
            const StateMat& P_cross = entry_j.P_cross_next;

            const State& x_s_jp1 = smoothed_states_[j+1];

            // Compute P_pred_jp1 = S_pred * S_pred^T via GEMM
            Eigen::MatrixXf P_pred_dyn = filtermath::gemm(
                Eigen::MatrixXf(S_pred_jp1), Eigen::MatrixXf(S_pred_jp1.transpose()));
            StateMat P_pred_jp1 = P_pred_dyn;

            // Smoothing Gain via SPD solve
            StateMat G_j = filtermath::kalman_gain(
                Eigen::MatrixXf(P_cross), Eigen::MatrixXf(P_pred_jp1));

            // State smoothing
            State diff_x = x_s_jp1 - x_pred_jp1;
            Eigen::VectorXf update_x = filtermath::mat_vec_mul(
                Eigen::MatrixXf(G_j), Eigen::VectorXf(diff_x));
            smoothed_states_[j] = x_f_j + State(update_x);

            // Covariance smoothing
            Eigen::MatrixXf P_f_dyn = filtermath::gemm(
                Eigen::MatrixXf(S_f_j), Eigen::MatrixXf(S_f_j.transpose()));
            Eigen::MatrixXf P_s_jp1_dyn = filtermath::gemm(
                Eigen::MatrixXf(smoothed_S_[j+1]), Eigen::MatrixXf(smoothed_S_[j+1].transpose()));

            StateMat diff_P = StateMat(P_s_jp1_dyn) - P_pred_jp1;
            Eigen::MatrixXf term1 = filtermath::gemm(Eigen::MatrixXf(G_j), Eigen::MatrixXf(diff_P));
            Eigen::MatrixXf term2 = filtermath::gemm(term1, Eigen::MatrixXf(G_j.transpose()));
            StateMat P_s_j = StateMat(P_f_dyn) + StateMat(term2);

            // Symmetrize
            P_s_j = 0.5f * (P_s_j + P_s_j.transpose());

            // Extract square root using accelerated Cholesky with fallback
            Eigen::MatrixXf L_s = filtermath::cholesky(P_s_j);
            if (L_s.size() == 0) {
                P_s_j += 1e-6f * StateMat::Identity();
                L_s = filtermath::cholesky(P_s_j);
            }
            if (L_s.size() == 0) {
                // Ultimate fallback to Eigen LLT
                Eigen::LLT<StateMat> llt(P_s_j);
                smoothed_S_[j] = llt.matrixL();
            } else {
                smoothed_S_[j] = L_s;
            }
        }
    }
};

} // namespace UKFCore

#endif // SRUKF_FIXED_LAG_SMOOTHER_H
