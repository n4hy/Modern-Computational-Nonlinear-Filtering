#ifndef SRUKF_FIXED_LAG_SMOOTHER_H
#define SRUKF_FIXED_LAG_SMOOTHER_H

#include <deque>
#include <vector>
#include <iostream>
#include <Eigen/Dense>
#include "SRUKF.h"

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

        // Compute square root of P0
        Eigen::LLT<StateMat> llt(P0);
        if (llt.info() != Eigen::Success) {
            StateMat P_jitter = P0 + 1e-6f * StateMat::Identity();
            llt.compute(P_jitter);
        }
        entry.S_filt = llt.matrixL();

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

            // Compute P_pred_jp1 = S_pred_jp1 * S_pred_jp1^T
            StateMat P_pred_jp1 = S_pred_jp1 * S_pred_jp1.transpose();

            // Smoothing Gain G_j = P_cross * P_pred_{j+1}^{-1}
            Eigen::LDLT<StateMat> ldlt(P_pred_jp1);
            StateMat G_j = P_cross * ldlt.solve(StateMat::Identity());

            // State smoothing
            State diff_x = x_s_jp1 - x_pred_jp1;
            smoothed_states_[j] = x_f_j + G_j * diff_x;

            // Covariance smoothing in square root form
            // P_s[j] = P_f[j] + G_j * (P_s[j+1] - P_pred[j+1]) * G_j^T
            // We need to compute the square root of this

            // For simplicity, compute full covariance and then factor
            StateMat P_f_j = S_f_j * S_f_j.transpose();
            StateMat P_s_jp1 = smoothed_S_[j+1] * smoothed_S_[j+1].transpose();

            StateMat diff_P = P_s_jp1 - P_pred_jp1;
            StateMat P_s_j = P_f_j + G_j * diff_P * G_j.transpose();

            // Symmetrize
            P_s_j = 0.5f * (P_s_j + P_s_j.transpose());

            // Extract square root
            Eigen::LLT<StateMat> llt(P_s_j);
            if (llt.info() != Eigen::Success) {
                // Add jitter if not positive definite
                P_s_j += 1e-6f * StateMat::Identity();
                llt.compute(P_s_j);
            }
            smoothed_S_[j] = llt.matrixL();
        }
    }
};

} // namespace UKFCore

#endif // SRUKF_FIXED_LAG_SMOOTHER_H
