#ifndef UNSCENTED_FIXED_LAG_SMOOTHER_H
#define UNSCENTED_FIXED_LAG_SMOOTHER_H

#include <deque>
#include <vector>
#include <iostream>
#include <Eigen/Dense>
#include "UKF.h"
#include <optmath/neon_kernels.hpp>

namespace UKFCore {

using namespace optmath::neon;

/**
 * Structure to hold historical filter state for UKF Smoothing.
 */
template<int NX>
struct UKFHistoryEntry {
    using State = Eigen::Matrix<float, NX, 1>;
    using StateMat = Eigen::Matrix<float, NX, NX>;

    State x_filt;      // x_{k|k}
    StateMat P_filt;   // P_{k|k}

    State x_pred_next; // x_{k+1|k} (Predicted state for NEXT step)
    StateMat P_pred_next; // P_{k+1|k} (Predicted cov for NEXT step)

    StateMat P_cross_next; // P_{x_k, x_{k+1}} (Computed during prediction of k+1 from k)
};

template<int NX, int NY>
class UnscentedFixedLagSmoother {
public:
    using UKFType = UKF<NX, NY>;
    using Model = typename UKFType::Model;
    using State = typename Model::State;
    using Observation = typename Model::Observation;
    using StateMat = typename Model::StateMat;

    UnscentedFixedLagSmoother(Model& model, int lag)
        : ukf_(model), lag_(lag) {}

    void initialize(const State& x0, const StateMat& P0) {
        ukf_.initialize(x0, P0);
        history_.clear();

        UKFHistoryEntry<NX> entry;
        entry.x_filt = x0;
        entry.P_filt = P0;
        history_.push_back(entry);
    }

    void step(float t_k, const Observation& y_k, const Eigen::Ref<const State>& u_k) {
        if (history_.empty()) {
            return;
        }

        UKFHistoryEntry<NX>& last_entry = history_.back();

        // Predict
        StateMat P_cross = ukf_.predict(t_k, u_k);

        last_entry.x_pred_next = ukf_.getState();
        last_entry.P_pred_next = ukf_.getCovariance();
        last_entry.P_cross_next = P_cross;

        // Update
        ukf_.update(t_k, y_k); // ukf_ is now at x_{k|k}

        UKFHistoryEntry<NX> new_entry;
        new_entry.x_filt = ukf_.getState();
        new_entry.P_filt = ukf_.getCovariance();
        history_.push_back(new_entry);

        if (history_.size() > static_cast<size_t>(lag_ + 2)) {
            history_.pop_front();
        }

        perform_smoothing();
    }

    // Accessors
    State get_filtered_state() const { return ukf_.getState(); }
    StateMat get_filtered_covariance() const { return ukf_.getCovariance(); }

    State get_smoothed_state(int lag) const {
        if (lag < 0 || lag >= static_cast<int>(history_.size())) return State::Zero();
        int idx = static_cast<int>(history_.size()) - 1 - lag;
        return smoothed_states_[idx];
    }

    StateMat get_smoothed_covariance(int lag) const {
         if (lag < 0 || lag >= static_cast<int>(history_.size())) return StateMat::Identity();
         int idx = static_cast<int>(history_.size()) - 1 - lag;
         return smoothed_covs_[idx];
    }

private:
    UKFType ukf_;
    int lag_;
    std::deque<UKFHistoryEntry<NX>> history_;

    std::vector<State> smoothed_states_;
    std::vector<StateMat> smoothed_covs_;

    void perform_smoothing() {
        int N = static_cast<int>(history_.size());
        if (N == 0) return;

        smoothed_states_.resize(N);
        smoothed_covs_.resize(N);

        smoothed_states_[N-1] = history_.back().x_filt;
        smoothed_covs_[N-1]   = history_.back().P_filt;

        for (int j = N - 2; j >= 0; --j) {
            const auto& entry_j = history_[j];

            const State& x_f_j = entry_j.x_filt;
            const StateMat& P_f_j = entry_j.P_filt;

            const State& x_pred_jp1 = entry_j.x_pred_next;
            const StateMat& P_pred_jp1 = entry_j.P_pred_next;
            const StateMat& P_cross = entry_j.P_cross_next;

            const State& x_s_jp1 = smoothed_states_[j+1];
            const StateMat& P_s_jp1 = smoothed_covs_[j+1];

            // Smoothing Gain G_j = P_cross * P_pred_{j+1}^{-1} using NEON inverse
            Eigen::MatrixXf P_pred_inv = neon_inverse(P_pred_jp1);
            StateMat G_j;
            if (P_pred_inv.size() > 0) {
                G_j = neon_gemm(Eigen::MatrixXf(P_cross), P_pred_inv);
            } else {
                // Fallback to Eigen LDLT
                Eigen::LDLT<StateMat> ldlt(P_pred_jp1);
                G_j = P_cross * ldlt.solve(StateMat::Identity());
            }

            // Update with NEON GEMM
            State diff_x = x_s_jp1 - x_pred_jp1;
            Eigen::MatrixXf update_x = neon_gemm(Eigen::MatrixXf(G_j), Eigen::MatrixXf(diff_x));
            smoothed_states_[j] = x_f_j + State(update_x);

            // Covariance smoothing using NEON GEMM
            StateMat diff_P = P_s_jp1 - P_pred_jp1;
            Eigen::MatrixXf term1 = neon_gemm(Eigen::MatrixXf(G_j), Eigen::MatrixXf(diff_P));
            Eigen::MatrixXf term2 = neon_gemm(term1, Eigen::MatrixXf(G_j.transpose()));
            smoothed_covs_[j] = P_f_j + StateMat(term2);
        }
    }
};

} // namespace UKFCore

#endif // UNSCENTED_FIXED_LAG_SMOOTHER_H
