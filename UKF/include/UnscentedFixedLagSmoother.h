#ifndef UNSCENTED_FIXED_LAG_SMOOTHER_H
#define UNSCENTED_FIXED_LAG_SMOOTHER_H

#include <deque>
#include <vector>
#include <iostream>
#include <Eigen/Dense>
#include "UKF.h"

namespace UKFCore {

/**
 * Structure to hold historical filter state for UKF Smoothing.
 */
template<int NX>
struct UKFHistoryEntry {
    using State = Eigen::Matrix<double, NX, 1>;
    using StateMat = Eigen::Matrix<double, NX, NX>;

    State x_filt;      // x_{k|k}
    StateMat P_filt;   // P_{k|k}

    State x_pred_next; // x_{k+1|k} (Predicted state for NEXT step)
    StateMat P_pred_next; // P_{k+1|k} (Predicted cov for NEXT step)

    StateMat P_cross_next; // P_{x_k, x_{k+1}} (Computed during prediction of k+1 from k)

    // For the very last entry (current time k), we might not have 'next' values yet until next step.
    // Actually, to smooth at time k, we look back.
    // The RTS smoother goes backward.
    // We need P_{x_k, x_{k+1}} which connects k to k+1.
    // This is computed when we predict k+1 from k.
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

        // Initial state is "filtered" state at k=0
        UKFHistoryEntry<NX> entry;
        entry.x_filt = x0;
        entry.P_filt = P0;
        // Other fields (pred_next, etc.) will be filled when we predict step 1
        history_.push_back(entry);
    }

    /**
     * Process a new measurement.
     * 1. Predict (k -> k+1) - Wait, standard loop is:
     *    Have x_{k|k}.
     *    Predict x_{k+1|k}.
     *    Update x_{k+1|k+1}.
     *
     * The history buffer needs to store the chain.
     * When we are at time k (just updated), we have x_{k|k}.
     * We also just computed the prediction from k-1 to k, so we have x_{k|k-1}, P_{k|k-1} and P_{x_{k-1}, x_k}.
     *
     * Let's align with the `step` function signature.
     * Input: t_k, y_k, u_{k-1} (control applied to get to k)?
     * Usually f(x_{k-1}, u_{k-1}) -> x_k.
     *
     * The prompt says: `step(double t_k, const Observation& y_k, const State& u_k)`
     * and "Internally calls predict + update".
     *
     * If we call predict(t_k, u_k), it means we are at time k-1 and predicting k?
     * Or are we at time k, predicting k+1?
     * Usually Kalman Filter implementations tracking a stream:
     *   Start: x_0|0
     *   Loop k=1..N:
     *     Predict: x_{k|k-1} from x_{k-1|k-1} using u_{k-1} (or u_k depending on notation)
     *     Update: x_{k|k} using y_k
     *
     * So `step` should perform the transition from previous state to current state, then update.
     */
    void step(double t_k, const Observation& y_k, const Eigen::Ref<const State>& u_k) {
        // We assume current state in UKF is x_{k-1|k-1}.
        // 1. Predict x_{k|k-1} and get P_{x_{k-1}, x_k}
        // Note: The `ukf_.predict` function updates the internal state to x_{k|k-1}.
        // We need to capture the state BEFORE prediction (x_{k-1|k-1}) to store in history?
        // Actually, the history already has x_{k-1|k-1} from the previous step.
        // We just need to update that entry with the prediction info (x_{k|k-1}, P_{k|k-1}, CrossCov).

        if (history_.empty()) {
            // Should have been initialized.
            return;
        }

        // Reference to the last entry (which is k-1)
        UKFHistoryEntry<NX>& last_entry = history_.back();

        // Predict
        StateMat P_cross = ukf_.predict(t_k, u_k); // ukf_ is now at x_{k|k-1}

        // Store prediction info into the *previous* entry (k-1) because it connects k-1 to k
        last_entry.x_pred_next = ukf_.getState(); // x_{k|k-1}
        last_entry.P_pred_next = ukf_.getCovariance(); // P_{k|k-1}
        last_entry.P_cross_next = P_cross;

        // 2. Update x_{k|k}
        ukf_.update(t_k, y_k); // ukf_ is now at x_{k|k}

        // 3. Add new entry for k
        UKFHistoryEntry<NX> new_entry;
        new_entry.x_filt = ukf_.getState();
        new_entry.P_filt = ukf_.getCovariance();
        history_.push_back(new_entry);

        // 4. Manage Buffer Size
        // We need L+1 entries to smooth L steps back?
        // If lag is L, we want to smooth x_{k-L}.
        // Buffer needs to hold enough to reach back.
        if (history_.size() > static_cast<size_t>(lag_ + 2)) {
            history_.pop_front();
        }

        // 5. Perform Smoothing
        perform_smoothing();
    }

    // Accessors
    State get_filtered_state() const { return ukf_.getState(); }
    StateMat get_filtered_covariance() const { return ukf_.getCovariance(); }

    State get_smoothed_state(int lag) const {
        if (lag < 0 || lag >= static_cast<int>(history_.size())) return State::Zero(); // Error
        // lag=0 means current time k (end of deque)
        // lag=1 means k-1
        int idx = static_cast<int>(history_.size()) - 1 - lag;
        // In this implementation, we will update x_filt/P_filt in the history with smoothed values?
        // The prompt says "Store smoothed at k: x_s[k]... For j=k-1...".
        // It's better to store smoothed values separately if we want to keep filtered values for debugging,
        // but typically Fixed-Lag smoothing overwrites or we just return the tail.
        // However, to do the recursion, we *need* the smoothed value of the future step to smooth the current step.
        // So we can store smoothed values in a separate structure or overwrite.
        // Given the prompt: "x_s[j] = ...".
        // Let's assume we want to output the smoothed estimate at lag L.

        // Wait, if we overwrite, we lose the filtered estimate which is the base for smoothing?
        // No, the RTS formula uses x_f[j] (filtered) and x_s[j+1] (smoothed future).
        // So we must preserve x_f[j].

        // For simplicity in this assignment, I will re-compute smoothing chain from k back to k-L every step.
        // This is O(L), which is fine.
        // I will return the smoothed value computed during that pass.
        // But `get_smoothed_state` implies I can query it later.
        // So I should store the smoothed values.

        // Let's add x_smooth, P_smooth to the history entry?
        // But x_smooth changes every time step as the window moves!
        // x_{k-L|k} is different from x_{k-L|k-1}.
        // The "Fixed-Lag Smoother" typically outputs x_{k-L|k}.
        // The intermediate smoothed values x_{k-1|k}, x_{k-2|k}... are transient computations usually.

        // However, to answer "get_smoothed_state(lag)", I need those transients.
        // So I'll compute them on demand or cache them.
        // Let's just re-run the smoothing loop for the requested lag?
        // No, `step` says "Runs fixed-lag smoothing".
        // So I should store the result of the smoothing pass.

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

    // Temporary storage for current smoothed estimates (parallel to history_)
    // Recomputed at every step.
    std::vector<State> smoothed_states_;
    std::vector<StateMat> smoothed_covs_;

    void perform_smoothing() {
        int N = static_cast<int>(history_.size());
        if (N == 0) return;

        smoothed_states_.resize(N);
        smoothed_covs_.resize(N);

        // Initialize last element (time k) with filtered values
        smoothed_states_[N-1] = history_.back().x_filt;
        smoothed_covs_[N-1]   = history_.back().P_filt;

        // Backward recursion
        for (int j = N - 2; j >= 0; --j) {
            // We are smoothing j using info from j+1
            const auto& entry_j = history_[j]; // contains x_f[j], P_f[j], and prediction info to j+1

            // Wait, entry_j stores prediction to j+1?
            // Yes: x_pred_next = x_{j+1|j}, P_pred_next = P_{j+1|j}, P_cross_next = Cov(x_j, x_{j+1})

            const State& x_f_j = entry_j.x_filt;
            const StateMat& P_f_j = entry_j.P_filt;

            const State& x_pred_jp1 = entry_j.x_pred_next;
            const StateMat& P_pred_jp1 = entry_j.P_pred_next;
            const StateMat& P_cross = entry_j.P_cross_next;

            const State& x_s_jp1 = smoothed_states_[j+1];
            const StateMat& P_s_jp1 = smoothed_covs_[j+1];

            // Smoothing Gain G_j = P_cross * P_pred_{j+1}^{-1}
            Eigen::LDLT<StateMat> ldlt(P_pred_jp1);
            if (ldlt.info() != Eigen::Success) {
                // Fallback
            }
            StateMat G_j = P_cross * ldlt.solve(StateMat::Identity());

            // Update
            smoothed_states_[j] = x_f_j + G_j * (x_s_jp1 - x_pred_jp1);
            smoothed_covs_[j]   = P_f_j + G_j * (P_s_jp1 - P_pred_jp1) * G_j.transpose();
        }
    }
};

} // namespace UKFCore

#endif // UNSCENTED_FIXED_LAG_SMOOTHER_H
