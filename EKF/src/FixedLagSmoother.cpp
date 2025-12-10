#include "FixedLagSmoother.h"
#include <iostream>

FixedLagSmoother::FixedLagSmoother(SystemModel* model, const Eigen::VectorXd& x0, const Eigen::MatrixXd& P0, int lag)
    : model_(model), ekf_(model, x0, P0), lag_(lag) {

    // Initialize buffer with initial state (conceptually k=0 before first prediction)
    // Actually, process loop assumes: Predict -> Update -> Store.
    // So we need to handle initialization carefully.
    // Let's assume the constructor sets x_0|0.
    // We start processing from k=1.

    // We need to store the initial state so we can re-filter from it if needed?
    // Or does the buffer only store post-update states?
    // RTS needs x_k|k, P_k|k, x_k+1|k, P_k+1|k.

    // We push the initial state as a "dummy" updated state for k=0
    FilterState init_state;
    init_state.x_upd = x0;
    init_state.P_upd = P0;
    // x_pred, P_pred, F, y_meas are not relevant for the very first anchor
    buffer_.push_back(init_state);
}

bool FixedLagSmoother::process(const Eigen::VectorXd& y, Eigen::VectorXd& x_out, Eigen::MatrixXd& P_out) {
    // 1. Store F for the *upcoming* prediction (evaluated at current posterior)
    // The RTS smoother needs F_k to relate x_k and x_k+1.
    // Specifically x_{k+1} ~ f(x_k). Linearized: x_{k+1} ~ F_k x_k.
    Eigen::MatrixXd F_k = model_->F(ekf_.getState());

    // 2. Standard EKF Step (Forward)
    // Store Prediction (Prior)
    Eigen::VectorXd x_prior = model_->f(ekf_.getState()); // Manual predict to capture intermediates
    Eigen::MatrixXd P_prior = F_k * ekf_.getCovariance() * F_k.transpose() + model_->Q();

    // Manually updating EKF internal state to match this logic
    ekf_.setState(x_prior);
    ekf_.setCovariance(P_prior);

    // Update
    ekf_.update(y);

    // 3. Add to Buffer
    FilterState state;
    state.F = F_k; // F used to get from prev to here? No, F used to get from here to next?
    // Wait, RTS uses P_k|k F_k^T P_{k+1|k}^-1.
    // F_k is Jacobian at k, transitioning k -> k+1.
    // So the state at index `i` (time k) should store the F that leads to `i+1`.
    // The F we just computed (step 1) was based on x_{k-1|k-1}. It transitions k-1 -> k.
    // So we should store that F with the *previous* entry in the buffer?
    // Let's clarify indexing.
    // Buffer[i] = State at time t_i.
    // We need F_i that maps t_i -> t_{i+1}.

    // Correction:
    // We computed F based on x_{k-1|k-1} to predict x_{k|k-1}.
    // So this F belongs to the *previous* step's record.
    if (!buffer_.empty()) {
        buffer_.back().F = F_k;
    }

    state.x_pred = x_prior;
    state.P_pred = P_prior;
    state.x_upd = ekf_.getState();
    state.P_upd = ekf_.getCovariance();
    state.y_meas = y;

    buffer_.push_back(state);

    // 4. Check Lag
    // We have initial state (idx 0) + N steps.
    // If buffer.size() > lag + 1 (meaning we have lag+1 transitions), we can smooth back lag steps.
    // Actually, user wants to output at T-lag.
    // If current is T. We need T-lag to be available.
    // Buffer: [T-lag, T-lag+1, ..., T]. Size = lag + 1.

    if (buffer_.size() <= (size_t)lag_) {
        // Accumulating
        return false;
    }

    // 5. Smoothing (RTS)
    // Smooth from end (T) down to start (T-lag)
    // We work on a copy or modify in place?
    // We need to modify in place to get the "Smoothed" value at T-lag.

    // For feedback, we normally don't want to corrupt the "filtered" history if we were doing pure RTS,
    // but here we are doing a re-filter anyway.
    // HOWEVER, RTS requires P_{k+1|k} (prediction) and P_{k|k} (filtered).
    // If we overwrite P_{k|k} with smoothed P, we can't re-smooth easily?
    // Wait, we only output the tail.

    // Create a temporary container for smoothed states?
    // No, let's just compute the smoothed estimate for the *tail* (index 0) by back-propagating.
    // We need the full chain.

    std::vector<Eigen::VectorXd> x_smooth(buffer_.size());
    std::vector<Eigen::MatrixXd> P_smooth(buffer_.size());

    // Initialize with last state
    x_smooth.back() = buffer_.back().x_upd;
    P_smooth.back() = buffer_.back().P_upd;

    // Backward pass
    for (int i = buffer_.size() - 2; i >= 0; --i) {
        // C_k = P_k|k * F_k^T * P_k+1|k^-1
        Eigen::MatrixXd& P_filt = buffer_[i].P_upd;
        Eigen::MatrixXd& F = buffer_[i].F;
        Eigen::MatrixXd& P_pred_next = buffer_[i+1].P_pred;

        Eigen::MatrixXd C = P_filt * F.transpose() * P_pred_next.inverse();

        // x_k|N = x_k|k + C_k * (x_k+1|N - x_k+1|k)
        x_smooth[i] = buffer_[i].x_upd + C * (x_smooth[i+1] - buffer_[i+1].x_pred);

        // P_k|N = P_k|k + C_k * (P_k+1|N - P_k+1|k) * C_k^T
        P_smooth[i] = P_filt + C * (P_smooth[i+1] - P_pred_next) * C.transpose();
    }

    // 6. Extract Output (at T-lag, which is buffer_[0])
    x_out = x_smooth[0];
    P_out = P_smooth[0];

    // 7. Feedback / Re-filtering
    // "The forward filter state uses the smooth operation input... updating the system state using the constant lag smoothing"
    // "Advance the forward filtered state from the smoothed variable"

    // This means: The state at T-lag is now DEFINITIVELY x_out.
    // We discard the old T-lag filtered state.
    // We restart the filter from T-lag (using x_out, P_out) and re-process measurements up to T.

    // Reset EKF to smoothed tail
    ekf_.setState(x_out);
    ekf_.setCovariance(P_out);

    // Re-process measurements from index 1 to End
    // Note: buffer_[0] is the state at T-lag.
    // We need to apply transition 0->1, update with y_1, then 1->2...

    // IMPORTANT: The buffer contains "old" predictions and updates. We must update them with the new path.
    // Update buffer_[0] to match the smoothed (now "truth") state.
    buffer_[0].x_upd = x_out;
    buffer_[0].P_upd = P_out;
    // F at index 0 is valid for 0->1 transition (based on old x_0).
    // Should we re-compute F based on smoothed x_0? Yes, ideally.
    buffer_[0].F = model_->F(x_out);

    for (size_t i = 1; i < buffer_.size(); ++i) {
        // Predict from i-1
        Eigen::VectorXd x_prev = buffer_[i-1].x_upd;
        Eigen::MatrixXd P_prev = buffer_[i-1].P_upd;
        Eigen::MatrixXd F_prev = buffer_[i-1].F; // Uses F from prev step

        // Predict
        // x_pred = f(x_prev)
        Eigen::VectorXd x_pred = model_->f(x_prev);
        Eigen::MatrixXd P_pred = F_prev * P_prev * F_prev.transpose() + model_->Q();

        // Update EKF internal (just primarily for the 'update' math helper if we used it, but we can do it manually or use the object)
        ekf_.setState(x_pred);
        ekf_.setCovariance(P_pred);
        ekf_.update(buffer_[i].y_meas);

        // Store back in buffer
        buffer_[i].x_pred = x_pred;
        buffer_[i].P_pred = P_pred;
        buffer_[i].x_upd = ekf_.getState();
        buffer_[i].P_upd = ekf_.getCovariance();

        // Recompute F for next step (unless this is the last step, but even then we need it for next iteration's smoothing)
        buffer_[i].F = model_->F(ekf_.getState());

        // Fix F of previous step?
        // Wait, in loop:
        // When we are at `i`, we used `buffer_[i-1].F` to get here.
        // That F was computed at `i-1`.
        // After updating `i-1` in the previous iteration, we recomputed `buffer_[i-1].F`.
        // So `F_prev` above IS the recomputed one. Correct.
    }

    // 8. Shift Window
    // We output buffer_[0]. We are done with it.
    buffer_.pop_front();

    return true;
}

bool FixedLagSmoother::flush(Eigen::VectorXd& x_out, Eigen::MatrixXd& P_out) {
    // "As near the endpoint time... we shorten the lag until it is zero"
    // This implies we pop the oldest element as "finalized" even if we don't have new measurements.

    if (buffer_.empty()) return false;

    // Perform smoothing on remaining buffer
    // (Same logic as process, just no new measurement added)

    std::vector<Eigen::VectorXd> x_smooth(buffer_.size());
    std::vector<Eigen::MatrixXd> P_smooth(buffer_.size());

    x_smooth.back() = buffer_.back().x_upd;
    P_smooth.back() = buffer_.back().P_upd;

    for (int i = buffer_.size() - 2; i >= 0; --i) {
        Eigen::MatrixXd& P_filt = buffer_[i].P_upd;
        Eigen::MatrixXd& F = buffer_[i].F;
        Eigen::MatrixXd& P_pred_next = buffer_[i+1].P_pred;
        Eigen::MatrixXd C = P_filt * F.transpose() * P_pred_next.inverse();
        x_smooth[i] = buffer_[i].x_upd + C * (x_smooth[i+1] - buffer_[i+1].x_pred);
        P_smooth[i] = P_filt + C * (P_smooth[i+1] - P_pred_next) * C.transpose();
    }

    x_out = x_smooth[0];
    P_out = P_smooth[0];

    // We don't strictly need to re-filter here because we are flushing (just outputting).
    // But if we wanted to be consistent for the *next* flush, we could.
    // However, since we are just emptying the queue, re-filtering the tail doesn't help future points
    // (since there are no future points added).
    // BUT: The next point in the buffer (current index 1) will become index 0 next call.
    // Does its value depend on the smoothing of index 0?
    // In RTS, the smoothed value at k depends on k+1. Not vice versa.
    // So re-filtering forward is not strictly needed if we are just flushing out.
    // EXCEPT: If we have [0, 1, 2]. Smooth 0<-1<-2. Output 0.
    // Next flush: [1, 2]. Smooth 1<-2. Output 1.
    // Does the smoothed value of 0 affect 1? No, 1 affects 0.
    // So for flushing, we can just smooth and pop.

    buffer_.pop_front();
    return true;
}
