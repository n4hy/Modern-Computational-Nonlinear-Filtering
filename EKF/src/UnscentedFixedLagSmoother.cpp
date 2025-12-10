#include "UnscentedFixedLagSmoother.h"
#include <iostream>

UnscentedFixedLagSmoother::UnscentedFixedLagSmoother(SystemModel* model, const Eigen::VectorXd& x0, const Eigen::MatrixXd& P0, int lag)
    : model_(model), ukf_(model, x0, P0), lag_(lag) {

    // Initialize buffer with initial state
    UKFFilterState init_state;
    init_state.x_upd = x0;
    init_state.P_upd = P0;
    // P_cross is undefined for initial state (no previous state)
    init_state.P_cross = Eigen::MatrixXd::Zero(x0.size(), x0.size());

    buffer_.push_back(init_state);
}

bool UnscentedFixedLagSmoother::process(const Eigen::VectorXd& y, Eigen::VectorXd& x_out, Eigen::MatrixXd& P_out) {

    // 1. UKF Predict
    // This returns P_{x_{k-1}, x_k} (cross covariance between PREVIOUS updated state and CURRENT predicted state)
    // The filter currently holds x_{k-1|k-1}.
    Eigen::MatrixXd P_cross = ukf_.predict();

    // Now filter holds x_{k|k-1}, P_{k|k-1}
    Eigen::VectorXd x_pred = ukf_.getState();
    Eigen::MatrixXd P_pred = ukf_.getCovariance();

    // Store P_cross in the PREVIOUS buffer entry, because it relates prev -> current
    if (!buffer_.empty()) {
        buffer_.back().P_cross = P_cross;
    }

    // 2. UKF Update
    ukf_.update(y);

    // 3. Store Current State
    UKFFilterState state;
    state.x_pred = x_pred;
    state.P_pred = P_pred;
    state.x_upd = ukf_.getState();
    state.P_upd = ukf_.getCovariance();
    state.y_meas = y;
    // state.P_cross will be filled next step

    buffer_.push_back(state);

    // 4. Check Lag
    if (buffer_.size() <= (size_t)lag_) {
        return false;
    }

    // 5. Unscented RTS Smoothing
    std::vector<Eigen::VectorXd> x_smooth(buffer_.size());
    std::vector<Eigen::MatrixXd> P_smooth(buffer_.size());

    x_smooth.back() = buffer_.back().x_upd;
    P_smooth.back() = buffer_.back().P_upd;

    for (int i = buffer_.size() - 2; i >= 0; --i) {
        // C_k = P_cross_{k, k+1} * P_{k+1|k}^-1
        // Note: buffer_[i].P_cross stores P_{x_i, x_{i+1}}
        Eigen::MatrixXd& P_cr = buffer_[i].P_cross;
        Eigen::MatrixXd& P_pred_next = buffer_[i+1].P_pred;

        // Use pseudo-inverse or inverse
        Eigen::MatrixXd C = P_cr * P_pred_next.inverse();

        x_smooth[i] = buffer_[i].x_upd + C * (x_smooth[i+1] - buffer_[i+1].x_pred);
        P_smooth[i] = buffer_[i].P_upd + C * (P_smooth[i+1] - P_pred_next) * C.transpose();
    }

    // 6. Extract Output
    x_out = x_smooth[0];
    P_out = P_smooth[0];

    // 7. Feedback / Re-filter
    ukf_.setState(x_out);
    ukf_.setCovariance(P_out);

    // Update head of buffer
    buffer_[0].x_upd = x_out;
    buffer_[0].P_upd = P_out;

    // Re-propagate forward
    // IMPORTANT: When we re-propagate, we generate NEW P_cross values.
    // We must update them in the buffer for the next smoothing cycle.

    for (size_t i = 1; i < buffer_.size(); ++i) {
        // Predict from i-1
        // This computes new P_cross_{i-1, i}
        Eigen::MatrixXd new_P_cross = ukf_.predict();

        // Update buffer
        buffer_[i-1].P_cross = new_P_cross;
        buffer_[i].x_pred = ukf_.getState();
        buffer_[i].P_pred = ukf_.getCovariance();

        // Update
        ukf_.update(buffer_[i].y_meas);

        buffer_[i].x_upd = ukf_.getState();
        buffer_[i].P_upd = ukf_.getCovariance();
    }

    // 8. Pop
    buffer_.pop_front();

    return true;
}

bool UnscentedFixedLagSmoother::flush(Eigen::VectorXd& x_out, Eigen::MatrixXd& P_out) {
    if (buffer_.empty()) return false;

    std::vector<Eigen::VectorXd> x_smooth(buffer_.size());
    std::vector<Eigen::MatrixXd> P_smooth(buffer_.size());

    x_smooth.back() = buffer_.back().x_upd;
    P_smooth.back() = buffer_.back().P_upd;

    for (int i = buffer_.size() - 2; i >= 0; --i) {
        Eigen::MatrixXd& P_cr = buffer_[i].P_cross;
        Eigen::MatrixXd& P_pred_next = buffer_[i+1].P_pred;
        Eigen::MatrixXd C = P_cr * P_pred_next.inverse();
        x_smooth[i] = buffer_[i].x_upd + C * (x_smooth[i+1] - buffer_[i+1].x_pred);
        P_smooth[i] = buffer_[i].P_upd + C * (P_smooth[i+1] - P_pred_next) * C.transpose();
    }

    x_out = x_smooth[0];
    P_out = P_smooth[0];

    buffer_.pop_front();
    return true;
}
