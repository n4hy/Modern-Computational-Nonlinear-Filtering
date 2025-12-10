#include "UnscentedFixedLagSmoother.h"
#include <iostream>

UnscentedFixedLagSmoother::UnscentedFixedLagSmoother(SystemModel* model, const Eigen::VectorXd& x0, const Eigen::MatrixXd& P0, int lag)
    : model_(model), ukf_(model, x0, P0), lag_(lag) {

    UKFFilterState init_state;
    init_state.x_upd = x0;
    init_state.P_upd = P0;
    init_state.P_cross = Eigen::MatrixXd::Zero(x0.size(), x0.size());

    buffer_.push_back(init_state);
}

bool UnscentedFixedLagSmoother::process(const Eigen::VectorXd& y,
                                        Eigen::VectorXd& x_smooth_out, Eigen::MatrixXd& P_smooth_out,
                                        Eigen::VectorXd& x_filt_out, Eigen::MatrixXd& P_filt_out) {

    // 1. UKF Predict
    Eigen::MatrixXd P_cross = ukf_.predict();

    // Now filter holds x_{k|k-1}, P_{k|k-1}
    Eigen::VectorXd x_pred = ukf_.getState();
    Eigen::MatrixXd P_pred = ukf_.getCovariance();

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

    buffer_.push_back(state);

    // 4. Check Lag
    if (buffer_.size() <= (size_t)lag_) {
        return false;
    }

    // Capture Filtered Estimate at output time
    x_filt_out = buffer_[0].x_upd;
    P_filt_out = buffer_[0].P_upd;

    // 5. Unscented RTS Smoothing
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

    // 6. Extract Output
    x_smooth_out = x_smooth[0];
    P_smooth_out = P_smooth[0];

    // 7. Feedback / Re-filter
    ukf_.setState(x_smooth_out);
    ukf_.setCovariance(P_smooth_out);

    // Update head of buffer
    buffer_[0].x_upd = x_smooth_out;
    buffer_[0].P_upd = P_smooth_out;

    // Re-propagate forward
    for (size_t i = 1; i < buffer_.size(); ++i) {
        Eigen::MatrixXd new_P_cross = ukf_.predict();

        buffer_[i-1].P_cross = new_P_cross;
        buffer_[i].x_pred = ukf_.getState();
        buffer_[i].P_pred = ukf_.getCovariance();

        ukf_.update(buffer_[i].y_meas);

        buffer_[i].x_upd = ukf_.getState();
        buffer_[i].P_upd = ukf_.getCovariance();
    }

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
