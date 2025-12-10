#include "FixedLagSmoother.h"
#include <iostream>

FixedLagSmoother::FixedLagSmoother(SystemModel* model, const Eigen::VectorXd& x0, const Eigen::MatrixXd& P0, int lag)
    : model_(model), ekf_(model, x0, P0), lag_(lag) {

    FilterState init_state;
    init_state.x_upd = x0;
    init_state.P_upd = P0;
    buffer_.push_back(init_state);
}

bool FixedLagSmoother::process(const Eigen::VectorXd& y,
                               Eigen::VectorXd& x_smooth_out, Eigen::MatrixXd& P_smooth_out,
                               Eigen::VectorXd& x_filt_out, Eigen::MatrixXd& P_filt_out) {

    // 1. Get F and Predict
    Eigen::MatrixXd F_k = model_->F(ekf_.getState());
    ekf_.predict();

    // 2. Store F in previous state
    if (!buffer_.empty()) {
        buffer_.back().F = F_k;
    }

    // Store prediction
    Eigen::VectorXd x_pred = ekf_.getState();
    Eigen::MatrixXd P_pred = ekf_.getCovariance();

    // 3. Update
    ekf_.update(y);

    // 4. Add to Buffer
    FilterState state;
    state.x_pred = x_pred;
    state.P_pred = P_pred;
    state.x_upd = ekf_.getState();
    state.P_upd = ekf_.getCovariance();
    state.y_meas = y;

    buffer_.push_back(state);

    // 5. Check Lag
    if (buffer_.size() <= (size_t)lag_) {
        return false;
    }

    // 6. Capture "Filtered" state at T-lag (which is buffer_[0])
    x_filt_out = buffer_[0].x_upd;
    P_filt_out = buffer_[0].P_upd;

    // 7. Smoothing (RTS)
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

    // 8. Extract Smoothed Output
    x_smooth_out = x_smooth[0];
    P_smooth_out = P_smooth[0];

    // 9. Feedback / Re-filtering
    ekf_.setState(x_smooth_out);
    ekf_.setCovariance(P_smooth_out);

    // Update head
    buffer_[0].x_upd = x_smooth_out;
    buffer_[0].P_upd = P_smooth_out;
    buffer_[0].F = model_->F(x_smooth_out);

    for (size_t i = 1; i < buffer_.size(); ++i) {
        // Predict from i-1
        Eigen::VectorXd x_prev = buffer_[i-1].x_upd;
        Eigen::MatrixXd F_prev = buffer_[i-1].F;

        Eigen::VectorXd x_pred = model_->f(x_prev);
        Eigen::MatrixXd P_pred = F_prev * buffer_[i-1].P_upd * F_prev.transpose() + model_->Q();

        ekf_.setState(x_pred);
        ekf_.setCovariance(P_pred);
        ekf_.update(buffer_[i].y_meas);

        buffer_[i].x_pred = x_pred;
        buffer_[i].P_pred = P_pred;
        buffer_[i].x_upd = ekf_.getState();
        buffer_[i].P_upd = ekf_.getCovariance();

        buffer_[i].F = model_->F(ekf_.getState());
    }

    buffer_.pop_front();

    return true;
}

bool FixedLagSmoother::flush(Eigen::VectorXd& x_out, Eigen::MatrixXd& P_out) {
    if (buffer_.empty()) return false;

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

    buffer_.pop_front();
    return true;
}
