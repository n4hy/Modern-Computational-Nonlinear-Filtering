#include "EKFFixedLag.h"
#include <iostream>
#include "FilterMath.h"

EKFFixedLag::EKFFixedLag(SystemModel* model, const Eigen::VectorXf& x0, const Eigen::MatrixXf& P0, int lag_L)
    : model_(model), ekf_(model, x0, P0), lag_L_(lag_L) {

    FilterState init_state;
    init_state.t = 0.0f;
    init_state.x_upd = x0;
    init_state.P_upd = P0;

    buffer_.push_back(init_state);
}

void EKFFixedLag::step(const Eigen::VectorXf& y_k, const Eigen::VectorXf& u_k, float t_k) {
    // 1. EKF Prediction
    Eigen::MatrixXf F_k = ekf_.predict(u_k, t_k);

    Eigen::VectorXf x_pred = ekf_.getPredictedState();
    Eigen::MatrixXf P_pred = ekf_.getPredictedCovariance();

    // 2. EKF Update
    ekf_.update(y_k, t_k);

    Eigen::VectorXf x_upd = ekf_.getState();
    Eigen::MatrixXf P_upd = ekf_.getCovariance();

    // 3. Store in Buffer
    FilterState state;
    state.t = t_k;
    state.x_pred = x_pred;
    state.P_pred = P_pred;
    state.x_upd = x_upd;
    state.P_upd = P_upd;
    state.F = F_k;

    buffer_.push_back(state);

    if (buffer_.size() > (size_t)(lag_L_ + 1)) {
        buffer_.pop_front();
    }

    // 4. Backward Smoothing Recursion (RTS)
    size_t N = buffer_.size();
    x_smooth_.resize(N);
    P_smooth_.resize(N);

    x_smooth_[N - 1] = buffer_[N - 1].x_upd;
    P_smooth_[N - 1] = buffer_[N - 1].P_upd;

    for (int j = (int)N - 2; j >= 0; --j) {
        const Eigen::VectorXf& x_j_filt = buffer_[j].x_upd;
        const Eigen::MatrixXf& P_j_filt = buffer_[j].P_upd;

        const Eigen::MatrixXf& F_j_plus_1 = buffer_[j + 1].F;
        const Eigen::VectorXf& x_j_plus_1_pred = buffer_[j + 1].x_pred;
        const Eigen::MatrixXf& P_j_plus_1_pred = buffer_[j + 1].P_pred;

        // Smoothing Gain: G_j = P_{j|j} * F_{j+1}^T * P_{j+1|j}^{-1}
        // Use SPD solve instead of explicit inverse
        Eigen::MatrixXf PFt = filtermath::gemm(P_j_filt, F_j_plus_1.transpose());
        Eigen::MatrixXf G_j = filtermath::kalman_gain(PFt, P_j_plus_1_pred);

        // Smoothed state
        Eigen::VectorXf diff_x = x_smooth_[j + 1] - x_j_plus_1_pred;
        x_smooth_[j] = x_j_filt + filtermath::mat_vec_mul(G_j, diff_x);

        // Smoothed covariance
        Eigen::MatrixXf diff_P = P_smooth_[j + 1] - P_j_plus_1_pred;
        Eigen::MatrixXf term1 = filtermath::gemm(G_j, diff_P);
        P_smooth_[j] = P_j_filt + filtermath::gemm(term1, G_j.transpose());
    }
}

std::pair<Eigen::VectorXf, Eigen::MatrixXf> EKFFixedLag::getFilteredState() const {
    if (buffer_.empty()) {
        return {Eigen::VectorXf(), Eigen::MatrixXf()};
    }
    return {buffer_.back().x_upd, buffer_.back().P_upd};
}

std::pair<Eigen::VectorXf, Eigen::MatrixXf> EKFFixedLag::getSmoothedState(int lag) const {
    if (x_smooth_.empty() || P_smooth_.empty()) {
        return {buffer_.back().x_upd, buffer_.back().P_upd};
    }

    size_t N = buffer_.size();
    if (lag < 0 || lag >= (int)N) {
        if (lag >= (int)N) lag = (int)N - 1;
        else lag = 0;
    }

    size_t index = N - 1 - lag;
    return {x_smooth_[index], P_smooth_[index]};
}
