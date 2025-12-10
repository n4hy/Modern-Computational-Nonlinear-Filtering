#include "EKFFixedLag.h"
#include <iostream>

EKFFixedLag::EKFFixedLag(SystemModel* model, const Eigen::VectorXd& x0, const Eigen::MatrixXd& P0, int lag_L)
    : model_(model), ekf_(model, x0, P0), lag_L_(lag_L) {

    // Initialize buffer with initial state (at implicit time t0)
    // We treat the initial state as a "filtered" state.
    FilterState init_state;
    init_state.t = 0.0; // Assume 0, or we could ask for t0
    init_state.x_upd = x0;
    init_state.P_upd = P0;
    // x_pred, P_pred, F are not really defined for k=0 (initial),
    // but they are needed for smoothing k-1 -> k.
    // For the very first element (k=0), these won't be used as "next" from k=-1.
    // But they might be used if we smoothed k=0 from k=1.

    buffer_.push_back(init_state);
}

void EKFFixedLag::step(const Eigen::VectorXd& y_k, const Eigen::VectorXd& u_k, double t_k) {
    // 1. EKF Prediction
    // predict() returns F used for this step (transition from k-1 to k)
    // and updates internal state to x_{k|k-1}, P_{k|k-1}
    Eigen::MatrixXd F_k = ekf_.predict(u_k, t_k);

    // Capture Predicted State
    Eigen::VectorXd x_pred = ekf_.getPredictedState();
    Eigen::MatrixXd P_pred = ekf_.getPredictedCovariance();

    // 2. EKF Update
    ekf_.update(y_k, t_k);

    // Capture Filtered State
    Eigen::VectorXd x_upd = ekf_.getState();
    Eigen::MatrixXd P_upd = ekf_.getCovariance();

    // 3. Store in Buffer
    FilterState state;
    state.t = t_k;
    state.x_pred = x_pred;
    state.P_pred = P_pred;
    state.x_upd = x_upd;
    state.P_upd = P_upd;
    state.F = F_k;

    buffer_.push_back(state);

    // Maintain window size.
    // We need at least 2 elements to smooth.
    // If buffer grows larger than lag_L + 1 (meaning we have k, k-1, ... k-L), we pop.
    // The requirement is "approximation to p(x_{k-L} | y_1...y_k)".
    // So we need history back to k-L.
    // buffer size = lag_L + 1 means indices 0..L. Index L is k, Index 0 is k-L.
    if (buffer_.size() > (size_t)(lag_L_ + 1)) {
        buffer_.pop_front();
    }

    // 4. Backward Smoothing Recursion (RTS)
    size_t N = buffer_.size();
    x_smooth_.resize(N);
    P_smooth_.resize(N);

    // Initialization at final time k
    x_smooth_[N - 1] = buffer_[N - 1].x_upd;
    P_smooth_[N - 1] = buffer_[N - 1].P_upd;

    // Backward pass
    for (int j = (int)N - 2; j >= 0; --j) {
        // We are smoothing step j using step j+1
        const Eigen::VectorXd& x_j_filt = buffer_[j].x_upd;
        const Eigen::MatrixXd& P_j_filt = buffer_[j].P_upd;

        // Transition from j to j+1
        const Eigen::MatrixXd& F_j_plus_1 = buffer_[j + 1].F;
        const Eigen::VectorXd& x_j_plus_1_pred = buffer_[j + 1].x_pred;
        const Eigen::MatrixXd& P_j_plus_1_pred = buffer_[j + 1].P_pred;

        // Smoothing Gain G_j = P_{j|j} * F_{j+1}^T * P_{j+1|j}^-1
        // Use robust solve
        Eigen::MatrixXd G_j = P_j_filt * F_j_plus_1.transpose() * P_j_plus_1_pred.completeOrthogonalDecomposition().pseudoInverse();

        // Smoothed state: x_{j|k} = x_{j|j} + G_j * (x_{j+1|k} - x_{j+1|j})
        x_smooth_[j] = x_j_filt + G_j * (x_smooth_[j + 1] - x_j_plus_1_pred);

        // Smoothed covariance: P_{j|k} = P_{j|j} + G_j * (P_{j+1|k} - P_{j+1|j}) * G_j^T
        P_smooth_[j] = P_j_filt + G_j * (P_smooth_[j + 1] - P_j_plus_1_pred) * G_j.transpose();
    }
}

std::pair<Eigen::VectorXd, Eigen::MatrixXd> EKFFixedLag::getFilteredState() const {
    if (buffer_.empty()) {
        // Should not happen after initialization
        return {Eigen::VectorXd(), Eigen::MatrixXd()};
    }
    return {buffer_.back().x_upd, buffer_.back().P_upd};
}

std::pair<Eigen::VectorXd, Eigen::MatrixXd> EKFFixedLag::getSmoothedState(int lag) const {
    // lag 0 means current time k (index N-1)
    // lag L means time k-L (index N-1-L)
    // We check if we have enough history

    size_t N = buffer_.size();
    if (lag < 0 || lag >= (int)N) {
        // Requested lag outside available window
        // Return filtered state as fallback or empty?
        // Let's return the oldest available smoothed state if lag is too large,
        // or current if lag is too small.
        if (lag >= (int)N) lag = (int)N - 1;
        else lag = 0;
    }

    size_t index = N - 1 - lag;
    return {x_smooth_[index], P_smooth_[index]};
}
