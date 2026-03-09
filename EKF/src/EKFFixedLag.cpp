#include "EKFFixedLag.h"
#include <iostream>
#include <optmath/neon_kernels.hpp>

EKFFixedLag::EKFFixedLag(SystemModel* model, const Eigen::VectorXf& x0, const Eigen::MatrixXf& P0, int lag_L)
    : model_(model), ekf_(model, x0, P0), lag_L_(lag_L) {

    // Initialize buffer with initial state (at implicit time t0)
    // We treat the initial state as a "filtered" state.
    FilterState init_state;
    init_state.t = 0.0f; // Assume 0, or we could ask for t0
    init_state.x_upd = x0;
    init_state.P_upd = P0;
    // x_pred, P_pred, F are not really defined for k=0 (initial),
    // but they are needed for smoothing k-1 -> k.
    // For the very first element (k=0), these won't be used as "next" from k=-1.
    // But they might be used if we smoothed k=0 from k=1.

    buffer_.push_back(init_state);
}

void EKFFixedLag::step(const Eigen::VectorXf& y_k, const Eigen::VectorXf& u_k, float t_k) {
    // 1. EKF Prediction
    // predict() returns F used for this step (transition from k-1 to k)
    // and updates internal state to x_{k|k-1}, P_{k|k-1}
    Eigen::MatrixXf F_k = ekf_.predict(u_k, t_k);

    // Capture Predicted State
    Eigen::VectorXf x_pred = ekf_.getPredictedState();
    Eigen::MatrixXf P_pred = ekf_.getPredictedCovariance();

    // 2. EKF Update
    ekf_.update(y_k, t_k);

    // Capture Filtered State
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
        const Eigen::VectorXf& x_j_filt = buffer_[j].x_upd;
        const Eigen::MatrixXf& P_j_filt = buffer_[j].P_upd;

        // Transition from j to j+1
        const Eigen::MatrixXf& F_j_plus_1 = buffer_[j + 1].F;
        const Eigen::VectorXf& x_j_plus_1_pred = buffer_[j + 1].x_pred;
        const Eigen::MatrixXf& P_j_plus_1_pred = buffer_[j + 1].P_pred;

        // Smoothing Gain G_j = P_{j|j} * F_{j+1}^T * P_{j+1|j}^-1
        Eigen::MatrixXf P_pred_inv = optmath::neon::neon_inverse(P_j_plus_1_pred);
        Eigen::MatrixXf PFt = optmath::neon::neon_gemm(P_j_filt, F_j_plus_1.transpose());
        Eigen::MatrixXf G_j;
        if (P_pred_inv.size() > 0) {
            G_j = optmath::neon::neon_gemm(PFt, P_pred_inv);
        } else {
            Eigen::LDLT<Eigen::MatrixXf> ldlt(P_j_plus_1_pred);
            G_j = PFt * ldlt.solve(Eigen::MatrixXf::Identity(P_j_plus_1_pred.rows(), P_j_plus_1_pred.cols()));
        }

        // Smoothed state: x_{j|k} = x_{j|j} + G_j * (x_{j+1|k} - x_{j+1|j})
        Eigen::VectorXf diff_x = x_smooth_[j + 1] - x_j_plus_1_pred;
        x_smooth_[j] = x_j_filt + optmath::neon::neon_mat_vec_mul(G_j, diff_x);

        // Smoothed covariance: P_{j|k} = P_{j|j} + G_j * (P_{j+1|k} - P_{j+1|j}) * G_j^T
        Eigen::MatrixXf diff_P = P_smooth_[j + 1] - P_j_plus_1_pred;
        Eigen::MatrixXf term1 = optmath::neon::neon_gemm(G_j, diff_P);
        P_smooth_[j] = P_j_filt + optmath::neon::neon_gemm(term1, G_j.transpose());
    }
}

std::pair<Eigen::VectorXf, Eigen::MatrixXf> EKFFixedLag::getFilteredState() const {
    if (buffer_.empty()) {
        // Should not happen after initialization
        return {Eigen::VectorXf(), Eigen::MatrixXf()};
    }
    return {buffer_.back().x_upd, buffer_.back().P_upd};
}

std::pair<Eigen::VectorXf, Eigen::MatrixXf> EKFFixedLag::getSmoothedState(int lag) const {
    // lag 0 means current time k (index N-1)
    // lag L means time k-L (index N-1-L)
    // We check if we have enough history

    // Check if smoothing was performed
    if (x_smooth_.empty() || P_smooth_.empty()) {
        // Return filtered state as fallback
        return {buffer_.back().x_upd, buffer_.back().P_upd};
    }

    size_t N = buffer_.size();
    if (lag < 0 || lag >= (int)N) {
        // Requested lag outside available window
        // Return the oldest available smoothed state if lag is too large,
        // or current if lag is too small.
        if (lag >= (int)N) lag = (int)N - 1;
        else lag = 0;
    }

    size_t index = N - 1 - lag;
    return {x_smooth_[index], P_smooth_[index]};
}
