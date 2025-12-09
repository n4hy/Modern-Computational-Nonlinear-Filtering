#ifndef FIXED_LAG_SMOOTHER_H
#define FIXED_LAG_SMOOTHER_H

#include <vector>
#include <deque>
#include <Eigen/Dense>
#include "EKF.h"
#include "SystemModel.h"

/**
 * Structure to hold historical filter state.
 */
struct FilterState {
    Eigen::VectorXd x_pred; // x_k|k-1
    Eigen::MatrixXd P_pred; // P_k|k-1
    Eigen::VectorXd x_upd;  // x_k|k
    Eigen::MatrixXd P_upd;  // P_k|k
    Eigen::VectorXd y_meas; // y_k (measurement at this step)
    Eigen::MatrixXd F;      // Jacobian F at k-1 (used to predict k)
};

/**
 * Fixed Lag Smoother with Feedback.
 *
 * Logic:
 * 1. Forward Filter advances to time T.
 * 2. If buffer size >= lag:
 *    a. Perform RTS smoothing backwards to T-lag.
 *    b. Extract smoothed state x_smooth at T-lag.
 *    c. FEEDBACK: Reset filter at T-lag to x_smooth.
 *    d. Re-filter from T-lag to T using stored measurements.
 * 3. Return the fully processed state at T-lag (smoothed).
 */
class FixedLagSmoother {
public:
    FixedLagSmoother(SystemModel* model, const Eigen::VectorXd& x0, const Eigen::MatrixXd& P0, int lag);

    /**
     * Process a new measurement y at the next time step.
     * Returns true if a smoothed estimate is available (buffer full or shrinking),
     * false if accumulating buffer.
     *
     * If returns true, x_out and P_out contain the smoothed estimate at output_time_step.
     */
    bool process(const Eigen::VectorXd& y, Eigen::VectorXd& x_out, Eigen::MatrixXd& P_out);

    // To handle end of episode: force reduce lag until 0
    // Call this repeatedly with empty measurement (or handle logic inside) until buffer empty?
    // User logic says: "As near the endpoint time... we shorten the lag until it is zero"
    // So user should probably just keep calling a "flush" method.
    bool flush(Eigen::VectorXd& x_out, Eigen::MatrixXd& P_out);

private:
    SystemModel* model_;
    EKF ekf_;
    int lag_;

    // Buffer for RTS smoothing
    // Index 0 is oldest (T-lag), Index N is current (T)
    std::deque<FilterState> buffer_;

    void smooth_backwards(std::deque<FilterState>& buf, int start_idx, int end_idx);
};

#endif // FIXED_LAG_SMOOTHER_H
