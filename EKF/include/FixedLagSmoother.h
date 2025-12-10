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
 */
class FixedLagSmoother {
public:
    FixedLagSmoother(SystemModel* model, const Eigen::VectorXd& x0, const Eigen::MatrixXd& P0, int lag);

    /**
     * Process a new measurement y.
     * x_smooth_out, P_smooth_out: The smoothed estimate at T-lag.
     * x_filt_out, P_filt_out: The filtered estimate at T-lag (before current smoothing pass).
     */
    bool process(const Eigen::VectorXd& y,
                 Eigen::VectorXd& x_smooth_out, Eigen::MatrixXd& P_smooth_out,
                 Eigen::VectorXd& x_filt_out, Eigen::MatrixXd& P_filt_out);

    // Overload for backward compatibility if needed, but we will update call sites.

    bool flush(Eigen::VectorXd& x_out, Eigen::MatrixXd& P_out);

private:
    SystemModel* model_;
    EKF ekf_;
    int lag_;

    // Buffer for RTS smoothing
    std::deque<FilterState> buffer_;
};

#endif // FIXED_LAG_SMOOTHER_H
