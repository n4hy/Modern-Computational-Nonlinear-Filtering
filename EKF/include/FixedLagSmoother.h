#ifndef FIXED_LAG_SMOOTHER_H
#define FIXED_LAG_SMOOTHER_H

#include <vector>
#include <deque>
#include <Eigen/Dense>
#include "EKF.h"
#include "SystemModel.h"

// Note: This struct definition conflicts with EKFFixedLag.h if both included.
// Assuming they are used independently.
struct FilterStateLegacy {
    Eigen::VectorXf x_pred; // x_k|k-1
    Eigen::MatrixXf P_pred; // P_k|k-1
    Eigen::VectorXf x_upd;  // x_k|k
    Eigen::MatrixXf P_upd;  // P_k|k
    Eigen::VectorXf y_meas; // y_k (measurement at this step)
    Eigen::MatrixXf F;      // Jacobian F at k-1 (used to predict k)
};

/**
 * Fixed Lag Smoother with Feedback.
 */
class FixedLagSmoother {
public:
    FixedLagSmoother(SystemModel* model, const Eigen::VectorXf& x0, const Eigen::MatrixXf& P0, int lag);

    /**
     * Process a new measurement y.
     * x_smooth_out, P_smooth_out: The smoothed estimate at T-lag.
     * x_filt_out, P_filt_out: The filtered estimate at T-lag (before current smoothing pass).
     */
    bool process(const Eigen::VectorXf& y,
                 Eigen::VectorXf& x_smooth_out, Eigen::MatrixXf& P_smooth_out,
                 Eigen::VectorXf& x_filt_out, Eigen::MatrixXf& P_filt_out);

    // Overload for backward compatibility if needed, but we will update call sites.

    bool flush(Eigen::VectorXf& x_out, Eigen::MatrixXf& P_out);

private:
    SystemModel* model_;
    EKF ekf_;
    int lag_;

    // Buffer for RTS smoothing
    std::deque<FilterStateLegacy> buffer_;
};

#endif // FIXED_LAG_SMOOTHER_H
