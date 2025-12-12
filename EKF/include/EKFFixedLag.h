#ifndef EKF_FIXED_LAG_H
#define EKF_FIXED_LAG_H

#include <vector>
#include <deque>
#include <Eigen/Dense>
#include "EKF.h"
#include "SystemModel.h"

/**
 * Structure to hold historical filter state for smoothing.
 */
struct FilterState {
    float t;                 // Time t_k
    Eigen::VectorXf x_pred;  // x_{k|k-1}
    Eigen::MatrixXf P_pred;  // P_{k|k-1}
    Eigen::VectorXf x_upd;   // x_{k|k} (filtered)
    Eigen::MatrixXf P_upd;   // P_{k|k} (filtered)
    Eigen::MatrixXf F;       // F_k (Jacobian used to predict x_{k|k-1} from x_{k-1|k-1})
};

/**
 * Fixed Lag Smoother (Windowed RTS Smoother).
 *
 * Maintains a window of size L. At each step:
 * 1. Runs EKF prediction and update.
 * 2. Stores the result in the window.
 * 3. Runs a backward RTS smoothing pass over the window.
 * 4. Provides smoothed estimate at lag L.
 */
class EKFFixedLag {
public:
    EKFFixedLag(SystemModel* model, const Eigen::VectorXf& x0, const Eigen::MatrixXf& P0, int lag_L);

    /**
     * Perform one time step.
     * @param y_k Measurement at current time
     * @param u_k Control input at current time
     * @param t_k Current time
     */
    void step(const Eigen::VectorXf& y_k, const Eigen::VectorXf& u_k, float t_k);

    /**
     * Get the filtered state at current time k (x_{k|k}).
     */
    std::pair<Eigen::VectorXf, Eigen::MatrixXf> getFilteredState() const;

    /**
     * Get the smoothed state at current time k (x_{k|k}).
     * This is the result of the backward pass starting at k, so it's equal to filtered at k,
     * but smoothed for previous steps.
     */
    std::pair<Eigen::VectorXf, Eigen::MatrixXf> getSmoothedState(int lag) const;

private:
    SystemModel* model_;
    EKF ekf_;
    int lag_L_;

    // Buffer: Index 0 is oldest (k-L), Index back is newest (k)
    std::deque<FilterState> buffer_;

    // Smoothed estimates cache (recomputed each step)
    // Organized same as buffer_: Index 0 corresponds to buffer_[0] (k-L)
    std::vector<Eigen::VectorXf> x_smooth_;
    std::vector<Eigen::MatrixXf> P_smooth_;
};

#endif // EKF_FIXED_LAG_H
