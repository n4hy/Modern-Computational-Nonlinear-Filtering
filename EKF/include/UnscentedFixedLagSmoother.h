#ifndef UNSCENTED_FIXED_LAG_SMOOTHER_H
#define UNSCENTED_FIXED_LAG_SMOOTHER_H

#include <vector>
#include <deque>
#include <Eigen/Dense>
#include "UKF.h"
#include "SystemModel.h"

/**
 * Structure to hold historical filter state for UKF Smoothing.
 */
struct UKFFilterState {
    Eigen::VectorXd x_pred; // x_k|k-1
    Eigen::MatrixXd P_pred; // P_k|k-1
    Eigen::VectorXd x_upd;  // x_k|k
    Eigen::MatrixXd P_upd;  // P_k|k
    Eigen::VectorXd y_meas; // y_k (measurement at this step)
    Eigen::MatrixXd P_cross; // Cross covariance P_{x_{k-1}, x_k} needed for RTS
};

/**
 * Fixed Lag Smoother using UKF/URTS.
 */
class UnscentedFixedLagSmoother {
public:
    UnscentedFixedLagSmoother(SystemModel* model, const Eigen::VectorXd& x0, const Eigen::MatrixXd& P0, int lag);

    bool process(const Eigen::VectorXd& y, Eigen::VectorXd& x_out, Eigen::MatrixXd& P_out);
    bool flush(Eigen::VectorXd& x_out, Eigen::MatrixXd& P_out);

private:
    SystemModel* model_;
    UKF ukf_;
    int lag_;

    std::deque<UKFFilterState> buffer_;

    void smooth_step(std::vector<Eigen::VectorXd>& x_smooth, std::vector<Eigen::MatrixXd>& P_smooth, int i);
};

#endif // UNSCENTED_FIXED_LAG_SMOOTHER_H
