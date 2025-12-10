#ifndef UKF_H
#define UKF_H

#include <Eigen/Dense>
#include <vector>
#include "SystemModel.h"

/**
 * Unscented Kalman Filter (UKF) Implementation.
 * Uses Merwe Scaled Sigma Points.
 */
class UKF {
public:
    UKF(SystemModel* model, const Eigen::VectorXd& x0, const Eigen::MatrixXd& P0);

    // Predict step
    // Returns the cross-covariance P_{x_k, x_{k+1}} needed for smoothing
    Eigen::MatrixXd predict();

    // Update step
    void update(const Eigen::VectorXd& y);

    // Getters
    Eigen::VectorXd getState() const { return x_; }
    Eigen::MatrixXd getCovariance() const { return P_; }

    // Setters
    void setState(const Eigen::VectorXd& x) { x_ = x; }
    void setCovariance(const Eigen::MatrixXd& P) { P_ = P; }

    // Constants for Sigma Points
    // Tunable if needed, but defaults provided in constructor
    double alpha_;
    double beta_;
    double kappa_;

private:
    SystemModel* model_;
    Eigen::VectorXd x_; // State estimate
    Eigen::MatrixXd P_; // State covariance
    int n_; // State dimension

    // Sigma Points
    std::vector<Eigen::VectorXd> sigma_points_;
    Eigen::VectorXd weights_m_;
    Eigen::VectorXd weights_c_;

    void generateSigmaPoints(const Eigen::VectorXd& x, const Eigen::MatrixXd& P, std::vector<Eigen::VectorXd>& out_sigmas);
};

#endif // UKF_H
