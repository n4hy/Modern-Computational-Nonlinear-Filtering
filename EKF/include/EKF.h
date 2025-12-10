#ifndef EKF_H
#define EKF_H

#include <Eigen/Dense>
#include "SystemModel.h"

/**
 * Extended Kalman Filter (EKF) Implementation.
 */
class EKF {
public:
    EKF(SystemModel* model, const Eigen::VectorXd& x0, const Eigen::MatrixXd& P0);

    // Predict step: x_k|k-1, P_k|k-1
    void predict();

    // Update step: x_k|k, P_k|k
    void update(const Eigen::VectorXd& y);

    // Getters
    Eigen::VectorXd getState() const { return x_; }
    Eigen::MatrixXd getCovariance() const { return P_; }

    // Setters (useful for re-filtering/feedback)
    void setState(const Eigen::VectorXd& x) { x_ = x; }
    void setCovariance(const Eigen::MatrixXd& P) { P_ = P; }

private:
    SystemModel* model_;
    Eigen::VectorXd x_; // State estimate
    Eigen::MatrixXd P_; // State covariance
    Eigen::MatrixXd I_; // Identity matrix
};

#endif // EKF_H
