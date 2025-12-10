#ifndef EKF_H
#define EKF_H

#include <Eigen/Dense>
#include "SystemModel.h"

/**
 * Extended Kalman Filter (EKF) Implementation.
 *
 * Implements a robust EKF with Joseph form covariance update and support
 * for time-varying systems and control inputs.
 */
class EKF {
public:
    EKF(SystemModel* model, const Eigen::VectorXd& x0, const Eigen::MatrixXd& P0);

    // Predict step: x_k|k-1, P_k|k-1
    // Returns the Jacobian F used for prediction (useful for smoothing)
    Eigen::MatrixXd predict(const Eigen::VectorXd& u, double t);

    // Update step: x_k|k, P_k|k
    void update(const Eigen::VectorXd& y, double t);

    // Getters
    Eigen::VectorXd getState() const { return x_; }
    Eigen::MatrixXd getCovariance() const { return P_; }

    // Getters for predicted state (needed for smoothing)
    Eigen::VectorXd getPredictedState() const { return x_pred_; }
    Eigen::MatrixXd getPredictedCovariance() const { return P_pred_; }

    // Setters (useful for re-filtering/feedback or initialization)
    void setState(const Eigen::VectorXd& x) { x_ = x; }
    void setCovariance(const Eigen::MatrixXd& P) { P_ = P; }

private:
    SystemModel* model_;
    Eigen::VectorXd x_;      // State estimate x_{k|k} or x_{k-1|k-1}
    Eigen::MatrixXd P_;      // State covariance P_{k|k} or P_{k-1|k-1}

    Eigen::VectorXd x_pred_; // Predicted state x_{k|k-1}
    Eigen::MatrixXd P_pred_; // Predicted covariance P_{k|k-1}

    Eigen::MatrixXd I_;      // Identity matrix
};

#endif // EKF_H
