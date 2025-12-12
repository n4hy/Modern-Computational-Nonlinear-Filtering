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
    EKF(SystemModel* model, const Eigen::VectorXf& x0, const Eigen::MatrixXf& P0);

    // Predict step: x_k|k-1, P_k|k-1
    // Returns the Jacobian F used for prediction (useful for smoothing)
    Eigen::MatrixXf predict(const Eigen::VectorXf& u, float t);

    // Update step: x_k|k, P_k|k
    void update(const Eigen::VectorXf& y, float t);

    // Getters
    Eigen::VectorXf getState() const { return x_; }
    Eigen::MatrixXf getCovariance() const { return P_; }

    // Getters for predicted state (needed for smoothing)
    Eigen::VectorXf getPredictedState() const { return x_pred_; }
    Eigen::MatrixXf getPredictedCovariance() const { return P_pred_; }

    // Setters (useful for re-filtering/feedback or initialization)
    void setState(const Eigen::VectorXf& x) { x_ = x; }
    void setCovariance(const Eigen::MatrixXf& P) { P_ = P; }

private:
    SystemModel* model_;
    Eigen::VectorXf x_;      // State estimate x_{k|k} or x_{k-1|k-1}
    Eigen::MatrixXf P_;      // State covariance P_{k|k} or P_{k-1|k-1}

    Eigen::VectorXf x_pred_; // Predicted state x_{k|k-1}
    Eigen::MatrixXf P_pred_; // Predicted covariance P_{k|k-1}

    Eigen::MatrixXf I_;      // Identity matrix
};

#endif // EKF_H
