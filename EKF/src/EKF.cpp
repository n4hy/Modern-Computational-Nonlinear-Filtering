#include "EKF.h"
#include <iostream>

EKF::EKF(SystemModel* model, const Eigen::VectorXd& x0, const Eigen::MatrixXd& P0)
    : model_(model), x_(x0), P_(P0) {
    I_ = Eigen::MatrixXd::Identity(x0.size(), x0.size());
    // Initialize predicted values to initial state for safety
    x_pred_ = x0;
    P_pred_ = P0;
}

Eigen::MatrixXd EKF::predict(const Eigen::VectorXd& u, double t) {
    // 1. Get Jacobian F at current state x_{k-1|k-1}
    // F_k = df/dx | x_{k-1}, u_k, t_k
    Eigen::MatrixXd F = model_->F(x_, u, t);

    // 2. Predict State
    // x_{k|k-1} = f(x_{k-1|k-1}, u_k, t_k)
    x_pred_ = model_->f(x_, u, t);

    // 3. Predict Covariance
    // P_{k|k-1} = F_k * P_{k-1|k-1} * F_k^T + Q_k
    Eigen::MatrixXd Q = model_->Q(t);
    P_pred_ = F * P_ * F.transpose() + Q;

    // Symmetrize
    P_pred_ = 0.5 * (P_pred_ + P_pred_.transpose());

    // Update internal state to predicted state (until update step confirms it)
    // Note: Standard EKF usage usually separates these, but often implementations
    // update x_ to x_pred_ here. However, for Joseph form in update(), we need P_pred_.
    // We will keep x_ and P_ as "current best estimate". After predict, that is x_pred_.
    x_ = x_pred_;
    P_ = P_pred_;

    return F;
}

void EKF::update(const Eigen::VectorXd& y, double t) {
    // 1. Get Jacobian H at predicted state x_{k|k-1}
    Eigen::MatrixXd H = model_->H(x_, t);

    // 2. Innovation
    Eigen::VectorXd y_pred = model_->h(x_, t);
    Eigen::VectorXd innov = y - y_pred;

    // 3. Innovation Covariance
    Eigen::MatrixXd R = model_->R(t);
    Eigen::MatrixXd S = H * P_ * H.transpose() + R;

    // 4. Kalman Gain: K = P * H^T * S^-1
    // Use robust decomposition
    Eigen::MatrixXd K = P_ * H.transpose() * S.completeOrthogonalDecomposition().pseudoInverse();
    // Alternatively for SPD S: S.llt().solve(H * P_).transpose() (if dimensions match)

    // 5. Update State
    x_ = x_ + K * innov;

    // 6. Update Covariance (Joseph form)
    // P_{k|k} = (I - K*H) * P_{k|k-1} * (I - K*H)^T + K * R * K^T
    Eigen::MatrixXd I_KH = I_ - K * H;
    P_ = I_KH * P_ * I_KH.transpose() + K * R * K.transpose();

    // Symmetrize
    P_ = 0.5 * (P_ + P_.transpose());
}
