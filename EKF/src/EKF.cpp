#include "EKF.h"
#include <iostream>

EKF::EKF(SystemModel* model, const Eigen::VectorXd& x0, const Eigen::MatrixXd& P0)
    : model_(model), x_(x0), P_(P0) {
    I_ = Eigen::MatrixXd::Identity(x0.size(), x0.size());
}

void EKF::predict() {
    // 1. Get Jacobian F at current state
    Eigen::MatrixXd F = model_->F(x_);

    // 2. Predict State
    x_ = model_->f(x_);

    // 3. Predict Covariance
    P_ = F * P_ * F.transpose() + model_->Q();
}

void EKF::update(const Eigen::VectorXd& y) {
    // 1. Get Jacobian H at current state
    Eigen::MatrixXd H = model_->H(x_);

    // 2. Innovation
    Eigen::VectorXd y_pred = model_->h(x_);
    Eigen::VectorXd innov = y - y_pred;

    // 3. Innovation Covariance
    Eigen::MatrixXd S = H * P_ * H.transpose() + model_->R();

    // 4. Kalman Gain
    Eigen::MatrixXd K = P_ * H.transpose() * S.inverse();

    // 5. Update State
    x_ = x_ + K * innov;

    // 6. Update Covariance
    P_ = (I_ - K * H) * P_;
}
