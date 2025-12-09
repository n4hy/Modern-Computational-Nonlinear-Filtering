#include "EKF.h"

EKF::EKF(SystemModel* model, const Eigen::VectorXd& x0, const Eigen::MatrixXd& P0)
    : model_(model), x_(x0), P_(P0) {
    int n = model_->getStateDim();
    I_ = Eigen::MatrixXd::Identity(n, n);
}

void EKF::predict() {
    // 1. Predict state: x_pred = f(x)
    Eigen::VectorXd x_pred = model_->f(x_);

    // 2. Compute Jacobian F at current state x
    Eigen::MatrixXd F = model_->F(x_);

    // 3. Predict covariance: P_pred = F * P * F^T + Q
    Eigen::MatrixXd P_pred = F * P_ * F.transpose() + model_->Q();

    // Update internal state
    x_ = x_pred;
    P_ = P_pred;
}

void EKF::update(const Eigen::VectorXd& y) {
    // 1. Compute measurement residual: y - h(x_pred)
    Eigen::VectorXd y_pred = model_->h(x_);
    Eigen::VectorXd y_res = y - y_pred;

    // 2. Compute Jacobian H at predicted state
    Eigen::MatrixXd H = model_->H(x_);

    // 3. Compute Innovation Covariance S = H * P * H^T + R
    Eigen::MatrixXd S = H * P_ * H.transpose() + model_->R();

    // 4. Compute Kalman Gain K = P * H^T * S^-1
    // Using LDLT for robust inverse, though standard inverse is often fine for small systems
    Eigen::MatrixXd K = P_ * H.transpose() * S.inverse();

    // 5. Update State x = x_pred + K * y_res
    x_ = x_ + K * y_res;

    // 6. Update Covariance P = (I - K * H) * P
    // Joseph form is numerically more stable: P = (I - KH)P(I - KH)^T + KRK^T
    // But standard form is usually sufficient for simple EKF tasks.
    // We stick to standard form here as requested for generic implementation unless stability issues arise.
    P_ = (I_ - K * H) * P_;
}
