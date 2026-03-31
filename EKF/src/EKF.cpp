#include "EKF.h"
#include <iostream>
#include "FilterMath.h"

EKF::EKF(SystemModel* model, const Eigen::VectorXf& x0, const Eigen::MatrixXf& P0)
    : model_(model), x_(x0), P_(P0) {
    I_ = Eigen::MatrixXf::Identity(x0.size(), x0.size());
    // Initialize predicted values to initial state for safety
    x_pred_ = x0;
    P_pred_ = P0;
}

Eigen::MatrixXf EKF::predict(const Eigen::VectorXf& u, float t) {
    // 1. Get Jacobian F at current state x_{k-1|k-1}
    Eigen::MatrixXf F = model_->F(x_, u, t);

    // 2. Predict State
    x_pred_ = model_->f(x_, u, t);

    // 3. Predict Covariance: P_{k|k-1} = F * P_{k-1|k-1} * F^T + Q
    Eigen::MatrixXf Q = model_->Q(t);

    Eigen::MatrixXf FP = filtermath::gemm(F, P_);
    Eigen::MatrixXf P_temp = filtermath::gemm(FP, F.transpose());

    P_pred_ = P_temp + Q;

    // Symmetrize
    P_pred_ = 0.5f * (P_pred_ + P_pred_.transpose());

    // Update internal state
    x_ = x_pred_;
    P_ = P_pred_;

    return F;
}

void EKF::update(const Eigen::VectorXf& y, float t) {
    // 1. Get Jacobian H at predicted state
    Eigen::MatrixXf H = model_->H(x_, t);

    // 2. Innovation
    Eigen::VectorXf y_pred = model_->h(x_, t);
    Eigen::VectorXf innov = y - y_pred;

    // 3. Innovation Covariance: S = H * P * H^T + R
    Eigen::MatrixXf R = model_->R(t);
    Eigen::MatrixXf HP = filtermath::gemm(H, P_);
    Eigen::MatrixXf S = filtermath::gemm(HP, H.transpose()) + R;

    // 4. Kalman Gain via SPD solve (more stable than explicit inverse)
    Eigen::MatrixXf PHt = filtermath::gemm(P_, H.transpose());
    Eigen::MatrixXf K = filtermath::kalman_gain(PHt, S);

    // 5. Update State
    x_ = x_ + K * innov;

    // 6. Update Covariance (Joseph form for numerical stability)
    // P_{k|k} = (I - K*H) * P_{k|k-1} * (I - K*H)^T + K * R * K^T
    Eigen::MatrixXf KH = filtermath::gemm(K, H);
    Eigen::MatrixXf I_KH = I_ - KH;

    Eigen::MatrixXf P_I_KH_T = filtermath::gemm(P_, I_KH.transpose());
    Eigen::MatrixXf Term1 = filtermath::gemm(I_KH, P_I_KH_T);

    Eigen::MatrixXf RKt = filtermath::gemm(R, K.transpose());
    Eigen::MatrixXf Term2 = filtermath::gemm(K, RKt);

    P_ = Term1 + Term2;

    // Symmetrize
    P_ = 0.5f * (P_ + P_.transpose());
}
