#include "EKF.h"
#include <iostream>
#include <optmath/neon_kernels.hpp>

EKF::EKF(SystemModel* model, const Eigen::VectorXf& x0, const Eigen::MatrixXf& P0)
    : model_(model), x_(x0), P_(P0) {
    I_ = Eigen::MatrixXf::Identity(x0.size(), x0.size());
    // Initialize predicted values to initial state for safety
    x_pred_ = x0;
    P_pred_ = P0;
}

Eigen::MatrixXf EKF::predict(const Eigen::VectorXf& u, float t) {
    // 1. Get Jacobian F at current state x_{k-1|k-1}
    // F_k = df/dx | x_{k-1}, u_k, t_k
    Eigen::MatrixXf F = model_->F(x_, u, t);

    // 2. Predict State
    // x_{k|k-1} = f(x_{k-1|k-1}, u_k, t_k)
    x_pred_ = model_->f(x_, u, t);

    // 3. Predict Covariance
    // P_{k|k-1} = F_k * P_{k-1|k-1} * F_k^T + Q_k
    Eigen::MatrixXf Q = model_->Q(t);

    // Use NEON GEMM: FP = F * P
    Eigen::MatrixXf FP = optmath::neon::neon_gemm(F, P_);

    // P_temp = FP * F^T
    Eigen::MatrixXf Ft = F.transpose();
    Eigen::MatrixXf P_temp = optmath::neon::neon_gemm(FP, Ft);

    P_pred_ = P_temp + Q;

    // Symmetrize
    P_pred_ = 0.5f * (P_pred_ + P_pred_.transpose());

    // Update internal state
    x_ = x_pred_;
    P_ = P_pred_;

    return F;
}

void EKF::update(const Eigen::VectorXf& y, float t) {
    // 1. Get Jacobian H at predicted state x_{k|k-1}
    Eigen::MatrixXf H = model_->H(x_, t);

    // 2. Innovation
    Eigen::VectorXf y_pred = model_->h(x_, t);
    Eigen::VectorXf innov = y - y_pred;

    // 3. Innovation Covariance
    Eigen::MatrixXf R = model_->R(t);
    // S = H * P * H^T + R
    Eigen::MatrixXf HP = optmath::neon::neon_gemm(H, P_);
    Eigen::MatrixXf Ht = H.transpose();
    Eigen::MatrixXf S_part = optmath::neon::neon_gemm(HP, Ht);
    Eigen::MatrixXf S = S_part + R;

    // 4. Kalman Gain: K = P * H^T * S^-1
    // Use robust decomposition. S inversion is not optimized by our kernels yet.
    Eigen::MatrixXf K = P_ * H.transpose() * S.completeOrthogonalDecomposition().pseudoInverse();

    // 5. Update State
    // x_ = x_ + K * innov
    // x_ = x_ + K * innov;
    Eigen::VectorXf correction = optmath::neon::neon_mat_vec_mul(K, innov);
    x_ = optmath::neon::neon_add(x_, correction);

    // 6. Update Covariance (Joseph form)
    // P_{k|k} = (I - K*H) * P_{k|k-1} * (I - K*H)^T + K * R * K^T

    // I_KH = I - K*H
    Eigen::MatrixXf KH = optmath::neon::neon_gemm(K, H);
    Eigen::MatrixXf I_KH = I_ - KH;

    // Term1 = I_KH * P * I_KH^T
    Eigen::MatrixXf P_I_KH_T = optmath::neon::neon_gemm(P_, I_KH.transpose()); // P * (I-KH)^T
    Eigen::MatrixXf Term1 = optmath::neon::neon_gemm(I_KH, P_I_KH_T); // (I-KH) * (P * (I-KH)^T)

    // Term2 = K * R * K^T
    Eigen::MatrixXf RKt = optmath::neon::neon_gemm(R, K.transpose());
    Eigen::MatrixXf Term2 = optmath::neon::neon_gemm(K, RKt);

    P_ = Term1 + Term2;

    // Symmetrize
    P_ = 0.5f * (P_ + P_.transpose());
}
