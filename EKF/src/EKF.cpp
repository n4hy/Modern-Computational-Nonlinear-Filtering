#include "EKF.h"
#include <iostream>
#include <optmath/neon_kernels.hpp>

using namespace optmath::neon;

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
    Eigen::MatrixXf FP = neon_gemm(F, P_);
    // P_temp = FP * F^T
    // Note: F.transpose() creates a temporary or expression. neon_gemm takes const Ref or Matrix.
    // Eigen expressions cast to MatrixXf usually eval.
    Eigen::MatrixXf Ft = F.transpose();
    Eigen::MatrixXf P_temp = neon_gemm(FP, Ft);

    // P_pred_ = P_temp + Q
    // We can use neon_add if we want, or just Eigen add.
    // neon_add returns VectorXf in the header I saw?
    // Header: Eigen::VectorXf neon_add(const Eigen::VectorXf& a, const Eigen::VectorXf& b);
    // It seems neon_add is for Vectors. For Matrices, Eigen default is likely fine or we can cast/map.
    // Let's stick to Eigen + for simplicity unless strictly required to use kernels for EVERYTHING.
    // The user said "optimize every component".
    // I'll stick to GEMM for the heavy lifting.
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
    Eigen::MatrixXf HP = neon_gemm(H, P_);
    Eigen::MatrixXf Ht = H.transpose();
    Eigen::MatrixXf S_part = neon_gemm(HP, Ht);
    Eigen::MatrixXf S = S_part + R;

    // 4. Kalman Gain: K = P * H^T * S^-1
    // Use robust decomposition. S inversion is not optimized by our kernels yet.
    Eigen::MatrixXf Ht_P = P_ * H.transpose(); // Or neon_gemm(P_, Ht)
    // K = Ht_P * S^-1
    // Eigen's solve: S * K^T = H * P
    // K = (S^-1 * H * P)^T ? No.
    // K = P H^T S^-1
    // Let's rely on Eigen for the solve.
    Eigen::MatrixXf K = P_ * H.transpose() * S.completeOrthogonalDecomposition().pseudoInverse();

    // 5. Update State
    // x_ = x_ + K * innov
    // K * innov is Matrix * Vector -> Vector. neon_gemm handles Matrix * Matrix.
    // Does neon_gemm handle Matrix * Vector?
    // Header says: Eigen::MatrixXf neon_gemm(const Eigen::MatrixXf& A, const Eigen::MatrixXf& B);
    // If we pass vector as Nx1 matrix, it works.
    // Eigen::VectorXf is a Matrix<float, Dynamic, 1>.
    // But neon_gemm returns MatrixXf. We might need to assign to VectorXf.
    // Let's try:
    // Eigen::MatrixXf correction_mat = neon_gemm(K, innov); // implicitly treated as matrix?
    // x_ = x_ + correction_mat;
    // Actually, simple vector operations are fast enough, but consistency...
    x_ = x_ + K * innov;

    // 6. Update Covariance (Joseph form)
    // P_{k|k} = (I - K*H) * P_{k|k-1} * (I - K*H)^T + K * R * K^T

    // I_KH = I - K*H
    Eigen::MatrixXf KH = neon_gemm(K, H);
    Eigen::MatrixXf I_KH = I_ - KH;

    // Term1 = I_KH * P * I_KH^T
    Eigen::MatrixXf P_I_KH_T = neon_gemm(P_, I_KH.transpose()); // P * (I-KH)^T
    Eigen::MatrixXf Term1 = neon_gemm(I_KH, P_I_KH_T); // (I-KH) * (P * (I-KH)^T)

    // Term2 = K * R * K^T
    Eigen::MatrixXf RKt = neon_gemm(R, K.transpose());
    Eigen::MatrixXf Term2 = neon_gemm(K, RKt);

    P_ = Term1 + Term2;

    // Symmetrize
    P_ = 0.5f * (P_ + P_.transpose());
}
