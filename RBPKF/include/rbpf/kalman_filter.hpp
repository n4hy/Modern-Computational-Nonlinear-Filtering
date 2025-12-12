#ifndef RBPF_KALMAN_FILTER_HPP
#define RBPF_KALMAN_FILTER_HPP

#include <Eigen/Dense>
#include <iostream>
#include <optmath/neon_kernels.hpp>

namespace rbpf {

using namespace optmath::neon;

/**
 * @brief Generic reusable Kalman Filter class.
 *
 * @tparam Types Typdef struct containing dimensions and matrix types.
 */
template<typename Types>
class LinearKalmanFilter {
public:
    using LinearState = typename Types::LinearState;
    using LinearCov   = typename Types::LinearCov;
    using Observation = typename Types::Observation;
    using ObsCov      = typename Types::ObsCov;

    LinearState x;
    LinearCov   P;

    /**
     * @brief Initialize the filter state and covariance.
     */
    void initialize(const LinearState& x0, const LinearCov& P0) {
        x = x0;
        P = P0;
    }

    /**
     * @brief Predict step: x = A*x + bias, P = A*P*A^T + Q
     *
     * @param A State transition matrix.
     * @param bias Deterministic input/bias term (B*u + etc).
     * @param Q Process noise covariance.
     */
    void predict(const Eigen::Ref<const Eigen::MatrixXf>& A,
                 const Eigen::Ref<const LinearState>& bias,
                 const LinearCov& Q) {
        // x_k|k-1 = A * x_{k-1|k-1} + bias
        // Use NEON GEMM for A*x (Matrix * Vector)
        // A is Nlin x Nlin, x is Nlin x 1.
        Eigen::MatrixXf Ax = optmath::neon::neon_gemm(A, x);
        x = Ax + bias;

        // P_k|k-1 = A * P_{k-1|k-1} * A^T + Q
        // AP = A * P
        Eigen::MatrixXf AP = optmath::neon::neon_gemm(A, P);
        // APAt = AP * A^T
        Eigen::MatrixXf APAt = optmath::neon::neon_gemm(AP, A.transpose());

        P = APAt + Q;

        // Ensure symmetry
        P = 0.5f * (P + P.transpose());
    }

    /**
     * @brief Update step using Joseph form for stability.
     *
     * @param y Measurement vector.
     * @param H Observation matrix.
     * @param offset Measurement offset (y = Hx + offset + v).
     * @param R Measurement noise covariance.
     */
    void update(const Eigen::Ref<const Observation>& y,
                const Eigen::Ref<const Eigen::MatrixXf>& H,
                const Eigen::Ref<const Observation>& offset,
                const ObsCov& R) {
        // Innovation: z = y - (H*x + offset)
        Eigen::MatrixXf Hx = optmath::neon::neon_gemm(H, x);
        Observation z = y - (Hx + offset);

        // Innovation covariance: S = H*P*H^T + R
        Eigen::MatrixXf HP = optmath::neon::neon_gemm(H, P);
        Eigen::MatrixXf HPHt = optmath::neon::neon_gemm(HP, H.transpose());
        ObsCov S = HPHt + R;

        // Kalman gain: K = P * H^T * S^{-1}
        // Using LDLT for inverse
        Eigen::Matrix<float, Types::Nlin, Types::Ny> K;
        K = P * H.transpose() * S.ldlt().solve(Eigen::Matrix<float, Types::Ny, Types::Ny>::Identity());

        // Update state: x = x + K*z
        Eigen::MatrixXf Kz = optmath::neon::neon_gemm(K, z);
        x = x + Kz;

        // Update covariance (Joseph form): P = (I - KH)P(I - KH)^T + KRK^T
        Eigen::Matrix<float, Types::Nlin, Types::Nlin> I = Eigen::Matrix<float, Types::Nlin, Types::Nlin>::Identity();

        Eigen::MatrixXf KH = optmath::neon::neon_gemm(K, H);
        Eigen::Matrix<float, Types::Nlin, Types::Nlin> I_KH = I - KH;

        // Term 1: (I-KH) * P * (I-KH)^T
        Eigen::MatrixXf P_I_KH_T = optmath::neon::neon_gemm(P, I_KH.transpose());
        Eigen::MatrixXf Term1 = optmath::neon::neon_gemm(I_KH, P_I_KH_T);

        // Term 2: K * R * K^T
        Eigen::MatrixXf RKt = optmath::neon::neon_gemm(R, K.transpose());
        Eigen::MatrixXf Term2 = optmath::neon::neon_gemm(K, RKt);

        P = Term1 + Term2;

        // Ensure symmetry
        P = 0.5f * (P + P.transpose());
    }
};

} // namespace rbpf

#endif // RBPF_KALMAN_FILTER_HPP
